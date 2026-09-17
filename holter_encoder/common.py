# -*- coding: utf-8 -*-
"""common.py — 분산 학습 / 체크포인트 / 로그 / 학습률 스케줄 공통 유틸."""

import csv
import math
import os
import time

import torch
import torch.distributed as dist


def limit_cpu_threads(n=1):
    """워커·랭크가 많을 때 numpy/OpenBLAS 가 프로세스마다 스레드를 띄워 서로 잡아먹는 것을 막는다.
    torch/numpy import 전에 환경변수를 잡아야 효과가 있으므로 학습 스크립트 맨 앞에서 부른다."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(k, str(n))
    torch.set_num_threads(n)


def cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def worker_init(_):
    torch.set_num_threads(1)


def setup_distributed(gpus=None):
    """torchrun 이면 프로세스 그룹을 만들고, 아니면 단일 프로세스. 반환 (rank, world, device).

    gpus: "0,2,3" 처럼 쓸 물리 GPU 번호. 공용 서버에서 남의 작업과 겹치지 않게 고른다.
      - 단일 프로세스: 해당 GPU 들만 보이게 하고 첫 번째를 쓴다
      - torchrun: rank 마다 목록에서 하나씩 맡는다 (nproc_per_node 와 개수를 맞출 것)
    CUDA 컨텍스트가 만들어지기 전에 설정해야 하므로 학습 스크립트 맨 앞에서 호출한다.
    """
    ids = [g.strip() for g in str(gpus).split(",") if g.strip()] if gpus else []
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    local = int(os.environ.get("LOCAL_RANK", 0))
    if ids:
        if torch.cuda.is_initialized():
            raise RuntimeError("--gpus 는 CUDA 초기화 전에 적용해야 합니다 (setup_distributed 를 먼저 호출)")
        os.environ["CUDA_VISIBLE_DEVICES"] = ids[local % len(ids)] if distributed else ",".join(ids)

    if distributed:
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        if ids and world > len(ids):
            print(f"  ** 경고: --gpus 에 {len(ids)}개인데 프로세스는 {world}개 — GPU 를 공유하게 됩니다 **")
        if torch.cuda.is_available():
            # --gpus 를 주면 프로세스마다 GPU 하나만 보이므로 논리 번호는 0
            index = 0 if ids else local
            torch.cuda.set_device(index)
            dist.init_process_group("nccl")
            device = torch.device("cuda", index)
        else:
            dist.init_process_group("gloo")
            device = torch.device("cpu")
        return rank, world, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def describe_device(device):
    if device.type != "cuda":
        return str(device)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "전체")
    return f"{device} ({torch.cuda.get_device_name(device)}, CUDA_VISIBLE_DEVICES={vis})"


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def all_reduce_mean(x, world):
    if world > 1:
        x = x.detach().clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= world
    return x


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(path, model, optimizer, scheduler, step, extra=None):
    """.tmp 에 쓰고 교체 — 저장 중 중단되어도 이전 체크포인트가 살아 있다."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"model": unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict() if optimizer else None,
                "scheduler": scheduler.state_dict() if scheduler else None,
                "step": step, "extra": extra or {}}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    unwrap(model).load_state_dict(ck["model"])
    if optimizer is not None and ck.get("optimizer"):
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and ck.get("scheduler"):
        scheduler.load_state_dict(ck["scheduler"])
    return ck.get("step", 0), ck.get("extra", {})


def cosine_with_warmup(optimizer, warmup, total, min_ratio=0.05):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def param_groups(model, lr, weight_decay, ssm_lr=None):
    """S4 의 dt/A(_optim 표시)는 weight decay 0 + 낮은 LR, bias·norm 도 decay 0."""
    ssm, no_decay, decay = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if hasattr(p, "_optim"):
            ssm.append(p)
        elif p.ndim <= 1 or n.endswith(".bias") or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": weight_decay, "lr": lr},
              {"params": no_decay, "weight_decay": 0.0, "lr": lr}]
    if ssm:
        groups.append({"params": ssm, "weight_decay": 0.0, "lr": ssm_lr or min(lr, 1e-3)})
    return groups


class CSVLogger:
    def __init__(self, path, enabled=True):
        self.path, self.enabled, self.cols = path, enabled, None
        self.t0 = time.time()

    def log(self, row):
        if not self.enabled:
            return
        row = {"time_s": round(time.time() - self.t0, 1), **row}
        new = not os.path.exists(self.path)
        if self.cols is None:
            self.cols = list(row)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.cols, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def autocast(device, enabled=True):
    if device.type == "cuda" and enabled:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)
