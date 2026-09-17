# -*- coding: utf-8 -*-
"""common.py — 분산 학습 / 체크포인트 / 로그 / 학습률 스케줄 공통 유틸."""

import csv
import math
import os
import time

import torch
import torch.distributed as dist


def setup_distributed():
    """torchrun 이면 프로세스 그룹을 만들고, 아니면 단일 프로세스. 반환 (rank, world, device)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            dist.init_process_group("nccl")
            device = torch.device("cuda", local)
        else:
            dist.init_process_group("gloo")
            device = torch.device("cpu")
        return rank, world, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


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
