# -*- coding: utf-8 -*-
"""
finetune.py — 사전학습 인코더를 downstream 라벨로 미세조정한다 (Stage C).

왜 필요한가
  선형 probe 는 record 임베딩(토큰 8,640개의 평균) 위에서만 동작한다. 평균을 내는 순간
  "하루 중 언제 무슨 일이 있었는지" 가 지워져, Stage B backbone 이 만든 시간 문맥이
  성능으로 이어지지 못했다 (stem 평균과 같은 수준: PSVT 0.682 vs 0.662~0.696).
  여기서는 attention pooling 과 분류기를 함께 학습하고, 필요하면 backbone 도 푼다.

  --freeze-backbone : backbone 고정, attention pooling + 분류기만 학습 (가볍고 빠름)
  기본              : backbone 도 함께 학습 (낮은 LR)

  stem 은 항상 고정이다. 입력이 캐시된 토큰이므로 원신호를 다시 읽지 않는다.

평가
  probe.py 와 같은 규칙: record 확률을 환자 단위로 평균, 환자 부트스트랩 95% CI.
  val 의 환자 AUROC 가 가장 좋은 epoch 을 골라 test 를 보고한다.

  python -m holter_encoder.finetune --splits $OUT/splits.csv --tokens $OUT/tokens_a \\
      --encoder $RUN/stage_b_mamba/encoder.pt --task psvt --out $RUN/ft_psvt --gpus 0
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from .common import (CSVLogger, autocast, cosine_with_warmup, describe_device, ensure_kernel_cache,
                     limit_cpu_threads, param_groups, save_checkpoint, setup_distributed, unwrap)
from .data import RawWindowDataset, TokenDataset, load_records
from .embed import build_from_checkpoint

TASK_COL = {"psvt": "y_psvt", "lqt": "y_lqt", "tof": "y_tof"}


def _labeled(records, col):
    return [r for r in records
            if r.get(col) is not None and np.isfinite(float(r[col]))]


class LabeledTokens(Dataset):
    """토큰(또는 원신호)에 라벨을 붙인다. 라벨이 없는 record 는 제외.

    raw_meta_dir 를 주면 원신호 경로로 동작한다 (stem 까지 미세조정할 때).
    """

    def __init__(self, records, token_dir, col, crop, random_crop, seed=0, raw_meta_dir=None):
        keep = _labeled(records, col)
        if raw_meta_dir:
            self.inner = RawWindowDataset(keep, raw_meta_dir, crop_seg=crop,
                                          random_crop=random_crop, seed=seed)
        else:
            self.inner = TokenDataset(keep, token_dir, crop=crop, random_crop=random_crop, seed=seed)
        self.y = np.array([float(r[col]) for r in self.inner.records], np.float32)
        self.pid = np.array([r["pid"] for r in self.inner.records])

    def set_epoch(self, e):
        self.inner.set_epoch(e)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        b = self.inner[i]
        b["y"] = torch.tensor(self.y[i])
        b["idx"] = torch.tensor(i)
        return b


class Classifier(nn.Module):
    """인코더 + attention pooling + 선형 분류기. 토큰 또는 원신호를 받는다."""

    def __init__(self, enc, dropout=0.2):
        super().__init__()
        self.enc = enc
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(enc.d_model, 1))

    def forward(self, batch, device, clip=20.0, checkpoint=True):
        tod, pad = batch["tod"].to(device), batch["pad"].to(device)
        if "tokens" in batch:
            out = self.enc(tokens=batch["tokens"].to(device), tod=tod, mask=pad)
        else:
            x = batch["x"].to(device)                       # (B, S, C, L) int16
            g = batch["gain"].to(device)
            x = x.float() * g[:, None, :, None]
            if clip:
                x = x.clamp_(-clip, clip)
            out = self.enc(segments=x, tod=tod, mask=pad, checkpoint=checkpoint)
        return self.head(out["record"]).squeeze(-1)


def patient_metrics(prob, y, pid, boot=1000, seed=0):
    import pandas as pd
    g = pd.DataFrame({"p": prob, "y": y, "pid": pid}).groupby("pid").agg(p=("p", "mean"), y=("y", "max"))
    pp, py = g["p"].values, g["y"].values
    if len(np.unique(py)) < 2:
        return float("nan"), float("nan"), (float("nan"), float("nan"))
    auc = roc_auc_score(py, pp)
    ap = average_precision_score(py, pp)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boot):
        idx = rng.integers(0, len(py), len(py))
        if len(np.unique(py[idx])) == 2:
            bs.append(roc_auc_score(py[idx], pp[idx]))
    ci = (np.percentile(bs, 2.5), np.percentile(bs, 97.5)) if bs else (np.nan, np.nan)
    return auc, ap, ci


@torch.no_grad()
def evaluate(model, ds, device, batch, amp, boot=1000):
    model.eval()
    probs = np.zeros(len(ds), np.float32)
    for b in DataLoader(ds, batch_size=batch, num_workers=2):
        with autocast(device, amp):
            logit = model(b, device, checkpoint=False)
        probs[b["idx"].numpy()] = torch.sigmoid(logit.float()).cpu().numpy()
    model.train()
    return probs, patient_metrics(probs, ds.y, ds.pid, boot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--task", default="psvt", choices=list(TASK_COL))
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=8640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=None,
                    help="평가 배치 (기본: 학습과 동일, 원신호 경로면 1)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4, help="head/pooling 학습률")
    ap.add_argument("--backbone-lr", type=float, default=3e-5, help="backbone 은 더 낮게")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--unfreeze-stem", action="store_true",
                    help="원신호로 stem 까지 미세조정한다. --meta-dir 필요, 훨씬 느리다")
    ap.add_argument("--meta-dir", default=None, help="prep_meta 결과 (원신호 경로에 필요)")
    ap.add_argument("--stem-lr", type=float, default=1e-5)
    ap.add_argument("--raw-crop", type=int, default=720,
                    help="원신호 학습 시 자를 세그먼트 수 (720 = 2시간)")
    ap.add_argument("--eval-crop", type=int, default=8640, help="평가 시 사용할 세그먼트 수")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ensure_kernel_cache(args.out)
    limit_cpu_threads(2)
    rank, world, device = setup_distributed(args.gpus)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    col = TASK_COL[args.task]

    if args.unfreeze_stem and not args.meta_dir:
        raise SystemExit("--unfreeze-stem 에는 --meta-dir (prep_meta 결과)가 필요합니다.")
    enc, cfg = build_from_checkpoint(args.encoder, device)
    model = Classifier(enc, args.dropout).to(device)
    if not args.unfreeze_stem:                     # 입력이 캐시 토큰이면 stem 은 쓰이지 않는다
        for p in model.enc.stem.parameters():
            p.requires_grad_(False)
    if args.freeze_backbone:
        for p in model.enc.backbone.parameters():
            p.requires_grad_(False)

    raw = args.meta_dir if args.unfreeze_stem else None
    # 평가는 24시간 전체를 통과시킨다. 원신호 경로에서는 한 건만 해도 8,640 세그먼트다.
    eval_batch = args.eval_batch or (1 if raw else args.batch)
    ds = {s: LabeledTokens(load_records(args.splits, s), args.tokens, col,
                           crop=(args.raw_crop if (raw and s == "train") else
                                 args.eval_crop if raw else args.crop),
                           random_crop=(s == "train"), seed=args.seed, raw_meta_dir=raw)
          for s in ("train", "val", "test")}
    n_pos = int(ds["train"].y.sum())
    print(f"[finetune] {args.task}  {describe_device(device)}  block={cfg.get('block')} "
          f"mid={cfg.get('mid_block')}  backbone={'고정' if args.freeze_backbone else '학습'}  "
          f"stem={'학습(원신호)' if args.unfreeze_stem else '고정(토큰)'}"
          + (f"  학습 crop {args.raw_crop} seg ({args.raw_crop/360:.1f}h)" if args.unfreeze_stem else ""))
    for s in ("train", "val", "test"):
        print(f"  {s:<5s} {len(ds[s]):>5,} record / 환자 {len(set(ds[s].pid)):>4,} / "
              f"양성 {int(ds[s].y.sum()):>4,} ({ds[s].y.mean():.1%})")
    if n_pos == 0 or n_pos == len(ds["train"]):
        raise SystemExit("train 에 양성/음성이 모두 있어야 합니다.")

    groups = param_groups(model, args.lr, args.weight_decay)
    if args.unfreeze_stem:                         # stem 은 가장 낮은 LR 로
        st = {id(p) for p in model.enc.stem.parameters()}
        for g in groups:
            g["params"] = [p for p in g["params"] if id(p) not in st]
        groups.append({"params": list(model.enc.stem.parameters()),
                       "lr": args.stem_lr, "weight_decay": 0.0})
    if not args.freeze_backbone:                   # backbone 은 낮은 LR 로 따로
        bb = {id(p) for p in model.enc.backbone.parameters()}
        for g in groups:
            g["params"] = [p for p in g["params"] if id(p) not in bb]
        groups.append({"params": [p for p in model.enc.backbone.parameters() if p.requires_grad],
                       "lr": args.backbone_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, 0.95))
    steps = max(1, len(ds["train"]) // args.batch) * args.epochs
    sched = cosine_with_warmup(opt, max(10, steps // 20), steps)
    pos_weight = torch.tensor([(len(ds["train"].y) - n_pos) / max(n_pos, 1)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger = CSVLogger(os.path.join(args.out, "log.csv"))
    print(f"  학습 파라미터 {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M"
          f"  pos_weight {pos_weight.item():.1f}  steps {steps:,}")

    best = {"val_auroc": -1.0, "epoch": 0}
    amp = not args.no_amp
    t0 = time.time()
    for ep in range(args.epochs):
        ds["train"].set_epoch(ep)
        model.train()
        tot = n = 0
        for b in DataLoader(ds["train"], batch_size=args.batch, shuffle=True, drop_last=True,
                            num_workers=args.workers):
            with autocast(device, amp):
                logit = model(b, device)
                loss = lossf(logit.float(), b["y"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); n += 1
        _, (va, vap, _) = evaluate(model, ds["val"], device, eval_batch, amp, boot=0)
        print(f"  epoch {ep+1:>3}/{args.epochs}  loss {tot/max(n,1):.4f}  "
              f"val 환자 AUROC {va:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        logger.log({"epoch": ep + 1, "loss": tot / max(n, 1), "val_pat_auroc": va, "val_pat_auprc": vap})
        # val 에 한쪽 클래스만 있으면 AUROC 가 NaN 이다. 그래도 첫 epoch 은 반드시 저장해
        # 마지막에 불러올 체크포인트가 있게 한다.
        better = (np.isfinite(va) and va > best["val_auroc"]) or ep == 0
        if better:
            best = {"val_auroc": va if np.isfinite(va) else float("nan"), "epoch": ep + 1}
            save_checkpoint(os.path.join(args.out, "best.pt"), model, None, None, ep + 1,
                            {"args": vars(args), "config": cfg})

    from .common import load_checkpoint
    load_checkpoint(os.path.join(args.out, "best.pt"), model, map_location=device)
    probs, (ta, tap, ci) = evaluate(model, ds["test"], device, eval_batch, amp, args.boot)
    print(f"\n[test] 환자 AUROC {ta:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]  환자 AUPRC {tap:.3f}"
          f"   (best epoch {best['epoch']}, val {best['val_auroc']:.3f})")
    res = {"task": args.task, "encoder": args.encoder, "freeze_backbone": args.freeze_backbone,
           "unfreeze_stem": args.unfreeze_stem,
           "best_epoch": best["epoch"], "val_pat_auroc": best["val_auroc"],
           "test_pat_auroc": ta, "test_pat_auprc": tap, "test_ci_lo": ci[0], "test_ci_hi": ci[1],
           "n_test_pos_pat": int(np.unique(ds["test"].pid[ds["test"].y == 1]).size)}
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(args.out, "test_probs.npy"), probs)
    print(f"[saved] {args.out}/result.json, best.pt, test_probs.npy")
    print("  probe.py 의 같은 태스크 CV/test 수치와 비교해 미세조정이 실제로 이득인지 본다.")


if __name__ == "__main__":
    main()
