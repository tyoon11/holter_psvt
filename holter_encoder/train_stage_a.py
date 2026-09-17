# -*- coding: utf-8 -*-
"""
train_stage_a.py — Stage A: 10초 세그먼트 stem 사전학습 (masked 파형 복원 + beat 보조).

데이터: splits.csv 의 train 적격 record (SSL 이라 downstream 라벨은 쓰지 않는다)
        val 은 손실 추적용으로만 쓴다.

  # 단일 GPU
  python -m holter_encoder.train_stage_a --splits $OUT/splits.csv --out runs/stage_a
  # 4 GPU
  torchrun --nproc_per_node 4 -m holter_encoder.train_stage_a --splits $OUT/splits.csv --out runs/stage_a

같은 --out 으로 다시 실행하면 last.pt 에서 이어서 학습한다.
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from .common import (CSVLogger, all_reduce_mean, autocast, cleanup_distributed, cosine_with_warmup,
                     describe_device, is_main, load_checkpoint, param_groups, save_checkpoint,
                     setup_distributed, unwrap)
from .data import SegmentDataset, load_records
from .ssl import StageAModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--batch", type=int, default=512, help="GPU 당 배치")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--beat-weight", type=float, default=0.3)
    ap.add_argument("--mask-ratio", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epoch-samples", type=int, default=200_000, help="가상 epoch 크기(GPU 당)")
    ap.add_argument("--val-samples", type=int, default=8192)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--gpus", default=None,
                    help='쓸 GPU 번호, 예: "0,2". torchrun 이면 rank 마다 하나씩 배정')
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rank, world, device = setup_distributed(args.gpus)
    torch.manual_seed(args.seed + rank)
    os.makedirs(args.out, exist_ok=True)
    main_proc = is_main(rank)

    train_recs = load_records(args.splits, "train")
    val_recs = load_records(args.splits, "val")
    if main_proc:
        print(f"[stage A] train {len(train_recs):,} record / val {len(val_recs):,}  world={world} device={describe_device(device)}")

    ds = SegmentDataset(train_recs, args.epoch_samples, seed=args.seed, rank=rank)
    dl_kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=device.type == "cuda",
                 drop_last=True, persistent_workers=False)
    val_ds = SegmentDataset(val_recs, args.val_samples, seed=12345, rank=0) if val_recs else None

    model = StageAModel(d_model=args.d_model).to(device)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None)
    opt = torch.optim.AdamW(param_groups(unwrap(model), args.lr, args.weight_decay), betas=(0.9, 0.95))
    sched = cosine_with_warmup(opt, args.warmup, args.steps)

    step = 0
    last = os.path.join(args.out, "last.pt")
    if os.path.exists(last):
        step, _ = load_checkpoint(last, model, opt, sched, map_location=device)
        if main_proc:
            print(f"  이어서 학습: step {step}")
    logger = CSVLogger(os.path.join(args.out, "log.csv"), main_proc)
    if main_proc:
        n = sum(p.numel() for p in unwrap(model).parameters())
        print(f"  파라미터 {n/1e6:.2f}M (stem {sum(p.numel() for p in unwrap(model).stem.parameters())/1e6:.2f}M)")

    use_amp = not args.no_amp
    epoch = step // max(1, args.epoch_samples // args.batch)
    t0, seen = time.time(), 0
    model.train()
    while step < args.steps:
        ds.set_epoch(epoch)
        for batch in DataLoader(ds, shuffle=False, **dl_kw):
            if step >= args.steps:
                break
            x = batch["x"].to(device, non_blocking=True)
            beat = batch["beat"].to(device, non_blocking=True)
            bmask = batch["beat_mask"].to(device, non_blocking=True)
            with autocast(device, use_amp):
                l_rec, l_beat = model(x, beat, bmask, mask_ratio=args.mask_ratio)
                loss = l_rec + args.beat_weight * l_beat
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            seen += x.shape[0] * world

            if step % args.log_every == 0:
                lr_ = all_reduce_mean(l_rec.detach(), world).item()
                lb_ = all_reduce_mean(l_beat.detach(), world).item()
                if main_proc:
                    el = time.time() - t0
                    print(f"  step {step:>6}/{args.steps}  rec {lr_:.4f}  beat {lb_:.4f}  "
                          f"lr {sched.get_last_lr()[0]:.2e}  {seen/el:,.0f} seg/s", flush=True)
                    logger.log({"step": step, "loss_rec": lr_, "loss_beat": lb_,
                                "lr": sched.get_last_lr()[0], "seg_per_s": seen / el})

            if main_proc and val_ds is not None and step % args.val_every == 0:
                m = unwrap(model); m.eval()
                tot_r = tot_b = n = 0
                g = torch.Generator(device="cpu").manual_seed(0)
                with torch.no_grad():
                    for vb in DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers):
                        with autocast(device, use_amp):
                            r, b = m(vb["x"].to(device), vb["beat"].to(device), vb["beat_mask"].to(device),
                                     mask_ratio=args.mask_ratio, generator=g)
                        tot_r += r.item(); tot_b += b.item(); n += 1
                m.train()
                print(f"  [val] step {step}  rec {tot_r/n:.4f}  beat {tot_b/n:.4f}", flush=True)
                logger.log({"step": step, "val_rec": tot_r / n, "val_beat": tot_b / n})

            if main_proc and step % args.ckpt_every == 0:
                save_checkpoint(last, model, opt, sched, step, {"args": vars(args)})
        epoch += 1

    if main_proc:
        save_checkpoint(last, model, opt, sched, step, {"args": vars(args)})
        # 토큰 캐시가 쓰는 stem 가중치만 따로
        torch.save({"stem": unwrap(model).stem.state_dict(), "d_model": args.d_model, "step": step},
                   os.path.join(args.out, "stem.pt"))
        print(f"[stage A] 완료 step {step} → {args.out}/stem.pt")
    cleanup_distributed()


if __name__ == "__main__":
    main()
