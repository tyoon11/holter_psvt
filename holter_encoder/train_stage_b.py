# -*- coding: utf-8 -*-
"""
train_stage_b.py — Stage B: 캐시 토큰(24h)으로 계층형 S4 backbone 사전학습.

  torchrun --nproc_per_node 4 -m holter_encoder.train_stage_b \\
      --splits $OUT/splits.csv --tokens $OUT/tokens_a --stem runs/stage_a/stem.pt --out runs/stage_b

끝나면 <out>/encoder.pt 를 만든다. model.HolterEncoder 의 state_dict 형식이라 downstream 에서
  enc = build_encoder(...); enc.load_state_dict(torch.load("encoder.pt")["state_dict"], strict=False)
로 바로 쓴다 (stem 은 Stage A, backbone/tod 는 Stage B. attention pool 은 학습되지 않았으므로
downstream 에서 학습하거나 평균 풀링을 쓴다).
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .common import (CSVLogger, all_reduce_mean, autocast, cleanup_distributed, cosine_with_warmup,
                     describe_device, is_main, load_checkpoint, param_groups, save_checkpoint,
                     setup_distributed, unwrap)
from .data import TokenDataset, load_records
from .ssl import StageBModel


def export_encoder(stem_path, stage_b_model, out_path, cfg):
    sd = {}
    stem = torch.load(stem_path, map_location="cpu", weights_only=False)
    for k, v in stem["stem"].items():
        sd["stem." + k] = v
    for k, v in unwrap(stage_b_model).state_dict().items():
        if k.startswith(("backbone.", "tod.")):
            sd[k] = v.cpu()
    torch.save({"state_dict": sd, "config": cfg}, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", choices=["s4", "mamba"], default="s4",
                    help="backbone 블록: s4(S4D, 시간 불변) | mamba(양방향 선택적 SSM, mamba-ssm 권장)")
    ap.add_argument("--mid-block", choices=["same", "attn", "mamba", "s4"], default="same",
                    help="최저 해상도(10분, 24h=144 토큰) 블록. attn=self-attention, "
                         "mamba=선택적 SSM(토큰이 적어 CUDA 커널 없이도 쓸 만함)")
    ap.add_argument("--mamba-d-state", type=int, default=16)
    ap.add_argument("--d-state", type=int, default=64)
    ap.add_argument("--depths", type=int, nargs=5, default=[2, 4, 4, 2, 2])
    ap.add_argument("--crop", type=int, default=8640, help="토큰 수 (8640 = 24h). 60 의 배수 권장")
    ap.add_argument("--batch", type=int, default=8, help="GPU 당 배치")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--mask-ratio", type=float, default=0.15)
    ap.add_argument("--span", type=int, nargs=2, default=[6, 36], help="마스크 구간 토큰 수 (6=1분)")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--gpus", default=None,
                    help='쓸 GPU 번호, 예: "0,2". torchrun 이면 rank 마다 하나씩 배정')
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rank, world, device = setup_distributed(args.gpus)
    torch.manual_seed(args.seed + rank)
    os.makedirs(args.out, exist_ok=True)
    main_proc = is_main(rank)
    d_model = torch.load(args.stem, map_location="cpu", weights_only=False)["d_model"]

    ds = TokenDataset(load_records(args.splits, "train"), args.tokens, crop=args.crop, seed=args.seed)
    val = TokenDataset(load_records(args.splits, "val"), args.tokens, crop=args.crop, random_crop=False)
    if main_proc:
        print(f"[stage B] train {len(ds):,} record (토큰 없음 {ds.missing}) / val {len(val):,}  "
              f"crop {args.crop}  world={world}  {describe_device(device)}")
        if ds.missing:
            print("  ** 토큰 캐시가 없는 train record 가 있습니다. cache_tokens 를 먼저 끝내세요. **")
    sampler = DistributedSampler(ds, world, rank, shuffle=True, seed=args.seed, drop_last=True) if world > 1 else None

    mid = None if args.mid_block == "same" else args.mid_block
    model = StageBModel(d_model, args.d_state, tuple(args.depths), dropout=args.dropout,
                        block=args.block, mid_block=mid, mamba_d_state=args.mamba_d_state).to(device)
    if main_proc and args.block == "mamba":
        from .mamba import HAS_MAMBA_SSM
        print(f"  Mamba 커널: {'mamba_ssm (CUDA)' if HAS_MAMBA_SSM and device.type == 'cuda' else '순수 PyTorch 참조 구현 (느림)'}")
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
        print(f"  파라미터 {sum(p.numel() for p in unwrap(model).parameters())/1e6:.2f}M")

    use_amp = not args.no_amp
    steps_per_epoch = max(1, len(ds) // (args.batch * world))
    epoch = step // steps_per_epoch
    t0, seen = time.time(), 0
    model.train()
    while step < args.steps:
        ds.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, shuffle=sampler is None,
                        num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True)
        for b in dl:
            if step >= args.steps:
                break
            tok = b["tokens"].to(device, non_blocking=True)
            tod = b["tod"].to(device, non_blocking=True)
            pad = b["pad"].to(device, non_blocking=True)
            sv = b["seg_valid"].to(device, non_blocking=True)
            with autocast(device, use_amp):
                loss, n_masked = model(tok, tod, pad, sv, mask_ratio=args.mask_ratio, span=tuple(args.span))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            seen += tok.shape[0] * world

            if step % args.log_every == 0:
                l = all_reduce_mean(loss.detach(), world).item()
                if main_proc:
                    el = time.time() - t0
                    print(f"  step {step:>6}/{args.steps}  loss {l:.4f}  masked {n_masked}  "
                          f"lr {sched.get_last_lr()[0]:.2e}  {seen/el:.2f} record/s", flush=True)
                    logger.log({"step": step, "loss": l, "lr": sched.get_last_lr()[0]})

            if main_proc and len(val) and step % args.val_every == 0:
                m = unwrap(model); m.eval()
                g = torch.Generator(device="cpu").manual_seed(0)
                tot = n = 0
                with torch.no_grad():
                    for vb in DataLoader(val, batch_size=args.batch, num_workers=args.workers):
                        with autocast(device, use_amp):
                            vl, _ = m(vb["tokens"].to(device), vb["tod"].to(device), vb["pad"].to(device),
                                      vb["seg_valid"].to(device), mask_ratio=args.mask_ratio,
                                      span=tuple(args.span), generator=g)
                        tot += vl.item(); n += 1
                m.train()
                print(f"  [val] step {step}  loss {tot/max(n,1):.4f}", flush=True)
                logger.log({"step": step, "val_loss": tot / max(n, 1)})

            if main_proc and step % args.ckpt_every == 0:
                save_checkpoint(last, model, opt, sched, step, {"args": vars(args)})
        epoch += 1

    if main_proc:
        save_checkpoint(last, model, opt, sched, step, {"args": vars(args)})
        cfg = {"d_model": d_model, "d_state": args.d_state, "depths": list(args.depths),
               "pool_factors": [6, 10], "block": args.block, "mid_block": mid,
               "mamba_d_state": args.mamba_d_state}
        export_encoder(args.stem, model, os.path.join(args.out, "encoder.pt"), cfg)
        print(f"[stage B] 완료 step {step} → {args.out}/encoder.pt")
    cleanup_distributed()


if __name__ == "__main__":
    main()
