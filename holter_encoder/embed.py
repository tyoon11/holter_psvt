# -*- coding: utf-8 -*-
"""
embed.py — 사전학습된 인코더를 고정하고 record 임베딩을 뽑는다 (downstream 평가용).

  python -m holter_encoder.embed --splits $OUT/splits.csv --tokens $OUT/tokens_a \\
      --encoder $RUN/stage_b_mamba/encoder.pt --out $RUN/stage_b_mamba/emb.npz --gpus 0

저장 내용 (npz)
  record, pid, cohort, split, y_lqt, y_tof, y_psvt
  X_mean  backbone 출력 토큰의 평균 (품질 통과 토큰만)  ← 기본 특징
  X_std   같은 토큰의 표준편차 (하루 동안의 변동)
  X_stem  stem 토큰 평균 (backbone 을 거치지 않은 비교용 = Stage A 만의 효과)

attention pool 은 사전학습되지 않았으므로 쓰지 않는다. 24h 전체를 한 번에 넣는다.
"""

import argparse
import os
import time

import numpy as np
import torch

from .common import (autocast, cleanup_distributed, describe_device, ensure_kernel_cache, is_main,
                     setup_distributed)
from .data import TokenDataset, load_records
from .model import HolterEncoder


def build_from_checkpoint(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["config"]
    enc = HolterEncoder(d_model=c["d_model"], d_state=c["d_state"], depths=tuple(c["depths"]),
                        pool_factors=tuple(c["pool_factors"]), block=c.get("block", "s4"),
                        mid_block=c.get("mid_block"), mamba_d_state=c.get("mamba_d_state", 16))
    missing = enc.load_state_dict(ck["state_dict"], strict=False)
    bad = [k for k in missing.missing_keys if not k.startswith("pool.")]
    if bad or missing.unexpected_keys:
        raise RuntimeError(f"가중치 불일치: missing {bad[:5]} / unexpected {missing.unexpected_keys[:5]}")
    return enc.to(device).eval(), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=17280, help="가장 긴 record 기준 (48h)")
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    ensure_kernel_cache()
    rank, world, device = setup_distributed(args.gpus)
    enc, cfg = build_from_checkpoint(args.encoder, device)
    recs = load_records(args.splits, split=None)
    ds = TokenDataset(recs, args.tokens, crop=args.max_tokens, random_crop=False)
    print(f"[embed] {len(ds):,} record (토큰 없음 {ds.missing})  {describe_device(device)}  "
          f"block={cfg.get('block')} mid={cfg.get('mid_block')}")

    out = {k: [] for k in ("record", "pid", "cohort", "split", "y_lqt", "y_tof", "y_psvt")}
    Xm, Xs, Xt = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            r = ds.records[i]
            valid = b["seg_valid"].to(device)[None]
            with autocast(device, not args.no_amp):
                o = enc(tokens=b["tokens"].to(device)[None], tod=b["tod"].to(device)[None])
            tok = o["tokens"].float()[0]                       # (L, d)
            stem = o["stem_tokens"].float()[0]
            m = valid[0]
            if m.sum() < 10:                                    # 품질 통과 토큰이 거의 없으면 전체 사용
                m = ~b["pad"].to(device)
            sel = tok[m]
            Xm.append(sel.mean(0).cpu().numpy())
            Xs.append(sel.std(0).cpu().numpy())
            Xt.append(stem[m].mean(0).cpu().numpy())
            for k in out:
                out[k].append(r.get(k if k != "record" else "record"))
            if (i + 1) % 200 == 0 or i + 1 == len(ds):
                el = time.time() - t0
                print(f"\r  {i+1:,}/{len(ds):,}  {el/(i+1)*1000:.0f} ms/record  "
                      f"ETA {(len(ds)-i-1)*el/(i+1)/60:.1f}분", end="", flush=True)
    print()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, X_mean=np.stack(Xm).astype(np.float32),
                        X_std=np.stack(Xs).astype(np.float32),
                        X_stem=np.stack(Xt).astype(np.float32),
                        **{k: np.array(v) for k, v in out.items()},
                        config=np.array(str(cfg)))
    print(f"[saved] {args.out}  X_mean {np.stack(Xm).shape}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
