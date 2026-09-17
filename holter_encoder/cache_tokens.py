# -*- coding: utf-8 -*-
"""
cache_tokens.py — Stage A stem 으로 모든 적격 record 를 토큰 시퀀스로 바꿔 저장한다.

  <token_dir>/<record>.npy        (n_seg, d) float16
  <token_dir>/<record>.valid.npy  (n_seg,)   bool  품질 통과 세그먼트

Stage B 는 train 만 쓰지만, downstream 평가(val/test)에도 같은 토큰을 쓰므로 전 split 을 만든다.
이미 있는 파일은 건너뛴다. torchrun 이면 record 를 rank 별로 나눠 처리한다.

  torchrun --nproc_per_node 4 -m holter_encoder.cache_tokens \\
      --splits $OUT/splits.csv --stem runs/stage_a/stem.pt --out $OUT/tokens_a
"""

import argparse
import os
import time

import numpy as np
import torch

from .common import (autocast, cleanup_distributed, describe_device, is_main, limit_cpu_threads,
                     setup_distributed)
from .data import iter_segments, load_records, token_paths
from .model import BeatCNNStem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--stem", required=True, help="train_stage_a 의 stem.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=1024, help="한 번에 stem 에 넣는 세그먼트 수")
    ap.add_argument("--meta-dir", default=None, help="prep_meta.py 결과 디렉토리")
    ap.add_argument("--clip", type=float, default=20.0)
    ap.add_argument("--gpus", default=None,
                    help='쓸 GPU 번호, 예: "0,2". torchrun 이면 rank 마다 하나씩 배정')
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    limit_cpu_threads(2)
    rank, world, device = setup_distributed(args.gpus)
    os.makedirs(args.out, exist_ok=True)
    ck = torch.load(args.stem, map_location="cpu", weights_only=False)
    stem = BeatCNNStem(3, ck["d_model"]).to(device).eval()
    stem.load_state_dict(ck["stem"])

    recs = load_records(args.splits, split=None)             # 전 split
    recs = sorted(recs, key=lambda r: r["record"])[rank::world]
    todo = [r for r in recs if not os.path.exists(token_paths(args.out, r["record"])[0])]
    if is_main(rank):
        print(f"[cache] rank0 담당 {len(recs):,} / 남은 {len(todo):,}  (world={world}, d={ck['d_model']}, {describe_device(device)})")

    t0 = time.time()
    for i, r in enumerate(todo):
        toks, valids = [], []
        with torch.no_grad():
            for _, x, v in iter_segments(r["path"], args.chunk, args.meta_dir, args.clip):
                with autocast(device, not args.no_amp):
                    tok, _ = stem(torch.from_numpy(x).to(device, non_blocking=True))
                toks.append(tok.float().cpu().numpy().astype(np.float16))
                valids.append(v)
        tp, vp = token_paths(args.out, r["record"])
        np.save(tp + ".tmp.npy", np.concatenate(toks))
        np.save(vp + ".tmp.npy", np.concatenate(valids))
        os.replace(vp + ".tmp.npy", vp)
        os.replace(tp + ".tmp.npy", tp)             # 토큰 파일이 마지막 — 있으면 완성본
        if is_main(rank) and ((i + 1) % 20 == 0 or i + 1 == len(todo)):
            el = time.time() - t0
            print(f"\r  rank0 {i + 1:,}/{len(todo):,}  {el / (i + 1):.1f}s/record  "
                  f"ETA {(len(todo) - i - 1) * el / (i + 1) / 60:.1f}분", end="", flush=True)
    if is_main(rank):
        print(f"\n[cache] rank0 완료 → {args.out}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
