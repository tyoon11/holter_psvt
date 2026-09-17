#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_loader.py — Stage A 데이터 로더만 떼어 병목을 찾는다 (GPU 없이).

서버에서 io_bench 는 단일 스레드로 2,459 record-open/s 를 냈는데 학습은 워커 48개로
초당 208 아이템에 그쳤다. 차이가 10배 이상이면 읽기가 아니라 로더 어딘가가 문제다.
세 층으로 나눠 잰다.

  1) 구성요소   record 고르기 / 핸들 얻기 / 신호 읽기 / beat 타깃 — 각각의 평균 시간
  2) 단일 프로세스  SegmentDataset[i] 반복 (워커·collate 없음)
  3) DataLoader    워커 수를 바꿔가며 실제 학습과 같은 경로

  python tools/bench_loader.py --splits $OUT/splits.csv --meta-dir $OUT/segmeta \\
      --segs-per-record 32 --batch 64 --workers 0 4 12 32
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from holter_encoder.common import cpu_count, limit_cpu_threads, worker_init  # noqa: E402
from holter_encoder.data import SegmentDataset, _Pack, _Rec, load_records, pack_exists  # noqa: E402


def time_components(recs, meta_dir, segs, n=200, seed=0):
    pack = _Pack(meta_dir) if pack_exists(meta_dir) else None
    rng0 = np.random.default_rng(seed)
    order = rng0.integers(0, len(recs), n)
    t = {}

    s = time.time()
    rngs = [np.random.default_rng([seed, 0, 0, i]) for i in range(n)]
    t["rng 생성"] = (time.time() - s) / n

    w = np.ones(len(recs)) / len(recs)
    s = time.time()
    for r in rngs:
        r.choice(len(recs), p=w)
    t["record 추첨(p 가중)"] = (time.time() - s) / n

    s = time.time()
    handles = [_Rec(recs[i]["path"], meta_dir, pack) for i in order]
    t["record 열기"] = (time.time() - s) / n

    s = time.time()
    for h in handles:
        lo = int(rngs[0].integers(0, max(1, h.n_seg - segs)))
        h.segments(lo, min(segs, h.n_seg - lo), None, True)
    t[f"신호 {segs}개 읽기"] = (time.time() - s) / n

    s = time.time()
    for h in handles:
        for j in range(segs):
            h.beat_target(j % h.n_seg)
    t["beat 타깃"] = (time.time() - s) / n
    for h in handles:
        h.close()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--meta-dir", required=True)
    ap.add_argument("--segs-per-record", type=int, default=32)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, nargs="*", default=[0, 4, 12, 32])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--random-segs", action="store_true")
    args = ap.parse_args()

    limit_cpu_threads(1)
    recs = load_records(args.splits, "train")
    print(f"[bench_loader] train {len(recs):,} record  CPU {cpu_count()}코어  "
          f"pack={'O' if pack_exists(args.meta_dir) else 'X'}  "
          f"segs-per-record={args.segs_per_record}  batch={args.batch}")

    print("\n[1] 아이템 하나를 만드는 데 드는 시간 (구성요소별, 평균)")
    t = time_components(recs, args.meta_dir, args.segs_per_record)
    for k, v in t.items():
        print(f"    {k:<22s} {v*1000:8.2f} ms")
    print(f"    {'합계(대략)':<22s} {sum(t.values())*1000:8.2f} ms "
          f"→ 워커 1개당 {1/max(sum(t.values()),1e-9):.0f} item/s")

    kw = dict(meta_dir=args.meta_dir, segs_per_item=args.segs_per_record,
              contiguous=not args.random_segs, raw=True)
    ds = SegmentDataset(recs, 1_000_000, seed=0, **kw)

    print("\n[2] 워커 없이 SegmentDataset[i] 만")
    s = time.time()
    n = 0
    for i in range(args.batch * 2):
        n += ds[i]["x"].shape[0]
    el = time.time() - s
    print(f"    {n/el:9.0f} seg/s   {args.batch*2/el:7.1f} item/s")

    print("\n[3] DataLoader (학습과 같은 경로)")
    for w in args.workers:
        dl = DataLoader(ds, batch_size=args.batch, num_workers=w, drop_last=True,
                        worker_init_fn=worker_init if w else None,
                        prefetch_factor=4 if w else None, persistent_workers=False)
        it = iter(dl)
        next(it)                                            # 워커 기동 제외
        s = time.time()
        segs = 0
        for _ in range(args.steps):
            b = next(it)
            segs += b["x"].shape[0] * b["x"].shape[1]
        el = time.time() - s
        print(f"    workers {w:>3}  {segs/el:9.0f} seg/s   {args.steps/el:6.2f} step/s   "
              f"(학습 1 step = GPU 수 × {args.batch * args.segs_per_record:,} 세그먼트)")
        del it, dl

    print("\n[읽는 법]")
    print("  - [1] 합계 × 워커 수 ≈ [3] 이면 순수 CPU 비용 → 워커를 늘리거나 segs 를 키운다")
    print("  - [3] 이 워커 수에 비례해 안 오르면 collate/전송(shm)·메인 프로세스가 병목")
    print("  - [2] 가 이미 느리면 [1] 에서 가장 큰 항목을 줄인다")


if __name__ == "__main__":
    main()
