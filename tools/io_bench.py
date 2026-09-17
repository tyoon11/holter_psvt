#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
io_bench.py — 학습 데이터 저장소의 읽기 특성을 잰다 (Stage A 병목 진단용).

Stage A 는 record 를 무작위로 열고 그 안에서 10초 구간을 읽는다. 네트워크 스토리지처럼
랜덤 접근 지연이 큰 곳에서는 대역폭이 남아도 IOPS 에 묶여 GPU 가 굶는다.
세 가지를 재서 어떤 설정이 맞는지 정한다.

  1) 순차      record 하나를 통째로 (24h, 약 60MB)
  2) 랜덤      흩어진 10초 구간 (7.5KB) 낱개
  3) 연속블록  10초 구간 16개를 한 번에 (120KB)  ← 현재 기본 동작

동시 읽기 수(--threads)를 바꿔가며 재면 워커 수를 정할 수 있다.

  python tools/io_bench.py --splits $OUT/splits.csv --meta-dir $OUT/segmeta --threads 1 8 32
"""

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from holter_encoder.data import _Pack, _Rec, load_records, pack_exists  # noqa: E402


def timed(fn, n_ops, threads):
    t = time.time()
    if threads == 1:
        for i in range(n_ops):
            fn(i)
    else:
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(fn, range(n_ops)))
    return time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--meta-dir", default=None)
    ap.add_argument("--threads", type=int, nargs="*", default=[1, 8, 32])
    ap.add_argument("--n-records", type=int, default=24, help="순차 읽기 표본")
    ap.add_argument("--n-reads", type=int, default=2000, help="랜덤/블록 읽기 횟수")
    ap.add_argument("--block", type=int, default=16, help="연속 블록의 세그먼트 수")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = load_records(args.splits, "train")
    random.seed(args.seed)
    picks = random.sample(recs, min(args.n_records, len(recs)))
    print(f"[io_bench] train {len(recs):,} record / 표본 {len(picks)}")
    st = os.statvfs(recs[0]["path"])
    print(f"  경로: {os.path.dirname(recs[0]['path'])}")
    packed = pack_exists(args.meta_dir)
    pack = _Pack(args.meta_dir) if packed else None
    print(f"  메타: {'묶음(pack)' if packed else ('record 별 npz' if args.meta_dir else '없음 (h5 직접 열기)')}")

    # 1) 순차
    def seq(i):
        r = _Rec(picks[i % len(picks)]["path"], args.meta_dir, pack)
        n = r.segments(0, r.n_seg).nbytes
        r.close()
        return n
    for th in args.threads:
        t = time.time()
        with ThreadPoolExecutor(th) as ex:
            nbytes = sum(ex.map(seq, range(len(picks))))
        el = time.time() - t
        print(f"  [순차]    threads {th:>3}  {nbytes/1e6/el:8.0f} MB/s  "
              f"{len(picks)/el:6.1f} record/s")

    # 2) 랜덤 / 3) 연속 블록 — 열려 있는 핸들 재사용 없이 매번 새 record (학습과 같은 조건)
    for mode, k in (("랜덤 10초", 1), (f"연속 {args.block}개", args.block)):
        for th in args.threads:
            rng = random.Random(args.seed)
            order = [rng.randrange(len(recs)) for _ in range(args.n_reads)]
            starts = [rng.random() for _ in range(args.n_reads)]

            def one(i):
                r = _Rec(recs[order[i]]["path"], args.meta_dir, pack)
                s = int(starts[i] * max(1, r.n_seg - k))
                x = r.segments(s, min(k, r.n_seg - s))
                r.close()
                return x.shape[0]
            el = timed(one, args.n_reads, th)
            segs = args.n_reads * k
            print(f"  [{mode}] threads {th:>3}  {segs/el:8.0f} seg/s  "
                  f"{args.n_reads/el:7.0f} record-open/s  {segs*7500/1e6/el:6.1f} MB/s")

    print("\n[읽는 법]")
    print("  - 연속 블록이 랜덤보다 크게 빠르면 디스크 랜덤 접근이 병목 → --segs-per-record 를 키운다")
    print("  - threads 를 늘릴 때 seg/s 가 계속 오르면 지연이 병목 → --workers 를 늘린다")
    print("  - record-open/s 가 스레드와 무관하게 일정하면 여는 비용이 상한이다 → 묶음 메타(pack)")
    print("  - 둘 다 안 오르면 대역폭 한계 → 로컬 NVMe 로 옮기는 것 외에는 방법이 없다")
    print("  - Stage A 목표: (GPU 수 × batch × segs-per-record) / 원하는 스텝 시간 만큼의 seg/s")


if __name__ == "__main__":
    main()
