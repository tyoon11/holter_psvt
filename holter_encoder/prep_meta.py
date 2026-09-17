# -*- coding: utf-8 -*-
"""
prep_meta.py — record 별 작은 메타 파일을 미리 만든다 (Stage A 데이터 로딩 가속).

문제
  Stage A 는 샘플마다 무작위 record 를 고른다. record 가 4,514 개인데 열어둔 h5 핸들
  캐시는 수십 개라 거의 매번 새로 연다. 그때마다 seg/quality(약 250KB)와 beat 주석
  (약 1MB)을 다시 읽어, 7.5KB 짜리 세그먼트 하나를 쓰려고 1.3MB 를 읽게 된다.
  서버에서 551 seg/s 로 GPU 가 굶었다.

해결
  record 당 한 번만 계산해 <record>.meta.npz 로 둔다 (약 100KB).
    valid        (n_seg,) bool    품질 통과 세그먼트
    norm         (n_sig,) float32 채널별 정규화 계수 (amp_std 중앙값)
    beat_targets (n_seg, 5) float16  구간 beat 요약 (log1p 정상/상심실/심실/전체, HR/100)
    has_beats, offset, shape, dtype, scale, n_seg, seg_len
  offset 을 담아두므로 학습 때는 h5py 를 열지 않고 numpy memmap 만 쓴다.

  python -m holter_encoder.prep_meta --splits $OUT/splits.csv --out $OUT/segmeta --workers 32
"""

import argparse
import os
import time
from multiprocessing import Pool

import h5py
import numpy as np

from .data import N_BEAT_TARGETS, SEG_SEC, _BEATS, _NORMAL, _SUPRA, _VENT, load_records, meta_path


def build_one(args):
    path, out_dir = args
    name = os.path.splitext(os.path.basename(path))[0]
    dst = meta_path(out_dir, name)
    if os.path.exists(dst):
        return "skip"
    try:
        with h5py.File(path, "r") as f:
            a = f.attrs
            n_seg, seg_len = int(a["n_seg"]), int(a["seg_len"])
            ds = f["signal"]
            offset = ds.id.get_offset()
            if offset is None or ds.chunks is not None or ds.compression is not None:
                offset = -1                       # memmap 불가 → 학습 때 h5py 로 읽는다
            if "seg/quality" in f:
                q = np.asarray(f["seg/quality"][:], np.float32)
                nan_ratio, amp_std = q[:, 0, :], q[:, 2, :]
                valid = ((np.nan_to_num(nan_ratio, nan=1.0) <= 0).all(1)
                         & (np.nan_to_num(amp_std) >= 1e-3).all(1))
                good = amp_std[valid]
                norm = np.median(good, 0) if len(good) else np.ones(amp_std.shape[1])
            else:
                valid = np.ones(n_seg, bool)
                norm = np.ones(int(a["n_sig"]))
            norm = np.where(np.isfinite(norm) & (norm > 1e-3), norm, 1.0).astype(np.float32)

            bt = np.zeros((n_seg, N_BEAT_TARGETS), np.float16)
            has_beats = "beat" in f
            if has_beats:
                off = f["beat/seg_offset"][:]
                sym = f["beat/symbol"][:]
                grp = np.zeros(256, np.int8)      # 심볼 코드 → 0 무시 / 1 정상 / 2 상심실 / 3 심실
                for c in _NORMAL: grp[c] = 1
                for c in _SUPRA: grp[c] = 2
                for c in _VENT: grp[c] = 3
                isbeat = np.zeros(256, bool)
                for c in _BEATS: isbeat[c] = True
                g, ib = grp[sym], isbeat[sym]
                for i in range(n_seg):
                    s = slice(int(off[i]), int(off[i + 1]))
                    gi, bi = g[s], ib[s]
                    n_all = int(bi.sum())
                    bt[i] = np.log1p([int((gi == 1).sum()), int((gi == 2).sum()),
                                      int((gi == 3).sum()), n_all]).tolist() + \
                            [n_all * (60.0 / SEG_SEC) / 100.0]
            np.savez(dst + ".tmp.npz", valid=valid, norm=norm, beat_targets=bt,
                     has_beats=np.array(has_beats), offset=np.array(int(offset)),
                     shape=np.array(ds.shape), dtype=np.array(str(ds.dtype)),
                     scale=np.array(__import__("json").loads(a["scale"]), np.float32),
                     n_seg=np.array(n_seg), seg_len=np.array(seg_len))
        os.replace(dst + ".tmp.npz", dst)
        return "ok"
    except Exception as e:
        return f"error {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = load_records(args.splits, split=None)
    jobs = [(r["path"], args.out) for r in recs]
    print(f"[prep_meta] {len(jobs):,} record → {args.out}")
    t0 = time.time()
    from collections import Counter
    stats = Counter()
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(build_one, jobs, chunksize=4)):
            stats[r.split(":")[0]] += 1
            if (i + 1) % 200 == 0 or i + 1 == len(jobs):
                el = time.time() - t0
                print(f"\r  {i+1:,}/{len(jobs):,}  {el:.0f}s  ETA {(len(jobs)-i-1)*el/(i+1):.0f}s",
                      end="", flush=True)
    print(f"\n[prep_meta] {dict(stats)}")


if __name__ == "__main__":
    main()
