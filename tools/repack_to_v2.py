#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repack_to_v2.py — 기존 HDF5 레코드를 v2 스키마로 다시 담는다.

두 종류의 입력을 모두 받아 하나의 포맷으로 통합한다.
  v1   h5_converter 가 만든 per-segment 스키마 (beat/fiducial/리포트 전부 보존)
  flat /ecg 같은 단일 2D 신호 데이터셋 (신호만; has_beats=False 로 표시)

원본은 읽기만 하고 건드리지 않는다. 출력은 .tmp 로 쓴 뒤 원자적으로 교체하므로
중단 후 재실행하면 이어서 진행된다.

사용:
  # 내 converter 산출물 (1,857개) 를 v2 로
  python repack_to_v2.py --src /home/coder/workspace/Holter_TOF/holter_h5 \
                         --dst /scratch/holter_v2 --workers 32

  # 신호만 있는 코호트도 같은 곳으로 (lead 순서는 profile_records.py 로 먼저 확인)
  python repack_to_v2.py --src /home/coder/workspace/Holter_TOF/nas1_Holter_PSVT/h5 \
                         --dst /scratch/holter_v2 --leads II,V1,V5 --workers 32

  python repack_to_v2.py --src ... --dst ... --limit 5 --workers 1   # 소규모 시험
"""

import argparse
import json
import os
import re
import sys
import time
import traceback

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from h5_converter.schema_v2 import (  # noqa: E402
    CANONICAL_LEADS, FIDUCIAL_FEATURES, QUALITY_FIELDS, SIMILARITY_FIELDS,
    RUN_FIELDS, write_v2, _to_str,
)

NUMERIC = re.compile(r"^\d+$")


# =============================================================================
# v1 (per-segment) 읽기
# =============================================================================
def _attr(obj, key, default=""):
    v = obj.attrs.get(key, default)
    return _to_str(v) if isinstance(v, (bytes, str)) else v


def flatten_report_h5(ecg):
    """v1 의 중첩 ECG/annotation group → 평탄 dict. 원본 JSON 없이 h5 에서 복원한다."""
    out = {}
    if "annotation" not in ecg:
        return out
    ann = ecg["annotation"]

    def _int(v, d=0):
        try:
            return int(v)
        except (TypeError, ValueError):
            return d

    out["ann_len"] = _int(ann.attrs.get("ann_len", 0))
    out["NoisePercentage"] = _to_str(ann.attrs.get("NoisePercentage", ""))
    out["AFAFLPercentage"] = _to_str(ann.attrs.get("AFAFLPercentage", ""))

    bc = ann.get("beat_count")
    if bc is None:
        return out
    for prefix, grp in (("vb", "VentricularBeat"), ("sb", "SupraventricularBeat")):
        if grp not in bc:
            continue
        g = bc[grp]
        out[f"{prefix}_total"] = _int(g.attrs.get("total", 0))
        for k in ("Isolated", "Couplets", "BigeminalCycles"):
            out[f"{prefix}_{k}"] = _int(g.attrs.get(k, 0))
        if "Runs" in g:
            r = g["Runs"]
            out[f"{prefix}_run_count"] = _int(r.attrs.get("count", 0))
            out[f"{prefix}_run_TotalBeats"] = _int(r.attrs.get("TotalBeats", 0))
            for k in RUN_FIELDS[2:]:
                out[f"{prefix}_run_{k}"] = _to_str(r.attrs.get(k, ""))
    for k in ("PacedBeats", "BBBeats", "JunctionalBeats", "AberrantBeats"):
        if k in bc:
            out[f"{k}_total"] = _int(bc[k].attrs.get("total", 0))
    return out


def read_v1(path):
    """v1 레코드를 통째로 읽어 write_v2 인자 dict 로 만든다."""
    with h5py.File(path, "r") as f:
        ecg = f["ECG"]
        segs = ecg["segments"]
        keys = sorted([k for k in segs.keys() if NUMERIC.match(k)], key=int)
        n_seg = len(keys)
        if n_seg == 0:
            raise ValueError("세그먼트 없음")

        # lead 순서에 함정이 둘 있다.
        #   (1) 신호는 signal/{이름} 으로 저장되므로 이름으로 읽으면 순서는 안전하다.
        #       단 h5py 의 group.keys() 는 알파벳 정렬이라 물리 순서가 아니다.
        #   (2) signal_quality/nan_ratio 같은 (3,) 배열은 .hea 의 물리적 lead 순서
        #       (이 데이터셋은 ['V5','V1','II'])를 따른다. 신호를 재배열하면 이 통계도
        #       같은 순열로 맞춰야 한다. 그 기준은 metadata/sig_name 이다.
        first_sig = segs[keys[0]]["signal"]
        available = list(first_sig.keys())
        orig_order = None
        if "metadata" in ecg and "sig_name" in ecg["metadata"]:
            orig_order = [_to_str(x) for x in ecg["metadata"]["sig_name"][:]]
        if not orig_order or set(orig_order) != set(available):
            orig_order = available          # sig_name 이 없거나 안 맞으면 포기하고 키 순서
        leads = [l for l in CANONICAL_LEADS if l in available]
        leads += [l for l in available if l not in leads]
        seg_len = int(first_sig[leads[0]].shape[-1])
        n_sig = len(leads)

        sig = np.empty((n_seg * seg_len, n_sig), dtype=np.float32)
        b_samp, b_sym, b_sub, b_chan, b_num, b_aux = [], [], [], [], [], []
        b_off = np.zeros(n_seg + 1, dtype=np.int32)
        f_samp, f_lab = [], []
        f_off = np.zeros(n_seg + 1, dtype=np.int32)
        ff = np.full((n_seg, len(FIDUCIAL_FEATURES)), np.nan, dtype=np.float32)
        qual = np.full((n_seg, len(QUALITY_FIELDS), n_sig), np.nan, dtype=np.float32)
        simi = np.full((n_seg, len(SIMILARITY_FIELDS), n_sig), np.nan, dtype=np.float32)

        for i, k in enumerate(keys):
            s = segs[k]
            base = i * seg_len
            sg = s["signal"]
            for j, l in enumerate(leads):
                sig[base:base + seg_len, j] = sg[l][:]

            added = 0
            if "beat_annotation" in s:
                ba = s["beat_annotation"]
                smp = np.asarray(ba["sample"][:], dtype=np.int64) + base
                added = len(smp)
                b_samp.append(smp)
                b_sym.append(ba["symbol"][:])
                for lst, key in ((b_sub, "subtype"), (b_chan, "chan"), (b_num, "num")):
                    lst.append(np.asarray(ba[key][:], np.int8) if key in ba
                               else np.zeros(added, np.int8))
                b_aux.append(ba["aux_note"][:] if "aux_note" in ba
                             else np.array([""] * added, dtype=object))
            b_off[i + 1] = b_off[i] + added

            added = 0
            if "fiducial_point" in s:
                fp = s["fiducial_point"]
                fs_ = np.asarray(fp["fsample"][:], dtype=np.int64) + base
                added = len(fs_)
                f_samp.append(fs_)
                f_lab.append(fp["fiducial"][:])
            f_off[i + 1] = f_off[i] + added

            if "fiducial_feature" in s:
                a = s["fiducial_feature"].attrs
                for j, name in enumerate(FIDUCIAL_FEATURES):
                    if name in a:
                        try:
                            ff[i, j] = float(a[name])
                        except (TypeError, ValueError):
                            pass

            if "signal_quality" in s:
                sq = s["signal_quality"]
                if "nan_ratio" in sq:
                    qual[i, 0, :] = np.asarray(sq["nan_ratio"][:], np.float32)[:n_sig]
                if "amplitude" in sq:
                    am = sq["amplitude"]
                    for j, name in enumerate(QUALITY_FIELDS[1:], start=1):
                        if name in am:
                            qual[i, j, :] = np.asarray(am[name][:], np.float32)[:n_sig]
                if "beat_similarity" in sq:
                    bs = sq["beat_similarity"]
                    for j, name in enumerate(SIMILARITY_FIELDS):
                        if name in bs:
                            simi[i, j, :] = np.asarray(bs[name][:], np.float32)[:n_sig]

        # (2) 에서 설명한 순열 보정. 기준은 group keys 가 아니라 metadata/sig_name.
        if orig_order != leads:
            perm = [orig_order.index(l) for l in leads]
            qual = qual[:, :, perm]
            simi = simi[:, :, perm]

        md = ecg.get("metadata")
        hea = {}
        fs_hz = 125.0
        if md is not None:
            for k in ("adc_gain", "baseline", "adc_res", "adc_zero"):
                if k in md:
                    hea[k] = np.asarray(md[k][:])
            for k in ("fmt", "units"):
                if k in md:
                    hea[k] = [_to_str(x) for x in md[k][:]]
            hea["base_date"] = _attr(md, "base_date")
            hea["base_time"] = _attr(md, "base_time")
            try:
                fs_hz = float(md.attrs.get("fs", 125))
            except (TypeError, ValueError):
                pass

        pat = f.get("patient")
        patient = ({"pid": _attr(pat, "pid"), "age": _attr(pat, "age"),
                    "gender": _attr(pat, "gender")} if pat is not None else None)

        beats = None
        if b_samp:
            beats = {"sample": np.concatenate(b_samp),
                     "symbol": np.concatenate([np.asarray(x, dtype=object) for x in b_sym]),
                     "subtype": np.concatenate(b_sub),
                     "chan": np.concatenate(b_chan),
                     "num": np.concatenate(b_num),
                     "aux_note": np.concatenate([np.asarray(x, dtype=object) for x in b_aux])}
        fid = None
        if f_samp:
            fid = {"sample": np.concatenate(f_samp),
                   "label": np.concatenate([np.asarray(x, dtype=object) for x in f_lab])}

        return dict(
            signal=sig, sig_name=leads, fs=fs_hz, seg_len=seg_len,
            record_name=_attr(f, "file_name") or os.path.splitext(os.path.basename(path))[0],
            created_by=_attr(f, "created_by"),
            patient=patient, beats=beats, beat_seg_offset=b_off,
            fiducial=fid, fid_seg_offset=f_off,
            fiducial_feat=ff, quality=qual, similarity=simi,
            hea_meta=hea or None, report=flatten_report_h5(ecg) or None,
        )


# =============================================================================
# flat (신호만) 읽기
# =============================================================================
def read_flat(path, leads, seg_len=1250):
    with h5py.File(path, "r") as f:
        best = None
        def visit(name, obj):
            nonlocal best
            if isinstance(obj, h5py.Dataset) and obj.size > 100_000:
                if best is None or obj.size > f[best].size:
                    best = name
        f.visititems(visit)
        if best is None:
            raise ValueError("신호 데이터셋 없음")
        ds = f[best]
        arr = ds[:]
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.shape[0] < arr.shape[1]:      # channel-first → time-major
            arr = arr.T
        fs = float(f.attrs.get("fs", 125))
        n_sig = arr.shape[1]
        names = leads[:n_sig] if leads else [f"ch{i}" for i in range(n_sig)]
        if leads and len(leads) != n_sig:
            raise ValueError(f"--leads {len(leads)}개 != 실제 채널 {n_sig}개")
        return dict(signal=arr.astype(np.float32), sig_name=names, fs=fs,
                    seg_len=seg_len,
                    record_name=os.path.splitext(os.path.basename(path))[0])


# =============================================================================
# 워커
# =============================================================================
def detect_kind(path):
    with h5py.File(path, "r") as f:
        for p in ("ECG/segments", "segments"):
            if p in f and isinstance(f[p], h5py.Group):
                if any(NUMERIC.match(k) for k in f[p].keys()):
                    return "v1"
        if f.attrs.get("schema_version", "") == "2.0":
            return "v2"
    return "flat"


def repack_one(src, dst_dir, leads=None, source_tag="", overwrite=False):
    name = os.path.splitext(os.path.basename(src))[0]
    dst = os.path.join(dst_dir, name + ".h5")
    if os.path.exists(dst) and not overwrite:
        return {"record": name, "status": "skip", "dst": dst}
    t0 = time.time()
    try:
        kind = detect_kind(src)
        if kind == "v1":
            kw = read_v1(src)
        elif kind == "flat":
            kw = read_flat(src, leads)
        else:
            return {"record": name, "status": "already_v2", "dst": src}
        kw["source"] = source_tag or os.path.basename(os.path.dirname(src))
        write_v2(dst, **kw)
        return {"record": name, "status": "ok", "kind": kind, "dst": dst,
                "src_bytes": os.path.getsize(src), "dst_bytes": os.path.getsize(dst),
                "sec": round(time.time() - t0, 2),
                "n_samples": int(kw["signal"].shape[0]),
                "leads": ",".join(kw["sig_name"])}
    except Exception as e:
        for junk in (dst + ".tmp",):
            if os.path.exists(junk):
                try:
                    os.remove(junk)
                except OSError:
                    pass
        return {"record": name, "status": "error", "sec": round(time.time() - t0, 2),
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, nargs="+")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--leads", default=None,
                    help="flat 코호트의 lead 순서, 예: II,V1,V5. v1 은 무시됨")
    ap.add_argument("--source-tag", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log", default=None, help="결과 CSV 경로")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    leads = args.leads.split(",") if args.leads else None

    files = []
    for root in args.src:
        if os.path.isfile(root):
            files.append(root)
        else:
            files += [e.path for e in os.scandir(root)
                      if e.is_file() and e.name.lower().endswith((".h5", ".hdf5"))]
    files.sort()
    if args.limit:
        files = files[:args.limit]
    print(f"[repack] 입력 {len(files):,}개 → {args.dst}  (workers={args.workers})")

    t0 = time.time()
    results = []
    if args.workers <= 1:
        for i, p in enumerate(files):
            results.append(repack_one(p, args.dst, leads, args.source_tag, args.overwrite))
            _progress(i + 1, len(files), results, t0)
    else:
        try:
            import ray
            ray.init(num_cpus=args.workers, ignore_reinit_error=True,
                     include_dashboard=False)
            remote = ray.remote(repack_one)
            pending = [remote.remote(p, args.dst, leads, args.source_tag, args.overwrite)
                       for p in files]
            done_n = 0
            while pending:
                ready, pending = ray.wait(pending, num_returns=min(20, len(pending)))
                results += ray.get(ready)
                done_n += len(ready)
                _progress(done_n, len(files), results, t0)
        except ImportError:
            from multiprocessing import Pool
            from functools import partial
            fn = partial(repack_one, dst_dir=args.dst, leads=leads,
                         source_tag=args.source_tag, overwrite=args.overwrite)
            with Pool(args.workers) as pool:
                for i, r in enumerate(pool.imap_unordered(fn, files)):
                    results.append(r)
                    _progress(i + 1, len(files), results, t0)

    print()
    _summary(results, time.time() - t0)
    if args.log:
        _write_log(args.log, results)
        print(f"[log] {args.log}")


def _progress(done, total, results, t0):
    el = time.time() - t0
    rate = done / max(el, 1e-9)
    eta = (total - done) / max(rate, 1e-9)
    nerr = sum(1 for r in results if r.get("status") == "error")
    print(f"\r  {done:,}/{total:,}  {rate:.2f} rec/s  경과 {el/60:.1f}분  "
          f"ETA {eta/60:.1f}분  오류 {nerr}", end="", flush=True)


def _summary(results, elapsed):
    from collections import Counter
    c = Counter(r["status"] for r in results)
    ok = [r for r in results if r["status"] == "ok"]
    print(f"[완료] {dict(c)}  총 {elapsed/60:.1f}분")
    if ok:
        s = sum(r["src_bytes"] for r in ok) / 1e9
        d = sum(r["dst_bytes"] for r in ok) / 1e9
        print(f"  용량 {s:.1f} GB → {d:.1f} GB  ({s/max(d,1e-9):.2f}x 축소)")
        print(f"  레코드당 평균 {np.mean([r['sec'] for r in ok]):.1f}초")
        lead_sets = Counter(r["leads"] for r in ok)
        print(f"  lead 구성: {dict(lead_sets)}")
    for r in results:
        if r["status"] == "error":
            print(f"  [오류] {r['record']}: {r['error']}")


def _write_log(path, results):
    import csv
    cols = sorted({k for r in results for k in r.keys() if k != "trace"})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)


if __name__ == "__main__":
    main()
