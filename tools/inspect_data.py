#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_data.py — Holter 데이터 디렉토리 구조/건수 조사

여러 후보 디렉토리를 훑어서 다음을 보고한다.
  1) 디렉토리별 파일 건수 / 확장자 분포 / 총 용량 / 하위 레이아웃 / mtime 범위
  2) 샘플 파일의 내부 구조 (HDF5 트리는 반복 그룹을 접어서 출력)
  3) 24h 전체 읽기 소요시간 + 메타데이터 오버헤드 비율
  4) 디렉토리 간 record 겹침 (어느 디렉토리가 상위집합인지)

사용:
  python inspect_data.py                       # 기본 후보 4곳
  python inspect_data.py --dirs /a /b          # 직접 지정
  python inspect_data.py --quick               # 내부 구조 probe 생략 (건수만)
  python inspect_data.py --samples 3           # 디렉토리당 샘플 파일 수
  python inspect_data.py --json out.json       # 결과 JSON 저장
"""

import argparse
import collections
import datetime as dt
import json
import os
import re
import sys
import time

DEFAULT_DIRS = [
    "/home/coder/workspace/Holter_TOF/nas1_Holter_PSVT/h5",
    "/home/coder/workspace/Holter_TOF/nas1_Holter_PSVT/final_denoised_3_lead",
    "/home/coder/workspace/Holter_TOF/nas1_Holter_PSVT/denoised_3_lead",
    "/home/coder/workspace/Holter_TOF/holter_h5",
]

NUMERIC = re.compile(r"^\d+$")
BAR = "=" * 78


# ----------------------------------------------------------------------------
# 유틸
# ----------------------------------------------------------------------------
def human(n):
    if n is None:
        return "?"
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or u == "TB":
            return f"{n:,.1f}{u}" if u != "B" else f"{int(n)}B"
        n /= 1024.0


def ts(x):
    try:
        return dt.datetime.fromtimestamp(x).strftime("%Y-%m-%d")
    except Exception:
        return "?"


def stem(name):
    """record 식별자 추출: 확장자 및 흔한 접미사 제거."""
    base = os.path.basename(name)
    for _ in range(2):  # .h5, .npy, .dat / .nii.gz 류 대비
        base, ext = os.path.splitext(base)
        if ext.lower() not in (".h5", ".hdf5", ".npy", ".npz", ".dat", ".hea",
                               ".json", ".csv", ".gz", ".mat", ".pkl"):
            base = base + ext
            break
    for suf in ("_denoised", "_final", "_clean", "_filtered", "_3lead", "_3_lead"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    return base


# ----------------------------------------------------------------------------
# 1) 디렉토리 스캔
# ----------------------------------------------------------------------------
def scan_dir(root, max_files=2_000_000):
    info = {
        "root": root,
        "exists": os.path.isdir(root),
        "n_files": 0,
        "n_dirs": 0,
        "total_bytes": 0,
        "by_ext": collections.Counter(),
        "bytes_by_ext": collections.Counter(),
        "top_level": collections.Counter(),
        "max_depth": 0,
        "mtime_min": None,
        "mtime_max": None,
        "size_min": None,
        "size_max": None,
        "examples": collections.defaultdict(list),
        "stems": set(),
        "truncated": False,
    }
    if not info["exists"]:
        return info

    stack = [(root, 0)]
    while stack:
        cur, depth = stack.pop()
        info["max_depth"] = max(info["max_depth"], depth)
        try:
            entries = list(os.scandir(cur))
        except (PermissionError, OSError) as e:
            info.setdefault("errors", []).append(f"{cur}: {e}")
            continue
        for e in entries:
            rel_top = os.path.relpath(e.path, root).split(os.sep)[0]
            try:
                if e.is_dir(follow_symlinks=False):
                    info["n_dirs"] += 1
                    stack.append((e.path, depth + 1))
                    continue
                st = e.stat(follow_symlinks=False)
            except OSError:
                continue

            ext = os.path.splitext(e.name)[1].lower() or "<noext>"
            info["n_files"] += 1
            info["total_bytes"] += st.st_size
            info["by_ext"][ext] += 1
            info["bytes_by_ext"][ext] += st.st_size
            if depth > 0:
                info["top_level"][rel_top] += 1
            if len(info["examples"][ext]) < 5:
                info["examples"][ext].append(e.path)
            if ext in (".h5", ".hdf5", ".npy", ".dat", ".mat"):
                info["stems"].add(stem(e.name))

            info["mtime_min"] = st.st_mtime if info["mtime_min"] is None else min(info["mtime_min"], st.st_mtime)
            info["mtime_max"] = st.st_mtime if info["mtime_max"] is None else max(info["mtime_max"], st.st_mtime)
            info["size_min"] = st.st_size if info["size_min"] is None else min(info["size_min"], st.st_size)
            info["size_max"] = st.st_size if info["size_max"] is None else max(info["size_max"], st.st_size)

            if info["n_files"] >= max_files:
                info["truncated"] = True
                stack = []
                break
    return info


def print_dir_report(info):
    root = info["root"]
    print(BAR)
    print(f"[DIR] {root}")
    print(BAR)
    if not info["exists"]:
        print("  ** 존재하지 않거나 접근 불가 **\n")
        return
    print(f"  파일 {info['n_files']:,}개 / 하위디렉토리 {info['n_dirs']:,}개 / "
          f"총 {human(info['total_bytes'])}  (최대깊이 {info['max_depth']})")
    if info["truncated"]:
        print("  ** max-files 도달, 일부만 집계됨 **")
    print(f"  mtime  {ts(info['mtime_min'])} ~ {ts(info['mtime_max'])}")
    print(f"  파일크기 {human(info['size_min'])} ~ {human(info['size_max'])}")

    print("  확장자별:")
    for ext, n in info["by_ext"].most_common(12):
        avg = info["bytes_by_ext"][ext] / max(n, 1)
        print(f"    {ext:10s} {n:>8,}개  합 {human(info['bytes_by_ext'][ext]):>9s}  평균 {human(avg):>9s}")

    if info["top_level"]:
        print(f"  1단계 하위 디렉토리 ({len(info['top_level'])}개):")
        for d, n in info["top_level"].most_common(8):
            print(f"    {d:40s} 파일 {n:,}")
        if len(info["top_level"]) > 8:
            print(f"    ... 외 {len(info['top_level']) - 8}개")

    print(f"  고유 record stem: {len(info['stems']):,}개")
    for ext, paths in list(info["examples"].items())[:4]:
        print(f"  예시({ext}): {os.path.basename(paths[0])}")
    for err in info.get("errors", [])[:3]:
        print(f"  [warn] {err}")
    print()


# ----------------------------------------------------------------------------
# 2) HDF5 내부 구조 (반복 그룹 접어서 출력)
# ----------------------------------------------------------------------------
def walk_h5(node, out, depth=0, max_depth=7, collapse_at=8, budget=None):
    """반복되는 숫자 이름 그룹은 첫 항목만 펼치고 나머지는 접는다.
    반환: 이 서브트리의 (그룹 수, 데이터셋 수, 속성 수) 추정치."""
    import h5py

    if budget is None:
        budget = [4000]
    ng = nd = na = 0
    na += len(node.attrs)

    if depth >= max_depth or budget[0] <= 0:
        return ng, nd, na

    try:
        keys = list(node.keys())
    except Exception:
        return ng, nd, na

    numeric = [k for k in keys if NUMERIC.match(k)]
    collapsed = len(numeric) >= collapse_at and len(numeric) > len(keys) * 0.8
    show = keys
    if collapsed:
        numeric_sorted = sorted(numeric, key=int)
        show = numeric_sorted[:1] + [k for k in keys if not NUMERIC.match(k)][:3]

    for k in show:
        budget[0] -= 1
        if budget[0] <= 0:
            out.append("  " * (depth + 1) + "... (출력 예산 초과)")
            break
        try:
            item = node[k]
        except Exception as e:
            out.append("  " * (depth + 1) + f"{k}  <읽기실패 {e}>")
            continue

        pad = "  " * (depth + 1)
        if isinstance(item, h5py.Group):
            extra = ""
            if collapsed and NUMERIC.match(k):
                extra = f"   <<< 동일 구조 {len(numeric):,}개 중 1개만 표시"
            attr_s = f"  [attrs: {', '.join(list(item.attrs)[:6])}]" if len(item.attrs) else ""
            out.append(f"{pad}{k}/{attr_s}{extra}")
            g, d, a = walk_h5(item, out, depth + 1, max_depth, collapse_at, budget)
            mult = len(numeric) if (collapsed and NUMERIC.match(k)) else 1
            ng += (1 + g) * mult
            nd += d * mult
            na += a * mult
        else:
            chunks = getattr(item, "chunks", None)
            comp = getattr(item, "compression", None)
            nb = item.dtype.itemsize * int(item.size)
            out.append(f"{pad}{k}  dset {tuple(item.shape)} {item.dtype} "
                       f"chunks={chunks} comp={comp} ({human(nb)})")
            mult = 1
            nd += 1
            na += len(item.attrs)
    return ng, nd, na


def find_signal(f):
    """파일 안에서 신호 데이터셋의 저장 방식을 판별."""
    import h5py

    found = {"mode": None, "path": None, "shape": None, "dtype": None,
             "chunks": None, "comp": None, "n_seg": None, "seg_len": None,
             "leads": None, "fs": None}

    # (a) flat: 큰 2D/1D 데이터셋
    big = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.size > 100_000:
            big.append((obj.size, name))
            if len(big) > 20:
                raise StopIteration

    try:
        f.visititems(visit)
    except StopIteration:
        pass
    except Exception:
        pass

    if big:
        big.sort(reverse=True)
        name = big[0][1]
        ds = f[name]
        found.update(mode="flat", path=name, shape=tuple(ds.shape), dtype=str(ds.dtype),
                     chunks=ds.chunks, comp=ds.compression)

    # (b) per-segment: ECG/segments/{i}/signal/{lead}
    for segpath in ("ECG/segments", "segments", "ECG/segment"):
        if segpath in f and isinstance(f[segpath], h5py.Group):
            g = f[segpath]
            keys = [k for k in g.keys() if NUMERIC.match(k)]
            if not keys:
                continue
            k0 = sorted(keys, key=int)[0]
            sub = g[k0]
            sig = sub["signal"] if "signal" in sub else sub
            leads = list(sig.keys()) if isinstance(sig, h5py.Group) else None
            ds0 = sig[leads[0]] if leads else sig
            found.update(mode="per_segment", path=f"{segpath}/{{i}}/signal",
                         n_seg=len(keys), seg_len=int(ds0.shape[-1]),
                         dtype=str(ds0.dtype), leads=leads,
                         chunks=ds0.chunks, comp=ds0.compression)
            break

    # fs / lead 이름 보강
    for p in ("ECG/metadata", "metadata"):
        if p in f:
            a = f[p].attrs
            if "fs" in a:
                try:
                    found["fs"] = float(a["fs"])
                except Exception:
                    pass
            if "sig_name" in f[p]:
                try:
                    found["leads"] = [x.decode() if isinstance(x, bytes) else str(x)
                                      for x in f[p]["sig_name"][:]]
                except Exception:
                    pass
    if found["fs"] is None and "fs" in f.attrs:
        try:
            found["fs"] = float(f.attrs["fs"])
        except Exception:
            pass
    return found


def probe_h5(path, time_full_read=True):
    import h5py
    import numpy as np

    rep = {"path": path, "file_bytes": os.path.getsize(path)}
    t0 = time.time()
    with h5py.File(path, "r") as f:
        rep["open_sec"] = time.time() - t0
        rep["root_attrs"] = {k: (v.decode() if isinstance(v, bytes) else
                                 (v.tolist() if hasattr(v, "tolist") else str(v)))
                             for k, v in list(f.attrs.items())[:12]}

        lines = ["/"]
        t0 = time.time()
        ng, nd, na = walk_h5(f, lines)
        rep["walk_sec"] = time.time() - t0
        rep["tree"] = lines
        rep["est_groups"], rep["est_datasets"], rep["est_attrs"] = ng, nd, na

        sig = find_signal(f)
        rep["signal"] = sig

        # 24h 전체 읽기 타이밍
        if time_full_read and sig["mode"]:
            try:
                t0 = time.time()
                if sig["mode"] == "flat":
                    arr = f[sig["path"]][:]
                    nbytes = arr.nbytes
                    # (N, C) time-major 와 (C, N) channel-first 를 모두 지원.
                    # 채널 수는 항상 작으므로(<=12) 긴 축을 시간축으로 본다.
                    if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                        rep["layout"] = "time_major (N, C)"
                        total_samples, n_ch = arr.shape
                        arr = arr.T                      # 이후 통계는 (C, N) 기준
                    else:
                        rep["layout"] = "channel_first (C, N)"
                        n_ch, total_samples = (arr.shape if arr.ndim == 2
                                               else (1, arr.shape[-1]))
                else:
                    g = f[sig["path"].split("/{i}")[0]]
                    keys = sorted([k for k in g.keys() if NUMERIC.match(k)], key=int)
                    leads = sig["leads"] or []
                    nbytes = 0
                    chunks = []
                    for k in keys:
                        s = g[k]["signal"] if "signal" in g[k] else g[k]
                        chunks.append(np.stack([s[l][:] for l in leads]))
                    arr = np.concatenate(chunks, axis=-1)
                    nbytes = arr.nbytes
                    rep["layout"] = "per_segment -> (C, N)"
                    total_samples = arr.shape[-1]
                dt_ = time.time() - t0
                rep["full_read_sec"] = dt_
                rep["signal_bytes"] = nbytes
                rep["read_MBps"] = (nbytes / 1e6) / max(dt_, 1e-9)
                rep["overhead_ratio"] = rep["file_bytes"] / max(nbytes, 1)
                fs = sig["fs"] or 125.0
                rep["duration_h"] = total_samples / fs / 3600.0
                rep["signal_shape"] = tuple(arr.shape)
                rep["nan_ratio"] = float(np.isnan(arr.astype(np.float32)).mean()) \
                    if np.issubdtype(arr.dtype, np.floating) else 0.0
                rep["value_range"] = [float(np.nanmin(arr)), float(np.nanmax(arr))]
            except Exception as e:
                rep["full_read_error"] = f"{type(e).__name__}: {e}"
    return rep


def print_h5_report(rep):
    print(f"  --- {os.path.basename(rep['path'])}  ({human(rep['file_bytes'])}) ---")
    if rep.get("root_attrs"):
        print(f"      root attrs: {rep['root_attrs']}")
    print("      [트리]")
    for ln in rep["tree"][:60]:
        print("      " + ln)
    if len(rep["tree"]) > 60:
        print(f"      ... ({len(rep['tree']) - 60}줄 생략)")
    print(f"      추정 객체수: group {rep['est_groups']:,} / dataset {rep['est_datasets']:,} "
          f"/ attr {rep['est_attrs']:,}   (트리순회 {rep['walk_sec']:.2f}s)")

    s = rep["signal"]
    print(f"      신호저장: mode={s['mode']} path={s['path']} dtype={s['dtype']} "
          f"chunks={s['chunks']} comp={s['comp']}")
    if s["mode"] == "per_segment":
        print(f"                n_seg={s['n_seg']:,} seg_len={s['seg_len']} leads={s['leads']} fs={s['fs']}")
    else:
        print(f"                shape={s['shape']} leads={s['leads']} fs={s['fs']}")

    if "full_read_sec" in rep:
        print(f"      >>> 24h 전체 읽기: {rep['full_read_sec']:.2f}초  "
              f"({rep['read_MBps']:.0f} MB/s, 신호 {human(rep['signal_bytes'])}, "
              f"{rep['duration_h']:.2f}h, shape={rep['signal_shape']}, "
              f"layout={rep.get('layout','?')})")
        print(f"      >>> 파일/신호 크기비 = {rep['overhead_ratio']:.2f}x  "
              f"(1.0에 가까울수록 좋음; 2.0이면 절반이 메타데이터)")
        print(f"      >>> 값 범위 {rep['value_range']}, NaN 비율 {rep['nan_ratio']:.4f}")
    elif "full_read_error" in rep:
        print(f"      [full read 실패] {rep['full_read_error']}")
    print()


# ----------------------------------------------------------------------------
# 3) npy / wfdb 샘플
# ----------------------------------------------------------------------------
def probe_npy(path):
    import numpy as np
    a = np.load(path, mmap_mode="r")
    return {"path": path, "shape": tuple(a.shape), "dtype": str(a.dtype),
            "bytes": os.path.getsize(path)}


def probe_hea(path):
    with open(path, "r", errors="replace") as f:
        return {"path": path, "head": [l.rstrip() for l in f.readlines()[:6]]}


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=DEFAULT_DIRS)
    ap.add_argument("--samples", type=int, default=2, help="디렉토리당 내부 probe할 파일 수")
    ap.add_argument("--quick", action="store_true", help="내부 구조 probe 생략")
    ap.add_argument("--no-timing", action="store_true", help="24h 전체 읽기 타이밍 생략")
    ap.add_argument("--max-files", type=int, default=2_000_000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    results = {}
    infos = {}

    for root in args.dirs:
        info = scan_dir(root, args.max_files)
        infos[root] = info
        print_dir_report(info)
        results[root] = {k: (dict(v) if isinstance(v, collections.Counter) else
                             (sorted(v) if isinstance(v, set) else v))
                         for k, v in info.items() if k != "examples"}
        results[root]["examples"] = {k: v for k, v in info["examples"].items()}

        if args.quick or not info["exists"] or info["n_files"] == 0:
            continue

        # 가장 흔한 데이터 확장자에서 샘플
        dominant = None
        for ext, _ in info["by_ext"].most_common():
            if ext in (".h5", ".hdf5", ".npy", ".dat", ".hea", ".mat"):
                dominant = ext
                break
        if dominant is None:
            continue

        print(f"  [샘플 {dominant} 파일 내부 구조]")
        sample_paths = info["examples"][dominant][: args.samples]
        for p in sample_paths:
            try:
                if dominant in (".h5", ".hdf5"):
                    rep = probe_h5(p, time_full_read=not args.no_timing)
                    print_h5_report(rep)
                    results[root].setdefault("samples", []).append(
                        {k: v for k, v in rep.items() if k != "tree"})
                elif dominant == ".npy":
                    r = probe_npy(p)
                    print(f"      {os.path.basename(p)}: shape={r['shape']} "
                          f"dtype={r['dtype']} ({human(r['bytes'])})")
                    results[root].setdefault("samples", []).append(r)
                elif dominant in (".hea",):
                    r = probe_hea(p)
                    print(f"      {os.path.basename(p)}:")
                    for l in r["head"]:
                        print(f"        {l}")
                    results[root].setdefault("samples", []).append(r)
                else:
                    print(f"      {os.path.basename(p)}: {human(os.path.getsize(p))}")
            except Exception as e:
                print(f"      [probe 실패] {p}: {type(e).__name__}: {e}")
        print()

    # ---- 디렉토리 간 record 겹침 ----
    print(BAR)
    print("[디렉토리 간 record 겹침]")
    print(BAR)
    live = [(r, i) for r, i in infos.items() if i["exists"] and i["stems"]]
    if len(live) < 2:
        print("  비교할 디렉토리가 부족합니다.\n")
    else:
        names = [os.path.basename(r.rstrip("/")) or r for r, _ in live]
        w = max(len(n) for n in names) + 2
        print(" " * w + "".join(f"{n[:14]:>16s}" for n in names))
        for (ra, ia), na in zip(live, names):
            row = f"{na:{w}s}"
            for (rb, ib), _ in zip(live, names):
                inter = len(ia["stems"] & ib["stems"])
                row += f"{inter:>16,}"
            print(row)
        print()
        print("  (대각선 = 해당 디렉토리 고유 record 수)")
        allsets = [i["stems"] for _, i in live]
        union = set().union(*allsets)
        common = set.intersection(*allsets)
        print(f"  전체 합집합 {len(union):,} / 모든 디렉토리 공통 {len(common):,}")
        for (r, i), n in zip(live, names):
            only = i["stems"] - set().union(*[s for s in allsets if s is not i["stems"]])
            print(f"  {n:{w}s} 고유 {len(i['stems']):>7,}  (이 디렉토리에만 있음: {len(only):,})")
    print()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"[saved] {args.json}")


if __name__ == "__main__":
    main()
