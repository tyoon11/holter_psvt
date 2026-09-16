#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plan_migration.py — 무엇을 로컬로 옮길지 정하고 복사 목록까지 만든다.

NAS 읽기가 86 MB/s 라 변환도 학습도 I/O 에 묶인다. 로컬로 옮기는 게 맞는데,
"무엇을" 옮길지는 (a) 임상 CSV 가 정의하는 대상 코호트, (b) 원본 raw 가 남아
있는지, (c) 이미 변환된 h5 가 있는지에 따라 달라진다. 이 셋을 대조해 결정한다.

하는 일
  1. 임상 CSV 구조를 읽고 ID 컬럼을 자동 탐지 (record 이름 / PID 양쪽 시도)
  2. raw(.dat/.hea/.json)를 record 단위로 묶어 목록과 용량 산출
  3. 기존 h5 코호트와 대조
  4. record 마다 상태를 매기고, 전략에 따라 복사 대상과 용량을 산출
  5. rsync --files-from 으로 바로 쓸 수 있는 목록 파일 생성

사용:
  python plan_migration.py \
      --raw  /home/coder/workspace/Holter_TOF \
      --h5   /home/coder/workspace/Holter_TOF/holter_h5 \
             /home/coder/workspace/Holter_TOF/nas1_Holter_PSVT/h5 \
      --clinical psvt=/home/coder/workspace/Holter_TOF/clinical_data_psvt.csv \
                 tof=/home/coder/workspace/Holter_TOF/clinical_data_tof.csv \
      --dest /home/coder/workspace/data/raw \
      --strategy all --out migration_plan
"""

import argparse
import collections
import csv as csvmod
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from survey_workspace import show_csv, human  # noqa: E402

# 원본 한 벌 = .hea + 신호 + (.ann) + (.json).
# 신호는 MARS export 의 .SIG 이며 WFDB 표준 .dat 가 아니다.
# .xml 은 별도 modality(10초 ECG)라 record 묶음에서 제외한다.
RAW_EXT = (".dat", ".sig", ".hea", ".json", ".ann", ".atr")
SIGNAL_EXT = (".sig", ".dat")
# h5_converter/utils.py 의 has_all_required_files 와 같은 조건.
# 넷 중 하나라도 없으면 convert_to_h5.py 가 건너뛴다.
CONVERT_REQ = (".hea", ".sig", ".ann", ".json")
BAR = "=" * 78


def stem(name):
    root, ext = os.path.splitext(os.path.basename(name))
    return root if ext else os.path.basename(name)


def pid_of(name):
    """DVD20130523_8_2522134 → 2522134 (fix_pid.py 규약: 파일명 끝 토큰이 PID)."""
    return name.rsplit("_", 1)[-1] if "_" in name else name


# =============================================================================
def scan_raw(roots):
    """raw 파일을 record stem 단위로 묶는다. {stem: {ext: (path, bytes)}}"""
    recs = collections.defaultdict(dict)
    for root in roots:
        if not os.path.isdir(root):
            print(f"  [skip] 없는 디렉토리: {root}")
            continue
        for dirpath, _, names in os.walk(root):
            for nm in names:
                ext = os.path.splitext(nm)[1].lower()
                if ext not in RAW_EXT:
                    continue
                p = os.path.join(dirpath, nm)
                try:
                    recs[stem(nm)][ext] = (p, os.path.getsize(p))
                except OSError:
                    pass
    return recs


def scan_h5(dirs):
    out = {}
    for d in dirs:
        if not os.path.isdir(d):
            print(f"  [skip] 없는 디렉토리: {d}")
            continue
        names = {}
        for e in os.scandir(d):
            if e.is_file() and e.name.lower().endswith((".h5", ".hdf5")):
                try:
                    names[stem(e.name)] = e.stat().st_size
                except OSError:
                    names[stem(e.name)] = 0
        out[os.path.basename(d.rstrip("/"))] = names
    return out


def read_csv_ids(path, universe_by_kind):
    """CSV 에서 ID 컬럼을 찾아 (컬럼명, 매칭기준, 값집합) 반환."""
    try:
        with open(path, "r", errors="replace", newline="") as f:
            sample = f.read(64 * 1024); f.seek(0)
            try:
                sep = csvmod.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csvmod.Error:
                sep = ","
            rows = list(csvmod.reader(f, delimiter=sep))
    except OSError as e:
        print(f"  [읽기 실패] {path}: {e}")
        return None
    if len(rows) < 2:
        return None
    header, body = rows[0], rows[1:]
    cols = list(zip(*body))
    best = None
    for i, name in enumerate(header):
        vals = {v.strip() for v in cols[i] if v.strip()}
        if len(vals) < 2:
            continue
        for kind, uni in universe_by_kind.items():
            hit = len(vals & uni)
            if hit and (best is None or hit > best[3]):
                best = (name, kind, vals, hit)
    return best


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="*", default=[], help="원본 탐색 루트 (재귀)")
    ap.add_argument("--h5", nargs="*", default=[], help="기존 h5 코호트 디렉토리")
    ap.add_argument("--clinical", nargs="*", default=[], help="name=path")
    ap.add_argument("--dest", default="/home/coder/workspace/data/raw")
    ap.add_argument("--strategy", default="all",
                    choices=["all", "convertible", "csv", "unconverted", "converted"],
                    help="all=신호 있는 것 전부 / convertible=4종 완비(권장) / "
                         "csv=임상 CSV 에 있는 것만 / unconverted=아직 h5 없는 것만 / "
                         "converted=이미 h5 있는 것만")
    ap.add_argument("--out", default="migration_plan")
    args = ap.parse_args()

    # ---- raw / h5 스캔 ----
    print(BAR); print("[1] 원본(raw) 스캔"); print(BAR)
    raw = scan_raw(args.raw) if args.raw else {}
    # 신호(.SIG/.dat) + .hea 가 있어야 최소한 읽을 수 있다.
    complete = {k: v for k, v in raw.items()
                if ".hea" in v and any(e in v for e in SIGNAL_EXT)}
    # 넷을 다 갖춰야 convert_to_h5.py 가 변환한다.
    convertible = {k: v for k, v in complete.items()
                   if all(e in v for e in CONVERT_REQ)}
    if raw:
        ext_n = collections.Counter(e for v in raw.values() for e in v)
        print(f"  raw record stem {len(raw):,}개")
        print(f"  확장자별 파일 수: {dict(ext_n)}")
        print(f"  신호+헤더 보유        {len(complete):,}")
        print(f"  변환 가능(.hea+.SIG+.ANN+.json)  {len(convertible):,}   "
              f"← utils.py has_all_required_files 조건")
        missing = collections.Counter()
        for k, v in complete.items():
            for e in CONVERT_REQ:
                if e not in v:
                    missing[e] += 1
        if missing:
            print(f"  변환 불가 사유(중복 집계): "
                  + "  ".join(f"{e} 없음 {n:,}" for e, n in missing.most_common()))
        tot = sum(sz for v in convertible.values() for _, sz in v.values())
        print(f"  변환 대상 총 용량 {human(tot)}  "
              f"(record당 평균 {human(tot/max(len(convertible),1))})")
        dirs = collections.Counter()
        for v in complete.values():
            sig = next((v[e][0] for e in SIGNAL_EXT if e in v), None)
            if sig:
                dirs[os.path.dirname(sig)] += 1
        for d, n in dirs.most_common(6):
            nconv = sum(1 for k, v in convertible.items()
                        if os.path.dirname(next(v[e][0] for e in SIGNAL_EXT if e in v)) == d)
            print(f"    {d}\n      신호 {n:,}개 중 변환 가능 {nconv:,}개")
    else:
        print("  ** 원본을 찾지 못했습니다 (--raw 미지정이거나 h5 만 남음) **")
    print()

    print(BAR); print("[2] 기존 h5 코호트"); print(BAR)
    h5 = scan_h5(args.h5) if args.h5 else {}
    for name, d in h5.items():
        print(f"  {name:28s} {len(d):>6,}개  {human(sum(d.values()))}")
    all_h5 = set().union(*h5.values()) if h5 else set()
    print(f"  h5 고유 record {len(all_h5):,}개")
    print()

    # ---- 임상 CSV ----
    print(BAR); print("[3] 임상 CSV"); print(BAR)
    universe = set(raw) | all_h5
    universe_by_kind = {
        "record이름": universe,
        "PID(파일명 끝토큰)": {pid_of(n) for n in universe if "_" in n},
    }
    keysets = {k: set(v) for k, v in h5.items()}
    if raw:
        keysets["<raw>"] = set(raw)

    csv_ids = {}
    for spec in args.clinical:
        if "=" not in spec:
            print(f"  [skip] 형식 오류: {spec}")
            continue
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"  [skip] 없는 파일: {path}")
            continue
        print(f"\n### {name}")
        show_csv(path, keysets)
        hit = read_csv_ids(path, universe_by_kind)
        if hit is None:
            print(f"  ** {name}: ID 컬럼을 찾지 못했습니다 **")
            continue
        col, kind, vals, nhit = hit
        # 값 → record 이름으로 환원
        if kind.startswith("PID"):
            back = {n for n in universe if pid_of(n) in vals}
        else:
            back = vals & universe
        csv_ids[name] = back
        print(f"  ▶ ID 컬럼 '{col}' ({kind}) — CSV {len(vals):,}개 중 "
              f"보유 데이터와 일치 {len(back):,}개, 미보유 {len(vals)-nhit:,}개")
    print()

    # ---- record 별 상태 ----
    print(BAR); print("[4] record 상태"); print(BAR)
    rows = []
    for name in sorted(universe):
        r = complete.get(name, {})
        row = {
            "record": name, "pid": pid_of(name),
            "has_raw": bool(r),
            "has_sig": any(e in r for e in SIGNAL_EXT),
            "has_ann": ".ann" in r,
            "has_json": ".json" in r,
            "convertible": name in convertible,
            "raw_dir": (os.path.dirname(next((r[e][0] for e in SIGNAL_EXT if e in r), ""))
                        if r else ""),
            "raw_bytes": sum(sz for _, sz in r.values()),
        }
        for c, d in h5.items():
            row[f"in_{c}"] = name in d
        row["in_any_h5"] = name in all_h5
        for cname, ids in csv_ids.items():
            row[f"csv_{cname}"] = name in ids
        row["in_any_csv"] = any(row.get(f"csv_{c}") for c in csv_ids) if csv_ids else False
        rows.append(row)

    def cnt(pred):
        return sum(1 for r in rows if pred(r))

    print(f"  전체 record(raw ∪ h5): {len(rows):,}")
    print(f"    raw 보유            {cnt(lambda r: r['has_raw']):>6,}"
          f"   (.ann {cnt(lambda r: r['has_ann']):,} / .json {cnt(lambda r: r['has_json']):,})")
    print(f"    변환 가능           {cnt(lambda r: r['convertible']):>6,}")
    print(f"    변환 가능 & h5 없음 {cnt(lambda r: r['convertible'] and not r['in_any_h5']):>6,}"
          f"   ← 변환하면 완전한 record 순증")
    print(f"    h5 보유             {cnt(lambda r: r['in_any_h5']):>6,}")
    print(f"    raw 만 (미변환)     {cnt(lambda r: r['has_raw'] and not r['in_any_h5']):>6,}"
          f"   ← 변환하면 순증")
    print(f"    h5 만 (raw 소실)    {cnt(lambda r: not r['has_raw'] and r['in_any_h5']):>6,}"
          f"   ← repack 만 가능")
    print(f"    둘 다               {cnt(lambda r: r['has_raw'] and r['in_any_h5']):>6,}")
    if csv_ids:
        print(f"    임상 CSV 에 있음    {cnt(lambda r: r['in_any_csv']):>6,}")
        for cname in csv_ids:
            print(f"      {cname:<10s}        {cnt(lambda r, c=cname: r.get(f'csv_{c}')):>6,}")
    print()

    # ---- 복사 계획 ----
    print(BAR); print(f"[5] 복사 계획  (전략: {args.strategy})"); print(BAR)
    if args.strategy == "all":
        sel = [r for r in rows if r["has_raw"]]
        why = "원본이 있는 record 전부"
    elif args.strategy == "convertible":
        sel = [r for r in rows if r["convertible"]]
        why = ".hea+.SIG+.ANN+.json 4종을 갖춰 바로 변환 가능한 record"
    elif args.strategy == "csv":
        sel = [r for r in rows if r["has_raw"] and r["in_any_csv"]]
        why = "임상 CSV 에 포함되고 원본이 있는 record"
    elif args.strategy == "unconverted":
        sel = [r for r in rows if r["has_raw"] and not r["in_any_h5"]]
        why = "원본은 있는데 아직 h5 가 없는 record"
    else:
        sel = [r for r in rows if r["has_raw"] and r["in_any_h5"]]
        why = "이미 h5 가 있고 원본도 있는 record (재변환용)"

    total = sum(r["raw_bytes"] for r in sel)
    print(f"  대상: {why}")
    print(f"  {len(sel):,} record   원본 {human(total)}")
    est_v2 = len(sel) * 65e6
    print(f"  변환 후 v2 예상 {human(est_v2)}  (record당 65 MB)")
    print(f"  필요 여유공간 ≈ {human(total + est_v2)}  (원본 + 산출물 동시 보관 시)")
    print(f"  → {args.dest}")

    if not sel and all_h5:
        print()
        print("  ** 원본이 없어 복사할 raw 가 없습니다. **")
        print("     이 경우 대안은 기존 h5 를 로컬로 옮긴 뒤 repack 하는 것입니다:")
        for name, d in h5.items():
            print(f"       {name:24s} {len(d):>6,}개  {human(sum(d.values()))}")
        print("     단 repack 은 원본 h5 에 없는 정보(주석 등)를 만들어내지 못합니다.")

    # ---- 산출물 ----
    files_txt = args.out + "_files.txt"
    table_csv = args.out + "_records.csv"
    if sel:
        srcs = sorted({r["raw_dir"] for r in sel if r["raw_dir"]})
        base = os.path.commonpath(srcs) if len(srcs) > 1 else srcs[0]
        with open(files_txt, "w") as f:
            for r in sel:
                for _, (p, _sz) in sorted(raw[r["record"]].items()):
                    f.write(os.path.relpath(p, base) + "\n")
        print(f"\n[saved] {files_txt}  (rsync 목록, 기준 경로 {base})")
        print(f"  실행:  rsync -a --info=progress2 --files-from={files_txt} \\")
        print(f"           {base}/ {args.dest}/")

    if rows:
        cols = sorted({k for r in rows for k in r})
        with open(table_csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(rows)
        print(f"[saved] {table_csv}  (record별 상태표 {len(rows):,}행)")


if __name__ == "__main__":
    main()
