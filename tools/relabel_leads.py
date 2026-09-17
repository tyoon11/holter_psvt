#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
relabel_leads.py — v2 파일의 lead 이름(sig_name attr)만 바꾼다. 신호는 건드리지 않는다.

run_conversion.py --lead-mode file 로 변환하면 채널이 원본 순서 그대로 ch0,ch1,ch2 로
저장된다. lead 가 확정되면 이 스크립트로 이름만 붙인다 (파일당 수 ms, 재변환 불필요).

코호트나 파일명 체계마다 순서가 다르면 --where 로 나눠 여러 번 실행한다.
바꾸기 전 이름은 sig_name_prev attr 에 남기므로 되돌릴 수 있다.

사용:
  python tools/relabel_leads.py --dir /home/coder/workspace/data/holter_v2 --names V5,V1,II --dry-run
  python tools/relabel_leads.py --dir ... --names V5,V1,II
  python tools/relabel_leads.py --dir ... --names II,V1,V5 --where cohort=nas1_Holter_TOF_250917
  python tools/relabel_leads.py --dir ... --revert
"""

import argparse
import json
import os
import sys
from collections import Counter

import h5py


def _s(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--names", help="파일 채널 순서대로의 lead 이름, 예: V5,V1,II")
    ap.add_argument("--where", action="append", default=[],
                    help="root attr 조건 key=value (여러 번 가능). 예: cohort=nas1_Holter_PSVT")
    ap.add_argument("--only-file-mode", action="store_true", default=True,
                    help="lead_mode=file 로 변환된 파일만 대상 (기본)")
    ap.add_argument("--revert", action="store_true", help="sig_name_prev 로 되돌림")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.revert and not args.names:
        sys.exit("--names 또는 --revert 가 필요합니다.")
    names = args.names.split(",") if args.names else None
    cond = dict(w.split("=", 1) for w in args.where)

    files = sorted(e.path for e in os.scandir(args.dir) if e.name.endswith(".h5"))
    stats = Counter()
    for p in files:
        with h5py.File(p, "r" if args.dry_run else "r+") as f:
            a = f.attrs
            if any(_s(a.get(k, "")) != v for k, v in cond.items()):
                stats["조건 불일치"] += 1; continue
            if args.only_file_mode and not args.revert and _s(a.get("lead_mode", "")) != "file":
                stats["lead_mode≠file (재배열된 파일이라 이름만 바꾸면 안 됨)"] += 1; continue
            cur = json.loads(_s(a["sig_name"]))
            if args.revert:
                if "sig_name_prev" not in a:
                    stats["되돌릴 기록 없음"] += 1; continue
                if not args.dry_run:
                    a["sig_name"] = _s(a["sig_name_prev"]); del a["sig_name_prev"]
                stats["되돌림"] += 1; continue
            if len(names) != len(cur):
                stats[f"채널 수 불일치({len(cur)})"] += 1; continue
            if cur == names:
                stats["이미 같음"] += 1; continue
            if not args.dry_run:
                a["sig_name_prev"] = json.dumps(cur)
                a["sig_name"] = json.dumps(names)
                a["lead_source"] = "relabeled"
            stats["변경"] += 1
    print(f"[relabel] {args.dir}  {len(files):,}개 {'(dry-run)' if args.dry_run else ''}")
    for k, v in stats.most_common():
        print(f"  {k:<50s} {v:>6,}")


if __name__ == "__main__":
    main()
