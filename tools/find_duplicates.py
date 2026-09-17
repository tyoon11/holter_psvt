#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_duplicates.py — 이름은 다르지만 신호(.SIG)가 같은 record 를 찾고, 남길 것을 정한다.

배경
  같은 Holter 가 디스크 번호만 바꿔(…_12_PID / …_13_PID), 혹은 다른 파일명 체계로
  (20230901_5_PID / 68_5_PID) 여러 번 내보내져 있다. 변환 결과가 중복되면
    - 사전학습에서 일부 기록이 두 배로 반영되고
    - record 단위로 split 하면 같은 기록이 train/test 양쪽에 들어간다
  (환자(PID) 단위 split 이면 누수는 막히지만 가중치 왜곡은 남는다.)

단계
  1) .SIG 크기가 같은 묶음 → 2) 앞 4MB 해시 → 3) 전체 SHA-1 로 확정 (쓰레드 병렬)
  확정된 묶음마다
    - .ANN / .json 이 같은지 (다르면 재분석·수정된 판독일 수 있어 따로 표시)
    - 파일명 PID 가 모두 같은지 (다르면 같은 신호가 다른 환자에 붙은 것 → 반드시 확인)
    - 남길 record 1개 (keep) 선정 규칙:
        .ANN 과 .json 을 모두 가진 것 > .ANN 파일이 큰 것(주석이 더 많음) > .json 가진 것
        > 이름 사전순 (재현 가능하도록)

출력
  <out>.csv     : 중복에 속한 record 전부 (group, keep, 판정 근거)
  build_manifest.py --duplicates <out>.csv 로 manifest 에 dup_group / dup_keep 을 붙인다.

사용:
  python tools/find_duplicates.py --raw /home/coder/workspace/data/raw --workers 16
"""

import argparse
import csv
import hashlib
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

BUF = 8 << 20


def sha1(path, limit=None):
    h = hashlib.sha1()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(BUF if limit is None else min(BUF, limit - n))
            if not b:
                break
            h.update(b)
            n += len(b)
            if limit is not None and n >= limit:
                break
    return h.hexdigest()


def maybe_sha1(path):
    return sha1(path) if os.path.exists(path) else ""


def cohort_of(path, root):
    rel = os.path.relpath(os.path.dirname(path), root)
    return os.path.basename(os.path.abspath(root)) if rel == "." else rel.split(os.sep)[0]


def pid_of(name):
    return name.rsplit("_", 1)[-1] if "_" in name else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="duplicates")
    args = ap.parse_args()
    root = os.path.abspath(args.raw)

    sigs = []
    for d, _, ns in os.walk(root):
        for n in ns:
            if n.endswith(".SIG"):
                p = os.path.join(d, n)
                sigs.append((p, os.path.getsize(p)))
    by_size = defaultdict(list)
    for p, s in sigs:
        by_size[s].append(p)
    cand = [p for ps in by_size.values() if len(ps) > 1 for p in ps]
    print(f"[1/3] .SIG {len(sigs):,}개 → 크기 같은 파일 {len(cand):,}개")

    with ThreadPoolExecutor(args.workers) as ex:
        part = dict(zip(cand, ex.map(lambda p: sha1(p, 4 << 20), cand)))
    by_part = defaultdict(list)
    for p in cand:
        by_part[(os.path.getsize(p), part[p])].append(p)
    cand2 = [p for ps in by_part.values() if len(ps) > 1 for p in ps]
    print(f"[2/3] 앞 4MB 까지 같은 파일 {len(cand2):,}개")

    done = [0]
    def full(p):
        r = sha1(p); done[0] += 1
        print(f"\r[3/3] 전체 해시 {done[0]:,}/{len(cand2):,}", end="", flush=True)
        return r
    with ThreadPoolExecutor(args.workers) as ex:
        fullh = dict(zip(cand2, ex.map(full, cand2)))
    print()
    groups = defaultdict(list)
    for p in cand2:
        groups[fullh[p]].append(p)
    groups = {h: sorted(ps) for h, ps in groups.items() if len(ps) > 1}
    partial_only = len({(os.path.getsize(p), part[p]) for p in cand2}) - len(groups)

    rows, stats = [], Counter()
    for gi, (h, ps) in enumerate(sorted(groups.items(), key=lambda kv: kv[1][0])):
        recs = []
        for p in ps:
            base = p[:-4]; name = os.path.basename(base)
            ann, js = base + ".ANN", base + ".json"
            recs.append({
                "group": gi, "record": name, "cohort": cohort_of(p, root),
                "path": os.path.relpath(p, root), "pid": pid_of(name),
                "has_ann": os.path.exists(ann), "has_json": os.path.exists(js),
                "ann_bytes": os.path.getsize(ann) if os.path.exists(ann) else 0,
                "ann_sha1": maybe_sha1(ann), "json_sha1": maybe_sha1(js), "sig_sha1": h,
            })
        keep = sorted(recs, key=lambda r: (-(r["has_ann"] and r["has_json"]), -r["ann_bytes"],
                                           -r["has_json"], r["record"]))[0]
        same_name = len({r["record"] for r in recs}) < len(recs)
        ann_same = len({r["ann_sha1"] for r in recs}) == 1
        json_same = len({r["json_sha1"] for r in recs}) == 1
        pid_same = len({r["pid"] for r in recs}) == 1
        cross = len({r["cohort"] for r in recs}) > 1
        notes = []
        if not pid_same: notes.append("PID 불일치")
        if not ann_same: notes.append(".ANN 다름")
        if not json_same: notes.append(".json 다름")
        if cross: notes.append("코호트 간")
        if same_name: notes.append("이름도 같음")
        # 이름까지 같은 파일은 변환기가 먼저 찾은 하나만 v2 로 만든다. 같은 이름은 한 record 로 보고
        # keep 도 이름 단위로 준다 (안 그러면 하나뿐인 v2 파일이 제외로 찍힐 수 있다).
        for r in recs:
            r.update(keep=(r["record"] == keep["record"]), group_size=len({x["record"] for x in recs}),
                     ann_same=ann_same,
                     json_same=json_same, pid_same=pid_same, cross_cohort=cross,
                     note=";".join(notes))
            rows.append(r)
        stats["묶음"] += 1
        stats["제외할 record"] += len({r["record"] for r in recs}) - 1
        for n in notes:
            stats[n] += 1

    cols = ["group", "keep", "record", "cohort", "pid", "path", "group_size", "has_ann",
            "has_json", "ann_bytes", "ann_same", "json_same", "pid_same", "cross_cohort",
            "note", "sig_sha1", "ann_sha1", "json_sha1"]
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 70)
    print(f"확정된 중복 묶음 {stats['묶음']:,}개  →  남길 것 {stats['묶음']:,} / 제외 {stats['제외할 record']:,}")
    print(f"  (앞 4MB 만 같고 뒤가 다른 묶음 {partial_only}개는 중복 아님)")
    print(f"  묶음 크기(파일 수): {dict(Counter(len(v) for v in groups.values()))}")
    for k in ("코호트 간", "이름도 같음", ".ANN 다름", ".json 다름", "PID 불일치"):
        print(f"  {k:<14s} {stats[k]:>5,} 묶음")
    if stats["PID 불일치"]:
        print("\n  ** 같은 신호에 다른 PID 가 붙은 묶음이 있습니다. 라벨이 엉킬 수 있으니 확인하세요: **")
        for r in rows:
            if not r["pid_same"] and r["keep"]:
                g = [x for x in rows if x["group"] == r["group"]]
                print("    " + "  |  ".join(f"{x['path']}" for x in g))
    if stats[".ANN 다름"]:
        print("\n  .ANN 이 다른 묶음 예시 (주석이 더 큰 쪽을 남김):")
        shown = 0
        for r in rows:
            if not r["ann_same"] and r["keep"] and shown < 5:
                g = [x for x in rows if x["group"] == r["group"]]
                print("    " + "  |  ".join(f"{x['record']}({x['ann_bytes']:,}B{'*' if x['keep'] else ''})" for x in g))
                shown += 1
    print(f"\n  [saved] {args.out}.csv   (* = keep)")


if __name__ == "__main__":
    main()
