#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
list_pids.py — 코호트별 PID 목록을 txt 로 뽑고, PID 출처끼리 어디서 어긋나는지 보여준다.

PID 출처는 셋이다.
  - 파일명 끝 토큰       10_50_2247355 → 2247355 (fix_pid.py 규약, h5 의 pid attr 도 이것)
  - .json 리포트          Holter Report/PatientInfo/PID
                          (fix_pid 적용본은 파일명으로 덮어써져 있으므로, 원래 값을 보려면
                           NAS 원본 경로를 --raw 로 준다)
  - 임상 CSV              --clinical name=path[:컬럼]. 컬럼을 안 주면 파일명 PID 와 가장 많이
                          겹치는 열을 고른다. 값이 record 이름(10_50_2247355)이면 끝 토큰을
                          PID 로 쓴다 (clinical_data_tof.csv 는 pid 열이 연구 번호이고
                          base_name 이 record 이름이다)

비교할 때는 공백 · 끝의 '.0' · 앞자리 0 을 무시한다. 그래야만 맞는 경우는 따로 센다
(엑셀/pandas 를 거치며 '0123456' → '123456' 이나 '123456.0' 이 되는 경우).

출력 (--out 디렉토리)
  pids_<코호트>.txt                     파일명 PID, 고유값 정렬, 한 줄에 하나
  records_<코호트>.tsv                  record, pid_file, pid_json, json_일치
  clinical_<name>_ids.txt               임상 CSV 에서 고른 열의 ID
  clinical_<name>_unmatched.txt         임상 CSV 에 있지만 어느 코호트에도 없는 ID
  <코호트>_not_in_<name>.txt            코호트에 있지만 임상 CSV 에 없는 PID
  pid_summary.txt                       위 내용 요약 (화면 출력과 같음)

사용:
  python tools/list_pids.py --raw /home/coder/workspace/Holter_TOF/holter_total/raw \\
      --clinical psvt=/home/coder/workspace/Holter_TOF/holter_total/labels/clinical_data_psvt.csv \\
                 tof=/home/coder/workspace/Holter_TOF/holter_total/labels/clinical_data_tof.csv \\
      --out pid_check

  # .json 의 원래 PID 와 비교하려면 NAS 원본을 (읽기만 한다)
  python tools/list_pids.py --raw /home/coder/workspace/Holter_TOF --cohorts nas1_Holter_PSVT \\
      nas1_Holter_TOF_250917 nas1_Holter_LQT_260615 --out pid_check_orig
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict


def pid_of(name):
    """DVD20130523_8_2522134 → 2522134"""
    return name.rsplit("_", 1)[-1] if "_" in name else name


def norm(v):
    """비교용 키: 공백 · 끝 '.0' · 앞자리 0 제거."""
    s = str(v).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s.lstrip("0") or s


def json_pid(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            d = json.load(f)
        return str(d.get("Holter Report", {}).get("PatientInfo", {}).get("PID", ""))
    except (OSError, ValueError, AttributeError) as e:
        return f"<읽기 실패: {type(e).__name__}>"


def scan(root, only):
    """root 아래 첫 디렉토리를 코호트로 보고 record(.hea 또는 .SIG 가 있는 stem)를 모은다."""
    root = os.path.abspath(root)
    recs = defaultdict(dict)                       # cohort → {stem: json 경로 or None}
    for dp, dirs, names in os.walk(root):
        rel = os.path.relpath(dp, root)
        cohort = os.path.basename(root) if rel == "." else rel.split(os.sep)[0]
        if only and cohort not in only:
            if rel == ".":
                dirs[:] = [d for d in dirs if d in only]
            continue
        nameset = set(names)
        for nm in names:
            stem, ext = os.path.splitext(nm)
            if ext not in (".hea", ".SIG"):
                continue
            j = stem + ".json"
            recs[cohort].setdefault(stem, os.path.join(dp, j) if j in nameset else None)
    return recs


def to_pid(v, stems):
    """임상 CSV 값 → 원래 표기의 PID. record 이름이면 끝 토큰을 쓴다."""
    s = str(v).strip()
    if s.endswith(".0") and s[:-2] in stems:
        s = s[:-2]
    return pid_of(s) if s in stems else s


def pick_column(rows, header, ref_keys, stems):
    """임상 CSV 에서 파일명 PID 와 가장 많이 겹치는 열. (일치 수, 고유값 수, 열, record열 여부)"""
    scores = []
    for c in header:
        raw = {str(r.get(c, "")).strip() for r in rows} - {""}
        if len(raw) < 2:
            continue
        vals = {norm(to_pid(x, stems)) for x in raw}
        n_rec = sum(x in stems or x[:-2] in stems for x in raw)
        scores.append((len(vals & ref_keys), len(raw), c, n_rec > len(raw) / 2))
    scores.sort(reverse=True)
    return scores


def write_lines(path, items):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(f"{x}\n" for x in items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="코호트 디렉토리들이 있는 루트")
    ap.add_argument("--cohorts", nargs="*", default=[], help="이 코호트만 (기본: 전부)")
    ap.add_argument("--clinical", nargs="*", default=[], help="name=path 또는 name=path:컬럼")
    ap.add_argument("--no-json", action="store_true", help=".json PID 읽기 생략 (빠름)")
    ap.add_argument("--out", default="pid_check")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    recs = scan(args.raw, set(args.cohorts))
    if not recs:
        sys.exit(f"record 를 찾지 못했습니다: {args.raw}")

    # ---- 코호트별 ----
    cohort_keys = {}                               # cohort → {norm pid: 원래 pid}
    all_stems = {s for v in recs.values() for s in v}
    log(f"[코호트] {os.path.abspath(args.raw)}")
    for cohort in sorted(recs):
        stems = recs[cohort]
        pids = sorted({pid_of(s) for s in stems})
        write_lines(os.path.join(args.out, f"pids_{cohort}.txt"), pids)
        cohort_keys[cohort] = {norm(p): p for p in pids}

        n_json = n_diff = 0
        diffs = []
        with open(os.path.join(args.out, f"records_{cohort}.tsv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["record", "pid_file", "pid_json", "json_match"])
            for s in sorted(stems):
                pf = pid_of(s)
                jp = stems[s]
                pj = json_pid(jp) if (jp and not args.no_json) else ""
                same = "" if not pj else ("Y" if norm(pj) == norm(pf) else "N")
                n_json += bool(pj)
                if same == "N":
                    n_diff += 1
                    diffs.append((s, pj))
                w.writerow([s, pf, pj, same])
        log(f"  {cohort:<28s} record {len(stems):>6,}  환자(PID) {len(pids):>6,}"
            + ("" if args.no_json else f"  json {n_json:,} 중 파일명과 다름 {n_diff:,}"))
        for s, pj in diffs[:3]:
            log(f"      예) {s}  json PID={pj}")

    # ---- 코호트 간 겹침 ----
    names = sorted(cohort_keys)
    if len(names) > 1:
        log("\n[코호트 간 같은 PID 환자 수]")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                n = len(cohort_keys[a].keys() & cohort_keys[b].keys())
                log(f"  {a} ∩ {b}: {n:,}")

    all_keys = set().union(*(set(v) for v in cohort_keys.values()))

    # ---- 임상 CSV ----
    for spec in args.clinical:
        if "=" not in spec:
            log(f"\n[skip] --clinical 형식 오류: {spec}")
            continue
        name, rest = spec.split("=", 1)
        path, col = (rest.rsplit(":", 1) + [None])[:2] if rest.count(":") else (rest, None)
        if not os.path.isfile(path):
            log(f"\n[skip] 없는 파일: {path}")
            continue
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
            rd = csv.DictReader(f)
            header = [h.strip() for h in (rd.fieldnames or [])]
            rows = [{(k or "").strip(): v for k, v in r.items()} for r in rd]
        log(f"\n[clinical:{name}] {os.path.basename(path)}  {len(rows):,}행")

        scores = pick_column(rows, header, all_keys, all_stems)
        log("  열별 파일명 PID 일치 (고유 일치 / 고유값):  " + "   ".join(
            f"{c}{'(record)' if isrec else ''} {hit:,}/{n:,}"
            for hit, n, c, isrec in scores[:5]))
        if col is None:
            if not scores or scores[0][0] == 0:
                log("  ** 파일명 PID 와 겹치는 열이 없습니다. name=path:컬럼 으로 지정하세요 **")
                continue
            col = scores[0][2]
        if col not in header:
            log(f"  ** '{col}' 열이 없습니다. 열: {header}")
            continue

        raw_ids = [to_pid(r.get(col, ""), all_stems) for r in rows]
        ids = sorted(set(raw_ids) - {""})
        exact = set(ids) & {p for v in cohort_keys.values() for p in v.values()}
        keys = {norm(x): x for x in ids}
        log(f"  사용 열 '{col}': 고유 ID {len(ids):,} (빈 칸 {raw_ids.count(''):,}행)")
        log(f"  어느 코호트든 일치: {len(keys.keys() & all_keys):,}"
            f"  (그대로 일치 {len(exact):,}, 앞자리0/.0 무시해야 일치 "
            f"{len(keys.keys() & all_keys) - len(exact):,})")
        write_lines(os.path.join(args.out, f"clinical_{name}_ids.txt"), ids)
        unmatched = sorted(keys[k] for k in keys.keys() - all_keys)
        write_lines(os.path.join(args.out, f"clinical_{name}_unmatched.txt"), unmatched)
        log(f"  임상 CSV 에만 있음: {len(unmatched):,}"
            + (f"  예) {', '.join(unmatched[:5])}" if unmatched else ""))
        for cohort in names:
            ck = cohort_keys[cohort]
            hit = len(ck.keys() & keys.keys())
            missing = sorted(ck[k] for k in ck.keys() - keys.keys())
            write_lines(os.path.join(args.out, f"{cohort}_not_in_{name}.txt"), missing)
            log(f"    {cohort:<28s} {hit:>6,}/{len(ck):,} 환자 일치  ({len(missing):,} 없음)")

    with open(os.path.join(args.out, "pid_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[saved] {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
