#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_splits.py — manifest 로 환자 단위 train/val/test split 과 downstream 라벨을 만든다.

정의 (2026-09-17 합의)
  SSL        : train 환자의 모든 적격 record (downstream 라벨 유무와 무관)
  LongQT     : LQT 코호트 record = 1, 그 외 코호트 = 0
  TOF        : TOF 코호트 record = 1, 그 외 코호트 = 0
  PSVT       : clinical_data_psvt 의 Label (psvt_Label) 1/0, 없으면 PSVT 태스크에서 제외(NaN)
  split      : 환자(PID) 단위 70/10/20. 한 환자의 record 는 모두 같은 split
  비율 유지  : 각 태스크의 val/test pos:neg 가 전체와 같도록 층화

적격 record
  - dup_keep == True                     (신호 중복 중 하나만)
  - duration_h >= --min-hours (기본 12)   (너무 짧은 기록 제외)
  - flat_seg_ratio <= --max-flat (기본 0.2) (무신호 구간 과다 제외)
  - dup_pid_conflict 인 record 는 SSL 에는 남기되 모든 downstream 라벨을 NaN 으로
    (같은 신호가 다른 PID 에 붙어 있어 어느 쪽 라벨이 맞는지 알 수 없음)

층화 방법
  환자마다 (소속 코호트 집합, PSVT 라벨 상태, 촬영 횟수 1/2/3+)로 층을 만들고,
  층 안에서 환자를 무작위로 섞은 뒤 record 수 기준으로 목표 비율에 가장 모자란
  split 부터 채운다. 크기순 정렬은 하지 않는다(다회 촬영 환자가 train 에 몰림).

출력
  <out>.csv          manifest 전체 열 + split, eligible, y_lqt, y_tof, y_psvt
  <out>_summary.txt  split × 태스크 pos/neg/유병률, 코호트 간 교란 요인 점검

사용:
  python tools/make_splits.py --manifest $OUT/manifest.csv --out $OUT/splits
"""

import argparse
import os
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

LQT = "nas1_Holter_LQT_260615"
TOF = "nas1_Holter_TOF_250917"
PSVT = "nas1_Holter_PSVT"
SPLITS = ("train", "val", "test")


def as_bool(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def to_label(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f if f in (0.0, 1.0) else np.nan


def allocate(df, ratios, seed):
    """환자 단위 층화 배정. 반환: pid → split"""
    rng = random.Random(seed)
    pat = df.groupby("pid").agg(
        n=("record_name", "size"),
        cohorts=("cohort", lambda x: "+".join(sorted(set(x)))),
        psvt=("y_psvt", lambda x: "pos" if (x == 1).any() else "neg" if (x == 0).any() else "na"),
    )
    # 촬영 횟수(1 / 2 / 3회 이상)도 층에 넣는다. 추적 관찰이 잦은 환자는 대개 더 아프므로
    # split 간에 이 분포가 달라지면 test 집단의 성격이 바뀐다.
    pat["nbin"] = pat["n"].clip(upper=3).astype(str)
    pat["stratum"] = pat["cohorts"] + "|" + pat["psvt"] + "|n" + pat["nbin"]
    target = np.array(ratios) / sum(ratios)
    assign = {}
    for _, g in pat.groupby("stratum"):
        pids = list(g.index)
        rng.shuffle(pids)
        # 크기 순으로 정렬하지 않는다. record 많은 환자부터 채우면 그런 환자(대개 추적 관찰이
        # 잦은, 더 아픈 환자)가 train 에 몰리고 val/test 는 1회 촬영 환자 위주가 된다
        # (합성 검증에서 환자 비율 54/18/28 로 확인). 무작위 순서 + 부족분 우선으로 채운다.
        filled = np.zeros(3)
        total = g["n"].sum()
        for p in pids:
            deficit = target * total - filled
            k = int(np.argmax(deficit))
            assign[p] = SPLITS[k]
            filled[k] += g.loc[p, "n"]
    return assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.1, 0.2])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-hours", type=float, default=12.0)
    ap.add_argument("--max-flat", type=float, default=0.2)
    ap.add_argument("--psvt-label-col", default="psvt_Label")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, low_memory=False)
    df["pid"] = df["pid"].astype(str)
    n0 = len(df)
    out_lines = []

    def log(msg=""):
        print(msg)
        out_lines.append(msg)

    # ---- 적격 ----
    keep = as_bool(df["dup_keep"]) if "dup_keep" in df else pd.Series(True, index=df.index)
    long_enough = pd.to_numeric(df["duration_h"], errors="coerce") >= args.min_hours
    flat = pd.to_numeric(df.get("flat_seg_ratio", 0), errors="coerce").fillna(0) <= args.max_flat
    df["eligible"] = keep & long_enough & flat
    conflict = as_bool(df["dup_pid_conflict"]) if "dup_pid_conflict" in df else pd.Series(False, index=df.index)

    log(f"[적격] manifest {n0:,} record")
    log(f"  중복 제외 {int((~keep).sum()):,} / {args.min_hours:g}h 미만 {int((keep & ~long_enough).sum()):,}"
        f" / 무신호>{args.max_flat:g} {int((keep & long_enough & ~flat).sum()):,}"
        f"  → 적격 {int(df['eligible'].sum()):,} record, 환자 {df.loc[df['eligible'], 'pid'].nunique():,}")

    # ---- 라벨 ----
    df["y_lqt"] = (df["cohort"] == LQT).astype(float)
    df["y_tof"] = (df["cohort"] == TOF).astype(float)
    if args.psvt_label_col not in df:
        sys.exit(f"'{args.psvt_label_col}' 열이 manifest 에 없습니다. build_manifest 의 --clinical psvt=... 확인")
    df["y_psvt"] = df[args.psvt_label_col].map(to_label)
    for c in ("y_lqt", "y_tof", "y_psvt"):
        df.loc[conflict, c] = np.nan
    log(f"  PID 불일치 record {int(conflict.sum())}개: 모든 downstream 라벨 NaN (SSL 에는 사용)")

    # 한 환자가 여러 코호트에 걸친 경우 (LQT/TOF 라벨이 환자 안에서 다름)
    el = df[df["eligible"]]
    multi = el.groupby("pid")["cohort"].nunique()
    if (multi > 1).any():
        log(f"  여러 코호트에 걸친 환자 {int((multi > 1).sum())}명 — record 라벨은 각 record 코호트를 따름")

    # ---- 배정 ----
    assign = allocate(el, args.ratios, args.seed)
    df["split"] = df["pid"].map(assign)
    df.loc[~df["eligible"], "split"] = np.nan

    # 검증: 환자가 두 split 에 걸치지 않는가
    leak = df.dropna(subset=["split"]).groupby("pid")["split"].nunique()
    assert (leak <= 1).all(), "환자가 여러 split 에 걸쳤습니다"

    # ---- 요약 ----
    el = df[df["eligible"]]
    log("\n[split] record / 환자")
    n_pat = el["pid"].nunique()
    for s in SPLITS:
        sub = el[el["split"] == s]
        np_ = sub["pid"].nunique()
        log(f"  {s:<5s} {len(sub):>6,} record ({100*len(sub)/len(el):4.1f}%)  "
            f"환자 {np_:>5,} ({100*np_/n_pat:4.1f}%)  환자당 record {len(sub)/max(np_,1):.2f}")
    log(f"  SSL 사전학습 = train {int((el['split']=='train').sum()):,} record "
        f"(downstream 라벨 없는 record 포함)")

    log("\n[태스크별 pos:neg]  (전체 유병률과 val/test 유병률이 같아야 함)")
    for task, col in (("LongQT", "y_lqt"), ("TOF", "y_tof"), ("PSVT", "y_psvt")):
        t = el.dropna(subset=[col])
        allp = t[col].mean()
        log(f"  {task}")
        for s in ("전체",) + SPLITS:
            sub = t if s == "전체" else t[t["split"] == s]
            pos, neg = int((sub[col] == 1).sum()), int((sub[col] == 0).sum())
            prev = pos / max(pos + neg, 1)
            flag = "" if s == "전체" or abs(prev - allp) < 0.02 else "  ← 차이 2%p 이상"
            log(f"    {s:<5s} pos {pos:>5,} : neg {neg:>5,}   유병률 {100*prev:5.2f}%  "
                f"(환자 pos {sub.loc[sub[col]==1,'pid'].nunique():,})"
                + flag)

    # ---- 교란 요인 점검 ----
    log("\n[교란 점검] 코호트별 연령·기록 연도·길이 (LongQT/TOF 가 코호트 소속으로 정의되므로)")
    age = pd.to_numeric(el.get("age"), errors="coerce").replace(-1, np.nan)
    year = pd.to_numeric(el["base_date"].astype(str).str[:4], errors="coerce")
    tmp = pd.DataFrame({"cohort": el["cohort"], "age": age, "year": year,
                        "dur": pd.to_numeric(el["duration_h"], errors="coerce"),
                        "hr": pd.to_numeric(el.get("mean_hr_bpm"), errors="coerce")})
    for c, g in tmp.groupby("cohort"):
        q = lambda s: f"{s.median():.0f} [{s.quantile(.25):.0f}-{s.quantile(.75):.0f}]" if s.notna().any() else "-"
        log(f"  {c:<26s} n={len(g):>5,}  나이 {q(g['age']):<14s} 기록연도 {q(g['year']):<18s}"
            f" 길이(h) {g['dur'].median():.1f}  평균HR {q(g['hr'])}")
    log("  ※ 코호트 간 나이/연도 분포가 크게 다르면 모델이 질환 대신 이것을 학습할 수 있다.")
    log("    보고 시 나이·연도 매칭 또는 층화 평가를 함께 제시할 것.")

    df.to_csv(args.out + ".csv", index=False)
    with open(args.out + "_summary.txt", "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\n[saved] {args.out}.csv  /  {args.out}_summary.txt")


if __name__ == "__main__":
    main()
