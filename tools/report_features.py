#!/usr/bin/env python3
"""벤더 리포트(.json) 에서 온 h5 attrs 를 downstream 라벨과 대조한다.

probe.py 의 `meta` 기준선에는 duration·n_beats·평균 HR·잡음 비율·인구학만 들어
있었다. 부정맥 지표는 하나도 없었다. 그런데 v2 attrs 에는 벤더가 계산한
상심실 이소성(sb_*), 심실 이소성(vb_*), 빈맥 비율(tachy_*), AF/AFL 비율이 있다.
상심실 이소성 부담은 PSVT 의 교과서적 예측 인자라, 인코더 성능은 이것들 대비로
해석해야 한다.

주의: 라벨이 이 리포트에서 파생된 것이라면 아래 AUROC 는 실력이 아니라 순환이다.
높게 나오면 라벨의 출처를 먼저 확인할 것.

사용:
  python tools/report_features.py --splits $OUT/splits.csv --out $OUT/report_features.csv
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# 리포트에서 가져올 수치 attrs. 파생 비율은 아래에서 따로 만든다.
FIELDS = ["hr_min", "hr_avg", "hr_max", "tachy_beats", "tachy_pct", "brady_beats", "brady_pct",
          "vb_total", "vb_Isolated", "vb_Couplets", "vb_BigeminalCycles", "vb_run_count",
          "vb_run_TotalBeats",
          "sb_total", "sb_Isolated", "sb_Couplets", "sb_BigeminalCycles", "sb_run_count",
          "sb_run_TotalBeats",
          "PacedBeats_total", "BBBeats_total", "JunctionalBeats_total", "AberrantBeats_total",
          "ann_len", "n_beats"]
PCT = ["NoisePercentage", "AFAFLPercentage"]   # 문자열로 저장된 백분율
TASKS = {"psvt": "y_psvt", "longqt": "y_lqt", "tof": "y_tof"}


def _num(v):
    try:
        f = float(str(v).strip().rstrip("%"))
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def read_attrs(path):
    row = {}
    try:
        with h5py.File(path, "r") as f:
            a = f.attrs
            row["has_report"] = bool(a.get("has_report", False))
            row["duration_h"] = _num(a.get("duration_h", np.nan))
            for k in FIELDS:
                if k in a:
                    row[k] = _num(a[k])
            for k in PCT:
                if k in a:
                    row[k] = _num(a[k])
    except (OSError, KeyError) as e:
        row["error"] = str(e)
    return row


def patient_auroc(score, y, pid):
    """record 점수를 환자 단위(평균)로 모아 AUROC. finetune.patient_metrics 와 같은 기준."""
    ok = np.isfinite(score)
    if ok.sum() < 20:
        return np.nan, 0
    g = pd.DataFrame({"p": score[ok], "y": y[ok], "pid": pid[ok]}).groupby("pid").agg(
        p=("p", "mean"), y=("y", "max"))
    if g["y"].nunique() < 2:
        return np.nan, len(g)
    return roc_auc_score(g["y"], g["p"]), len(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", help="AUROC 를 잴 split (기본 test)")
    args = ap.parse_args()

    df = pd.read_csv(args.splits)
    print(f"{len(df):,} record 의 attrs 를 읽는다 ...")
    rows = []
    for i, r in enumerate(df.itertuples(index=False), 1):
        rows.append(read_attrs(r.path))
        if i % 500 == 0 or i == len(df):
            print(f"  {i:,}/{len(df):,}", end="\r", flush=True)
    print()
    feat = pd.DataFrame(rows)
    out = pd.concat([df.reset_index(drop=True), feat], axis=1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    cov = feat["has_report"].mean() if "has_report" in feat else 0.0
    print(f"[저장] {args.out}   리포트 보유 {cov:.1%}")

    # 부담 지표는 절대 수보다 시간당 비율이 낫다 (기록 길이가 제각각이다).
    dur = out.get("duration_h", pd.Series(np.nan, index=out.index)).replace(0, np.nan)
    derived = {}
    for k in ("sb_total", "sb_run_count", "sb_Couplets", "vb_total", "vb_run_count"):
        if k in out:
            derived[f"{k}_per_h"] = out[k] / dur
    for k, v in derived.items():
        out[k] = v

    sub = out[out["split"] == args.split]
    print(f"\n[{args.split}] 단일 지표 환자 단위 AUROC — 인코더는 이 수치들을 넘어야 한다")
    for task, col in TASKS.items():
        if col not in sub.columns:
            continue
        y = pd.to_numeric(sub[col], errors="coerce").values
        keep = np.isfinite(y)
        if keep.sum() == 0 or len(np.unique(y[keep])) < 2:
            continue
        yy, pid = y[keep], sub["pid"].values[keep]
        scored = []
        for c in FIELDS + PCT + list(derived):
            if c not in sub.columns:
                continue
            auc, n = patient_auroc(pd.to_numeric(sub[c], errors="coerce").values[keep], yy, pid)
            if np.isfinite(auc):
                scored.append((max(auc, 1 - auc), auc, c, n))
        scored.sort(reverse=True)
        print(f"\n  {task}  (환자 {len(set(pid)):,}, 양성 record {int(yy.sum()):,})")
        for _, auc, c, n in scored[:8]:
            arrow = "↑" if auc >= 0.5 else "↓"
            print(f"    {c:<26s} {auc:.3f} {arrow}")
    print("\n라벨이 이 리포트에서 파생된 것이면 위 수치는 순환이다. 출처를 확인할 것.")


if __name__ == "__main__":
    main()
