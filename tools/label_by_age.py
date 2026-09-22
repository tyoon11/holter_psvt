#!/usr/bin/env python3
"""과제별 라벨을 10세 구간으로 쪼개 본다.

나이를 맞추면(--age-band) PSVT 성능이 무너지는데 전체 데이터에서는 선형 demo
기준선이 우연 수준이었다. 양성이 특정 연령대에 뭉쳐 있으면 둘 다 성립한다.
표를 직접 보는 편이 AUROC 보다 빠르다.

사용: python tools/label_by_age.py --splits $OUT/splits.csv
"""
import argparse

import numpy as np
import pandas as pd

TASKS = {"psvt": "y_psvt", "longqt": "y_lqt", "tof": "y_tof"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--band", type=float, nargs=2, default=[15, 45],
                    help="probe --age-band 와 같은 구간을 표시한다")
    args = ap.parse_args()

    df = pd.read_csv(args.splits, low_memory=False, dtype={"pid": str})
    df["age_num"] = pd.to_numeric(df.get("age"), errors="coerce").where(lambda s: s >= 0)
    edges = list(range(0, 90, 10)) + [200]
    labels = [f"{edges[i]}-{edges[i+1]-1}" for i in range(len(edges) - 2)] + ["80+"]
    df["band"] = pd.cut(df["age_num"], bins=edges, right=False, labels=labels)

    for task, col in TASKS.items():
        if col not in df.columns:
            continue
        # 환자 단위로 본다 (한 환자의 여러 record 가 표를 부풀리지 않도록).
        pat = df.dropna(subset=[col]).groupby("pid").agg(
            y=(col, "max"), band=("band", "first"), age=("age_num", "median"))
        if pat.empty:
            continue
        g = pat.groupby("band", observed=False)["y"].agg(n="size", pos="sum")
        g["rate"] = g["pos"] / g["n"].replace(0, np.nan)
        pos_age = pat.loc[pat["y"] == 1, "age"]
        neg_age = pat.loc[pat["y"] == 0, "age"]
        print(f"\n=== {task}  (환자 {len(pat):,}, 양성 {int(pat['y'].sum()):,}) ===")
        print(f"    양성 나이 중앙값 {pos_age.median():.0f} (IQR {pos_age.quantile(.25):.0f}-"
              f"{pos_age.quantile(.75):.0f})   음성 {neg_age.median():.0f} "
              f"(IQR {neg_age.quantile(.25):.0f}-{neg_age.quantile(.75):.0f})")
        lo, hi = args.band
        for band, row in g.iterrows():
            if not row["n"]:
                continue
            mark = ""
            try:                                  # --age-band 구간에 드는 칸을 표시
                b0 = float(str(band).split("-")[0].rstrip("+"))
                mark = " ←" if lo <= b0 <= hi else ""
            except ValueError:
                pass
            bar = "█" * int(round((row["rate"] or 0) * 40))
            print(f"    {str(band):>6s}  n {int(row['n']):>5,}  양성 {int(row['pos']):>4,}"
                  f"  {row['rate'] if pd.notna(row['rate']) else 0:>6.1%}  {bar}{mark}")
        inband = pat[(pat["age"] >= lo) & (pat["age"] <= hi)]
        if len(inband):
            print(f"    {lo:g}~{hi:g}세 안: 환자 {len(inband):,} / 양성 {int(inband['y'].sum()):,}"
                  f" ({inband['y'].mean():.1%}) — 전체 양성의 "
                  f"{inband['y'].sum() / max(pat['y'].sum(), 1):.0%}")


if __name__ == "__main__":
    main()
