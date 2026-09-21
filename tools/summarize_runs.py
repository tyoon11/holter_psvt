#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_runs.py — 실험 결과를 한 화면으로 모은다 (공유용).

터미널 전체를 복사하면 경고에 묻히고 잘린다. 미세조정 result.json 과 probe CSV 의
핵심 행만 뽑아 짧게 출력한다.

  python tools/summarize_runs.py --runs $RUN/ft_* --probe $RUN/probe_all.csv
"""

import argparse
import glob
import json
import os


def show_finetune(paths):
    rows = []
    for d in paths:
        f = os.path.join(d, "result.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        r["name"] = os.path.basename(d.rstrip("/"))
        r["enc"] = os.path.basename(os.path.dirname(r.get("encoder", "")))
        rows.append(r)
    if not rows:
        return
    print("=" * 96)
    print("[미세조정] test 환자 AUROC (probe 의 같은 태스크 CV 값과 비교할 것)")
    print("=" * 96)
    print(f"  {'run':<22s}{'task':<7s}{'encoder':<20s}{'backbone':<9s}"
          f"{'val':>7s}{'test AUROC':>12s}{'95% CI':>18s}{'AUPRC':>8s}{'ep':>4s}")
    for r in sorted(rows, key=lambda x: -(x.get("test_pat_auroc") or 0)):
        ci = f"[{r.get('test_ci_lo', float('nan')):.3f}-{r.get('test_ci_hi', float('nan')):.3f}]"
        print(f"  {r['name'][:21]:<22s}{r.get('task', ''):<7s}{r['enc'][:19]:<20s}"
              f"{'고정' if r.get('freeze_backbone') else '학습':<9s}"
              f"{r.get('val_pat_auroc', float('nan')):>7.3f}"
              f"{r.get('test_pat_auroc', float('nan')):>12.3f}{ci:>18s}"
              f"{r.get('test_pat_auprc', float('nan')):>8.3f}{r.get('best_epoch', 0):>4d}")
    print(f"  (test 양성 환자 {rows[0].get('n_test_pos_pat', '?')}명)")


def show_curves(paths, last=6):
    import csv
    for d in paths:
        f = os.path.join(d, "log.csv")
        if not os.path.exists(f):
            continue
        rows = [r for r in csv.DictReader(open(f)) if r.get("val_pat_auroc")]
        if not rows:
            continue
        v = [f"{float(r['val_pat_auroc']):.3f}" for r in rows]
        line = " → ".join(v) if len(v) <= last else \
            " → ".join(v[:3]) + " … " + " → ".join(v[-3:])
        print(f"  {os.path.basename(d.rstrip('/'))[:24]:<26s} val 곡선({len(v)} epoch): {line}")


def show_probe(path, tasks=None):
    """probe CSV 에서 CV 행만 뽑아 요약한다 (pandas 없이)."""
    import csv
    if not path or not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path)))
    cv = [r for r in rows if str(r.get("train_frac")) == "cv"]
    if not cv:
        cv = [r for r in rows if str(r.get("train_frac")) in ("1.0", "1")]
    if not cv:
        return
    print("\n" + "=" * 96)
    print(f"[probe] {os.path.basename(path)} — CV 환자 AUROC")
    print("=" * 96)
    order = {}
    for r in cv:
        order.setdefault(r["task"], []).append(r)
    for task, g in order.items():
        if tasks and task not in tasks:
            continue
        print(f"  {task}")
        seen_base = set()
        for r in g:
            fs = r["feature_set"]
            if fs in ("demo", "meta"):
                if fs in seen_base:            # 기준선은 특징 종류와 무관하므로 한 번만
                    continue
                seen_base.add(fs)
                tag = fs
            else:
                tag = f"{r['emb']}/{r['features']}/{fs}"
            try:
                print(f"    {tag:<46s} {float(r['pat_auroc']):.3f} "
                      f"[{float(r['pat_ci_lo']):.3f}-{float(r['pat_ci_hi']):.3f}]")
            except (ValueError, KeyError):
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="미세조정 출력 디렉토리들 (glob 가능)")
    ap.add_argument("--probe", default=None, help="probe 결과 CSV")
    ap.add_argument("--tasks", nargs="*", default=None)
    args = ap.parse_args()
    paths = sorted({p for pat in args.runs for p in glob.glob(pat) if os.path.isdir(p)})
    show_finetune(paths)
    if paths:
        print("\n[epoch 별 val 환자 AUROC]")
        show_curves(paths)
    show_probe(args.probe, args.tasks)


if __name__ == "__main__":
    main()
