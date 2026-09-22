# -*- coding: utf-8 -*-
"""
probe.py — 고정된 인코더 임베딩으로 downstream 선형 probe 평가.

  python -m holter_encoder.probe --emb $RUN/stage_b_mamba/emb.npz --splits $OUT/splits.csv \\
      --out $RUN/stage_b_mamba/probe

평가 규칙 (2026-09-17 합의 반영)
  - train 라벨로 로지스틱 회귀 학습 → val AUROC 로 C 선택 → test 보고
  - record 는 한 환자 안에서 상관되므로 **환자 단위**로도 집계한다 (환자 확률 = record 평균)
  - 신뢰구간은 **환자 단위 부트스트랩** (LongQT test 양성 환자 14명)
  - 세 가지 특징을 나란히 비교한다
      demo    나이 + 평균HR + 성별        ← 교란 기준선. 인코더는 이걸 넘어야 의미가 있다
      enc     인코더 임베딩
      enc+demo 둘 다
    TOF 는 demo 만으로 AUROC 0.88 이 나오므로 enc 단독 수치만으로 판단하면 안 된다.
  - 라벨 효율: train 환자의 10% / 25% / 100% 로 학습해 비교 (SSL 의 핵심 이점)
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# sklearn 의 deprecation 안내가 fold 마다 쏟아져 표를 덮는다. 결과에는 영향이 없다.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

TASKS = [("LongQT", "y_lqt"), ("TOF", "y_tof"), ("PSVT", "y_psvt")]
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def patient_level(prob, y, pid):
    df = pd.DataFrame({"p": prob, "y": y, "pid": pid})
    g = df.groupby("pid").agg(p=("p", "mean"), y=("y", "max"))
    return g["p"].values, g["y"].values


def boot_ci(y, p, pid, n=1000, seed=0):
    """환자 단위 부트스트랩 AUROC 95% CI."""
    rng = np.random.default_rng(seed)
    pids = np.unique(pid)
    idx = {q: np.flatnonzero(pid == q) for q in pids}
    out = []
    for _ in range(n):
        take = np.concatenate([idx[q] for q in rng.choice(pids, len(pids))])
        if len(np.unique(y[take])) == 2:
            out.append(roc_auc_score(y[take], p[take]))
    return (np.percentile(out, 2.5), np.percentile(out, 97.5)) if out else (np.nan, np.nan)


def fit_eval(Xtr, ytr, Xva, yva, Xte, yte, pid_te, seed=0):
    best, best_auc = None, -1
    for C in CS:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        clf.fit(Xtr, ytr)
        a = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1]) if len(np.unique(yva)) == 2 else 0.5
        if a > best_auc:
            best, best_auc = clf, a
    p = best.predict_proba(Xte)[:, 1]
    pp, py = patient_level(p, yte, pid_te)
    return {
        "val_auroc": best_auc,
        "rec_auroc": roc_auc_score(yte, p), "rec_auprc": average_precision_score(yte, p),
        "pat_auroc": roc_auc_score(py, pp), "pat_auprc": average_precision_score(py, pp),
        "pat_ci": boot_ci(yte, p, pid_te, seed=seed),
        "n_test_pos_pat": int(py.sum()), "n_test_pat": len(py),
    }


def cv_eval(X, y, pid, folds=5, seed=0, boot=1000):
    """SSL 이 보지 않은 환자 풀(val+test)에서 환자 단위 교차검증.

    LongQT 처럼 test 양성 환자가 10여 명뿐이면 단일 test 추정치가 크게 흔들린다
    (합성 검증에서 val 0.93 인데 test 0.41). 같은 데이터를 fold 로 돌려 쓰면
    추정이 안정된다. C 는 각 학습 fold 안에서 내부 교차검증으로 고른다.
    """
    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y, groups=pid):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegressionCV(Cs=CS, cv=3, max_iter=3000,
                                                 class_weight="balanced", scoring="roc_auc"))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    m = np.isfinite(oof)
    pp, py = patient_level(oof[m], y[m], pid[m])
    lo, hi = boot_ci(y[m], oof[m], pid[m], n=boot, seed=seed)
    return {"val_auroc": np.nan,
            "rec_auroc": roc_auc_score(y[m], oof[m]), "rec_auprc": average_precision_score(y[m], oof[m]),
            "pat_auroc": roc_auc_score(py, pp), "pat_auprc": average_precision_score(py, pp),
            "pat_ci": (lo, hi), "n_test_pos_pat": int(py.sum()), "n_test_pat": len(py)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, nargs="+", help="embed.py 결과 npz (여러 개면 비교)")
    ap.add_argument("--splits", required=True, help="나이/HR/성별 등 메타")
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", nargs="+", default=["mean"],
                    choices=["mean", "mean+std", "stem"],
                    help="여러 개를 주면 한 번에 비교한다. stem 은 Stage A 만의 표현이라 "
                         "Stage B backbone 의 기여를 분리해서 볼 수 있다")
    ap.add_argument("--fractions", type=float, nargs="*", default=[0.1, 0.25, 1.0])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--cv", action="store_true",
                    help="val+test 환자를 모아 환자 단위 5-fold 교차검증도 수행 "
                         "(LongQT 처럼 양성이 적을 때 권장)")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--no-meta", action="store_true",
                    help="촬영 조건(meta) 기준선을 계산하지 않는다")
    ap.add_argument("--report-csv", default=None,
                    help="tools/report_features.py 결과. 벤더 리포트 지표 기준선을 추가한다")
    ap.add_argument("--age-band", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="이 나이 구간의 record 만 사용 (코호트 간 나이 차이 교란 확인용)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = pd.read_csv(args.splits, low_memory=False, dtype={"pid": str})
    meta["age_num"] = pd.to_numeric(meta.get("age"), errors="coerce").where(lambda s: s >= 0)
    meta["hr"] = pd.to_numeric(meta.get("mean_hr_bpm"), errors="coerce")
    meta["male"] = meta.get("gender", pd.Series(index=meta.index, dtype=str)).astype(str).str.lower().str[:1].map({"m": 1.0, "f": 0.0})
    # 촬영 조건만으로 만든 특징. LongQT/TOF 는 라벨이 "어느 코호트 폴더인가" 이므로,
    # 장비·시기·전처리 차이(배치 효과)만으로도 높은 점수가 나올 수 있다. 이 특징이
    # 인코더와 비슷한 점수를 내면 그 태스크의 성능은 질환 정보의 근거가 되지 못한다.
    hh = meta.get("hookup_time", pd.Series(index=meta.index, dtype=str)).astype(str).str[:2]
    meta["hour"] = pd.to_numeric(hh, errors="coerce")
    META_COLS = ["duration_h", "n_seg", "n_beats", "hr", "nan_ratio_mean", "amp_std_mean",
                 "flat_seg_ratio", "ann_len", "file_bytes", "hour",
                 "scale_II", "scale_V1", "scale_V5"]
    for c in META_COLS:
        if c in meta:
            meta[c] = pd.to_numeric(meta[c], errors="coerce")
    meta = meta.set_index("record_name")
    meta_cols = [c for c in META_COLS if c in meta.columns]

    # 벤더 리포트 지표. meta 기준선에는 부정맥 지표가 없어서, 인코더가 "리포트에 이미
    # 있는 것"을 재발견한 것인지 구분하지 못한다. 단일 지표 AUROC 와 달리 여기서는
    # 전체를 넣어 적합시키므로 공정한 기준선이 된다.
    REPORT_COLS = ["hr_min", "hr_avg", "hr_max", "tachy_beats", "tachy_pct",
                   "brady_beats", "brady_pct", "vb_total", "vb_Isolated", "vb_Couplets",
                   "vb_BigeminalCycles", "vb_run_count", "vb_run_TotalBeats",
                   "sb_total", "sb_Isolated", "sb_Couplets", "sb_BigeminalCycles",
                   "sb_run_count", "sb_run_TotalBeats", "PacedBeats_total", "BBBeats_total",
                   "JunctionalBeats_total", "AberrantBeats_total",
                   "NoisePercentage", "AFAFLPercentage"]
    rep, rep_cols = None, []
    if args.report_csv:
        rep = pd.read_csv(args.report_csv, low_memory=False, dtype={"pid": str})
        rep = rep.set_index("record_name")
        dur = pd.to_numeric(rep.get("duration_h"), errors="coerce").replace(0, np.nan)
        for c in REPORT_COLS:
            if c in rep.columns:
                rep[c] = pd.to_numeric(rep[c], errors="coerce")
        for c in ("sb_total", "sb_run_count", "sb_Couplets", "vb_total", "vb_run_count"):
            if c in rep.columns:                       # 기록 길이가 제각각이라 시간당 비율도
                rep[f"{c}_per_h"] = rep[c] / dur
                REPORT_COLS.append(f"{c}_per_h")
        rep_cols = [c for c in REPORT_COLS if c in rep.columns]
        print(f"  리포트 지표 {len(rep_cols)}개 ({args.report_csv})")

    rows = []
    for path in args.emb:
        z = np.load(path, allow_pickle=True)
        rec = z["record"].astype(str)
        Xs = {"mean": z["X_mean"], "mean+std": np.concatenate([z["X_mean"], z["X_std"]], 1),
              "stem": z["X_stem"]}
        split = z["split"].astype(str)
        pid = z["pid"].astype(str)
        m = meta.reindex(rec)
        def fill(A):
            med = np.nanmedian(np.where(np.isfinite(A), A, np.nan), axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            return np.where(np.isfinite(A), A, med)

        D = fill(np.stack([m["age_num"].values, m["hr"].values, m["male"].values], 1))
        feats = {"demo": D}          # enc / enc+demo 는 특징 종류마다 아래에서 채운다
        R = None
        if rep_cols:
            R = fill(rep.reindex(rec)[rep_cols].to_numpy(dtype=float))
            feats["report"] = np.concatenate([R, D], 1)    # 벤더 리포트 + 인구학
        if meta_cols and not args.no_meta:
            M = fill(m[meta_cols].to_numpy(dtype=float))
            feats["meta"] = np.concatenate([M, D], 1)      # 촬영 조건 + 인구학
        keep_all = np.ones(len(rec), bool)
        if args.age_band:
            lo, hi = args.age_band
            a = m["age_num"].values
            keep_all = np.isfinite(a) & (a >= lo) & (a <= hi)
            print(f"  나이 {lo:g}~{hi:g}세로 제한: {int(keep_all.sum()):,}/{len(rec):,} record")
        name = path.split("/")[-2] if "/" in path else path
        for feat_name in args.features:
            X = Xs[feat_name]
            feats = {k: v for k, v in feats.items() if not k.startswith("enc")}
            feats["enc"] = X
            feats["enc+demo"] = np.concatenate([X, D], 1)
            if R is not None:
                feats["enc+report"] = np.concatenate([X, R, D], 1)
            print(f"\n{'='*78}\n[{name}]  특징={feat_name}  X {X.shape}\n{'='*78}")
            run_tasks(name, feat_name, X, D, feats, z, split, pid, keep_all, args, rows)

    df = pd.DataFrame(rows).drop(columns=["pat_ci"])
    df.to_csv(args.out + ".csv", index=False)
    print(f"\n[saved] {args.out}.csv")
    print(HOWTO)


HOWTO = """
[읽는 법]
  - enc 가 demo 를 못 넘으면 인코더가 질환 정보를 못 담은 것이다 (특히 TOF)
  - enc+demo 가 demo 보다 얼마나 올라가는지가 인코더의 순수 기여분이다
  - meta(촬영 조건: 길이·전극 이득·잡음·시각 등)가 enc 와 비슷하면 그 태스크는
    질환이 아니라 배치 효과를 재는 것이다. LongQT/TOF 는 라벨이 코호트 소속이라 특히 위험
  - --age-band 로 나이를 맞춘 뒤에도 남는 성능이 나이 교란을 뺀 실력이다
  - 10% 라벨에서 demo 대비 격차가 크면 SSL 사전학습이 제 몫을 한 것이다
  - 특징 stem(Stage A 만) 대비 mean(Stage B 통과)이 나아지지 않으면 backbone 이
    기여하지 못한 것이다. stem 은 세 백본에서 같은 값이라 한 번만 보면 된다
  - report(벤더 리포트 지표: 상심실·심실 이소성, 빈맥·AF 비율, 잡음)를 enc 가 못 넘으면
    인코더는 리포트에 이미 있는 것을 재발견한 것이다. enc+report 가 report 보다 오르는
    만큼이 파형에서 새로 얻은 정보다
  - CV 행은 SSL 이 보지 않은 val+test 환자만으로 돌린 교차검증이다. 양성이 적은
    태스크(LongQT)는 단일 test 점추정보다 이쪽을 근거로 삼는다
"""


def run_tasks(name, feat_name, X, D, feats, z, split, pid, keep_all, args, rows):
        for task, col in TASKS:
            y = z[col].astype(float)
            ok = np.isfinite(y) & keep_all
            print(f"\n  {task}  (라벨 있는 record {int(ok.sum()):,})")
            print(f"    {'특징':<9s}{'비율':>6s}{'val':>7s}{'test AUROC':>12s}{'환자 AUROC':>13s}"
                  f"{'95% CI':>18s}{'환자 AUPRC':>12s}")
            for fname, F in feats.items():
                for frac in (args.fractions if fname.startswith("enc") else [1.0]):
                    tr = ok & (split == "train"); va = ok & (split == "val"); te = ok & (split == "test")
                    if frac < 1.0:                       # 환자 단위로 train 일부만
                        rng = np.random.default_rng(args.seed)
                        tp = np.unique(pid[tr]); keep = set(rng.choice(tp, max(2, int(len(tp) * frac)), replace=False))
                        tr = tr & np.array([p in keep for p in pid])
                    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                        continue
                    r = fit_eval(F[tr], y[tr], F[va], y[va], F[te], y[te], pid[te], seed=args.seed)
                    lo, hi = r["pat_ci"]
                    print(f"    {fname:<9s}{frac:>6.0%}{r['val_auroc']:>7.3f}{r['rec_auroc']:>12.3f}"
                          f"{r['pat_auroc']:>13.3f}{f'[{lo:.3f}-{hi:.3f}]':>18s}{r['pat_auprc']:>12.3f}")
                    rows.append({"emb": name, "features": feat_name, "task": task,
                                 "feature_set": fname, "train_frac": frac, **r,
                                 "pat_ci_lo": lo, "pat_ci_hi": hi})
            if args.cv:
                pool = ok & np.isin(split, ["val", "test"])
                for fname, F in feats.items():
                    if len(np.unique(y[pool])) < 2:
                        continue
                    r = cv_eval(F[pool], y[pool], pid[pool], args.cv_folds, args.seed, args.boot)
                    lo, hi = r["pat_ci"]
                    print(f"    {fname:<9s}{'CV':>6s}{'-':>7s}{r['rec_auroc']:>12.3f}"
                          f"{r['pat_auroc']:>13.3f}{f'[{lo:.3f}-{hi:.3f}]':>18s}{r['pat_auprc']:>12.3f}")
                    rows.append({"emb": name, "features": feat_name, "task": task,
                                 "feature_set": fname, "train_frac": "cv", **r,
                                 "pat_ci_lo": lo, "pat_ci_hi": hi})
            npos = rows[-1]["n_test_pos_pat"] if rows else 0
            if npos < 30:
                print(f"    ** test 양성 환자 {npos}명 — CI 가 넓으니 점추정만으로 비교하지 말 것 **")

if __name__ == "__main__":
    main()
