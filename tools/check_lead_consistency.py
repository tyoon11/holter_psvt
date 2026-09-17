#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_lead_consistency.py — 원본 3채널의 순서가 record 마다 같은지 확인한다.

왜 따로 필요한가
  check_lead_order.py 는 "V1 = QRS 가 가장 음성" 같은 형태 규칙으로 lead 를 판별한다.
  실제 데이터에서는 이 규칙이 자주 틀렸다.
    - TOF 교정 환자는 대부분 우각차단이라 V1 QRS 가 오히려 양성(rSR')이다
    - 소아는 V1 에서 R 파가 큰 것이 정상이다
    - Holter 변형 쌍극 유도는 P 파가 작아 II/V5 구분 근거가 약하다
  그래서 "각 채널이 무엇인가"와 "순서가 일정한가"를 분리한다. 이 스크립트는 후자만 본다.

방법 (lead 이름을 가정하지 않는다)
  1) record 마다 채널별 median beat 템플릿을 만든다 (check_lead_order.templates 재사용).
     각 채널은 QRS 최대 절댓값으로 나눠 크기만 맞추고 부호는 유지한다.
  2) 같은 그룹(코호트 × 파일명 체계) 안에서 채널별 기준 템플릿 = 전 record 중앙값.
  3) record 마다 6가지 채널 순열 중 기준과 가장 잘 맞는(상관 평균 최대) 순열을 찾고,
     그 순열로 정렬해 기준을 다시 만든다. 몇 번 반복한다.
  4) 순서가 고정이면 거의 모든 record 의 최적 순열이 "그대로(0,1,2)"로 나온다.
     특정 순열이 무더기로 나오면 그 묶음은 순서가 다르다.
  5) 그룹 기준끼리도 비교해 코호트·체계 간 순서가 같은지 본다.
  6) 그룹별 채널 평균 파형을 그린다. 수백 명 평균이라 개인 질환 차이가 희석되어
     개별 record 보다 lead 특성이 잘 드러난다.

주의
  같은 그룹 전체가 한꺼번에 다른 순서라면 그룹 내부에서는 "고정"으로 보인다.
  그 경우는 5) 그룹 간 비교와 6) 평균 파형으로 드러난다.

사용:
  python tools/check_lead_consistency.py --raw /home/coder/workspace/data/raw --per-group 300 --workers 16
  python tools/check_lead_consistency.py --raw ... --all --workers 32
  python tools/check_lead_consistency.py --load lead_consistency_templates.npz   # 재계산 없이
"""

import argparse
import csv
import itertools
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_lead_order import collect, name_scheme, templates  # noqa: E402

PERMS = list(itertools.permutations(range(3)))
IDENT = (0, 1, 2)


def _one(args):
    base, cohort = args
    name = os.path.basename(base)
    try:
        tmpl, fs, nb = templates(base)
        if tmpl is None or tmpl.shape[0] != 3:
            return name, cohort, None, fs, nb
        return name, cohort, tmpl.astype(np.float32), fs, nb
    except Exception:
        return name, cohort, None, 125, 0


def normalize(T, fs):
    """(N, 3, L) → record 마다 세 채널에 같은 배율을 적용해 QRS 최대 절댓값을 1 로.

    채널마다 따로 맞추면 II 와 V5 처럼 모양이 비슷한 채널을 바꿔도 구분이 안 된다
    (합성 검증에서 확인). 한 배율을 공유해 채널 간 상대 크기를 단서로 남긴다.
    """
    pre = int(0.30 * fs)
    q0, q1 = pre - int(0.06 * fs), pre + int(0.06 * fs)
    s = np.abs(T[:, :, q0:q1 + 1]).max(axis=(1, 2), keepdims=True) + 1e-6
    T = T / s
    w0, w1 = pre - int(0.12 * fs), pre + int(0.45 * fs)     # P 말단 ~ T 끝
    return T[:, :, w0:w1 + 1]


def cosine(a, b):
    """a (N, D), b (D,) → (N,)"""
    return (a @ b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b) + 1e-9)


def perm_scores(X, ref):
    """X (N,3,L), ref (3,L) → (N, 6). 세 채널을 이어붙인 벡터의 코사인 유사도.

    모양과 채널 간 상대 크기를 함께 본다.
    """
    N = X.shape[0]
    r = ref.reshape(-1)
    out = np.zeros((N, len(PERMS)), np.float32)
    for k, p in enumerate(PERMS):
        out[:, k] = cosine(X[:, list(p), :].reshape(N, -1), r)
    return out


SWAPS = {"ch0↔ch1": (1, 0, 2), "ch0↔ch2": (2, 1, 0), "ch1↔ch2": (0, 2, 1)}


def align(X, iters=6):
    """반복 정렬. 반환: 최적 순열 인덱스(N,), 점수(N,6), 기준(3,L)"""
    best = np.full(X.shape[0], PERMS.index(IDENT))
    for _ in range(iters):
        aligned = np.stack([X[i, list(PERMS[best[i]]), :] for i in range(X.shape[0])])
        ref = np.median(aligned, axis=0)
        sc = perm_scores(X, ref)
        new = sc.argmax(1)
        if np.array_equal(new, best):
            break
        best = new
    # 기준의 채널 순서 자체가 뒤섞이지 않도록, 다수 순열이 identity 가 되게 재표시
    major = Counter(best.tolist()).most_common(1)[0][0]
    if PERMS[major] != IDENT:
        inv = np.argsort(PERMS[major])
        ref = ref[inv]
        sc = perm_scores(X, ref)
        best = sc.argmax(1)
    return best, sc, ref


def plot_refs(groups_out, fs, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib 없음 — 그림 생략)")
        return
    keys = sorted(groups_out)
    cols = 3
    rows = (len(keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.2 * rows), squeeze=False)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for ax, k in zip(axes.flat, keys):
        g = groups_out[k]
        A = g["aligned"]                        # (N,3,L) 다수 순서 기준 정렬본
        L = A.shape[2]
        t = (np.arange(L) - int(0.12 * fs)) / fs * 1000
        for c in range(3):
            med = np.median(A[:, c], 0)
            lo, hi = np.percentile(A[:, c], [25, 75], axis=0)
            off = -2.4 * c
            ax.fill_between(t, lo + off, hi + off, color=colors[c], alpha=0.2, lw=0)
            ax.plot(t, med + off, color=colors[c], lw=1.4)
            ax.text(t[0], off + 0.7, f"ch{c}", color=colors[c], fontsize=8)
        ax.axvline(0, color="gray", lw=0.5, ls=":")
        ax.set_title(f"{k[0]} / {k[1]}  n={A.shape[0]}  identity {g['ident_frac']*100:.0f}%",
                     fontsize=8)
        ax.set_yticks([]); ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    fig.suptitle("Group median beat per file channel (aligned; band = IQR; x = ms, 0 = R)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"  [saved] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw")
    ap.add_argument("--per-group", type=int, default=300)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-group", type=int, default=15, help="이보다 작은 그룹은 정렬 생략")
    ap.add_argument("--margin", type=float, default=0.02,
                    help="최적 순열과 차선 순열의 코사인 차가 이 이상이어야 '확실한 판정'")
    ap.add_argument("--load", help="이전에 저장한 템플릿 npz 로 재분석")
    ap.add_argument("--out", default="lead_consistency")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    if args.load:
        z = np.load(args.load, allow_pickle=True)
        names, cohorts, T, fs = list(z["names"]), list(z["cohorts"]), z["templates"], int(z["fs"])
        print(f"[load] {len(names):,}개 템플릿")
    else:
        if not args.raw:
            sys.exit("--raw 또는 --load 가 필요합니다.")
        recs = collect(args.raw)
        groups = defaultdict(list)
        for b, c in recs:
            groups[(c, name_scheme(os.path.basename(b))[0])].append((b, c))
        sample = recs if args.all else [x for v in groups.values()
                                        for x in random.sample(v, min(args.per_group, len(v)))]
        print(f"[템플릿 추출] {len(sample):,}개 / 전체 {len(recs):,}  (workers={args.workers})")
        from multiprocessing import Pool
        names, cohorts, tl, fss = [], [], [], []
        with Pool(args.workers) as pool:
            for i, (n, c, t, f, nb) in enumerate(pool.imap_unordered(_one, sample, chunksize=2)):
                if t is not None:
                    names.append(n); cohorts.append(c); tl.append(t); fss.append(f)
                print(f"\r  {i + 1:,}/{len(sample):,}  (유효 {len(tl):,})", end="", flush=True)
        print()
        L = min(t.shape[1] for t in tl)
        T = np.stack([t[:, :L] for t in tl])
        fs = int(Counter(fss).most_common(1)[0][0])
        np.savez_compressed(args.out + "_templates.npz", names=np.array(names),
                            cohorts=np.array(cohorts), templates=T, fs=fs)
        print(f"  [saved] {args.out}_templates.npz  (다음부터 --load 로 재분석)")

    X = normalize(T, fs)
    schemes = [name_scheme(n)[0] for n in names]
    groups = defaultdict(list)
    for i, (c, s) in enumerate(zip(cohorts, schemes)):
        groups[(c, s)].append(i)

    print("\n" + "=" * 78)
    print("[그룹 내부] 최적 채널 순열 분포   (0,1,2)=파일 순서 그대로")
    print("=" * 78)
    rows, groups_out = [], {}
    for k in sorted(groups):
        idx = np.array(groups[k])
        if len(idx) < args.min_group:
            print(f"  {k[0]:<26s} {k[1]:<7s} n={len(idx):>4}  (표본 부족, 생략)")
            continue
        best, sc, ref = align(X[idx])
        cnt = Counter(PERMS[b] for b in best)
        top2 = np.sort(sc, 1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        bestc = sc.max(1)
        ident = cnt[IDENT] / len(idx)
        reliable = margin >= args.margin
        is_ident = np.array([PERMS[b] == IDENT for b in best])
        n_rel = int(reliable.sum())
        print(f"  {k[0]:<26s} {k[1]:<7s} n={len(idx):>4}  적합도 중앙값 {np.median(bestc):.2f}")
        print(f"      확실한 판정(margin≥{args.margin}) {n_rel}개 중  그대로 {int((is_ident & reliable).sum())}"
              f"  / 다른 순서 {int((~is_ident & reliable).sum())}"
              f"      불확실 {len(idx) - n_rel}개")
        rel_other = Counter(PERMS[b] for b, rr in zip(best, reliable) if rr and PERMS[b] != IDENT)
        if rel_other:
            print(f"      확실하게 다른 순서: " + ", ".join(f"{p} {n}" for p, n in rel_other.most_common()))
        # 채널 쌍별 구분 가능성: 다수 순서 기준에서 두 채널을 바꿨을 때 점수가 얼마나 떨어지나
        ident_sc = sc[:, PERMS.index(IDENT)]
        parts = []
        for nm, p in SWAPS.items():
            drop = ident_sc - sc[:, PERMS.index(p)]
            parts.append(f"{nm} 점수차 중앙값 {np.median(drop):.3f}")
        print("      " + "   ".join(parts) + "   (0 에 가까우면 그 두 채널은 구분 불가)")
        aligned = np.stack([X[i, list(PERMS[b]), :] for i, b in zip(idx, best)])
        groups_out[k] = {"ref": ref, "aligned": aligned, "ident_frac": ident}
        for j, i in enumerate(idx):
            rows.append({"record": names[i], "cohort": k[0], "scheme": k[1],
                         "best_perm": "".join(map(str, PERMS[best[j]])),
                         "best_corr": round(float(bestc[j]), 3),
                         "margin": round(float(margin[j]), 3),
                         "identity_corr": round(float(sc[j, PERMS.index(IDENT)]), 3)})

    print("\n[그룹 간] 각 그룹 기준 파형을 첫 그룹에 맞췄을 때의 순열")
    print("  (코호트마다 질환 형태가 다르면(예: TOF 우각차단) 이 비교는 약하다. 참고로만)")
    keys = sorted(groups_out)
    if keys:
        base = groups_out[keys[0]]["ref"]
        print(f"  기준: {keys[0]}")
        for k in keys:
            sc = perm_scores(groups_out[k]["ref"][None], base)[0]
            b = int(sc.argmax())
            print(f"  {k[0]:<26s} {k[1]:<7s} 최적 {PERMS[b]}  상관 {sc[b]:.2f}"
                  f"  (그대로 {sc[PERMS.index(IDENT)]:.2f})")

    csv_path = args.out + "_records.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\n  [saved] {csv_path}")
    plot_refs(groups_out, fs, args.out + "_group_templates.png")

    print("\n[읽는 법]")
    print("  - '확실한 판정' 중 거의 전부가 '그대로'면 그 그룹의 순서는 고정")
    print("  - '확실하게 다른 순서'가 무더기면 그 record 들은 순서가 다름 → _records.csv 로 추림")
    print("  - 쌍별 점수차가 0 에 가까운 채널 쌍은 이 방법으로는 서로 구분할 수 없다")
    print("  - '각 채널이 어떤 lead 인지'는 이 결과로 정해지지 않는다. 그룹 평균 파형 그림과")
    print("    장비 설정 또는 같은 환자의 12유도 ECG 대조로 확정한다")


if __name__ == "__main__":
    main()
