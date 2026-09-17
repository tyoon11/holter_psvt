#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_lead_order.py — 원본 .SIG 3채널이 실제로 어떤 lead 인지 파형으로 판별한다.

배경
  원본 .hea 의 채널 설명이 전부 "MARS export" 라 lead 이름이 없다. h5_converter/utils.py
  는 이 경우 3채널이면 ["V5","V1","II"] 를 하드코딩한다. v2 는 이 이름으로 채널을
  II,V1,V5 순서로 재배열하므로, 가정이 틀리면 모든 파일의 lead 이름이 틀린다.
  파일명 체계(DVD2013… / 202003_…)나 시기에 따라 순서가 다를 가능성도 확인해야 한다.

방법 (record 마다)
  1) .ANN 의 정상 박동(N) 위치를 기준점으로 채널별 median beat 템플릿을 만든다.
     기록 초반(부착 직후 잡음)을 피해 여러 구간에서 뽑는다.
  2) 채널별 특징
       qrs_net  = (QRS 최대 양성 − QRS 최대 음성) / QRS 폭     [-1, 1]
       p_ratio  = P파 부호 있는 진폭 / QRS 폭
       t_ratio  = T파 부호 있는 진폭 / QRS 폭                  (참고용, 판별에는 안 씀)
  3) 판별
       V1 = qrs_net 가 가장 음수인 채널                 (rS 패턴)
       II = 나머지 둘 중 p_ratio 가 더 큰 채널           (동리듬 P파는 II 에서 가장 양성)
       V5 = 남은 채널
     각 단계의 차이(margin)를 신뢰도로 남긴다.
  4) 코호트 / 파일명 체계 / 연도별로 판별 패턴을 집계하고, 하드코딩 가정과 비교한다.
  5) 무작위 record 의 채널별 템플릿을 그림으로 저장한다 — 최종 판단은 눈으로 확인할 것.

※ 실제 데이터(348 record 표본)에서는 결과가 그룹 안에서도 30~60% 로 흩어졌다.
  TOF 교정 환자의 우각차단(V1 QRS 양성), 소아의 V1 R 우세, Holter 변형 유도의 작은 P파
  때문에 형태 규칙이 자주 틀린다. 순서 일관성은 check_lead_consistency.py 로 확인할 것.

주의
  휴리스틱이다. 심방세동(P파 없음), 우축편위, 소아(V1 R 우세 가능), 우각차단 등에서는
  틀릴 수 있어 개별 record 보다 "다수 패턴"과 신뢰도를 본다. 확정은 템플릿 그림과
  장비(MARS) 내보내기 설정 확인으로 한다.

사용:
  python tools/check_lead_order.py --raw /home/coder/workspace/data/raw --per-group 40
  python tools/check_lead_order.py --raw ... --all --workers 16        # 전체
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np

FS_DEFAULT = 125
ASSUMED = ("V5", "V1", "II")          # utils.py 하드코딩: 파일 ch0, ch1, ch2


def name_scheme(name):
    if re.match(r"^DVD\d{8}_", name):
        return "DVD", name[3:7]
    m = re.match(r"^(\d{4})(\d{2})_", name)
    if m:
        return "YYYYMM", m.group(1)
    return "기타", ""


def cohort_of(dirpath, root):
    rel = os.path.relpath(dirpath, root)
    return os.path.basename(os.path.abspath(root)) if rel == "." else rel.split(os.sep)[0]


def collect(root):
    recs = []
    for dirpath, _, names in os.walk(root):
        ns = set(names)
        for nm in names:
            if nm.endswith(".hea"):
                b = nm[:-4]
                if b + ".SIG" in ns and b + ".ANN" in ns:
                    recs.append((os.path.join(dirpath, b), cohort_of(dirpath, root)))
    return recs


# =============================================================================
def templates(base, n_windows=6, win_min=5, max_beats=250):
    """채널별 median beat. 반환 (tmpl[C, L], fs, n_beats)"""
    import wfdb
    hdr = wfdb.rdheader(base)
    fs = int(hdr.fs or FS_DEFAULT)
    n = int(hdr.sig_len)
    pre, post = int(0.30 * fs), int(0.50 * fs)
    win = int(win_min * 60 * fs)
    start0 = min(int(0.5 * 3600 * fs), max(0, n - win * n_windows))   # 앞 30분 건너뜀
    starts = np.linspace(start0, max(start0, n - win - 1), n_windows).astype(int)
    beats = []
    for s in starts:
        e = min(n, s + win)
        if e - s < fs * 30:
            continue
        rec = wfdb.rdrecord(base, sampfrom=int(s), sampto=int(e), physical=True)
        x = np.nan_to_num(rec.p_signal.T.astype(np.float32))          # (C, T)
        ann = wfdb.rdann(base, "ANN", sampfrom=int(s), sampto=int(e))
        samp = np.asarray(ann.sample) - s
        sym = np.asarray(ann.symbol)
        # 정상 박동 + 앞뒤 RR 이 비슷한 것만 (조기수축 주변 제외)
        idx = np.where(sym == "N")[0]
        for k in idx[1:-1]:
            r = samp[k]
            rr1, rr2 = samp[k] - samp[k - 1], samp[k + 1] - samp[k]
            if r - pre < 0 or r + post >= x.shape[1] or rr1 <= 0 or rr2 <= 0:
                continue
            if abs(rr1 - rr2) > 0.2 * max(rr1, rr2) or sym[k - 1] != "N" or sym[k + 1] != "N":
                continue
            beats.append(x[:, r - pre:r + post])
            if len(beats) >= max_beats * n_windows:
                break
    if len(beats) < 30:
        return None, fs, len(beats)
    b = np.stack(beats)                                               # (B, C, L)
    # 박동별 기준선: PR 분절(-90 ~ -50 ms)
    i0, i1 = pre - int(0.09 * fs), pre - int(0.05 * fs)
    b = b - np.median(b[:, :, i0:i1 + 1], axis=2, keepdims=True)
    return np.median(b, axis=0), fs, len(beats)


def features(tmpl, fs):
    pre = int(0.30 * fs)
    q0, q1 = pre - int(0.06 * fs), pre + int(0.06 * fs)
    p0, p1 = pre - int(0.25 * fs), pre - int(0.10 * fs)
    t0, t1 = pre + int(0.16 * fs), pre + int(0.42 * fs)
    out = []
    for c in range(tmpl.shape[0]):
        y = tmpl[c]
        qpos, qneg = float(y[q0:q1 + 1].max()), float(-y[q0:q1 + 1].min())
        span = max(qpos + qneg, 1e-6)
        pw = y[p0:p1 + 1]; tw = y[t0:t1 + 1]
        p = float(pw[np.argmax(np.abs(pw))]); t = float(tw[np.argmax(np.abs(tw))])
        out.append({"qrs_pos": qpos, "qrs_neg": qneg, "qrs_span": span,
                    "qrs_net": (qpos - qneg) / span, "p_ratio": p / span, "t_ratio": t / span})
    return out


def classify(feat):
    nets = [f["qrs_net"] for f in feat]
    order = np.argsort(nets)
    v1 = int(order[0])
    v1_margin = float(nets[order[1]] - nets[order[0]])
    rest = [c for c in range(len(feat)) if c != v1]
    ii = max(rest, key=lambda c: feat[c]["p_ratio"])
    v5 = [c for c in rest if c != ii][0]
    ii_margin = float(feat[ii]["p_ratio"] - feat[v5]["p_ratio"])
    names = [""] * len(feat)
    names[v1], names[ii], names[v5] = "V1", "II", "V5"
    return names, v1_margin, ii_margin


def analyze(args_tuple):
    base, cohort = args_tuple
    name = os.path.basename(base)
    scheme, year = name_scheme(name)
    row = {"record": name, "cohort": cohort, "scheme": scheme, "year": year}
    try:
        tmpl, fs, nb = templates(base)
        row["n_beats"] = nb
        if tmpl is None:
            row["status"] = "beats<30"
            return row, None
        feat = features(tmpl, fs)
        names, m1, m2 = classify(feat)
        row.update(status="ok", pattern=",".join(names), v1_margin=round(m1, 3),
                   ii_margin=round(m2, 3), matches_assumed=(tuple(names) == ASSUMED))
        for c, f in enumerate(feat):
            for k in ("qrs_net", "p_ratio", "t_ratio", "qrs_span"):
                row[f"ch{c}_{k}"] = round(f[k], 3)
        return row, (tmpl, fs)
    except Exception as e:
        row["status"] = f"error: {type(e).__name__}: {e}"
        return row, None


def scan_json_for_leads(recs, k=40):
    """.json 안에 lead/channel 정보가 있는지 키 이름으로 훑는다."""
    hits = Counter()
    def walk(o, path=""):
        if isinstance(o, dict):
            for kk, v in o.items():
                p = f"{path}/{kk}"
                if re.search(r"lead|chan|deriv|electrode", kk, re.I):
                    hits[f"{p} = {str(v)[:40]}"] += 1
                walk(v, p)
        elif isinstance(o, list):
            for v in o[:5]:
                walk(v, path + "[]")
    for base, _ in random.sample(recs, min(k, len(recs))):
        try:
            walk(json.load(open(base + ".json", errors="replace")))
        except Exception:
            pass
    return hits


def plot(examples, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib 없음 — 그림 생략)")
        return
    n = len(examples)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.8 * rows), squeeze=False)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for ax, (row, (tmpl, fs)) in zip(axes.flat, examples):
        t = (np.arange(tmpl.shape[1]) - int(0.30 * fs)) / fs * 1000
        for c in range(tmpl.shape[0]):
            ax.plot(t, tmpl[c] / max(np.abs(tmpl[c]).max(), 1e-6) - 2.2 * c,
                    color=colors[c % 3], lw=1.2)
            ax.text(t[0], -2.2 * c + 0.55, f"ch{c}", color=colors[c % 3], fontsize=8)
        ax.axvline(0, color="gray", lw=0.5, ls=":")
        ax.set_title(f"{row['record'][:24]}\n{row['cohort'][:22]} | est {row['pattern']}"
                     f" (V1m {row['v1_margin']:.2f}, IIm {row['ii_margin']:.2f})", fontsize=7)
        ax.set_yticks([]); ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    # 서버에 한글 글꼴이 없을 수 있어 그림 안 문구는 영어로 둔다
    fig.suptitle("Median beat per channel (normalized to channel max; x = ms, 0 = R peak)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"  [saved] {path}")


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--per-group", type=int, default=40,
                    help="코호트 × 파일명체계 조합마다 표본 수")
    ap.add_argument("--all", action="store_true", help="전체 record 분석")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="lead_check")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    recs = collect(args.raw)
    groups = defaultdict(list)
    for base, cohort in recs:
        groups[(cohort, name_scheme(os.path.basename(base))[0])].append((base, cohort))
    print(f"[대상] .hea+.SIG+.ANN record {len(recs):,}개")
    for (c, s), v in sorted(groups.items()):
        print(f"  {c:<28s} {s:<7s} {len(v):>6,}")

    if args.all:
        sample = recs
    else:
        sample = []
        for v in groups.values():
            sample += random.sample(v, min(args.per_group, len(v)))
    print(f"\n[분석] {len(sample):,}개 (workers={args.workers})")

    rows, examples = [], []
    from multiprocessing import Pool
    with Pool(args.workers) as pool:
        for i, (row, tp) in enumerate(pool.imap_unordered(analyze, sample, chunksize=2)):
            rows.append(row)
            if tp is not None and len(examples) < 400:
                examples.append((row, tp))
            print(f"\r  {i + 1:,}/{len(sample):,}", end="", flush=True)
    print()

    ok = [r for r in rows if r.get("status") == "ok"]
    bad = Counter(r["status"].split(":")[0] for r in rows if r.get("status") != "ok")
    print(f"  판별 성공 {len(ok):,} / 실패 {dict(bad)}")

    # ---- 요약 ----
    print("\n" + "=" * 78)
    print(f"[판별 패턴]  파일 채널 순서 ch0,ch1,ch2 의 추정 lead   (코드 가정: {','.join(ASSUMED)})")
    print("=" * 78)
    total = Counter(r["pattern"] for r in ok)
    for p, n in total.most_common():
        mark = "  ← 코드 가정" if tuple(p.split(",")) == ASSUMED else ""
        print(f"  {p:<12s} {n:>6,}  ({100 * n / max(len(ok), 1):5.1f}%){mark}")

    confident = [r for r in ok if r["v1_margin"] >= 0.3 and r["ii_margin"] >= 0.05]
    tc = Counter(r["pattern"] for r in confident)
    print(f"\n  신뢰도 높은 것만 (V1 margin≥0.3, II margin≥0.05): {len(confident):,}개")
    for p, n in tc.most_common(4):
        print(f"    {p:<12s} {n:>6,}  ({100 * n / max(len(confident), 1):5.1f}%)")

    print("\n  V1 위치만 (가장 견고한 지표):")
    v1pos = Counter(r["pattern"].split(",").index("V1") for r in ok)
    for c, n in sorted(v1pos.items()):
        print(f"    ch{c} = V1   {n:>6,}  ({100 * n / max(len(ok), 1):5.1f}%)")

    print("\n[그룹별] 코호트 × 파일명체계")
    by = defaultdict(Counter)
    for r in ok:
        by[(r["cohort"], r["scheme"])][r["pattern"]] += 1
    for k in sorted(by):
        c = by[k]; n = sum(c.values()); top, tn = c.most_common(1)[0]
        print(f"  {k[0]:<28s} {k[1]:<7s} n={n:>4}  최다 {top} {100 * tn / n:5.1f}%   "
              f"가정일치 {100 * c[','.join(ASSUMED)] / n:5.1f}%")

    print("\n[연도별]")
    byy = defaultdict(Counter)
    for r in ok:
        byy[r["year"] or "?"][r["pattern"]] += 1
    for y in sorted(byy):
        c = byy[y]; n = sum(c.values()); top, tn = c.most_common(1)[0]
        print(f"  {y:<6s} n={n:>4}  최다 {top} {100 * tn / n:5.1f}%   "
              f"V1=ch1 {100 * sum(v for p, v in c.items() if p.split(',')[1] == 'V1') / n:5.1f}%")

    print("\n[채널별 특징 중앙값]  qrs_net(음수=V1형)  p_ratio(클수록 II형)")
    for c in range(3):
        qn = np.median([r[f"ch{c}_qrs_net"] for r in ok]) if ok else float("nan")
        pr = np.median([r[f"ch{c}_p_ratio"] for r in ok]) if ok else float("nan")
        tr = np.median([r[f"ch{c}_t_ratio"] for r in ok]) if ok else float("nan")
        print(f"  ch{c}  qrs_net {qn:+.2f}   p_ratio {pr:+.3f}   t_ratio {tr:+.2f}")

    hits = scan_json_for_leads(recs)
    print("\n[.json 안의 lead/channel 관련 키]")
    print("  " + ("\n  ".join(f"{k}  ({v})" for k, v in hits.most_common(10)) if hits else "없음"))

    # ---- 저장 ----
    csv_path = args.out + "_records.csv"
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k.startswith("ch"), k))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"\n  [saved] {csv_path}")
    # 그림: 그룹마다 고르게 최대 24개
    pick, seen = [], Counter()
    random.shuffle(examples)
    for row, tp in examples:
        g = (row["cohort"], row["scheme"])
        if seen[g] < 4 and len(pick) < 24:
            pick.append((row, tp)); seen[g] += 1
    plot(pick, args.out + "_templates.png")


if __name__ == "__main__":
    main()
