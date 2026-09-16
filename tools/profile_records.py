#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_records.py — 코호트 전체 프로파일 + lead 순서 판별 + I/O 벤치마크.

inspect_data.py 가 "무엇이 있는지"를 봤다면, 이 스크립트는 repack 설계와
코호트 통합에 필요한 세 가지를 정량화한다.

  (A) 전체 헤더 스캔    길이/dtype/채널수 분포. 데이터를 읽지 않고 shape 만 본다.
  (B) lead 순서 판별    nas1/h5 는 sig_name 이 없어 II/V1/V5 순서를 모른다.
                        채널별 통계 프로파일로 코호트 간 채널을 정렬한다.
  (C) I/O 벤치마크      전체 순차 읽기 / 랜덤 10초 윈도우 읽기 처리량.
                        Stage A(랜덤 crop)와 Stage C(24h 통짜)의 실제 상한.

사용:
  python profile_records.py --scan-all                 # 전 파일 헤더 스캔 포함
  python profile_records.py --samples 20 --windows 40  # lead 판별 표본 수
  python profile_records.py --skip-bench               # 벤치마크 생략
"""

import argparse
import collections
import os
import random
import re
import time

import h5py
import numpy as np

DEFAULT_DIRS = [
    "/home/coder/workspace/Holter_TOF/nas1_Holter_PSVT/h5",
    "/home/coder/workspace/Holter_TOF/nas1_Holter_PSVT/final_denoised_3_lead",
    "/home/coder/workspace/Holter_TOF/nas1_Holter_PSVT/denoised_3_lead",
    "/home/coder/workspace/Holter_TOF/holter_h5",
]
NUMERIC = re.compile(r"^\d+$")
BAR = "=" * 78

try:
    from scipy.signal import find_peaks as _find_peaks
except ImportError:  # scipy 없을 때의 대체 구현
    _find_peaks = None


# =============================================================================
# 레코드 접근 추상화 — flat(time-major/channel-first) 과 per-segment 를 모두 흡수
# =============================================================================
class RecordReader:
    """어떤 스키마든 (C, L) float32 윈도우를 돌려주는 공통 인터페이스."""

    def __init__(self, path):
        self.path = path
        self.f = h5py.File(path, "r")
        self.fs = 125.0
        self.leads = None
        self.mode = None

        # --- per-segment 스키마 ---
        for segpath in ("ECG/segments", "segments"):
            if segpath in self.f and isinstance(self.f[segpath], h5py.Group):
                g = self.f[segpath]
                keys = sorted([k for k in g.keys() if NUMERIC.match(k)], key=int)
                if keys:
                    self.mode = "per_segment"
                    self.segkeys = keys
                    self.seg_grp = g
                    s0 = g[keys[0]]["signal"]
                    self.leads = list(s0.keys())
                    self.seg_len = int(s0[self.leads[0]].shape[-1])
                    self.n_ch = len(self.leads)
                    self.n_samples = self.seg_len * len(keys)
                    self.dtype = s0[self.leads[0]].dtype
                    break

        # --- flat 스키마 ---
        if self.mode is None:
            best = None
            def visit(name, obj):
                nonlocal best
                if isinstance(obj, h5py.Dataset) and obj.size > 100_000:
                    if best is None or obj.size > self.f[best].size:
                        best = name
            self.f.visititems(visit)
            if best is None:
                raise ValueError(f"신호 데이터셋을 찾지 못함: {path}")
            ds = self.f[best]
            self.mode = "flat"
            self.dspath = best
            self.dtype = ds.dtype
            self.chunks, self.comp = ds.chunks, ds.compression
            if ds.ndim == 1:
                self.time_major, self.n_samples, self.n_ch = True, ds.shape[0], 1
            elif ds.shape[0] >= ds.shape[1]:
                self.time_major, self.n_samples, self.n_ch = True, ds.shape[0], ds.shape[1]
            else:
                self.time_major, self.n_samples, self.n_ch = False, ds.shape[1], ds.shape[0]

        # fs / lead 이름 보강
        for p in ("ECG/metadata", "metadata"):
            if p in self.f:
                a = self.f[p].attrs
                if "fs" in a:
                    self.fs = float(a["fs"])
                if "sig_name" in self.f[p]:
                    try:
                        self.leads = [x.decode() if isinstance(x, bytes) else str(x)
                                      for x in self.f[p]["sig_name"][:]]
                    except Exception:
                        pass
        if "fs" in self.f.attrs:
            try:
                self.fs = float(self.f.attrs["fs"])
            except Exception:
                pass

    @property
    def duration_h(self):
        return self.n_samples / self.fs / 3600.0

    @property
    def nbytes(self):
        return self.n_samples * self.n_ch * self.dtype.itemsize

    def window(self, start, length):
        """(C, length) float32."""
        if self.mode == "flat":
            ds = self.f[self.dspath]
            if ds.ndim == 1:
                return ds[start:start + length][None, :].astype(np.float32)
            if self.time_major:
                return ds[start:start + length, :].astype(np.float32).T
            return ds[:, start:start + length].astype(np.float32)
        # per-segment: 걸치는 세그먼트를 모아 잘라낸다
        i0, i1 = start // self.seg_len, (start + length - 1) // self.seg_len
        parts = []
        for i in range(i0, min(i1 + 1, len(self.segkeys))):
            s = self.seg_grp[self.segkeys[i]]["signal"]
            parts.append(np.stack([s[l][:] for l in self.leads]))
        arr = np.concatenate(parts, axis=-1)
        off = start - i0 * self.seg_len
        return arr[:, off:off + length].astype(np.float32)

    def full(self):
        return self.window(0, self.n_samples)

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def list_h5(root, limit=None):
    if not os.path.isdir(root):
        return []
    out = [e.path for e in os.scandir(root)
           if e.is_file() and e.name.lower().endswith((".h5", ".hdf5"))]
    out.sort()
    return out[:limit] if limit else out


# =============================================================================
# (A) 전체 헤더 스캔
# =============================================================================
def scan_headers(root, files):
    print(f"  전체 {len(files):,}개 파일 헤더 스캔 중...", flush=True)
    durs, shapes, dtypes, chans, bad = [], collections.Counter(), collections.Counter(), collections.Counter(), []
    t0 = time.time()
    for i, p in enumerate(files):
        try:
            r = RecordReader(p)
            durs.append(r.duration_h)
            shapes[(r.n_samples, r.n_ch)] += 1
            dtypes[str(r.dtype)] += 1
            chans[r.n_ch] += 1
            r.close()
        except Exception as e:
            bad.append((os.path.basename(p), f"{type(e).__name__}: {e}"))
        if (i + 1) % 500 == 0:
            print(f"    {i+1:,}/{len(files):,}  ({time.time()-t0:.0f}s)", flush=True)

    d = np.array(durs)
    print(f"  스캔 완료 {len(durs):,}개 / 실패 {len(bad)}개  ({time.time()-t0:.0f}s)")
    if len(d):
        q = np.percentile(d, [0, 1, 25, 50, 75, 99, 100])
        print(f"  기록 길이(h)  min={q[0]:.2f}  p1={q[1]:.2f}  p25={q[2]:.2f}  "
              f"median={q[3]:.2f}  p75={q[4]:.2f}  p99={q[5]:.2f}  max={q[6]:.2f}")
        for thr in (1, 6, 12, 20):
            n = int((d < thr).sum())
            if n:
                print(f"    {thr}시간 미만: {n:,}개 ({100*n/len(d):.1f}%)")
        print(f"  총 신호 시간: {d.sum():,.0f} 시간 ({d.sum()/24:,.0f} record-day)")
    print(f"  dtype: {dict(dtypes)}   채널수: {dict(chans)}")
    print(f"  고유 shape 수: {len(shapes)}  (상위 3개: {shapes.most_common(3)})")
    for name, err in bad[:5]:
        print(f"    [실패] {name}: {err}")
    return {"durations": durs, "dtypes": dict(dtypes), "bad": bad}


# =============================================================================
# (B) lead 순서 판별
# =============================================================================
def find_peaks_simple(x, height, distance):
    if _find_peaks is not None:
        return _find_peaks(x, height=height, distance=distance)[0]
    idx = np.where(x > height)[0]
    out, last = [], -distance
    for i in idx:
        if i - last >= distance:
            out.append(i)
            last = i
    return np.array(out, dtype=int)


def channel_features(win, fs=125.0):
    """(C, L) → 채널별 특징 벡터. 진폭 스케일에 불변이 되도록 MAD 로 정규화한다.

    V1 은 rS 패턴이라 R 시점 진폭이 음수/양음 혼재이고 왜도가 음(-)으로 치우친다.
    II, V5 는 R 이 뚜렷하게 양(+)이다. 이 차이로 채널을 구분한다.
    """
    C, L = win.shape
    feats = []
    med = np.median(win, axis=1, keepdims=True)
    x = win - med
    mad = np.median(np.abs(x), axis=1, keepdims=True) + 1e-8
    z = x / (1.4826 * mad)

    # R 피크는 "모든 채널 공통 시점"이어야 극성 비교가 의미를 갖는다.
    ref = np.abs(z).sum(0)
    peaks = find_peaks_simple(ref, height=np.percentile(ref, 99) * 0.5,
                              distance=int(0.3 * fs))
    for c in range(C):
        zc = z[c]
        amp = zc[peaks] if len(peaks) else np.array([0.0])
        feats.append([
            float(np.mean(zc ** 3)),                                  # 왜도
            float(np.mean(zc ** 4)),                                  # 첨도
            float(np.mean(amp) / (np.mean(np.abs(amp)) + 1e-8)),      # R 극성 (-1..1)
            float(np.percentile(zc, 99)),
            float(np.percentile(zc, 1)),
            float(np.std(np.diff(zc)) / (np.std(zc) + 1e-8)),         # 고주파 비율
        ])
    return np.array(feats), len(peaks)


FEAT_NAMES = ["skew", "kurt", "R극성", "p99", "p01", "HF비"]


def profile_leads(root, files, n_records=20, n_windows=30, win_sec=10, seed=0):
    rng = random.Random(seed)
    picks = rng.sample(files, min(n_records, len(files)))
    acc, leads_seen, n_pk = [], None, []
    for p in picks:
        try:
            r = RecordReader(p)
            if r.leads:
                leads_seen = r.leads
            L = int(win_sec * r.fs)
            if r.n_samples < L * 2:
                r.close(); continue
            per_rec = []
            for _ in range(n_windows):
                st = rng.randrange(0, r.n_samples - L)
                if r.mode == "per_segment":
                    st = (st // r.seg_len) * r.seg_len
                f, k = channel_features(r.window(st, L), r.fs)
                per_rec.append(f); n_pk.append(k)
            acc.append(np.nanmean(np.stack(per_rec), axis=0))
            r.close()
        except Exception as e:
            print(f"    [skip] {os.path.basename(p)}: {type(e).__name__}: {e}")
    if not acc:
        return None
    prof = np.nanmean(np.stack(acc), axis=0)          # (C, F)
    print(f"  표본 {len(acc)}개 record × {n_windows} window, "
          f"윈도우당 평균 R피크 {np.mean(n_pk):.0f}개")
    print(f"  파일에 기록된 lead 이름: {leads_seen}")
    hdr = "        " + "".join(f"{n:>10s}" for n in FEAT_NAMES)
    print(hdr)
    for c in range(prof.shape[0]):
        name = leads_seen[c] if leads_seen and c < len(leads_seen) else f"ch{c}"
        print(f"    {name:>4s}" + "".join(f"{v:>10.3f}" for v in prof[c]))
    return {"profile": prof, "leads": leads_seen}


def match_channels(ref_prof, ref_leads, other_prof):
    """채널 특징 프로파일이 가장 잘 맞는 순열을 찾아 lead 이름을 이식한다."""
    import itertools
    a = (ref_prof - ref_prof.mean(0)) / (ref_prof.std(0) + 1e-8)
    b = (other_prof - other_prof.mean(0)) / (other_prof.std(0) + 1e-8)
    best, best_d = None, None
    for perm in itertools.permutations(range(b.shape[0])):
        d = float(np.abs(a - b[list(perm)]).mean())
        if best_d is None or d < best_d:
            best, best_d = perm, d
    names = [ref_leads[i] if ref_leads else f"ch{i}" for i in range(len(best))]
    mapping = {f"ch{best[i]}": names[i] for i in range(len(best))}
    return mapping, best_d


# =============================================================================
# (C) I/O 벤치마크
# =============================================================================
def bench_io(root, files, n_full=3, n_win=200, win_sec=10, seed=0):
    rng = random.Random(seed)
    picks = rng.sample(files, min(n_full, len(files)))

    # 순차: 24h 통짜 (Stage C 접근 패턴)
    secs, mbs = [], []
    for p in picks:
        r = RecordReader(p)
        t0 = time.time()
        arr = r.full()
        dt = time.time() - t0
        secs.append(dt); mbs.append(r.nbytes / 1e6)
        r.close()
        del arr
    print(f"  [순차] 24h 통짜 읽기  {np.mean(secs):.2f}s/record  "
          f"({np.mean(mbs)/np.mean(secs):.0f} MB/s, 평균 {np.mean(mbs):.0f} MB)")
    seq = np.mean(secs)

    # 랜덤: 10초 윈도우 (Stage A 접근 패턴)
    p = picks[0]
    r = RecordReader(p)
    L = int(win_sec * r.fs)
    t0 = time.time()
    got = 0
    for _ in range(n_win):
        st = rng.randrange(0, max(1, r.n_samples - L))
        if r.mode == "per_segment":
            st = (st // r.seg_len) * r.seg_len
        r.window(st, L); got += 1
    dt = time.time() - t0
    r.close()
    print(f"  [랜덤] 10초 윈도우 {got}개  {dt:.2f}s  → {got/dt:.0f} window/s  "
          f"({1000*dt/got:.1f} ms/window)")
    return {"seq_sec_per_record": seq, "rand_win_per_sec": got / dt}


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=DEFAULT_DIRS)
    ap.add_argument("--scan-all", action="store_true", help="전 파일 헤더 스캔 (수 분 소요)")
    ap.add_argument("--samples", type=int, default=20, help="lead 판별용 record 수")
    ap.add_argument("--windows", type=int, default=30, help="record 당 윈도우 수")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="디렉토리당 파일 수 상한")
    args = ap.parse_args()

    profiles = {}
    for root in args.dirs:
        files = list_h5(root, args.limit)
        print(BAR); print(f"[{root}]  ({len(files):,} files)"); print(BAR)
        if not files:
            print("  파일 없음\n"); continue

        if args.scan_all:
            print("--- (A) 헤더 스캔 ---")
            scan_headers(root, files)
            print()

        print("--- (B) 채널 프로파일 ---")
        pr = profile_leads(root, files, args.samples, args.windows)
        if pr:
            profiles[root] = pr
        print()

        if not args.skip_bench:
            print("--- (C) I/O 벤치마크 ---")
            try:
                bench_io(root, files)
            except Exception as e:
                print(f"  [실패] {type(e).__name__}: {e}")
            print()

    # lead 이름이 있는 코호트를 기준으로 나머지 채널 순서를 추정
    print(BAR); print("[lead 순서 정렬]"); print(BAR)
    ref = next((r for r, p in profiles.items() if p["leads"]), None)
    if ref is None:
        print("  lead 이름이 기록된 디렉토리가 없어 절대 이름을 붙일 수 없습니다.")
    else:
        print(f"  기준: {os.path.basename(ref)}  leads={profiles[ref]['leads']}")
        for root, p in profiles.items():
            if root == ref:
                continue
            m, d = match_channels(profiles[ref]["profile"], profiles[ref]["leads"],
                                  p["profile"])
            flag = "신뢰가능" if d < 0.6 else "불확실 — 육안 확인 필요"
            print(f"  {os.path.basename(root):28s} → {m}   (거리 {d:.3f}, {flag})")
    print()
    print("  참고: V1 은 rS 패턴이라 R극성이 음수/0 근처, II·V5 는 뚜렷한 양수입니다.")
    print("       위 표의 'R극성' 열로 직접 교차검증하세요.")


if __name__ == "__main__":
    main()
