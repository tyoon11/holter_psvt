# -*- coding: utf-8 -*-
"""
data.py — v2 h5 와 splits.csv 로 SSL 학습 데이터를 공급한다.

  load_records     splits.csv 에서 적격 record 목록 (split 지정)
  SegmentDataset   Stage A: 무작위 10초 구간 (3, 1250) + 구간 beat 요약 타깃
  TokenDataset     Stage B: 캐시된 stem 토큰 시퀀스 (L, d) + time-of-day + 유효 마스크
  iter_segments    토큰 캐시용: record 전체를 세그먼트 청크로 순차 공급

설계 메모
  - 신호는 RecordV2 처럼 contiguous /signal 을 memmap 으로 연다. h5py 핸들은 DataLoader
    워커 안에서 처음 쓸 때 연다 (fork 로 공유되면 깨진다). 워커마다 LRU 로 핸들을 재사용한다.
  - 정규화: record·채널마다 seg/quality 의 amp_std 중앙값으로 나눈다. 전극·이득 차이로
    진폭이 record 마다 크게 달라서, 형태 학습에 진폭 차이가 섞이지 않게 한다.
  - 품질: nan_ratio > 0 이거나 어느 채널이든 amp_std < 1e-3 (무신호)인 세그먼트는
    Stage A 에서 뽑지 않고, Stage B 에서는 손실 계산에서 뺀다.
  - time-of-day: .json hookup_time 기준 (splits.csv 의 hookup_time). 없으면 NaN.
"""

import json
import os
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SEG_LEN = 1250
FS = 125.0
SEG_SEC = SEG_LEN / FS

# WFDB 심볼 그룹 (h5_converter/schema_v2.WFDB_SYMBOLS 의 코드 순서를 따른다)
_SYMS = list("NLRBAaJSVrFejnE/fQ?[]!x()ptu`'^|~+sTD=\"@")
_NORMAL = {_SYMS.index(c) for c in "NLRB"}          # 정상, 좌/우각차단, 분류불가 BBB
_SUPRA = {_SYMS.index(c) for c in "AaJSejn"}        # 심방/결절/상심실 조기·이탈 박동
_VENT = {_SYMS.index(c) for c in "VrFE"}
_BEATS = _NORMAL | _SUPRA | _VENT | {_SYMS.index(c) for c in "/fQ?"}
N_BEAT_TARGETS = 5      # log1p(정상), log1p(상심실성), log1p(심실성), log1p(전체), 심박수/100


def _truthy(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def parse_hhmmss(v):
    """'11:17:00' / '11:17' → 자정 이후 초. 파싱 실패 시 NaN."""
    if not isinstance(v, str):
        return float("nan")
    parts = v.strip().split(":")
    try:
        h, m = int(parts[0]), int(parts[1])
        s = float(parts[2]) if len(parts) > 2 else 0.0
    except (ValueError, IndexError):
        return float("nan")
    if not (0 <= h < 24 and 0 <= m < 60):
        return float("nan")
    return h * 3600 + m * 60 + s


def meta_path(meta_dir, record):
    return os.path.join(meta_dir, record + ".meta.npz")


def load_records(splits_csv, split=None, require_eligible=True):
    """splits.csv → [{record, path, pid, cohort, split, hookup_sec, y_*}]"""
    df = pd.read_csv(splits_csv, low_memory=False, dtype={"pid": str})
    if require_eligible and "eligible" in df:
        df = df[_truthy(df["eligible"])]
    if split is not None:
        splits = [split] if isinstance(split, str) else list(split)
        df = df[df["split"].isin(splits)]
    recs = []
    for r in df.itertuples(index=False):
        d = r._asdict()
        recs.append({
            "record": d["record_name"], "path": d["path"], "pid": str(d.get("pid", "")),
            "cohort": d.get("cohort", ""), "split": d.get("split", ""),
            "hookup_sec": parse_hhmmss(d.get("hookup_time")),
            "n_seg_hint": float(d["n_seg"]) if pd.notna(d.get("n_seg")) else 8640.0,
            "y_lqt": d.get("y_lqt"), "y_tof": d.get("y_tof"), "y_psvt": d.get("y_psvt"),
        })
    return recs


# =============================================================================
# record 핸들
# =============================================================================
class _Pack:
    """묶음 메타. 워커마다 한 번만 열고 이후 record 열기는 슬라이스로 끝난다."""

    def __init__(self, meta_dir):
        i = np.load(os.path.join(meta_dir, "pack_index.npz"), allow_pickle=False)
        self.row = {str(r): k for k, r in enumerate(i["record"])}
        self.i = {k: i[k] for k in ("row", "n_seg", "seg_len", "offset", "n_samples",
                                    "n_sig", "dtype", "has_beats")}
        self.scale, self.norm = i["scale"], i["norm"]
        self.valid = np.load(os.path.join(meta_dir, "pack_valid.npy"), mmap_mode="r")
        self.beats = np.load(os.path.join(meta_dir, "pack_beats.npy"), mmap_mode="r")

    def get(self, record):
        k = self.row.get(record)
        if k is None:
            return None
        r0, n = int(self.i["row"][k]), int(self.i["n_seg"][k])
        return {"n_seg": n, "seg_len": int(self.i["seg_len"][k]),
                "offset": int(self.i["offset"][k]), "dtype": str(self.i["dtype"][k]),
                "shape": (int(self.i["n_samples"][k]), int(self.i["n_sig"][k])),
                "scale": self.scale[k], "norm": self.norm[k],
                "has_beats": bool(self.i["has_beats"][k]),
                "valid": np.asarray(self.valid[r0:r0 + n]), "beats": self.beats[r0:r0 + n]}


def pack_exists(meta_dir):
    return bool(meta_dir) and os.path.exists(os.path.join(meta_dir, "pack_index.npz"))


class _Rec:
    """record 하나. prep_meta 로 만든 <record>.meta.npz 가 있으면 h5py 를 열지 않는다.

    메타가 없으면 예전처럼 h5 에서 직접 읽는다 (샘플마다 1MB 이상을 읽어 매우 느리다).
    """

    def __init__(self, path, meta_dir=None, pack=None):
        self.path = path
        self.f = None
        m = None
        if pack is not None:                      # 묶음 메타 (가장 빠름)
            d = pack.get(os.path.splitext(os.path.basename(path))[0])
            if d is not None:
                self.n_seg, self.seg_len = d["n_seg"], d["seg_len"]
                self.scale, self.norm = d["scale"], d["norm"]
                self.valid, self._bt, self.has_beats = d["valid"], d["beats"], d["has_beats"]
                if d["offset"] >= 0:
                    self.sig = np.memmap(path, dtype=d["dtype"], mode="r",
                                         offset=d["offset"], shape=d["shape"])
                else:
                    self.f = h5py.File(path, "r")
                    self.sig = self.f["signal"]
                self.valid_idx = np.flatnonzero(self.valid)
                return
        if meta_dir:
            mp = meta_path(meta_dir, os.path.splitext(os.path.basename(path))[0])
            if os.path.exists(mp):
                try:
                    m = np.load(mp, allow_pickle=False)
                except Exception:
                    m = None
        if m is not None:
            self.n_seg, self.seg_len = int(m["n_seg"]), int(m["seg_len"])
            self.scale = m["scale"].astype(np.float32)
            self.norm = m["norm"].astype(np.float32)
            self.valid = m["valid"]
            self._bt = m["beat_targets"]
            self.has_beats = bool(m["has_beats"])
            off = int(m["offset"])
            if off >= 0:
                self.sig = np.memmap(path, dtype=str(m["dtype"]), mode="r",
                                     offset=off, shape=tuple(m["shape"]))
            else:
                self.f = h5py.File(path, "r")
                self.sig = self.f["signal"]
        else:
            self._init_from_h5(path)
        self.valid_idx = np.flatnonzero(self.valid)

    def _init_from_h5(self, path):
        self.f = h5py.File(path, "r")
        a = self.f.attrs
        self.n_seg = int(a["n_seg"])
        self.seg_len = int(a["seg_len"])
        self.scale = np.asarray(json.loads(a["scale"]), dtype=np.float32)
        ds = self.f["signal"]
        off = ds.id.get_offset()
        if ds.chunks is None and ds.compression is None and off is not None:
            self.sig = np.memmap(path, dtype=ds.dtype, mode="r", offset=off, shape=ds.shape)
        else:
            self.sig = ds
        q = self.f["seg/quality"][:].astype(np.float32) if "seg/quality" in self.f else None
        if q is not None:
            nan_ratio, amp_std = q[:, 0, :], q[:, 2, :]
            self.valid = ((np.nan_to_num(nan_ratio, nan=1.0) <= 0).all(1)
                          & (np.nan_to_num(amp_std) >= 1e-3).all(1))
            good = amp_std[self.valid]
            norm = np.median(good, axis=0) if len(good) else np.ones(amp_std.shape[1])
        else:
            self.valid = np.ones(self.n_seg, bool)
            norm = np.ones(len(self.scale))
        self.norm = np.where(np.isfinite(norm) & (norm > 1e-3), norm, 1.0).astype(np.float32)
        self.has_beats = "beat" in self.f
        self._bt = None

    @property
    def gain(self):
        """int16 원값 → 정규화된 물리 단위로 바꾸는 채널별 계수."""
        return (self.scale / self.norm).astype(np.float32)

    def segments(self, start, count, clip=None, raw=False):
        """(count, C, seg_len). raw=True 면 int16 원값 그대로 (GPU 에서 변환).

        워커가 float32 로 바꿔 보내면 전송량이 2배가 된다 (int16 2바이트 vs float32 4바이트).
        학습 경로는 raw=True 로 받아 GPU 에서 gain 을 곱하고 clip 한다.
        """
        a, b = start * self.seg_len, (start + count) * self.seg_len
        x = np.asarray(self.sig[a:b])                                   # (T, C)
        if raw and x.dtype == np.int16:
            return np.ascontiguousarray(x.reshape(count, self.seg_len, -1).transpose(0, 2, 1))
        x = x.astype(np.float32) * self.gain
        if clip:
            np.clip(x, -clip, clip, out=x)
        return np.ascontiguousarray(x.reshape(count, self.seg_len, -1).transpose(0, 2, 1))

    def beat_target(self, seg):
        if not self.has_beats:
            return np.zeros(N_BEAT_TARGETS, np.float32), 0.0
        if self._bt is not None:                     # prep_meta 로 미리 계산됨
            return self._bt[seg].astype(np.float32), 1.0
        if self.f is None:
            self.f = h5py.File(self.path, "r")
        if not hasattr(self, "_beats"):
            self._beats = (self.f["beat/seg_offset"][:], self.f["beat/symbol"][:])
        off, sym = self._beats
        s = sym[off[seg]:off[seg + 1]]
        n_norm = sum(int(c in _NORMAL) for c in s)
        n_sup = sum(int(c in _SUPRA) for c in s)
        n_ven = sum(int(c in _VENT) for c in s)
        n_all = sum(int(c in _BEATS) for c in s)
        t = np.log1p([n_norm, n_sup, n_ven, n_all]).tolist() + [n_all * (60.0 / SEG_SEC) / 100.0]
        return np.asarray(t, np.float32), 1.0

    def close(self):
        self.sig = None
        if self.f is not None:
            try:
                self.f.close()
            except Exception:
                pass


class _Handles:
    """워커별 LRU 핸들 캐시. fork 이후 각 워커에서 새로 연다."""

    def __init__(self, max_open=256, meta_dir=None):
        self.max_open = max_open
        self.meta_dir = meta_dir
        self.pack = None
        self.cache = OrderedDict()
        self.pid = None

    def get(self, path):
        if self.pid != os.getpid():            # fork 된 새 프로세스 → 물려받은 핸들 폐기
            self.cache = OrderedDict()
            self.pid = os.getpid()
        if self.pack is None and pack_exists(self.meta_dir):
            self.pack = _Pack(self.meta_dir)      # 워커마다 최초 1회
        r = self.cache.get(path)
        if r is None:
            r = _Rec(path, self.meta_dir, self.pack)
            self.cache[path] = r
            if len(self.cache) > self.max_open:
                _, old = self.cache.popitem(last=False)
                old.close()
        else:
            self.cache.move_to_end(path)
        return r


# =============================================================================
# Stage A
# =============================================================================
class SegmentDataset(Dataset):
    """무작위 (record, 세그먼트) 샘플. 길이는 가상 epoch 크기.

    record 는 세그먼트 수에 비례해 뽑는다 (긴 기록이 더 자주 나오도록 = 시간 균등).
    한 번 고른 record 에서 segs_per_item 개를 뽑아 파일 여는 비용을 나눠 갚는다
    (서버에서 샘플마다 새 record 를 여느라 551 seg/s 로 GPU 가 굶었다).
    같은 (seed, rank, epoch, index) 면 같은 샘플이 나온다.
    """

    def __init__(self, records, samples_per_epoch=200_000, seed=0, rank=0, max_open=256,
                 meta_dir=None, segs_per_item=1, clip=20.0, contiguous=True, raw=True):
        self.records = records
        self.segs = max(1, int(segs_per_item))
        self.n = max(1, samples_per_epoch // self.segs)
        self.seed, self.rank, self.epoch = seed, rank, 0
        self.clip = clip
        self.contiguous = contiguous
        self.raw = raw
        self.h = _Handles(max_open, meta_dir)
        w = np.array([max(1.0, r.get("n_seg_hint", 8640)) for r in records], np.float64)
        self.p = w / w.sum()

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.rank, self.epoch, i])
        for _ in range(10):                      # 유효 세그먼트가 없는 record 면 다시 뽑는다
            rec = self.records[rng.choice(len(self.records), p=self.p)]
            r = self.h.get(rec["path"])
            if len(r.valid_idx):
                break
        if not len(r.valid_idx):
            segs = np.zeros(self.segs, int)
            xs = np.stack([r.segments(int(s), 1, self.clip, self.raw)[0] for s in segs])
        elif self.contiguous and self.segs > 1:
            # 흩어진 K 번 읽기 대신 연속 블록 한 번 읽기. 디스크 랜덤 접근이 K 배 줄어든다
            # (서버에서 GPU util 10~14%, 초당 40MB 수준으로 IO 에 묶였다).
            # 같은 배치 안에서 몇 분 간격의 세그먼트가 함께 오지만, SSL 에는 문제되지 않는다.
            lo = int(rng.choice(r.valid_idx))
            start = min(max(0, lo - int(rng.integers(0, self.segs))), max(0, r.n_seg - self.segs))
            count = min(self.segs, r.n_seg - start)
            block = r.segments(start, count, self.clip, self.raw)
            idx = np.arange(start, start + count)
            keep = r.valid[start:start + count]
            if keep.any():                       # 블록 안의 무효 세그먼트는 유효한 것으로 대체
                block, idx = block[keep], idx[keep]
            take = rng.choice(len(idx), self.segs, replace=len(idx) < self.segs)
            xs, segs = block[take], idx[take]
        else:
            segs = rng.choice(r.valid_idx, self.segs, replace=len(r.valid_idx) < self.segs)
            xs = np.stack([r.segments(int(s), 1, self.clip, self.raw)[0] for s in segs])
        bt, bm = zip(*(r.beat_target(int(s)) for s in segs))
        return {"x": torch.from_numpy(xs),                       # (K, C, L) int16 또는 float32
                "gain": torch.from_numpy(r.gain),                # (C,) int16 → 물리 단위
                "beat": torch.from_numpy(np.stack(bt)),          # (K, 5)
                "beat_mask": torch.tensor(bm, dtype=torch.float32)}


# =============================================================================
# 토큰 캐시 / Stage B
# =============================================================================
def iter_segments(path, chunk=512, meta_dir=None, clip=None):
    """record 전체를 (start, (n, C, L) 배열, valid(n,)) 청크로 공급."""
    r = _Rec(path, meta_dir, _Pack(meta_dir) if pack_exists(meta_dir) else None)
    try:
        for s in range(0, r.n_seg, chunk):
            n = min(chunk, r.n_seg - s)
            yield s, r.segments(s, n, clip), r.valid[s:s + n]
    finally:
        r.close()


class RawWindowDataset(Dataset):
    """원신호에서 연속 구간(기본 2시간)을 잘라 온다. stem 까지 미세조정할 때 쓴다.

    24시간 전체를 gradient 와 함께 통과시키면 메모리가 크다. 학습 때는 무작위 구간을
    쓰고(증강 효과도 있다), 평가 때는 전체를 gradient 없이 통과시킨다.
    반환은 int16 원값 + 채널별 gain (GPU 에서 변환).
    """

    def __init__(self, records, meta_dir, crop_seg=720, random_crop=True, seed=0,
                 max_open=128, clip=20.0):
        self.records, self.crop, self.random_crop = records, crop_seg, random_crop
        self.seed, self.epoch, self.clip = seed, 0, clip
        self.h = _Handles(max_open, meta_dir)

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        r = self.h.get(rec["path"])
        L = min(self.crop, r.n_seg)
        if self.random_crop and r.n_seg > L:
            rng = np.random.default_rng([self.seed, self.epoch, i])
            start = int(rng.integers(0, r.n_seg - L + 1))
        else:
            start = 0
        x = r.segments(start, L, self.clip, raw=True)          # (L, C, seg_len) int16
        pad = np.zeros(self.crop, bool)
        valid = np.asarray(r.valid[start:start + L])
        if L < self.crop:                                      # 짧은 record 는 뒤를 패딩
            x = np.concatenate([x, np.zeros((self.crop - L,) + x.shape[1:], x.dtype)])
            pad[L:] = True
            valid = np.concatenate([valid, np.zeros(self.crop - L, bool)])
        hs = rec.get("hookup_sec", float("nan"))
        if np.isfinite(hs):
            tod = ((hs + (start + np.arange(self.crop)) * SEG_SEC) % 86400.0) / 86400.0
        else:
            tod = np.full(self.crop, np.nan)
        return {"x": torch.from_numpy(np.ascontiguousarray(x)),
                "gain": torch.from_numpy(r.gain),
                "tod": torch.from_numpy(tod.astype(np.float32)),
                "pad": torch.from_numpy(pad),
                "seg_valid": torch.from_numpy(valid & ~pad)}


def token_paths(token_dir, record):
    return os.path.join(token_dir, record + ".npy"), os.path.join(token_dir, record + ".valid.npy")


class TokenDataset(Dataset):
    """캐시 토큰 (S, d) 에서 길이 crop 창을 자른다. S < crop 이면 뒤를 패딩한다.

    반환: tokens (crop, d) float32, tod (crop,) 0..1 또는 NaN, pad (crop,) True=패딩,
          seg_valid (crop,) True=품질 통과
    """

    def __init__(self, records, token_dir, crop=8640, random_crop=True, seed=0):
        self.records = [r for r in records if os.path.exists(token_paths(token_dir, r["record"])[0])]
        self.missing = len(records) - len(self.records)
        self.dir, self.crop, self.random_crop, self.seed = token_dir, crop, random_crop, seed
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        tp, vp = token_paths(self.dir, rec["record"])
        tok = np.load(tp, mmap_mode="r")
        valid = np.load(vp) if os.path.exists(vp) else np.ones(len(tok), bool)
        S, d = tok.shape
        L = self.crop
        if S > L:
            if self.random_crop:
                rng = np.random.default_rng([self.seed, self.epoch, i])
                st = int(rng.integers(0, S - L + 1))
            else:
                st = 0
            t = np.asarray(tok[st:st + L], np.float32)
            v = valid[st:st + L]
            pad = np.zeros(L, bool)
        else:
            st = 0
            t = np.zeros((L, d), np.float32); t[:S] = tok
            v = np.zeros(L, bool); v[:S] = valid
            pad = np.ones(L, bool); pad[:S] = False
        hs = rec.get("hookup_sec", float("nan"))
        if np.isfinite(hs):
            tod = ((hs + (st + np.arange(L)) * SEG_SEC) % 86400.0) / 86400.0
        else:
            tod = np.full(L, np.nan)
        return {"tokens": torch.from_numpy(t), "tod": torch.from_numpy(tod.astype(np.float32)),
                "pad": torch.from_numpy(pad), "seg_valid": torch.from_numpy(v & ~pad)}
