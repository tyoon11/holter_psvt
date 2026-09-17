# -*- coding: utf-8 -*-
"""
schema_v2.py — Holter HDF5 저장 구조 v2 (writer + reader).

v1 문제점
    세그먼트마다 group 8개 + dataset 18개 + attr 21개를 만들어, 24시간 record 하나가
    dataset 141,271 / group 62,798 / attr 157,006 개가 된다. 실측 결과
      - 파일/신호 크기비 3.88x  (파일의 3/4가 메타데이터)
      - 24h 전체 읽기 13 MB/s  (flat 저장 코호트의 1/6.6)
    24시간 전체를 입력으로 쓰는 학습에서는 GPU 가 아니라 여기서 막힌다.

v2 설계
    세그먼트 축을 배열의 한 축으로 흡수해 dataset 을 15개 이하로 줄인다.
    가변 길이(beat, fiducial)는 concat + CSR offset 으로 편다.
    정보는 하나도 버리지 않는다. v1 의 모든 필드가 대응된다.

레이아웃
    /signal   (n_samples, n_sig) int16, contiguous, 무압축   ← time-major
        time-major 인 이유: 랜덤 10초 윈도우가 전 lead 에 걸쳐 연속 블록 한 번에
        읽힌다 (1250*3*2 = 7.5KB). channel-first 면 3번 떨어진 곳을 읽어야 한다.
        nas1 코호트가 이미 (N, 3) 이라 그쪽 repack 도 단순 캐스팅이 된다.
        contiguous + 무압축이라 open_signal_memmap() 으로 memmap 이 가능하다.

    /beat/{sample,symbol,subtype,chan,num,aux_code}  (n_beats,)
    /beat/seg_offset   (n_seg+1,) int32     ← CSR. seg i = sample[off[i]:off[i+1]]
    /fid/{sample,label}, /fid/seg_offset
    /seg/fiducial_feat (n_seg, 19) float16
    /seg/quality       (n_seg, 5, n_sig) float16
    /seg/similarity    (n_seg, 2, n_sig) float16
    /meta/{adc_gain,baseline,adc_res,adc_zero}

    나머지(환자, 리포트 집계)는 전부 root attrs 로 올려 평탄화한다.

int16 + per-lead scale
    v1 은 float16 이었다. int16 은 같은 크기지만 ±32767 을 lead 최대 진폭에 맞춰
    균등 배분하므로 큰 진폭 구간에서 float16 보다 정밀하다.
    복원: x_mV = signal.astype(float32) * scale[lead]
"""

import json
import os
from datetime import datetime

import h5py
import numpy as np

SCHEMA_VERSION = "2.0"

# 정규 lead 순서. v1 파일은 sig_name 이 ['V5','V1','II'] 로 역순인 경우가 있어
# 반드시 이름으로 인덱싱해 이 순서로 재배열한다.
CANONICAL_LEADS = ["II", "V1", "V5"]

# v1 fiducial_feature 의 19개 필드. 배열 열 순서를 여기서 고정한다.
FIDUCIAL_FEATURES = [
    "p_amp", "q_amp", "r_amp", "s_amp", "t_amp",
    "p_dur", "pr_seg", "qrs_dur", "st_seg", "t_dur",
    "pr_int", "qt_int", "qtc_baz", "qtc_frid",
    "rr_int", "tp_seg", "p_axis", "r_axis", "t_axis",
]
QUALITY_FIELDS = ["nan_ratio", "amp_mean", "amp_std", "amp_skewness", "amp_kurtosis"]
SIMILARITY_FIELDS = ["bs_correlation", "bs_dtw"]

# WFDB 표준 annotation 심볼 → uint8 코드. 255 = 미지.
WFDB_SYMBOLS = list("NLRBAaJSVrFejnE/fQ?[]!x()ptu`'^|~+sTD=\"@")
SYMBOL_TO_CODE = {s: i for i, s in enumerate(WFDB_SYMBOLS)}
CODE_UNKNOWN = 255

# 리포트(JSON) 집계 필드 — root attr 로 평탄화할 때 쓰는 접두사 규칙
RUN_FIELDS = ["count", "TotalBeats", "LongestRunBeats", "LongestRunBPM",
              "LongestRunTimestamp", "FastestRunBeats", "FastestRunBPM",
              "FastestRunTimestamp"]


# =============================================================================
# 인코딩 유틸
# =============================================================================
def _to_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    return "" if x is None else str(x)


def encode_symbols(symbols):
    """심볼 문자열 배열 → uint8 코드 배열."""
    return np.array([SYMBOL_TO_CODE.get(_to_str(s), CODE_UNKNOWN) for s in symbols],
                    dtype=np.uint8)


def build_vocab(values):
    """문자열 배열 → (코드 배열 uint16, vocab 리스트). aux_note/fiducial 라벨용.

    자유 문자열을 vlen 문자열 데이터셋으로 두면 읽기가 느려지므로,
    레코드 안에서 고유값 사전을 만들고 정수 코드만 저장한다.
    """
    vocab, out = [], []
    index = {}
    for v in values:
        s = _to_str(v)
        if s not in index:
            index[s] = len(vocab)
            vocab.append(s)
        out.append(index[s])
    return np.array(out, dtype=np.uint16), vocab


def quantize_int16(sig):
    """(n_samples, n_sig) float → (int16 배열, per-lead scale float32).

    lead 마다 최대 절댓값을 32767 에 맞춘다. 전량 0인 lead 는 scale=1 로 둔다.
    """
    sig = np.asarray(sig, dtype=np.float32)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    peak = np.abs(sig).max(axis=0)
    scale = np.where(peak > 0, peak / 32767.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(sig / scale), -32767, 32767).astype(np.int16)
    return q, scale


def _set_attrs(obj, d):
    for k, v in d.items():
        if v is None:
            v = ""
        if isinstance(v, (list, tuple)):
            v = json.dumps(v, ensure_ascii=False)
        try:
            obj.attrs[k] = v
        except TypeError:
            obj.attrs[k] = str(v)


# =============================================================================
# Writer
# =============================================================================
def write_v2(
    path,
    signal,                      # (n_samples, n_sig) float, lead 순서 = sig_name
    sig_name,
    fs,
    seg_len=1250,
    record_name="",
    source="",
    created_by="",
    patient=None,                # {pid, age, gender}
    beats=None,                  # {sample(절대), symbol, subtype, chan, num, aux_note}
    beat_seg_offset=None,        # (n_seg+1,) CSR. None 이면 sample 로부터 계산
    fiducial=None,               # {sample(절대), label}
    fid_seg_offset=None,
    fiducial_feat=None,          # (n_seg, 19) float
    quality=None,                # (n_seg, 5, n_sig) float
    similarity=None,             # (n_seg, 2, n_sig) float
    hea_meta=None,               # {adc_gain, baseline, adc_res, adc_zero, fmt, units,
                                 #  base_date, base_time}
    report=None,                 # 평탄화된 리포트 dict (flatten_report 결과)
    extra_attrs=None,
):
    """v2 포맷으로 한 레코드를 쓴다. 부분 정보(신호만 있는 코호트)도 허용한다."""
    signal = np.asarray(signal)
    if signal.ndim == 1:
        signal = signal[:, None]
    n_samples, n_sig = signal.shape
    n_seg = int(np.ceil(n_samples / seg_len))

    q, scale = quantize_int16(signal)

    tmp = path + ".tmp"
    with h5py.File(tmp, "w", libver="latest") as f:
        # ---- 신호: contiguous, 무압축 (memmap 가능 조건) ----
        f.create_dataset("signal", data=q)          # chunks/compression 미지정 = contiguous

        root = {
            "schema_version": SCHEMA_VERSION,
            "record_name": record_name,
            "source": source,
            "created_by": created_by,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fs": float(fs),
            "n_sig": int(n_sig),
            "n_samples": int(n_samples),
            "n_seg": int(n_seg),
            "seg_len": int(seg_len),
            "sig_name": list(sig_name),
            "scale": scale.tolist(),
            "dtype": "int16",
            "duration_h": float(n_samples / fs / 3600.0),
            "symbol_table": WFDB_SYMBOLS,
            "fiducial_feature_names": FIDUCIAL_FEATURES,
            "quality_names": QUALITY_FIELDS,
            "similarity_names": SIMILARITY_FIELDS,
        }
        if patient:
            root.update({"pid": _to_str(patient.get("pid")),
                         "age": _to_str(patient.get("age")),
                         "gender": _to_str(patient.get("gender"))})
        if hea_meta:
            for k in ("base_date", "base_time"):
                root[k] = _to_str(hea_meta.get(k))
        root["has_beats"] = bool(beats is not None and len(beats.get("sample", [])) > 0)
        root["has_fiducial"] = bool(fiducial is not None and len(fiducial.get("sample", [])) > 0)
        root["has_report"] = bool(report)
        if report:
            root.update(report)
        if extra_attrs:
            root.update(extra_attrs)
        _set_attrs(f, root)

        # ---- beat annotation (CSR) ----
        if root["has_beats"]:
            g = f.create_group("beat")
            samp = np.asarray(beats["sample"], dtype=np.int32)
            g.create_dataset("sample", data=samp)
            g.create_dataset("symbol", data=encode_symbols(beats["symbol"]))
            for k, dt in (("subtype", np.int8), ("chan", np.int8), ("num", np.int8)):
                if k in beats and beats[k] is not None:
                    g.create_dataset(k, data=np.asarray(beats[k], dtype=dt))
            if beats.get("aux_note") is not None:
                code, vocab = build_vocab(beats["aux_note"])
                g.create_dataset("aux_code", data=code)
                f.attrs["aux_vocab"] = json.dumps(vocab, ensure_ascii=False)
            off = (np.asarray(beat_seg_offset, dtype=np.int32)
                   if beat_seg_offset is not None
                   else np.searchsorted(samp, np.arange(n_seg + 1) * seg_len).astype(np.int32))
            g.create_dataset("seg_offset", data=off)

        # ---- fiducial point (CSR) ----
        if root["has_fiducial"]:
            g = f.create_group("fid")
            samp = np.asarray(fiducial["sample"], dtype=np.int32)
            g.create_dataset("sample", data=samp)
            code, vocab = build_vocab(fiducial.get("label", []))
            g.create_dataset("label", data=code)
            f.attrs["fiducial_vocab"] = json.dumps(vocab, ensure_ascii=False)
            off = (np.asarray(fid_seg_offset, dtype=np.int32)
                   if fid_seg_offset is not None
                   else np.searchsorted(samp, np.arange(n_seg + 1) * seg_len).astype(np.int32))
            g.create_dataset("seg_offset", data=off)

        # ---- 세그먼트 단위 파생 배열 ----
        seg = None
        for name, arr in (("fiducial_feat", fiducial_feat),
                          ("quality", quality),
                          ("similarity", similarity)):
            if arr is None:
                continue
            seg = seg or f.create_group("seg")
            seg.create_dataset(name, data=np.asarray(arr, dtype=np.float16))

        # ---- .hea 원본 메타 ----
        if hea_meta:
            g = f.create_group("meta")
            for k, dt in (("adc_gain", np.float32), ("baseline", np.int32),
                          ("adc_res", np.int32), ("adc_zero", np.int32)):
                if hea_meta.get(k) is not None:
                    g.create_dataset(k, data=np.asarray(hea_meta[k], dtype=dt))
            for k in ("fmt", "units"):
                if hea_meta.get(k) is not None:
                    g.attrs[k] = json.dumps([_to_str(x) for x in hea_meta[k]])

    os.replace(tmp, path)    # 원자적 교체 — 중단되어도 깨진 파일이 남지 않는다
    return path


def flatten_report(annotation_data):
    """v1 의 중첩 annotation group 을 평탄한 dict 로 바꾼다.

    v1 writer 는 General / Ventriculars / Supraventriculars 만 읽었는데, 실제 .json 에는
    PatientInfo 와 HeartRates 도 들어있다. 둘 다 학습에 바로 쓸 수 있어 함께 뽑는다.
      - HeartRates: min/avg/max 심박수와 그 시각, 빈맥·서맥 beat 수와 비율
        → SSL 보조 타깃(L_beat 계열)과 QC 필터로 유용하다.
      - PatientInfo/HookupDate+HookupTime: 기록 시작 절대 시각
        → time-of-day 임베딩의 기준. .hea 의 base_date/base_time 이 비어 있을 때 대체된다.

    값에 "Unknown", "< 1" 같은 문자열이 섞여 있으므로 숫자 변환은 실패해도 죽지 않는다.
    """
    out = {}
    hr_root = (annotation_data or {}).get("Holter Report", {})
    gen = hr_root.get("General", {})

    def _int(v, default=0):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default

    def _num(v):
        """숫자면 float, 아니면 빈 문자열. 'Unknown' 을 0 으로 오해하지 않게 한다."""
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    # ---- 환자 / 기록 정보 ----
    pi = hr_root.get("PatientInfo", {})
    for src, dst in (("PID", "report_pid"), ("Age", "report_age"),
                     ("Gender", "report_gender"), ("Duration", "report_duration"),
                     ("HookupDate", "hookup_date"), ("HookupTime", "hookup_time")):
        if src in pi:
            out[dst] = _to_str(pi[src])

    # ---- 전체 집계 ----
    out["ann_len"] = _int(gen.get("QRScomplexes", 0))
    out["NoisePercentage"] = _to_str(gen.get("NoisePercentage", ""))
    out["AFAFLPercentage"] = _to_str(gen.get("AFAFLPercentage", ""))

    # ---- 심박수 ----
    hrs = hr_root.get("HeartRates", {})
    for key, prefix in (("MinimumRate", "hr_min"), ("AverageRate", "hr_avg"),
                        ("MaximumRate", "hr_max")):
        d = hrs.get(key, {})
        if d:
            out[prefix] = _num(d.get("Value"))
            if "Timestamp" in d:
                out[f"{prefix}_ts"] = _to_str(d.get("Timestamp"))
    for key, prefix, pct in (("TachycardiaBeats", "tachy", "TachycardiaPercentage"),
                             ("BradycardiaBeats", "brady", "BradycardiaPercentage")):
        d = hrs.get(key, {})
        if d:
            out[f"{prefix}_beats"] = _num(d.get("Value"))
            out[f"{prefix}_pct"] = _num(d.get(pct))

    # ---- 심실성 / 상심실성 ----
    for prefix, key, sub in (("vb", "VentricularBeats", "Ventriculars"),
                             ("sb", "SupraventricularBeats", "Supraventriculars")):
        d = hr_root.get(sub, {})
        out[f"{prefix}_total"] = _int(gen.get(key, 0))
        for k in ("Isolated", "Couplets", "BigeminalCycles"):
            out[f"{prefix}_{k}"] = _int(d.get(k, 0))
        out[f"{prefix}_run_count"] = _int(d.get("Runs", 0))
        # run 안의 beat 총수는 Ventriculars/TotalBeats 다.
        # v1 writer 는 여기에 General/VentricularBeats(=전체 이소성 beat 수)를 넣었는데
        # 서로 다른 값이다(예: General 9 vs TotalBeats 0). 원본 필드를 우선한다.
        out[f"{prefix}_run_TotalBeats"] = _int(d.get("TotalBeats", gen.get(key, 0)))
        for k in RUN_FIELDS[2:]:
            out[f"{prefix}_run_{k}"] = _to_str(d.get(k, ""))

    for k in ("PacedBeats", "BBBeats", "JunctionalBeats", "AberrantBeats"):
        out[f"{k}_total"] = _int(gen.get(k, 0))
    return out


# =============================================================================
# Reader
# =============================================================================
class RecordV2:
    """v2 레코드 읽기. 신호는 memmap 으로 열어 h5py 락과 GIL 을 피한다."""

    def __init__(self, path, mmap=True):
        self.path = path
        self.f = h5py.File(path, "r")
        a = self.f.attrs
        self.fs = float(a["fs"])
        self.n_samples = int(a["n_samples"])
        self.n_sig = int(a["n_sig"])
        self.n_seg = int(a["n_seg"])
        self.seg_len = int(a["seg_len"])
        self.sig_name = json.loads(a["sig_name"]) if isinstance(a["sig_name"], str) else list(a["sig_name"])
        self.scale = np.array(json.loads(a["scale"]), dtype=np.float32)
        self.record_name = _to_str(a.get("record_name", ""))
        self.has_beats = bool(a.get("has_beats", False))

        self._mm = self._open_memmap() if mmap else None

    def _open_memmap(self):
        """contiguous + 무압축이면 파일 내 바이트 오프셋으로 직접 memmap 한다.
        .npy memmap 과 동일한 경로를 타므로 OS page cache 가 그대로 먹는다."""
        ds = self.f["signal"]
        if ds.chunks is not None or ds.compression is not None:
            return None
        off = ds.id.get_offset()
        if off is None:
            return None
        return np.memmap(self.path, dtype=ds.dtype, mode="r",
                         offset=off, shape=ds.shape)

    def window(self, start, length, mv=True):
        """(length, n_sig). mv=True 면 scale 을 곱해 물리 단위로 돌린다."""
        src = self._mm if self._mm is not None else self.f["signal"]
        x = np.asarray(src[start:start + length])
        return x.astype(np.float32) * self.scale if mv else x

    def segment(self, i, mv=True):
        return self.window(i * self.seg_len, self.seg_len, mv)

    def beats_in(self, seg_from, seg_to):
        """[seg_from, seg_to) 구간의 beat (절대 sample, 심볼코드)."""
        if not self.has_beats:
            return np.empty(0, np.int32), np.empty(0, np.uint8)
        off = self.f["beat/seg_offset"]
        lo, hi = int(off[seg_from]), int(off[min(seg_to, self.n_seg)])
        return self.f["beat/sample"][lo:hi], self.f["beat/symbol"][lo:hi]

    def close(self):
        self._mm = None
        try:
            self.f.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# =============================================================================
# convert_to_h5.py 용 어댑터
# =============================================================================
def create_h5_structure_v2(
    path, sig_name, n_sig, seg_len=1, dataset="", created_by="", datetime="",
    record_filename="", patient_id="", age="", gender="",
    signal=None, beat_annotation=None, sig_stats=None, beat_sims=None,
    fiducial_point=None, fiducial_feature=None, metadata=None, annotation_data=None,
    extra_attrs=None,
):
    """create_h5_structure() 와 같은 인자를 받아 v2 로 쓴다.

    차이는 두 가지뿐이다.
      - 첫 인자가 h5py File 객체가 아니라 출력 경로다 (원자적 교체를 위해).
      - seg_len 은 v1 에서 '세그먼트 개수'를 뜻했다. v2 의 seg_len(=샘플 수)과
        이름이 겹치므로 여기서 n_seg 로 받아 구분한다.
    """
    n_seg = int(seg_len)
    seg_samples = int(np.asarray(signal[0]).shape[0])

    # (n_seg, seg_samples, n_sig) → (n_samples, n_sig), 그다음 정규 lead 순서로
    sig = np.concatenate([np.asarray(s, dtype=np.float32) for s in signal], axis=0)
    src = list(sig_name)
    leads = [l for l in CANONICAL_LEADS if l in src] + [l for l in src if l not in CANONICAL_LEADS]
    perm = [src.index(l) for l in leads]
    sig = sig[:, perm]

    def _stack(lst, keys):
        """세그먼트별 dict 리스트 → (n_seg, len(keys), n_sig). lead 순열도 적용."""
        if not lst:
            return None
        out = np.full((n_seg, len(keys), n_sig), np.nan, dtype=np.float32)
        for i, d in enumerate(lst[:n_seg]):
            for j, k in enumerate(keys):
                v = np.asarray(d.get(k, np.nan), dtype=np.float32).ravel()
                if v.size >= n_sig:
                    out[i, j, :] = v[:n_sig][perm]
        return out

    beats = fid = None
    b_off = np.zeros(n_seg + 1, dtype=np.int32)
    f_off = np.zeros(n_seg + 1, dtype=np.int32)
    if beat_annotation:
        cols = {k: [] for k in ("sample", "symbol", "subtype", "chan", "num", "aux_note")}
        for i, ba in enumerate(beat_annotation[:n_seg]):
            smp = np.asarray(ba.get("sample", []), dtype=np.int64) + i * seg_samples
            cols["sample"].append(smp)
            for k in ("symbol", "aux_note"):
                v = ba.get(k, [])
                cols[k].append(np.asarray(v if len(v) == len(smp) else [""] * len(smp),
                                          dtype=object))
            for k in ("subtype", "chan", "num"):
                v = np.asarray(ba.get(k, []), dtype=np.int64)
                cols[k].append(v[:len(smp)] if v.size >= len(smp) else np.zeros(len(smp), np.int64))
            b_off[i + 1] = b_off[i] + len(smp)
        if b_off[-1] > 0:
            beats = {k: np.concatenate(v) for k, v in cols.items()}
    if fiducial_point:
        fs_, fl_ = [], []
        for i, fp in enumerate(fiducial_point[:n_seg]):
            s = np.asarray(fp.get("fsample", []), dtype=np.int64) + i * seg_samples
            fs_.append(s)
            lab = fp.get("fiducial", [])
            fl_.append(np.asarray(lab if len(lab) == len(s) else [""] * len(s), dtype=object))
            f_off[i + 1] = f_off[i] + len(s)
        if f_off[-1] > 0:
            fid = {"sample": np.concatenate(fs_), "label": np.concatenate(fl_)}

    ff = None
    if fiducial_feature:
        ff = np.full((n_seg, len(FIDUCIAL_FEATURES)), np.nan, dtype=np.float32)
        for i, d in enumerate(fiducial_feature[:n_seg]):
            for j, k in enumerate(FIDUCIAL_FEATURES):
                try:
                    ff[i, j] = float(d.get(k, np.nan))
                except (TypeError, ValueError):
                    pass

    hea = None
    if metadata:
        hea = {k: metadata.get(k) for k in
               ("adc_gain", "baseline", "adc_res", "adc_zero", "fmt", "units",
                "base_date", "base_time")}
        # .hea 의 배열들도 lead 순열을 따라야 한다
        for k in ("adc_gain", "baseline", "adc_res", "adc_zero", "fmt", "units"):
            v = hea.get(k)
            if v is not None and len(v) == n_sig:
                hea[k] = [v[i] for i in perm]

    return write_v2(
        path, signal=sig, sig_name=leads,
        fs=float((metadata or {}).get("fs", 125) or 125),
        seg_len=seg_samples, record_name=record_filename, source=dataset,
        created_by=created_by,
        patient={"pid": patient_id, "age": age, "gender": gender},
        beats=beats, beat_seg_offset=b_off if beats else None,
        fiducial=fid, fid_seg_offset=f_off if fid else None,
        fiducial_feat=ff,
        quality=_stack(sig_stats, QUALITY_FIELDS),
        similarity=_stack(beat_sims, ["bs_corr", "bs_dtw"]),
        hea_meta=hea,
        report=flatten_report(annotation_data) if annotation_data else None,
        # v1 은 metadata/sig_len 에 .hea 의 원래 샘플 수를 남겼다. v2 의 n_samples 는
        # 10초 단위로 자른 뒤 길이라(마지막 10초 미만은 버린다) 원래 값을 따로 보존한다.
        extra_attrs={**({"sig_len": int((metadata or {}).get("sig_len") or 0)}),
                     **(extra_attrs or {})},
    )
