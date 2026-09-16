# -*- coding: utf-8 -*-
"""
test_schema_v2.py — v2 스키마 회귀 테스트.

두 경로를 모두 검증한다.
  (1) repack  : 기존 v1 파일 → tools/repack_to_v2.py → v2
  (2) adapter : convert_to_h5.py 가 넘기는 자료구조 → create_h5_structure_v2 → v2

특히 lead 순열을 집중적으로 본다. 실제 데이터의 물리적 lead 순서가
['V5','V1','II'] 로 정규 순서(II,V1,V5)와 역순이라, 신호는 이름으로 읽어 맞더라도
signal_quality/adc_gain 같은 (3,) 배열은 순열을 따로 맞춰야 조용히 어긋나지 않는다.

    python -m h5_converter.test_schema_v2        (repo 루트에서)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from h5_converter.schema_v2 import (  # noqa: E402
    RecordV2, WFDB_SYMBOLS, create_h5_structure_v2,
)

UTF8 = h5py.string_dtype(encoding="utf-8")
SEG, NSEG = 1250, 40
PHYS = ["V5", "V1", "II"]            # 물리적 lead 순서 (정규 순서와 역순)
QUAL_TRUTH = {"V5": 0.11, "V1": 0.22, "II": 0.33}
BEATS_PER = {0: 3, 3: 1, 4: 5, 17: 2, 39: 4}   # 일부 세그먼트만 → CSR 경계 검증
SYMS = ["N", "S", "V", "N", "A"]

_results = []


def chk(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"  [{'OK' if cond else 'FAIL'}] {name}  {detail}")


def make_truth(seed=7):
    rng = np.random.default_rng(seed)
    return {"II": rng.normal(0, 1.0, NSEG * SEG).astype(np.float32),
            "V1": rng.normal(0, 2.0, NSEG * SEG).astype(np.float32) - 3.0,
            "V5": rng.normal(0, 0.5, NSEG * SEG).astype(np.float32) + 5.0}


def write_v1_fixture(path, truth):
    p = path
    with h5py.File(p, "w") as f:
        f.attrs["dataset_version"]="1.0"; f.attrs["created_by"]="tester"
        f.attrs["file_name"]="REC_TEST"
        pg=f.create_group("patient"); pg.attrs["pid"]="P123"; pg.attrs["age"]="67"; pg.attrs["gender"]="F"
        ecg=f.create_group("ECG"); segs=ecg.create_group("segments"); segs.attrs["seg_len"]=NSEG
        for i in range(NSEG):
            s=segs.create_group(str(i)); sg=s.create_group("signal")
            for l in PHYS:
                sg.create_dataset(l, data=truth[l][i*SEG:(i+1)*SEG].astype(np.float16))
            if i in BEATS_PER:
                n=BEATS_PER[i]; ba=s.create_group("beat_annotation")
                ba.create_dataset("sample", data=(np.arange(n)*100+50).astype(np.int16))
                ba.create_dataset("symbol", data=np.array(SYMS[:n],dtype=UTF8),dtype=UTF8)
                ba.create_dataset("subtype", data=np.zeros(n,np.int16))
                ba.create_dataset("chan", data=np.zeros(n,np.int16))
                ba.create_dataset("num", data=np.zeros(n,np.int16))
                ba.create_dataset("aux_note", data=np.array([""]*n,dtype=UTF8),dtype=UTF8)
            fp=s.create_group("fiducial_point"); fp.attrs["extraction_method"]="nk"
            fp.create_dataset("fsample", data=np.array([10,20],np.int16))
            fp.create_dataset("fiducial", data=np.array(["P","R"],dtype=UTF8),dtype=UTF8)
            ff=s.create_group("fiducial_feature")
            ff.attrs["qt_int"]=np.float16(0.38); ff.attrs["rr_int"]=np.float16(0.85)
            sq=s.create_group("signal_quality")
            sq.create_dataset("nan_ratio", data=np.array([QUAL_TRUTH[l] for l in PHYS],np.float16))
            am=sq.create_group("amplitude")
            for k in ["amp_mean","amp_std","amp_skewness","amp_kurtosis"]:
                am.create_dataset(k, data=np.zeros(3,np.float16))
            bs=sq.create_group("beat_similarity")
            for k in ["bs_correlation","bs_dtw"]: bs.create_dataset(k, data=np.zeros(3,np.float16))
        md=ecg.create_group("metadata"); md.attrs["fs"]=125; md.attrs["record_name"]="REC_TEST"
        md.attrs["base_date"]="2024-05-01"; md.attrs["base_time"]="09:30:00"
        md.create_dataset("sig_name", data=np.array(PHYS,dtype=UTF8),dtype=UTF8)
        for k,v in [("adc_gain",np.ones(3,np.float16)),("baseline",np.zeros(3,np.int16)),
                    ("adc_res",np.full(3,16,np.int16)),("adc_zero",np.zeros(3,np.int16))]:
            md.create_dataset(k, data=v)
        md.create_dataset("fmt", data=np.array(["16"]*3,dtype=UTF8),dtype=UTF8)
        md.create_dataset("units", data=np.array(["mV"]*3,dtype=UTF8),dtype=UTF8)
        an=ecg.create_group("annotation"); an.attrs["ann_len"]=1234
        an.attrs["NoisePercentage"]="2.5"; an.attrs["AFAFLPercentage"]="11.0"
        bc=an.create_group("beat_count")
        for gname,tot in [("VentricularBeat",77),("SupraventricularBeat",55)]:
            g=bc.create_group(gname); g.attrs["total"]=tot
            for k in ["Isolated","Couplets","BigeminalCycles"]: g.attrs[k]=1
            r=g.create_group("Runs"); r.attrs["count"]=2; r.attrs["TotalBeats"]=tot
            for k in ["LongestRunBeats","LongestRunBPM","LongestRunTimestamp",
                      "FastestRunBeats","FastestRunBPM","FastestRunTimestamp"]:
                r.attrs[k]=f"{k}_val"
        for k in ["PacedBeats","BBBeats","JunctionalBeats","AberrantBeats"]:
            bc.create_group(k).attrs["total"]=0
    return p


def test_repack():
    print("=" * 70); print("(1) v1 → repack_to_v2.py → v2"); print("=" * 70)
    work = tempfile.mkdtemp(prefix="v2test_")
    try:
        src_dir, dst_dir = os.path.join(work, "v1"), os.path.join(work, "v2")
        os.makedirs(src_dir)
        truth = make_truth()
        src = write_v1_fixture(os.path.join(src_dir, "REC_TEST.h5"), truth)

        n1 = [0]
        with h5py.File(src, "r") as f:
            f.visititems(lambda n, o: n1.__setitem__(0, n1[0] + 1))

        r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "repack_to_v2.py"),
                            "--src", src_dir, "--dst", dst_dir, "--workers", "1"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-500:]); print(r.stderr[-1500:])
            chk("repack 실행", False); return

        out = os.path.join(dst_dir, "REC_TEST.h5")
        n2 = [0]
        with h5py.File(out, "r") as f:
            f.visititems(lambda n, o: n2.__setitem__(0, n2[0] + 1))
        rec = RecordV2(out); A = dict(rec.f.attrs)

        chk("객체 수 감소", n2[0] < n1[0] / 10, f"v1 {n1[0]:,} → v2 {n2[0]:,}")
        chk("memmap 가능", rec._mm is not None)
        chk("lead 정규화", rec.sig_name == ["II", "V1", "V5"], f"{PHYS} → {rec.sig_name}")

        sig = rec.window(0, rec.n_samples)
        e = {l: float(np.abs(sig[:, j] - truth[l].astype(np.float16).astype(np.float32)).max())
             for j, l in enumerate(rec.sig_name)}
        chk("신호 왕복", all(v < 0.01 for v in e.values()),
            ", ".join(f"{k}={v:.1e}" for k, v in e.items()))

        q = {l: float(rec.f["seg/quality"][0, 0, j]) for j, l in enumerate(rec.sig_name)}
        chk("quality 순열 보정", all(abs(q[l] - QUAL_TRUTH[l]) < 0.01 for l in q),
            ", ".join(f"{k}={v:.2f}" for k, v in q.items()))

        off = rec.f["beat/seg_offset"][:]
        exp = np.cumsum([0] + [BEATS_PER.get(i, 0) for i in range(NSEG)])
        chk("CSR offset", np.array_equal(off, exp), f"총 beat {off[-1]}")
        s0, y0 = rec.beats_in(0, 1); s1, _ = rec.beats_in(1, 3); s4, _ = rec.beats_in(4, 5)
        chk("세그먼트별 조회", len(s0) == 3 and len(s1) == 0 and len(s4) == 5
            and s4[0] == 4 * SEG + 50, f"seg0={len(s0)} seg1~2={len(s1)} seg4={len(s4)}")
        chk("심볼 코드", [WFDB_SYMBOLS[c] for c in y0] == ["N", "S", "V"])
        chk("환자 정보", A["pid"] == "P123" and A["gender"] == "F")
        chk("리포트 평탄화", A["vb_total"] == 77 and A["sb_total"] == 55
            and A["ann_len"] == 1234 and A["AFAFLPercentage"] == "11.0")
        ffn = json.loads(A["fiducial_feature_names"])
        chk("fiducial feature",
            abs(float(rec.f["seg/fiducial_feat"][0, ffn.index("qt_int")]) - 0.38) < 0.01)
        chk("fiducial CSR", rec.f["fid/seg_offset"][-1] == NSEG * 2)
        chk("용량 감소", os.path.getsize(out) < os.path.getsize(src),
            f"{os.path.getsize(src)/1e6:.1f}MB → {os.path.getsize(out)/1e6:.1f}MB")
        rec.close()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_adapter():
    print()
    print("=" * 70); print("(2) convert_to_h5.py 자료구조 → create_h5_structure_v2"); print("=" * 70)
    work = tempfile.mkdtemp(prefix="v2test_")
    try:
        truth = make_truth()
        signal = [np.stack([truth[l][i*SEG:(i+1)*SEG] for l in PHYS], axis=1)
                  for i in range(NSEG)]
        ba, ss, bs, fp, ff = [], [], [], [], []
        for i in range(NSEG):
            n = BEATS_PER.get(i, 0)
            ba.append({"sample": [j*100+50 for j in range(n)], "symbol": SYMS[:n],
                       "subtype": [0]*n, "chan": [0]*n, "num": [0]*n, "aux_note": [""]*n})
            ss.append({"nan_ratio": [QUAL_TRUTH[l] for l in PHYS], "amp_mean": [0, 0, 0],
                       "amp_std": [1, 2, 3], "amp_skewness": [0, 0, 0],
                       "amp_kurtosis": [0, 0, 0]})
            bs.append({"bs_corr": [.9, .8, .7], "bs_dtw": [1, 2, 3]})
            fp.append({"fsample": [10, 20], "fiducial": ["P", "R"], "extraction_method": "nk"})
            ff.append({"qt_int": 0.38, "rr_int": 0.85})

        out = os.path.join(work, "ADAPTER.h5")
        create_h5_structure_v2(
            out, sig_name=PHYS, n_sig=3, seg_len=NSEG, dataset="SNUH", created_by="t",
            datetime="2026-01-01 00:00:00", record_filename="ADAPTER",
            patient_id="P123", age="67", gender="F", signal=signal,
            beat_annotation=ba, sig_stats=ss, beat_sims=bs,
            fiducial_point=fp, fiducial_feature=ff,
            metadata={"fs": 125, "record_name": "ADAPTER", "base_date": "2024-05-01",
                      "base_time": "09:30:00", "adc_gain": [1, 1, 1], "baseline": [0, 0, 0],
                      "adc_res": [16]*3, "adc_zero": [0]*3, "fmt": ["16"]*3,
                      "units": ["mV"]*3},
            annotation_data={"Holter Report": {"General": {
                "QRScomplexes": "1234", "VentricularBeats": "77",
                "SupraventricularBeats": "55", "AFAFLPercentage": "11.0"},
                "Ventriculars": {"Runs": "2"}, "Supraventriculars": {"Runs": "2"}}})

        rec = RecordV2(out); A = dict(rec.f.attrs)
        chk("lead 정규화", rec.sig_name == ["II", "V1", "V5"])
        sig = rec.window(0, rec.n_samples)
        e = {l: float(np.abs(sig[:, j] - truth[l]).max()) for j, l in enumerate(rec.sig_name)}
        chk("신호 왕복 (int16 양자화 오차 이내)", all(v < 0.01 for v in e.values()),
            ", ".join(f"{k}={v:.1e}" for k, v in e.items()))
        q = {l: float(rec.f["seg/quality"][0, 0, j]) for j, l in enumerate(rec.sig_name)}
        chk("quality 순열", all(abs(q[l] - QUAL_TRUTH[l]) < 0.01 for l in q))
        astd = [float(rec.f["seg/quality"][0, 2, j]) for j in range(3)]
        chk("amp_std 순열", astd == [3.0, 2.0, 1.0], f"{astd} (물리 [1,2,3] → 정규 [3,2,1])")
        off = rec.f["beat/seg_offset"][:]
        exp = np.cumsum([0] + [BEATS_PER.get(i, 0) for i in range(NSEG)])
        chk("CSR offset", np.array_equal(off, exp))
        s4, y4 = rec.beats_in(4, 5)
        chk("beat 절대 인덱스", len(s4) == 5 and s4[0] == 4 * SEG + 50)
        chk("심볼", [WFDB_SYMBOLS[c] for c in y4] == SYMS)
        chk("리포트", A["vb_total"] == 77 and A["ann_len"] == 1234)
        chk("memmap 가능", rec._mm is not None)
        rec.close()
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    test_repack()
    test_adapter()
    print()
    print("전체 통과" if all(_results) else f"실패 {_results.count(False)}건")
    sys.exit(0 if all(_results) else 1)
