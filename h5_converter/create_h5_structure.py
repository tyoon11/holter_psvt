import h5py
import numpy as np

# UTF-8 문자열 타입 정의 (HDF5 저장용)
UTF8 = h5py.string_dtype(encoding="utf-8")


def create_h5_structure(
    h5_file,
    sig_name,
    n_sig,
    seg_len=1,
    dataset="",
    created_by="",
    datetime="",
    record_filename="",
    patient_id="",
    age="",
    gender="",
    signal=None,
    beat_annotation=None,
    sig_stats=None,
    beat_sims=None,
    fiducial_point=None,
    fiducial_feature=None,
    metadata=None,
    annotation_data=None,
):
    """
    ECG 세그먼트 데이터를 HDF5 구조로 저장하는 함수

    Parameters:
        h5_file: h5py File 객체
        signal: 세그먼트 리스트 [np.array(샘플 x 채널)]
        beat_annotation: 세그먼트 리스트 [{dict}]
        sig_stats: 세그먼트 리스트 [{dict}]
        beat_sims: 세그먼트 리스트 [{dict}]
        fiducial_point: 세그먼트 리스트 [{dict}]
        fiducial_feature: 세그먼트 리스트 [{dict}]
        metadata: .hea 기반 메타데이터
        annotation_data: JSON(Holter Report) 기반 요약 정보
    """

    # ───────────── 루트 level attribute 기록 ─────────────
    h5_file.attrs["dataset_version"] = "1.0"
    h5_file.attrs["created_by"] = created_by
    h5_file.attrs["created_at"] = datetime
    h5_file.attrs["file_name"] = record_filename

    # ───────────── patient 그룹 생성 ─────────────
    patient_grp = h5_file.create_group("patient")
    patient_grp.attrs["pid"] = patient_id
    patient_grp.attrs["age"] = age
    patient_grp.attrs["gender"] = gender

    # ───────────── ECG 그룹 및 segments 그룹 생성 ─────────────
    ecg_grp = h5_file.create_group("ECG")
    segments_grp = ecg_grp.create_group("segments")
    segments_grp.attrs["seg_len"] = seg_len

    # 각 세그먼트 그룹에 대한 리스트 준비
    seg_grp = [None] * seg_len
    signal_grp = [None] * seg_len
    beatannot_grp = [None] * seg_len
    fidupoint_grp = [None] * seg_len
    fidufeature_grp = [None] * seg_len
    sigquality_grp = [None] * seg_len
    amplitude_grp = [None] * seg_len
    beatsim_grp = [None] * seg_len

    # ───────────── 세그먼트 별로 그룹 생성 및 데이터 저장 ─────────────
    for i in range(seg_len):
        seg_grp[i] = segments_grp.create_group(str(i))  # segments/0, 1, ...

        # 1️⃣ 신호 저장: 12채널 기준 + PTB 확장
        signal_grp[i] = seg_grp[i].create_group("signal")
        for j, label in enumerate(sig_name):
            if label not in signal_grp[i]:
                signal_grp[i].create_dataset(
                    label, data=signal[i][:, j], dtype=np.float16
                )

        # 2️⃣ R-peak 어노테이션 저장
        if beat_annotation is not None:
            beatannot_grp[i] = seg_grp[i].create_group("beat_annotation")
            ba = beat_annotation[i]
            beatannot_grp[i].create_dataset(
                "sample", data=np.array(ba["sample"], dtype=np.int16)
            )
            beatannot_grp[i].create_dataset(
                "symbol", data=np.array(ba["symbol"], dtype=UTF8), dtype=UTF8
            )
            beatannot_grp[i].create_dataset(
                "subtype", data=np.array(ba["subtype"], dtype=np.int16)
            )
            beatannot_grp[i].create_dataset(
                "chan", data=np.array(ba["chan"], dtype=np.int16)
            )
            beatannot_grp[i].create_dataset(
                "num", data=np.array(ba["num"], dtype=np.int16)
            )
            beatannot_grp[i].create_dataset(
                "aux_note", data=np.array(ba["aux_note"], dtype=UTF8), dtype=UTF8
            )

        # 3️⃣ fiducial point 저장 (e.g. R peak 위치)
        if fiducial_point is not None:
            fp = fiducial_point[i]
            fidupoint_grp[i] = seg_grp[i].create_group("fiducial_point")
            # 🔧 문자열로 변환하여 저장 (안하면 h5py TypeError 발생)
            val = fp.get("extraction_method", "")
            fidupoint_grp[i].attrs["extraction_method"] = (
                str(val) if val is not None else ""
            )

            fidupoint_grp[i].create_dataset(
                "fsample", data=np.array(fp["fsample"], dtype=np.int16)
            )
            fidupoint_grp[i].create_dataset(
                "fiducial", data=np.array(fp["fiducial"], dtype=UTF8), dtype=UTF8
            )

        # 4️⃣ fiducial feature 저장 (QT, PR 등 interval/amplitude)
        if fiducial_feature is not None:
            ff = fiducial_feature[i]
            fidufeature_grp[i] = seg_grp[i].create_group("fiducial_feature")
            for key in [
                "p_amp",
                "q_amp",
                "r_amp",
                "s_amp",
                "t_amp",
                "p_dur",
                "pr_seg",
                "qrs_dur",
                "st_seg",
                "t_dur",
                "pr_int",
                "qt_int",
                "qtc_baz",
                "qtc_frid",
                "rr_int",
                "tp_seg",
                "p_axis",
                "r_axis",
                "t_axis",
            ]:
                val = ff.get(key, np.nan)
                try:
                    fidufeature_grp[i].attrs[key] = np.float16(val)
                except (TypeError, ValueError):
                    fidufeature_grp[i].attrs[key] = np.float16(np.nan)

        # 5️⃣ signal_quality 저장 (NaN 비율 + 통계량)
        if sig_stats is not None:
            stats = sig_stats[i]
            sigquality_grp[i] = seg_grp[i].create_group("signal_quality")
            sigquality_grp[i].create_dataset(
                "nan_ratio", data=np.array(stats["nan_ratio"], dtype=np.float16)
            )

            amplitude_grp[i] = sigquality_grp[i].create_group("amplitude")
            # 🔁 이 부분 수정 필요
            amplitude_grp[i].create_dataset(
                "amp_mean", data=np.array(stats["amp_mean"], dtype=np.float16)
            )
            amplitude_grp[i].create_dataset(
                "amp_std", data=np.array(stats["amp_std"], dtype=np.float16)
            )
            amplitude_grp[i].create_dataset(
                "amp_skewness", data=np.array(stats["amp_skewness"], dtype=np.float16)
            )
            amplitude_grp[i].create_dataset(
                "amp_kurtosis", data=np.array(stats["amp_kurtosis"], dtype=np.float16)
            )

        # 6️⃣ beat 유사도 저장 (correlation / DTW)
        if beat_sims is not None:
            sim = beat_sims[i]
            beatsim_grp[i] = sigquality_grp[i].create_group("beat_similarity")
            beatsim_grp[i].create_dataset(
                "bs_correlation", data=np.array(sim["bs_corr"], dtype=np.float16)
            )
            beatsim_grp[i].create_dataset(
                "bs_dtw", data=np.array(sim["bs_dtw"], dtype=np.float16)
            )

    # ───────────── 메타데이터 (.hea 기반) 저장 ─────────────
    if metadata is not None:
        metadata_grp = ecg_grp.create_group("metadata")
        metadata_grp.attrs["record_name"] = metadata["record_name"] or ""
        metadata_grp.attrs["n_sig"] = n_sig or ""
        metadata_grp.attrs["fs"] = metadata["fs"] or ""
        metadata_grp.attrs["sig_len"] = metadata["sig_len"] or ""
        metadata_grp.attrs["base_time"] = metadata["base_time"] or ""
        metadata_grp.attrs["base_date"] = metadata["base_date"] or ""
        metadata_grp.attrs["dtype"] = "fp16"
        metadata_grp.create_dataset(
            "sig_name", data=np.array(sig_name, dtype=UTF8), dtype=UTF8
        )
        metadata_grp.create_dataset(
            "fmt", data=np.array(metadata["fmt"], dtype=UTF8), dtype=UTF8
        )
        metadata_grp.create_dataset(
            "adc_gain", data=np.array(metadata["adc_gain"], dtype=np.float16)
        )
        metadata_grp.create_dataset(
            "baseline", data=np.array(metadata["baseline"], dtype=np.int16)
        )
        metadata_grp.create_dataset(
            "units", data=np.array(metadata["units"], dtype=UTF8), dtype=UTF8
        )
        metadata_grp.create_dataset(
            "adc_res", data=np.array(metadata["adc_res"], dtype=np.int16)
        )
        metadata_grp.create_dataset(
            "adc_zero", data=np.array(metadata["adc_zero"], dtype=np.int16)
        )

    # ───────────── JSON 기반 리포트 어노테이션 저장 ─────────────
    if annotation_data is not None:
        annotation_grp = ecg_grp.create_group("annotation")
        qrs = (
            annotation_data.get("Holter Report", {})
            .get("General", {})
            .get("QRScomplexes", "0")
        )
        try:
            annotation_grp.attrs["ann_len"] = int(qrs)
        except ValueError:
            annotation_grp.attrs["ann_len"] = 0

        beat_count_grp = annotation_grp.create_group("beat_count")

        ventricular_grp = beat_count_grp.create_group("VentricularBeat")
        general = annotation_data.get("Holter Report", {}).get("General", {})
        ventricular = annotation_data.get("Holter Report", {}).get("Ventriculars", {})
        try:
            ventricular_grp.attrs["total"] = int(general.get("VentricularBeats", 0))
        except ValueError:
            ventricular_grp.attrs["total"] = 0
        for key in ["Isolated", "Couplets", "BigeminalCycles"]:
            val = ventricular.get(key, "0")
            try:
                ventricular_grp.attrs[key] = int(val)
            except ValueError:
                ventricular_grp.attrs[key] = val

        ventricular_runs_grp = ventricular_grp.create_group("Runs")
        vent_runs = ventricular.get("Runs", "0")
        try:
            ventricular_runs_grp.attrs["count"] = int(vent_runs)
        except ValueError:
            ventricular_runs_grp.attrs["count"] = vent_runs
        ventricular_runs_grp.attrs["TotalBeats"] = int(
            general.get("VentricularBeats", 0)
        )
        for attr in [
            "LongestRunBeats",
            "LongestRunBPM",
            "LongestRunTimestamp",
            "FastestRunBeats",
            "FastestRunBPM",
            "FastestRunTimestamp",
        ]:
            ventricular_runs_grp.attrs[attr] = ventricular.get(attr, "Unknown")

        supraventricular_grp = beat_count_grp.create_group("SupraventricularBeat")
        general_supra = annotation_data.get("Holter Report", {}).get("General", {})
        supra = annotation_data.get("Holter Report", {}).get("Supraventriculars", {})
        try:
            supraventricular_grp.attrs["total"] = int(
                general_supra.get("SupraventricularBeats", 0)
            )
        except ValueError:
            supraventricular_grp.attrs["total"] = 0
        for key in ["Isolated", "Couplets", "BigeminalCycles"]:
            val = supra.get(key, "0")
            try:
                supraventricular_grp.attrs[key] = int(val)
            except ValueError:
                supraventricular_grp.attrs[key] = val

        supraventricular_runs_grp = supraventricular_grp.create_group("Runs")
        supra_runs = supra.get("Runs", "0")
        try:
            supraventricular_runs_grp.attrs["count"] = int(supra_runs)
        except ValueError:
            supraventricular_runs_grp.attrs["count"] = supra_runs
        supraventricular_runs_grp.attrs["TotalBeats"] = int(
            general_supra.get("SupraventricularBeats", 0)
        )
        for attr in [
            "LongestRunBeats",
            "LongestRunBPM",
            "LongestRunTimestamp",
            "FastestRunBeats",
            "FastestRunBPM",
            "FastestRunTimestamp",
        ]:
            supraventricular_runs_grp.attrs[attr] = supra.get(attr, "Unknown")

        for group_name, key in zip(
            ["PacedBeats", "BBBeats", "JunctionalBeats", "AberrantBeats"],
            ["PacedBeats", "BBBeats", "JunctionalBeats", "AberrantBeats"],
        ):
            grp = beat_count_grp.create_group(group_name)
            val = general.get(key, "0")
            try:
                grp.attrs["total"] = int(val)
            except ValueError:
                grp.attrs["total"] = val

        annotation_grp.attrs["NoisePercentage"] = general.get(
            "NoisePercentage", "Unknown"
        )
        annotation_grp.attrs["AFAFLPercentage"] = general.get(
            "AFAFLPercentage", "Unknown"
        )
