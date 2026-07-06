import numpy as np
import neurokit2 as nk
from scipy.stats import skew, kurtosis
from dtw import dtw
import wfdb
import os
from neurokit2.misc import NeuroKitWarning
import warnings
import json
import pandas as pd

warnings.filterwarnings(
    "ignore",
    category=NeuroKitWarning,
    message="Too few peaks detected to compute the rate.*",
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered in divide"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="Mean of empty slice"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in scalar divide",
)


def safe_mean_duration(a, b, sampling_rate):
    try:
        a = np.array(a)
        b = np.array(b)
        if len(a) == 0 or len(b) == 0:
            return np.nan
        diff = b - a
        if len(diff) == 0:
            return np.nan
        # 경고 제거
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = np.nanmean(diff) / sampling_rate
        return result
    except:
        return np.nan


# 📌 JSON에서 환자 정보 추출
def extract_patient_info(json_path):
    """
    - patient_id: 파일명에서 추출 (예: DVD20160909_33_2411480 → 2411480)
    - age, gender: JSON에서 추출 (없거나 오류 시 fallback)
    """
    # patient_id: 파일명에서 추출
    base = os.path.basename(json_path)
    name_only = os.path.splitext(base)[0]
    parts = name_only.split("_")
    patient_id = parts[-1] if len(parts) >= 3 else name_only

    # age, gender: JSON에서 추출 시도
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        patient_data = data.get("Holter Report", {}).get("PatientInfo", {})
        age = patient_data.get("Age", "")
        gender = patient_data.get("Gender", "")
        age = int(age) if str(age).isdigit() else -1
        gender = gender or ""
    except:
        age = -1
        gender = ""

    return {
        "patient_id": patient_id,
        "age": age,
        "gender": gender,
    }


# =============================================================================
# 1. 신호 품질 통계 계산 함수
# =============================================================================


def signal_statistics(signal):
    signal = np.array(signal)

    nan_ratios = np.mean(np.isnan(signal), axis=1)
    n_channels = signal.shape[0]
    means = np.array(
        [
            (
                np.nanmean(signal[i])
                if signal[i].size > 0 and not np.all(np.isnan(signal[i]))
                else np.nan
            )
            for i in range(n_channels)
        ]
    )
    stds = np.array(
        [
            (
                np.nanstd(signal[i])
                if signal[i].size > 0 and not np.all(np.isnan(signal[i]))
                else np.nan
            )
            for i in range(n_channels)
        ]
    )
    skews = np.array(
        [
            (
                skew(signal[i], nan_policy="omit")
                if signal[i].size > 0 and not np.all(np.isnan(signal[i]))
                else np.nan
            )
            for i in range(n_channels)
        ]
    )
    kurtoses = np.array(
        [
            (
                kurtosis(signal[i], nan_policy="omit")
                if signal[i].size > 0 and not np.all(np.isnan(signal[i]))
                else np.nan
            )
            for i in range(n_channels)
        ]
    )

    means = np.nan_to_num(means, nan=0.0)
    stds = np.nan_to_num(stds, nan=0.0)
    skews = np.nan_to_num(skews, nan=0.0)
    kurtoses = np.nan_to_num(kurtoses, nan=0.0)

    result = {
        "nan_ratio": list(nan_ratios),
        "amp_mean": list(means),
        "amp_std": list(stds),
        "amp_skewness": list(skews),
        "amp_kurtosis": list(kurtoses),
    }

    return result


# =============================================================================
# 2. beat 간 유사도 계산 함수 (correlation + DTW)
# =============================================================================
def beat_similarity(signal, sampling_rate=500, use_dummy=True):
    if use_dummy:
        n_channel = signal.shape[0]
        return {
            "bs_corr": [np.nan] * n_channel,
            "bs_dtw": [np.nan] * n_channel,
        }

    signal = np.array(signal)
    fixed_length = sampling_rate * 2
    n_channel = signal.shape[0]
    mean_corrs = [np.nan for i in range(n_channel)]
    mean_dtws = [np.nan for i in range(n_channel)]

    for idx in range(n_channel):
        _, rpeaks = nk.ecg_peaks(signal[idx], sampling_rate=sampling_rate)
        rpeaks = rpeaks.get("ECG_R_Peaks", [])

        if len(rpeaks) > 3:
            beats = nk.ecg_segment(signal[idx], rpeaks, sampling_rate=sampling_rate)

            if len(beats) > 3:
                beat_matrix = []
                for beat in beats.values():
                    beat_resampled = nk.signal_resample(
                        np.array(beat, dtype=float), desired_length=fixed_length
                    )
                    std = np.std(beat_resampled)
                    if std == 0 or np.isnan(std):
                        beat_resampled = np.zeros_like(beat_resampled)
                    else:
                        beat_resampled = (
                            beat_resampled - np.mean(beat_resampled)
                        ) / np.std(beat_resampled)
                    beat_matrix.append(beat_resampled.squeeze())
                beat_matrix = np.array(beat_matrix)

                # Beat correlation
                correlations = []
                for i in range(len(beat_matrix) - 1):
                    if not np.any(np.isnan(beat_matrix[i])) and not np.any(
                        np.isnan(beat_matrix[i + 1])
                    ):
                        corr = np.corrcoef(beat_matrix[i], beat_matrix[i + 1])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)

                valid_corrs = [c for c in correlations if not np.isnan(c)]
                mean_corrs[idx] = (
                    np.mean(valid_corrs) if len(valid_corrs) > 0 else np.nan
                )

                # Beat DTW
                dtw_distances = []
                for i in range(len(beat_matrix) - 1):
                    if np.any(np.isnan(beat_matrix[i])) or np.any(
                        np.isnan(beat_matrix[i + 1])
                    ):
                        continue
                    try:
                        alignment = dtw(
                            beat_matrix[i],
                            beat_matrix[i + 1],
                            # dist=lambda x, y: norm(x - y)  # 유클라디안 거리 추가
                        )
                        normalized_distance = alignment.distance / fixed_length
                        dtw_distances.append(normalized_distance)
                    except:
                        pass

                mean_dtws[idx] = (
                    np.nanmean(dtw_distances) if len(dtw_distances) > 0 else np.nan
                )

    return {"bs_corr": mean_corrs, "bs_dtw": mean_dtws}


def compute_signal_quality(signal, sampling_rate, use_dummy_similarity=True):
    sig_qual = {}
    sig_qual.update(signal_statistics(signal))
    sig_qual.update(
        beat_similarity(signal, sampling_rate, use_dummy=use_dummy_similarity)
    )
    return sig_qual


def amplitude_calc(signal, points1, points2):
    if (len(points1) == len(points2)) and (len(points1) > 3):
        points1, points2 = np.array(points1, dtype=np.float32), np.array(
            points2, dtype=np.float32
        )
        signal = np.array(signal, dtype=np.float32)
        valid_mask = ~np.isnan(points1) & ~np.isnan(points2)
        if not np.any(valid_mask):
            return np.nan
        return np.nanmean(
            signal[points1[valid_mask].astype(int)]
            - signal[points2[valid_mask].astype(int)]
        )
    else:
        return np.nan


def interval_calc(arr1, arr2):
    if (len(arr1) == len(arr2)) and (len(arr1) > 3):
        arr1, arr2 = np.array(arr1, dtype=np.float32), np.array(arr2, dtype=np.float32)
        valid_mask = ~np.isnan(arr1) & ~np.isnan(arr2)
        if not np.any(valid_mask):
            return np.nan
        return np.nanmean(arr2[valid_mask] - arr1[valid_mask])
    else:
        return np.nan


def axis_calc(points, sig1, sig2):
    points = [point for point in points if type(point) is int]
    if len(points) > 0:
        amps1 = sig1[points]
        amps2 = sig2[points]
        angles = np.degrees(np.arctan2(amps2, amps1))
        return np.nanmean(angles)
    else:
        return np.nan


# =============================================================================
# 3. fiducial point 및 ECG feature 추출 함수
# =============================================================================
def get_fiducial_points(waves_dwt):
    fiducial_point = {"extraction_method": None, "fsample": [], "fiducial": []}

    if waves_dwt is not None:
        fiducial_names = [
            "ECG_P_Onsets",
            "ECG_P_Peaks",
            "ECG_Q_Onsets",
            "ECG_P_Offsets",
            "ECG_Q_Peaks",
            "ECG_R_Peaks",
            "ECG_S_Peaks",
            "ECG_R_Offsets",
            "ECG_L_Points",
            "ECG_T_Onsets",
            "ECG_T_Peaks",
            "ECG_T_Offsets",
        ]
        fsample_list, fiducial_list = [], []
        for label in fiducial_names:
            points = waves_dwt.get(label)
            if points is not None:
                points = np.array(points, dtype=np.float32)
                valid_indices = ~np.isnan(points)
                points = points[valid_indices].astype(int)
                labels = [label] * len(points)
                fsample_list.extend(points)
                fiducial_list.extend(labels)

        sorted_indices = np.argsort(fsample_list)

        fiducial_point["extraction_method"] = "neurokit2-dwt"
        fiducial_point["fsample"] = list(np.array(fsample_list)[sorted_indices])
        fiducial_point["fiducial"] = list(np.array(fiducial_list)[sorted_indices])

    return fiducial_point


def get_fiducial_features(waves_dwt, signal_lead1, signal_lead2, sampling_rate):
    fiducial_feature = {
        "p_amp": None,
        "q_amp": None,
        "r_amp": None,
        "s_amp": None,
        "t_amp": None,
        "p_dur": None,
        "pr_seg": None,
        "qrs_dur": None,
        "st_seg": None,
        "t_dur": None,
        "pr_int": None,
        "qt_int": None,
        "rr_int": None,
        "tp_seg": None,
        "rr_int": None,
        "qtc_baz": None,
        "qtc_frid": None,
        "p_axis": None,
        "r_axis": None,
        "t_axis": None,
    }

    if waves_dwt is not None:
        # Amplitudes
        fiducial_feature["p_amp"] = amplitude_calc(
            signal_lead2, waves_dwt["ECG_P_Peaks"], waves_dwt["ECG_P_Onsets"]
        )
        fiducial_feature["q_amp"] = amplitude_calc(
            signal_lead2, waves_dwt["ECG_Q_Peaks"], waves_dwt["ECG_P_Onsets"]
        )
        fiducial_feature["r_amp"] = (
            amplitude_calc(
                signal_lead2, waves_dwt["ECG_R_Peaks"], waves_dwt["ECG_Q_Peaks"]
            )
            / 2
            + amplitude_calc(
                signal_lead2, waves_dwt["ECG_R_Peaks"], waves_dwt["ECG_S_Peaks"]
            )
            / 2
        )
        fiducial_feature["s_amp"] = amplitude_calc(
            signal_lead2, waves_dwt["ECG_S_Peaks"], waves_dwt["ECG_P_Onsets"]
        )
        fiducial_feature["t_amp"] = amplitude_calc(
            signal_lead2, waves_dwt["ECG_T_Peaks"], waves_dwt["ECG_T_Onsets"]
        )

        # Intervals
        fiducial_feature["p_dur"] = (
            interval_calc(waves_dwt["ECG_P_Onsets"], waves_dwt["ECG_P_Offsets"])
            / sampling_rate
        )
        fiducial_feature["pr_seg"] = (
            interval_calc(waves_dwt["ECG_P_Offsets"], waves_dwt["ECG_Q_Peaks"])
            / sampling_rate
        )
        fiducial_feature["qrs_dur"] = (
            interval_calc(waves_dwt["ECG_Q_Peaks"], waves_dwt["ECG_S_Peaks"])
            / sampling_rate
        )
        fiducial_feature["st_seg"] = (
            interval_calc(waves_dwt["ECG_S_Peaks"], waves_dwt["ECG_T_Onsets"])
            / sampling_rate
        )
        fiducial_feature["t_dur"] = (
            interval_calc(waves_dwt["ECG_T_Onsets"], waves_dwt["ECG_T_Offsets"])
            / sampling_rate
        )
        fiducial_feature["pr_int"] = (
            interval_calc(waves_dwt["ECG_P_Onsets"], waves_dwt["ECG_Q_Peaks"])
            / sampling_rate
        )
        fiducial_feature["qt_int"] = (
            interval_calc(waves_dwt["ECG_Q_Peaks"], waves_dwt["ECG_T_Offsets"])
            / sampling_rate
        )
        fiducial_feature["rr_int"] = (
            interval_calc(waves_dwt["ECG_R_Peaks"][:-1], waves_dwt["ECG_R_Peaks"][1:])
            / sampling_rate
        )
        fiducial_feature["tp_seg"] = (
            interval_calc(
                waves_dwt["ECG_T_Offsets"][:-1], waves_dwt["ECG_P_Onsets"][1:]
            )
            / sampling_rate
        )
        if fiducial_feature["rr_int"] > 0:
            fiducial_feature["qtc_baz"] = fiducial_feature["qt_int"] / np.sqrt(
                fiducial_feature["rr_int"]
            )
            fiducial_feature["qtc_frid"] = fiducial_feature[
                "qt_int"
            ] / fiducial_feature["rr_int"] ** (1 / 3)
        else:
            fiducial_feature["qtc_baz"] = np.nan
            fiducial_feature["qtc_frid"] = np.nan

        # Axes
        if signal_lead1 is not None:
            fiducial_feature["p_axis"] = axis_calc(
                waves_dwt["ECG_P_Peaks"], signal_lead1, signal_lead2
            )
            fiducial_feature["r_axis"] = axis_calc(
                waves_dwt["ECG_R_Peaks"], signal_lead1, signal_lead2
            )
            fiducial_feature["t_axis"] = axis_calc(
                waves_dwt["ECG_T_Peaks"], signal_lead1, signal_lead2
            )
        else:
            fiducial_feature["p_axis"] = np.nan
            fiducial_feature["r_axis"] = np.nan
            fiducial_feature["t_axis"] = np.nan
        fiducial_feature = {
            k: (
                np.float16(v)
                if isinstance(v, (float, int, np.number))
                else np.float16(np.nan)
            )
            for k, v in fiducial_feature.items()
        }

    return fiducial_feature


def extract_ecg_features(signal, sampling_rate, leads, use_dummy=False):
    """
    Extract ECG fiducial points, interval-related features, and axis.

    Inputs:
        signal (np.array or list): shape (num_leads, signal_length)
        sampling_rate (int): ECG sampling rate
        leads (list): lead names, e.g., ["I", "II", "III", "aVR", "aVL", "aVF"]

    Outputs:
        dict: fiducial points and features
    """
    if signal is None or sampling_rate is None or leads is None:
        raise ValueError("Signal, sampling_rate, and leads cannot be None.")

    signal = np.array(signal)
    if signal.size == 0:
        raise ValueError("Signal cannot be empty.")

    fiducial_feature_keys = [
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
        "rr_int",
        "tp_seg",
        "qtc_baz",
        "qtc_frid",
        "p_axis",
        "r_axis",
        "t_axis",
    ]
    if use_dummy:
        fiducial_feature = {key: np.nan for key in fiducial_feature_keys}
        fiducial_point = {"extraction_method": "", "fsample": [], "fiducial": []}
        return {"fiducial_point": fiducial_point, "fiducial_feature": fiducial_feature}

    fiducial_feature = {key: np.nan for key in fiducial_feature_keys}
    fiducial_point = {"extraction_method": "", "fsample": [], "fiducial": []}

    signal_lead1 = signal_lead2 = None

    if "I" in leads:
        idx_lead1 = leads.index("I")
        if signal[idx_lead1].size > 0:
            signal_lead1 = nk.ecg_clean(signal[idx_lead1], sampling_rate=sampling_rate)

    if "II" in leads:
        idx_lead2 = leads.index("II")
        if signal[idx_lead2].size > 0:
            signal_lead2 = nk.ecg_clean(signal[idx_lead2], sampling_rate=sampling_rate)
            _, rpeaks = nk.ecg_peaks(signal_lead2, sampling_rate=sampling_rate)
            rpeaks = rpeaks.get("ECG_R_Peaks", [])

            if len(rpeaks) > 3:
                _, waves_dwt = nk.ecg_delineate(
                    signal_lead2,
                    rpeaks,
                    sampling_rate=sampling_rate,
                    method="dwt",
                    show=False,
                )
                waves_dwt["ECG_R_Peaks"] = rpeaks
            else:
                waves_dwt = None

            fiducial_point = get_fiducial_points(waves_dwt)
            fiducial_feature = get_fiducial_features(
                waves_dwt, signal_lead1, signal_lead2, sampling_rate
            )

    return {"fiducial_point": fiducial_point, "fiducial_feature": fiducial_feature}


# =============================================================================
# 4. JSON 어노테이션에서 "Holter Report" 항목 추출
# =============================================================================
def parse_annotation_json(report: dict):
    return report


# =============================================================================
# 5. HEA 파일 메타데이터 파싱
# =============================================================================
def parse_hea(record_path):
    base_path = os.path.splitext(record_path)[0]

    # ✅ force 없이 단순히 기본 사용 (확장자 자동 인식)
    record = wfdb.rdrecord(base_path)  # force=False, pb_dir 사용 X

    d = record.__dict__
    n_sig = d["n_sig"]
    lead_names = record.sig_name

    if len(set(lead_names)) == 1 and lead_names[0].lower() in [
        "mars export",
        "unnamed",
        "",
    ]:
        if n_sig == 12:
            record.sig_name = [
                "I",
                "II",
                "III",
                "aVR",
                "aVL",
                "aVF",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
            ]
        elif n_sig == 3:
            record.sig_name = ["V5", "V1", "II"]
        else:
            record.sig_name = [f"lead{i}" for i in range(n_sig)]

    meta = {
        "record_name": d["record_name"],
        "n_sig": n_sig,
        "fs": d["fs"],
        "sig_len": d["sig_len"],
        "base_time": str(d["base_time"]),
        "base_date": str(d["base_date"]),
        "fmt": d["fmt"],
        "adc_gain": d["adc_gain"],
        "baseline": d["baseline"],
        "units": d["units"],
        "adc_res": d["adc_res"],
        "adc_zero": d["adc_zero"],
        "sig_name": record.sig_name,
    }

    return record, meta


# =============================================================================
# 6. ANN 어노테이션 파일 파싱 (심볼, 샘플 등)
# =============================================================================
def parse_ann(record_path):
    try:
        ann = wfdb.rdann(record_path, extension="ANN")  # ✅ .ANN 확장자 직접 지정
        return {
            "sample": ann.sample,
            "symbol": ann.symbol,
            "subtype": ann.subtype,
            "chan": ann.chan,
            "num": ann.num,
            "aux_note": ann.aux_note,
        }
    except:
        return {
            "sample": [],
            "symbol": [],
            "subtype": [],
            "chan": [],
            "num": [],
            "aux_note": [],
        }


# =============================================================================
# 7. 어노테이션 세그먼트 단위로 자르기
# =============================================================================
def slice_ann_by_segment(ann_dict, start, end):
    sample = np.array(ann_dict["sample"])
    idx = (sample >= start) & (sample < end)
    result = {
        k: np.array(v)[idx].tolist() if isinstance(v, (np.ndarray, list)) else []
        for k, v in ann_dict.items()
    }
    result["sample"] = (np.array(result["sample"]) - start).tolist()
    return result


# =============================================================================
# 8. 유효 레코드 목록 생성 함수
# =============================================================================


def has_all_required_files(base_path):
    return all(
        os.path.exists(base_path + ext) for ext in [".hea", ".SIG", ".ANN", ".json"]
    )


def generate_valid_records(input_dir, output_csv):
    hea_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".hea")]
    record_names = [os.path.splitext(f)[0] for f in hea_files]

    valid_records = []
    for name in record_names:
        base_path = os.path.join(input_dir, name)
        if has_all_required_files(base_path):
            valid_records.append(name)

    df = pd.DataFrame({"record_name": valid_records})
    df.to_csv(output_csv, index=False)
    print(f"✅ 유효한 레코드 {len(valid_records)}개 저장됨: {output_csv}")
