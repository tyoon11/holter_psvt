import os
import json
import h5py
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import ray
import logging

from utils import (
    signal_statistics,
    beat_similarity,
    extract_ecg_features,
    parse_hea,
    parse_ann,
    parse_annotation_json,
    slice_ann_by_segment,
    extract_patient_info,
    generate_valid_records,
)
from create_h5_structure import create_h5_structure


# ✅ 레코드 하나 처리 (Ray 전용, JSON 정보까지 포함)
@ray.remote
def convert_one_record_ray(
    record_path,
    sampling_rate,
    segment_sec,
    max_segments,
    use_dummy_fiducial=True,
    use_dummy_similarity=True,
):
    record_name = os.path.splitext(os.path.basename(record_path))[0]
    record_path_no_ext = os.path.splitext(record_path)[0]

    try:
        print(f"[\U0001f504 INFO] Processing: {record_name}", flush=True)
        record, metadata = parse_hea(record_path)
        leads = record.sig_name
        full_signal = record.p_signal.T
        total_length = full_signal.shape[1]

        ann_data = parse_ann(record_path_no_ext)
        try:
            with open(record_path_no_ext + ".json") as f:
                annotation_json = json.load(f)
                annotation_data = parse_annotation_json(annotation_json)
        except:
            annotation_data = {}

        # 📌 환자 정보도 파싱
        patient_info = extract_patient_info(record_path_no_ext + ".json")

        segment_len = sampling_rate * segment_sec
        total_segments = total_length // segment_len
        if max_segments is not None:
            total_segments = min(total_segments, max_segments)

        seg_signals, seg_annotations, seg_stats = [], [], []
        seg_sims, seg_fidupoints, seg_fidufeatures = [], [], []

        for i in range(total_segments):
            start = i * segment_len
            end = (i + 1) * segment_len
            seg_signal = full_signal[:, start:end]
            seg_ann = slice_ann_by_segment(ann_data, start, end)

            stats = signal_statistics(seg_signal)
            sims = beat_similarity(
                seg_signal, sampling_rate, use_dummy=use_dummy_similarity
            )
            fidu = extract_ecg_features(
                seg_signal, sampling_rate, leads, use_dummy=use_dummy_fiducial
            )

            seg_signals.append(seg_signal.T)
            seg_annotations.append(seg_ann)
            seg_stats.append(stats)
            seg_sims.append(sims)
            seg_fidupoints.append(fidu["fiducial_point"])
            seg_fidufeatures.append(fidu["fiducial_feature"])

        return {
            "record_name": record_name,
            "leads": leads,
            "signal": seg_signals,
            "beat_annotation": seg_annotations,
            "sig_stats": seg_stats,
            "beat_sims": seg_sims,
            "fiducial_point": seg_fidupoints,
            "fiducial_feature": seg_fidufeatures,
            "metadata": metadata,
            "annotation_data": annotation_data,
            "segment_cnt": total_segments,
            "patient_info": patient_info,
        }

    except Exception as e:
        logging.exception(f"[❌ EXCEPTION] {record_name} - {e}")
        return None


# ✅ 전체 폴더 변환 (Ray 병렬 + 직렬 저장 + 환자 정보 포함)
def convert_folder_to_h5_ray(
    input_dir,
    output_dir,
    csv_path,
    sampling_rate=125,
    segment_sec=10,
    max_segments=None,
    log_path="conversion.log",
    valid_list_path="valid_records.csv",
    use_dummy_fiducial=True,
    use_dummy_similarity=True,
):
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    os.makedirs(output_dir, exist_ok=True)

    # ✅ 유효 레코드 목록 CSV가 없으면 자동 생성
    if not os.path.exists(valid_list_path):
        logging.info(f"📄 valid_list_path가 없어 자동 생성 중 → {valid_list_path}")
        generate_valid_records(input_dir, valid_list_path)

    ray.init(num_cpus=64)
    logging.info(f"🧠 Ray initialized (CPUs: {ray.available_resources().get('CPU')})")

    df = pd.read_csv(valid_list_path)
    record_names = df["record_name"].tolist()
    record_paths = [os.path.join(input_dir, f"{name}.hea") for name in record_names]

    existing_h5 = {
        os.path.splitext(f)[0]
        for f in os.listdir(output_dir)
        if f.lower().endswith(".h5")
    }
    pending_record_paths = [
        path
        for path in record_paths
        if os.path.splitext(os.path.basename(path))[0] not in existing_h5
    ]
    logging.info(
        f"🚀 변환 대상: 총 {len(record_paths)}개 중 {len(pending_record_paths)}개 처리 예정"
    )

    futures = [
        convert_one_record_ray.remote(
            path,
            sampling_rate,
            segment_sec,
            max_segments,
            use_dummy_fiducial,
            use_dummy_similarity,
        )
        for path in pending_record_paths
    ]

    saved_files = []
    for future in tqdm(futures, desc="📦 Processing + Saving"):
        try:
            data = ray.get(future)
            if data:
                h5_name = f"{data['record_name']}.h5"
                h5_path = os.path.join(output_dir, h5_name)

                with h5py.File(h5_path, "w") as h5f:
                    create_h5_structure(
                        h5_file=h5f,
                        sig_name=data["leads"],
                        n_sig=len(data["leads"]),
                        seg_len=data["segment_cnt"],
                        dataset="SNUH",
                        created_by="",
                        datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        record_filename=data["record_name"],
                        patient_id=data["patient_info"]["patient_id"],
                        age=data["patient_info"]["age"],
                        gender=data["patient_info"]["gender"],
                        signal=data["signal"],
                        beat_annotation=data["beat_annotation"],
                        sig_stats=data["sig_stats"],
                        beat_sims=data["beat_sims"],
                        fiducial_point=data["fiducial_point"],
                        fiducial_feature=data["fiducial_feature"],
                        metadata=data["metadata"],
                        annotation_data=data["annotation_data"],
                    )

                logging.info(f"[📂 SAVED] {h5_name}")
                saved_files.append(data["record_name"])

        except Exception as e:
            logging.exception(f"[❌ SAVE FAILED] {e}")

    pd.DataFrame({"file": [f"{name}.h5" for name in saved_files]}).to_csv(
        csv_path, index=False
    )
    logging.info(f"📄 저장된 파일 목록: {csv_path}")
    ray.shutdown()
    logging.info("🛑 Ray 종료 완료")


# ✅ 실행 예시
if __name__ == "__main__":
    convert_folder_to_h5_ray(
        input_dir="/your/raw/data",
        output_dir="/your/output/h5",
        csv_path="/your/path/output_h5_list.csv",
        sampling_rate=125,
        segment_sec=10,
        log_path="/your/path/conversion_log.txt",
        valid_list_path="/your/path/valid_records.csv",
        use_dummy_fiducial=True,
        use_dummy_similarity=True,
    )

