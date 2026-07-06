# gn_pipeline.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GN 버전의 H5 데이터셋 이용 버전
이 스크립트는 Holter 24시간 ECG를 10초 세그먼트로 분할해 만든 HDF5 데이터셋을 이용해
PSVT 여부를 이진 분류하는 Multimodal MIL(ECG bag + metadata) 학습/평가 파이프라인이다.

구성 요약
- Sampling:
  * 각 record(HDF5 파일)에서 K개의 segment key를 선택해 하나의 bag을 구성
  * "S" 심볼(예: supraventricular) 포함 segment를 기준으로 select_keys()가 segment를 선택
  * segment_seed에 따라 같은 설정이면 같은 segment들이 선택되도록 결정론적으로 동작

- 캐시:
  * record별로 선택된 segment key 리스트를 JSONL로 저장/재사용
  * (method, radius, K, fill, segseed, upsample_factor, split_seed, split) 조합으로 캐시 분리

- Upsampling:
  * Positive record(label==1)만 upsample_factor번 반복하여 서로 다른 bag을 만들도록 scan_record 호출
  * train/val에 적용 여부를 bool로 제어
  * test는 항상 미적용

- 학습:
  * 여러 encoder/head 조합을 순회
  * AUROC 기준 early stopping
  * BCEWithLogitsLoss의 pos_weight를 fixed_train 분포 기반으로 동적으로 계산 가능

- 집계:
  * per_run_summary.csv에 모든 run 결과를 append
  * 일정 run마다(per AGG_EVERY) segseed_agg / overall / summary 재계산하여 CSV로 저장
"""

# =========================
# 0) 환경/재현성 관련 환경변수
# =========================
import os

# PyTorch CUDA 결정론을 위한 설정.
# CUBLAS_WORKSPACE_CONFIG는 CUDA matmul/cublas의 deterministic 동작을 위해 필요할 수 있다.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# Ray가 컨테이너/호스트 CPU 코어 수 감지 관련 경고를 내는 것을 줄이기 위한 옵션들.
# (성능/기능에 영향이 있을 수 있으나, 현재 목적은 로그/경고 억제 및 안정적 실행)
os.environ["RAY_USE_MULTIPROCESSING_CPU_COUNT"] = "1"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"

# =========================
# 1) 외부 라이브러리 import
# =========================
import ray
import h5py
import json
import torch
import random
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)

# =========================
# 2) 내부 모듈 import
# =========================
# build_model: encoder/head 조합으로 Multimodal MIL 모델을 구성하는 빌더 함수.
# select_keys: HDF5의 segment key 목록(all_keys)과 S-segment 목록(s_keys_all)을 받아,
#              method/radius/fill 규칙에 따라 K개 segment key를 선택하는 로직.
from models import build_model             # 변경 금지
from sampling import select_keys

# =========================
# 3) 전역 설정: 시드, 실험 옵션
# =========================
# SEED: 전체 코드에서 공통으로 사용하는 베이스 시드
# - 모델 초기화, DataLoader worker seed, 일부 랜덤 연산의 기준값
SEED = 42

# SEGMENT_SEEDS: 세그먼트 샘플링(=bag 구성) 결정에 사용하는 시드 목록
# - 동일한 설정이라도 segment_seed를 바꾸면 다른 bag이 선택될 수 있다.
SEGMENT_SEEDS = [42]

# Upsampling 실험 설정
# - 1이면 upsampling 없음
# - 5, 10이면 positive record에 대해 동일 record를 여러 bag으로 확장 (upsample_index로 bag 다양화)
UPSAMPLE_FACTORS = [1, 5, 10]

# Train/Val 각각에 대해 upsampling 적용 여부
# - True면 train/val에 upsample_factor 적용
# - False면 해당 split은 항상 upsample_factor=1로 강제(즉, upsampling 비활성)
APPLY_UPSAMPLE_TRAIN = True
APPLY_UPSAMPLE_VAL   = True
# Test는 항상 upsampling 미적용(평가 고정)

# Dynamic pos_weight 설정
# - BCEWithLogitsLoss(pos_weight=...)에 들어갈 값을 fixed_train의 클래스 분포로 동적으로 계산
USE_DYNAMIC_POS_WEIGHT = True
POS_WEIGHT_CAP = 20.0           # pos_weight의 상한 (너무 큰 가중치로 학습 불안정 방지 목적)
POS_WEIGHT_FALLBACK = 15.0      # pos 또는 neg가 0인 경우(정상적 계산 불가) 사용할 대체 값

# =========================
# 4) 재현성 세팅 (PyTorch / numpy / random)
# =========================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# cudnn 설정: deterministic / benchmark off
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# PyTorch가 제공하는 deterministic 알고리즘 강제
# - 일부 연산이 deterministic 구현이 없으면 에러가 날 수 있음
torch.use_deterministic_algorithms(True)

# =========================
# 5) 디바이스 및 Ray 초기화
# =========================
# GPU가 있으면 cuda:0 사용, 없으면 CPU 사용
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Ray 초기화
# - _temp_dir: Ray가 임시 파일/로그를 쓰는 위치
# - include_dashboard=False: 대시보드 비활성화
# - log_to_driver=False: 드라이버로 로그 전달 억제
ray.init(
    ignore_reinit_error=True,
    _temp_dir="/home/coder/workspace/data/ray_tmp",
    logging_level="ERROR",
    log_to_driver=False,
    include_dashboard=False,
)

# =========================
# 6) 경로/입력/출력 설정
# =========================
# CSV_PATH: record별 메타데이터(나이/성별/beat 통계 등)가 들어있는 요약 CSV
CSV_PATH    = "/home/coder/workspace/data_readwrite/10s_segment_final/h5_metadata_summary.csv"

# BASE_DIR: 원본 HDF5 데이터가 위치한 루트
BASE_DIR    = "/home/coder/workspace/data_readwrite/10s_segment_final"

# RESULT_ROOT: 이번 실행 결과를 저장할 디렉토리 (타임스탬프 포함)
RESULT_ROOT = f"/home/coder/workspace/data_readwrite/tykim/results0109/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# SPLIT_DIR: patient-wise split 결과 CSV들이 있는 디렉토리
# - 파일명 규칙: split_seed{00}.csv
SPLIT_DIR   = "/home/coder/workspace/data_readwrite/10s_segment_final/seed_splits1215"

# SPLIT_SEEDS: 어떤 split 파일들을 돌릴지 결정
SPLIT_SEEDS = [500]

# CACHE_DIR: fixed bag(=record별 selected_keys) 캐시를 저장할 디렉토리
CACHE_DIR   = "/home/coder/workspace/data_readwrite/tykim/seed_caches0109"

# SLIKE_DIR: fill="slike" 전략에서 사용하는 외부 score/랭킹 파일 위치(선택 로직에서 사용)
SLIKE_DIR = "/home/coder/workspace/data_readwrite/hjlee/VectorScore_Best500_20260106"

os.makedirs(RESULT_ROOT, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# =========================
# 7) 학습 하이퍼파라미터
# =========================
HIDDEN_DIM = 64    # MIL head/encoder 내부 latent 크기(모델 빌더에서 사용)
META_DIM = 32      # metadata encoder 출력 차원(모델 빌더에서 사용)
DROPOUT = 0.1
LR = 0.0008

# SEGMENT_LENGTH: 각 segment의 time length (샘플 수)
# - 신호가 더 짧으면 pad, 더 길면 crop
SEGMENT_LENGTH = 1250

BATCH_SIZE = 16
EPOCHS = 50

# Early stopping 관련
EARLY_PATIENCE = 30  # 개선 없으면 stop
MIN_DELTA = 0.001    # 개선으로 인정할 최소 향상 폭

# ReduceLROnPlateau 사용 여부
USE_PLATEAU_LR = True

# FORCE_RESCAN: True면 캐시가 있어도 다시 스캔해서 캐시를 덮어쓴다.
FORCE_RESCAN = False

# =========================
# 8) 메타 피처 / 라벨 정의
# =========================
FEATURE_COLS = [
    "Age","Gender","DurationHours","QRScomplexes",
    "SupraventricularBeats","VentricularBeats","NoisePercentage",
    "AverageRate","MaximumRate","MinimumRate",
    "TachycardiaBeats","TachycardiaPercentage",
    "SV_Isolated","SV_Couplets","SV_Runs","SV_TotalBeats","SV_BigeminalCycles",
    "V_Isolated","V_Couplets","V_Runs","V_TotalBeats",
]
LABEL_COL = "Label"

# =========================
# 9) 모델 조합 정의
# =========================
# (enc_name, head_name) 조합을 전부 돌며 성능 비교
MODEL_CONFIGS = [
    ("basic", "simple"),
    ("basic", "abmil"),
    ("resnet", "simple"),
    ("resnet", "abmil"),
    ("inception", "simple"),
    ("inception", "abmil"),
]

# =========================
# 10) Sampling 조합 정의
# =========================
# SEGPERS: bag에 담을 segment 개수 K 후보들
SEGPERS = list(range(20, 181, 40))  # 20, 60, 100, 140, 180

# BASE_RADII: S-beat 주변 탐색 반경 후보
BASE_RADII = [0, 1, 2]

# FILLS: segment 부족 시 보완 전략 후보
FILLS = ["normal", "duplicate", "half", "slike"]

# SAMPLING_CONFIGS: (K, radius, method, fill) 모든 조합
SAMPLING_CONFIGS = []
for k in SEGPERS:
    for r in BASE_RADII:
        for f in FILLS:
            SAMPLING_CONFIGS.append((k, r, "base", f))

# 전체 실험 조합 수(=K×radius×fill)
print(f"Total configs: {len(SAMPLING_CONFIGS)}")

# =========================
# 11) 입력 ECG 리드 설정
# =========================
# 선택된 리드 순서/구성은 build_model 내부에서 입력 채널 수 결정에 사용될 수 있다.
SELECTED_LEADS = ["V5", "V1", "II"]

# =========================
# 12) 메타데이터 로딩 및 정규화
# =========================
# record별 metadata를 meta_lookup[basename] = normalized_vector로 저장한다.
# Dataset에서 fp의 basename으로 lookup하여 모델 입력에 제공한다.
df = pd.read_csv(CSV_PATH)

# feature와 label이 모두 존재하는 row만 사용
df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])

# Gender를 LabelEncoder로 수치화(문자열 카테고리 → 정수)
df["Gender"] = LabelEncoder().fit_transform(df["Gender"].astype(str))

meta_lookup = {}

# 전체 데이터 기준으로 feature z-score normalization을 위한 평균/표준편차 계산
mean_vec = df[FEATURE_COLS].mean()
std_vec  = df[FEATURE_COLS].std()

# 각 파일(row["File"])의 basename을 key로 하여 정규화된 meta 벡터를 저장
for _, row in df.iterrows():
    key = os.path.basename(row["File"]).replace(".h5", "")
    vec = ((row[FEATURE_COLS] - mean_vec) / std_vec).values.astype(np.float32)
    meta_lookup[key] = vec

# =========================
# 13) Summary CSV 경로 및 저장 정책
# =========================
PER_RUN_CSV      = os.path.join(RESULT_ROOT, "per_run_summary.csv")     # 모든 run을 1행씩 append
SEGSEED_AGG_CSV  = os.path.join(RESULT_ROOT, "segseed_agg_summary.csv") # split_seed 내 seg_seed 반복 집계
OVERALL_CSV      = os.path.join(RESULT_ROOT, "overall_summary.csv")     # split×segseed 전체 집계
SUMMARY_CSV      = os.path.join(RESULT_ROOT, "summary.csv")             # 위 3개를 합친 단일 CSV

# AGG_EVERY: run을 몇 번 돌 때마다 집계 파일을 재계산할지 결정
# - 너무 자주 하면 IO/계산 부담 증가
# - 너무 드물면 중간 결과 확인이 어려움
AGG_EVERY = 5

def _ensure_parent(path: str):
    """파일 저장 전, 부모 디렉토리 생성 보장"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _csv_exists(path: str) -> bool:
    """CSV가 존재하고 내용이 비어있지 않은지 확인"""
    return os.path.exists(path) and os.path.getsize(path) > 0

def append_per_run_row(row: dict):
    """
    per_run_summary.csv에 1행 append.
    - 파일이 없으면 header 포함해 새로 생성
    - 파일이 있으면 header 없이 append
    """
    _ensure_parent(PER_RUN_CSV)
    df_ = pd.DataFrame([row])
    write_header = not _csv_exists(PER_RUN_CSV)
    df_.to_csv(PER_RUN_CSV, mode="a", index=False, header=write_header, float_format="%.4f")

def recompute_aggregates_from_per_run():
    """
    per_run_summary.csv를 읽어서 다음을 전부 재계산 후 저장:
    - segseed_agg_summary.csv
    - overall_summary.csv
    - summary.csv(합본)

    집계 방식
    - 동일 그룹에 대해 mean/std/95% CI를 계산한다.
    - CI는 정규 근사로 1.96 * (std / sqrt(n)) 사용
    """
    if not _csv_exists(PER_RUN_CSV):
        return

    per_run_df = pd.read_csv(PER_RUN_CSV)

    # -------------------------
    # 13-1) segseed_agg (split_seed 고정, seg_seed 반복 집계)
    # -------------------------
    rows_segseed_agg = []
    group_cols_segseed = [
        "config_id","method","radius","seg_per_record","fill",
        "encoder","head",
        "upsample_factor","upsample_train","upsample_val",
        "split_seed"
    ]

    for keys, gdf in per_run_df.groupby(group_cols_segseed):
        (cfg, m, r, k, f, enc, head, upf, uptr, upva, split_sd) = keys
        n = len(gdf)

        # 특정 metric 컬럼에 대해 mean/std/CI를 계산하는 내부 함수
        def _ci(col):
            vals = gdf[col].values
            n_ = len(vals)

            # nanmean/nanstd 사용: 혹시 NaN이 섞여도 집계가 가능하도록 함
            m_ = float(np.nanmean(vals)) if n_ > 0 else float("nan")
            s_ = float(np.nanstd(vals, ddof=1)) if n_ > 1 else 0.0

            # 95% CI half-width
            h_ = 1.96 * (s_ / max(1, np.sqrt(n_))) if n_ > 0 else 0.0
            return m_, s_, (m_ - h_, m_ + h_)

        # validation best AUROC 기준 집계
        val_m, val_s, (val_lo, val_hi)     = _ci("val_best_auc")
        # test AUROC 기준 집계
        test_m, test_s, (test_lo, test_hi) = _ci("test_auc")

        rows_segseed_agg.append({
            "config_id": cfg, "method": m, "radius": r, "seg_per_record": k, "fill": f,
            "encoder": enc, "head": head,
            "upsample_factor": int(upf),
            "upsample_train": int(uptr),
            "upsample_val": int(upva),
            "split_seed": split_sd,
            "n_segseeds": n,
            "val_auc_mean": val_m,  "val_auc_std": val_s,  "val_auc_ci95_lo": val_lo,  "val_auc_ci95_hi": val_hi,
            "test_auc_mean": test_m, "test_auc_std": test_s, "test_auc_ci95_lo": test_lo, "test_auc_ci95_hi": test_hi,
        })

    segseed_agg_df = pd.DataFrame(rows_segseed_agg)
    if len(segseed_agg_df) > 0:
        segseed_agg_df = segseed_agg_df.sort_values(group_cols_segseed)
    segseed_agg_df.to_csv(SEGSEED_AGG_CSV, index=False, float_format="%.4f")

    # -------------------------
    # 13-2) overall (split_seed까지 합쳐 전체 run 집계)
    # -------------------------
    rows_overall = []
    group_cols_overall = [
        "config_id","method","radius","seg_per_record","fill",
        "encoder","head",
        "upsample_factor","upsample_train","upsample_val",
    ]

    for keys, gdf in per_run_df.groupby(group_cols_overall):
        (cfg, m, r, k, f, enc, head, upf, uptr, upva) = keys
        n = len(gdf)

        def _ci(col):
            vals = gdf[col].values
            n_ = len(vals)
            m_ = float(np.nanmean(vals)) if n_ > 0 else float("nan")
            s_ = float(np.nanstd(vals, ddof=1)) if n_ > 1 else 0.0
            h_ = 1.96 * (s_ / max(1, np.sqrt(n_))) if n_ > 0 else 0.0
            return m_, s_, (m_ - h_, m_ + h_)

        val_m, val_s, (val_lo, val_hi)     = _ci("val_best_auc")
        test_m, test_s, (test_lo, test_hi) = _ci("test_auc")

        rows_overall.append({
            "config_id": cfg, "method": m, "radius": r, "seg_per_record": k, "fill": f,
            "encoder": enc, "head": head,
            "upsample_factor": int(upf),
            "upsample_train": int(uptr),
            "upsample_val": int(upva),
            "n_runs": n,
            "val_auc_mean": val_m,  "val_auc_std": val_s,  "val_auc_ci95_lo": val_lo,  "val_auc_ci95_hi": val_hi,
            "test_auc_mean": test_m, "test_auc_std": test_s, "test_auc_ci95_lo": test_lo, "test_auc_ci95_hi": test_hi,
        })

    overall_df = pd.DataFrame(rows_overall)
    if len(overall_df) > 0:
        overall_df = overall_df.sort_values(group_cols_overall)
    overall_df.to_csv(OVERALL_CSV, index=False, float_format="%.4f")

    # -------------------------
    # 13-3) summary.csv (합본)
    # -------------------------
    # summary_type으로 각 row가 어떤 요약 단계인지 구분 가능하게 한다.
    per_run_df2   = per_run_df.copy()
    per_run_df2.insert(0, "summary_type", "per_run")

    segseed_agg2  = segseed_agg_df.copy()
    segseed_agg2.insert(0, "summary_type", "segseed_agg")

    overall_df2   = overall_df.copy()
    overall_df2.insert(0, "summary_type", "overall")

    summary_df    = pd.concat([segseed_agg2, overall_df2, per_run_df2], ignore_index=True)
    summary_df.to_csv(SUMMARY_CSV, index=False, float_format="%.4f")


# =========================
# 14) 캐시/파일 관련 유틸
# =========================
def _cache_tag(method: str, radius: int, k: int, fill: str, segseed: int, upsample_factor: int = 1) -> str:
    """
    캐시 파일명을 구분하기 위한 tag 문자열 생성.
    - segment_seed(SS)와 upsample_factor(UP)를 포함하여 다른 설정이 섞이지 않도록 한다.
    """
    if int(upsample_factor) > 1:
        return f"{method}_R{radius}_K{k}_F{fill}_SS{segseed}_UP{int(upsample_factor)}"
    return f"{method}_R{radius}_K{k}_F{fill}_SS{segseed}"

def _result_dir_for_config(k: int, radius: int, method: str, fill: str, upsample_factor: int) -> str:
    """
    결과 디렉토리를 실험 조합별로 분리.
    - upsample_factor에 따라 결과가 섞이지 않도록 폴더를 따로 만든다.
    """
    if int(upsample_factor) > 1:
        return os.path.join(RESULT_ROOT, f"{method}_R{radius}_K{k}_F{fill}_UP{int(upsample_factor)}")
    return os.path.join(RESULT_ROOT, f"{method}_R{radius}_K{k}_F{fill}_UP1")

def load_split_records(seed):
    """
    split_seed{seed:02d}.csv를 읽어서 train/val/test record 목록을 만든다.

    반환 형식
    - train_records: [(file_path, label), ...]
    - val_records  : [(file_path, label), ...]
    - test_records : [(file_path, label), ...]

    CSV에서 사용하는 컬럼
    - PID: 환자 ID (중복 제거에 사용)
    - file_path: HDF5 파일 경로
    - split: train/val/test
    - label: 0/1
    """
    path = os.path.join(SPLIT_DIR, f"split_seed{seed:02d}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    d = pd.read_csv(path, usecols=["PID", "file_path", "split", "label"]).drop_duplicates()
    d["label"] = d["label"].astype(int)

    mk = lambda part: [
        (fp, lb)
        for fp, lb in zip(d[d["split"]==part]["file_path"], d[d["split"]==part]["label"])
    ]
    return mk("train"), mk("val"), mk("test")

def cache_path(tag, split_seed, split):
    """
    캐시 파일 경로 생성.
    - tag에 segseed/upsample_factor가 들어있고,
    - 파일명에 split_seed 및 split(train/val/test)을 포함한다.
    """
    return os.path.join(CACHE_DIR, f"{tag}_split{split_seed:02d}_{split}.jsonl")

def load_fixed_cache(tag, split_seed, split, method, radius, k, fill, upsample_factor: int = 1):
    """
    캐시(jsonl)에서 fixed records를 로드한다.

    캐시의 핵심 목적
    - scan_record + select_keys 결과(=selected_keys)를 재사용하여
      같은 실험 설정에서 반복 실행 시 HDF5 scan 비용을 줄인다.

    로드 시 검증
    - 첫 줄 메타(method/radius/k/fill/upsample_factor)가 현재 설정과 맞는지 확인
    - selected_keys 길이가 K인지 확인
    """
    p = cache_path(tag, split_seed, split)

    # 캐시 파일이 없거나 FORCE_RESCAN이면 캐시 사용하지 않음
    if not os.path.exists(p) or FORCE_RESCAN:
        return None

    out = []
    with open(p, "r", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            return None

        obj0 = json.loads(first)

        # 메타 검증: 캐시가 현재 설정과 동일한지 확인
        m  = obj0.get("method")
        rd = obj0.get("radius")
        kk = obj0.get("seg_per_record")
        fl = obj0.get("fill")
        up = int(obj0.get("upsample_factor", 1))

        # 하나라도 다르면 캐시 무효로 처리(None 반환)하여 재생성하게 만든다.
        if not (m == method and rd == radius and kk == k and fl == fill and up == int(upsample_factor)):
            return None

        # 첫 줄 포함 전체 로드
        out.append((obj0["file_path"], int(obj0["label"]), obj0["selected_keys"]))
        for ln in f:
            o = json.loads(ln)
            out.append((o["file_path"], int(o["label"]), o["selected_keys"]))

    # 길이 검증: bag 크기는 항상 K여야 한다.
    # (select_keys에서 fill로 맞추더라도, 결과가 K가 아니면 캐시 무효)
    if any(len(keys) != k for _, _, keys in out):
        return None

    return out

def save_fixed_cache(tag, split_seed, split, records_fixed, method, radius, k, fill, upsample_factor: int = 1):
    """
    fixed records를 JSONL 형식으로 저장한다.

    각 줄의 구조(사실상 record 단위 bag 정의)
    - file_path
    - label
    - selected_keys: bag에 포함될 segment key 목록
    - method/radius/seg_per_record/fill/upsample_factor 등 설정 메타
    - created_at: 캐시 생성 시각
    - seed: split seed 기록
    - split: train/val/test
    """
    p = cache_path(tag, split_seed, split)
    with open(p, "w", encoding="utf-8") as f:
        for fp, lb, keys in records_fixed:
            f.write(json.dumps({
                "file_path": fp,
                "label": int(lb),
                "selected_keys": list(keys),
                "method": method,
                "radius": radius,
                "seg_per_record": k,
                "fill": fill,
                "upsample_factor": int(upsample_factor),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "seed": split_seed,
                "split": split,
            }, ensure_ascii=False) + "\n")


# =========================
# 15) 결정론 보조 유틸(해시 기반 시드, key 파서)
# =========================
def _stable_seed(*items) -> int:
    """
    입력 요소들을 문자열로 결합한 뒤 SHA256 해시로 정수 시드를 만든다.
    같은 items 조합은 항상 같은 시드를 반환한다.
    """
    s = "||".join(map(str, items))
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)

def _parse_idx_local(k: str) -> int:
    """
    segment key가 'seg_000123' 형태라고 가정하고 인덱스만 정수로 추출.
    - HDF5 group key를 숫자 기준으로 정렬하기 위한 보조 함수
    """
    return int(k.split("_")[-1])


# =========================
# 16) Record 스캐너 (Ray remote)
# =========================
@ray.remote
def scan_record(fp, label, seg_per_record, method, radius, fill, segment_seed, upsample_index=0):
    """
    하나의 record(HDF5 파일)에서 bag을 만들기 위한 segment key들을 선택한다.

    반환
    - (fp, label, selected_keys)

    결정론 설계
    - seed = stable_hash(SEED, segment_seed, 파일명, method, radius, K, fill, upsample_index)
    - 같은 설정/seed면 항상 같은 rng 시퀀스를 사용하게 된다.

    segment key 정렬
    - HDF5 내부 key 순서는 저장/환경에 따라 바뀔 수 있으므로,
      'seg_000123'의 숫자 인덱스를 기준으로 정렬하여 순서 의존성을 제거한다.

    S-beat 필터
    - 각 segment의 annotation/Symbol에 'S'(byte string)가 2개 미만이면 record 자체를 사용하지 않는다(None).
    """
    base = os.path.basename(fp)

    # file + 설정 + upsample_index에 의해 고정되는 시드
    seed = _stable_seed(SEED, segment_seed, base, method, radius, seg_per_record, fill, int(upsample_index))
    rng = random.Random(seed)

    with h5py.File(fp, "r") as f:
        # 예상 HDF5 구조: root에 "segments" 그룹이 있어야 한다.
        if "segments" not in f:
            return None

        raw_keys = list(f["segments"].keys())
        if not raw_keys:
            return None

        # 숫자 인덱스 기준 정렬을 위해 idx_map 생성
        idx_map = {_parse_idx_local(k): k for k in raw_keys}
        all_keys = [idx_map[i] for i in sorted(idx_map)]

        # S 심볼이 포함된 segment key만 모은다.
        s_keys_all = []
        for k in all_keys:
            grp = f["segments"][k]
            if "annotation" in grp and "Symbol" in grp["annotation"]:
                syms = grp["annotation"]["Symbol"][...]
                # HDF5에 저장된 Symbol이 bytes로 들어있다는 전제 하에서 b"S"로 비교
                if any(s == b"S" for s in syms):
                    s_keys_all.append(k)

        # S segment가 너무 적으면 이 record는 실험에서 제외
        if len(s_keys_all) < 2:
            return None

        # 실제 K개 선택 로직은 v2_sampling.select_keys에서 담당
        selected = select_keys(
            all_keys=all_keys,
            s_keys_all=s_keys_all,
            method=method,
            radius=radius,
            seg_per_record=seg_per_record,
            fill=fill,
            rng=rng,
            file_basename=os.path.splitext(os.path.basename(fp))[0],
            slike_dir=SLIKE_DIR,
        )

    return (fp, label, selected)


# =========================
# 17) fixed records 생성 (Ray 병렬 스캔)
# =========================
def build_fixed_records_with_ray(records, method, radius, seg_per_record, fill, segment_seed, upsample_factor: int = 1):
    """
    입력 records[(fp,label),...]에 대해 scan_record를 Ray로 병렬 실행하여
    fixed_records[(fp,label,selected_keys),...]를 만든다.

    upsample_factor 동작
    - upsample_factor == 1:
        * 각 record 당 scan_record 1회
    - upsample_factor > 1:
        * label==1(positive) record만 upsample_factor회 반복 스캔
        * upsample_index가 0..upf-1로 바뀌며 seed에 반영 → 서로 다른 bag 생성 가능
        * label==0(negative)는 1회만 스캔 (데이터 불균형을 완화)

    반환 값 정렬
    - Ray 완료 순서는 비결정적이므로,
      결과를 (basename, selected_keys 문자열) 기준으로 정렬해 저장 순서를 고정한다.
    """
    tasks = []
    upf = int(upsample_factor)

    for fp, lb in records:
        lb_i = int(lb)

        # positive record만 upf번 생성하여 확장
        if lb_i == 1 and upf > 1:
            for up_i in range(upf):
                tasks.append(scan_record.remote(fp, lb_i, seg_per_record, method, radius, fill, segment_seed, up_i))
        else:
            tasks.append(scan_record.remote(fp, lb_i, seg_per_record, method, radius, fill, segment_seed, 0))

    results = []

    # tqdm 진행 표시용 문자열
    desc = f"[Scan→Cache][{method} R={radius} K={seg_per_record} F={fill} SS{segment_seed}"
    if upf > 1:
        desc += f" UP={upf}"
    desc += "]"

    # Ray task를 하나씩 수거하면서 progress bar 업데이트
    with tqdm(total=len(tasks), desc=desc) as pbar:
        while tasks:
            done, tasks = ray.wait(tasks, num_returns=1)
            res = ray.get(done[0])
            if res:
                results.append(res)
            pbar.update(1)

    # 결과를 결정론적으로 정렬:
    # - 같은 fp가 upsampling 때문에 여러 번 들어올 수 있음
    # - selected_keys 조합까지 포함해 정렬 키를 구성하여 순서를 고정
    results.sort(key=lambda x: (os.path.basename(x[0]), "|".join(map(str, x[2]))))
    return results


# =========================
# 18) Dataset: fixed_records 기반 MIL bag 로딩
# =========================
class MILBagDatasetFixed(Dataset):
    """
    records_fixed: [(fp, label, selected_keys), ...]
    - 각 item은 하나의 bag(=K개의 segment) + metadata + label로 구성된다.
    """
    def __init__(self, records_fixed):
        self.records = records_fixed

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        """
        반환:
        - bag_tensor: (K, C, T)
        - meta_vec  : (len(FEATURE_COLS),)
        - label     : scalar float32
        """
        fp, label, keys = self.records[idx]

        bag = []
        with h5py.File(fp, "r") as f:
            for k in keys:
                # HDF5 저장 신호 shape가 (T, C)라고 가정하고 permute(1,0) → (C,T)
                x = torch.tensor(f["segments"][k]["signal"][:], dtype=torch.float32).permute(1, 0)

                # 길이 맞추기
                # - 짧으면 오른쪽 pad
                # - 길면 앞부분을 사용하도록 crop
                x = (F.pad(x, (0, SEGMENT_LENGTH - x.shape[1]))
                     if x.shape[1] < SEGMENT_LENGTH else x[:, :SEGMENT_LENGTH])

                bag.append(x)

        bag_tensor = torch.stack(bag)  # (K, C, T)

        # metadata 벡터 lookup
        basename = os.path.basename(fp).replace(".h5", "")
        meta_vec = meta_lookup.get(basename, np.zeros(len(FEATURE_COLS), dtype=np.float32))

        return bag_tensor, torch.tensor(meta_vec), torch.tensor(label, dtype=torch.float32)


# =========================
# 19) Dynamic pos_weight 계산 유틸
# =========================
def compute_pos_weight_from_fixed_train(fixed_train):
    """
    fixed_train을 기준으로 class imbalance를 계산하여 pos_weight를 반환한다.

    입력
    - fixed_train: [(fp, label, selected_keys), ...]

    계산
    - pos = sum(labels)
    - neg = len(labels) - pos
    - pos_weight = neg / pos

    예외 처리
    - pos 또는 neg가 0이면 정상 계산이 불가능 → POS_WEIGHT_FALLBACK 사용
    - pos_weight는 POS_WEIGHT_CAP으로 상한을 둔다.

    반환
    - pos_weight(float), pos_cnt(int), neg_cnt(int), pos_ratio(float)
    """
    labels = [int(lb) for _, lb, _ in fixed_train]
    pos = int(sum(labels))
    neg = int(len(labels) - pos)

    if pos <= 0 or neg <= 0:
        return float(POS_WEIGHT_FALLBACK), pos, neg, 0.0

    base = neg / max(1, pos)
    pw = min(float(base), float(POS_WEIGHT_CAP))
    ratio = pos / max(1, len(labels))
    return pw, pos, neg, ratio


# =========================
# 20) Train / Eval
# =========================
def train(model, tr_loader, va_loader, save_dir, pos_weight: float = 10.0):
    """
    모델 학습 루프.

    입력
    - model: build_model로 생성한 Multimodal MIL 모델
    - tr_loader: 학습 DataLoader
    - va_loader: 검증 DataLoader
    - save_dir: best 모델 저장 디렉토리
    - pos_weight: BCEWithLogitsLoss의 pos_weight 값

    학습/검증
    - 각 epoch마다 train step 수행
    - 검증은 AUROC로 성능 평가
    - best_auc가 개선되면 best.pt 저장
    - EARLY_PATIENCE 동안 개선이 없으면 early stop

    LR scheduler
    - USE_PLATEAU_LR=True면 ReduceLROnPlateau 사용
    - metric은 validation AUROC
    """
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # BCEWithLogitsLoss는 logits 입력을 기대하며,
    # pos_weight는 positive class에 대한 가중치를 의미한다.
    pos_weight_t = torch.tensor([float(pos_weight)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    scheduler = None
    if USE_PLATEAU_LR:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=5, threshold=1e-4, threshold_mode="abs"
        )

    best_auc = 0.0
    no_improve = 0

    for ep in range(1, EPOCHS + 1):
        # ---------------------
        # Train
        # ---------------------
        model.train()
        for b, m, y in tqdm(tr_loader, desc=f"[Epoch {ep}]"):
            # b: (B, K, C, T) 형태를 기대(데이터셋이 (K,C,T)를 반환하므로 batching 후 (B,K,C,T))
            # m: (B, meta_dim_input)
            # y: (B,)
            b, m, y = b.to(DEVICE), m.to(DEVICE), y.to(DEVICE)

            # model(b, m) → (logits, attention_or_aux)
            out, _ = model(b, m)

            # BCEWithLogitsLoss: out과 y shape이 맞아야 함
            loss = criterion(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        # ---------------------
        # Validation (AUROC)
        # ---------------------
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for b, m, y in va_loader:
                b, m = b.to(DEVICE), m.to(DEVICE)
                out, _ = model(b, m)

                # logits → sigmoid 확률
                probs = torch.sigmoid(out)

                # sklearn metric을 위해 python list로 누적
                all_probs += probs.cpu().tolist()
                all_labels += y.tolist()

        auc = roc_auc_score(all_labels, all_probs)
        print(f"Epoch {ep} Validation AUROC: {auc:.4f}")

        # plateau scheduler가 있으면 AUROC 기반으로 step
        if scheduler is not None:
            scheduler.step(auc)

        # best 업데이트 조건: MIN_DELTA 이상 개선
        if auc > best_auc + MIN_DELTA:
            best_auc = auc
            no_improve = 0

            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
        else:
            no_improve += 1
            if no_improve >= EARLY_PATIENCE:
                print(f"[EarlyStopping] No improvement for {EARLY_PATIENCE} epochs. Stop at epoch {ep}.")
                break

    return best_auc

def evaluate(model, loader, out_dir):
    """
    Test/Validation 평가 함수 (여기서는 test에 사용).

    출력 metric
    - AUROC
    - AUPRC
    - Accuracy, Precision, Recall, F1
    - Confusion matrix 및 classification report를 eval.txt로 저장

    임계값
    - p >= 0.5 → positive로 이진화
    """
    model.to(DEVICE).eval()

    preds, gts = [], []
    with torch.no_grad():
        for b, m, y in tqdm(loader, desc="[Test]"):
            b, m = b.to(DEVICE), m.to(DEVICE)
            out, _ = model(b, m)
            prob = torch.sigmoid(out)
            preds += prob.cpu().tolist()
            gts += y.tolist()

    bin_preds = [1 if p >= 0.5 else 0 for p in preds]

    auc  = roc_auc_score(gts, preds)
    ap   = average_precision_score(gts, preds)
    acc  = accuracy_score(gts, bin_preds)
    f1   = f1_score(gts, bin_preds)
    prec = precision_score(gts, bin_preds, zero_division=0)
    rec  = recall_score(gts, bin_preds, zero_division=0)

    cm = confusion_matrix(gts, bin_preds)
    cls_report = classification_report(gts, bin_preds, digits=4)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval.txt"), "w") as f:
        f.write(f"Test AUROC : {auc:.4f}\n")
        f.write(f"Test AUPRC : {ap:.4f}\n")
        f.write(f"Accuracy   : {acc:.4f}\n")
        f.write(f"Precision  : {prec:.4f}\n")
        f.write(f"Recall     : {rec:.4f}\n")
        f.write(f"F1-score   : {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(cls_report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")

    return {"auc": auc, "ap": ap, "acc": acc, "f1": f1, "precision": prec, "recall": rec}


# =========================
# 21) DataLoader 결정론 유틸
# =========================
def _worker_init_fn(worker_id):
    """
    DataLoader worker별 시드 설정.
    - worker_id를 더해 각 worker가 서로 다른 seed를 갖게 하되,
      전체적으로는 재현 가능한 패턴이 되도록 한다.
    """
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def _make_generator(seed_val: int) -> torch.Generator:
    """
    DataLoader shuffle의 결정론성을 위해 generator를 명시적으로 생성/시드 부여.
    - train_loader에서 shuffle=True일 때 특히 중요
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed_val)
    return g


# =========================
# 22) main: 전체 실험 루프
# =========================
def main():
    """
    전체 실험의 중첩 루프 구조

    루프 순서
    1) sampling config: (K, R, METHOD, FILL)
    2) upsample_factor
    3) segseed
    4) split_seed
    5) model config: (encoder, head)

    각 조합에 대해
    - fixed_train/val/test를 캐시 로드하거나 Ray로 생성
    - (선택) fixed_train 기준 pos_weight 계산
    - 모델 학습(train) → best.pt 저장
    - best.pt 로드 후 test 평가(evaluate)
    - per_run CSV에 결과 1행 기록
    - 주기적으로 집계 CSV 재계산
    """
    run_counter = 0  # per_run_row를 몇 번 쌓았는지 카운트하여 AGG_EVERY마다 집계

    for cfg_id, (K, R, METHOD, FILL) in enumerate(SAMPLING_CONFIGS):
        for upsample_factor in UPSAMPLE_FACTORS:
            for segseed in SEGMENT_SEEDS:
                # 실험 조합별 결과 폴더(upsample_factor별로 분리)
                result_dir_root = _result_dir_for_config(K, R, METHOD, FILL, upsample_factor)
                os.makedirs(result_dir_root, exist_ok=True)

                print(
                    f"\n===== Config {cfg_id:02d}: method={METHOD}, radius={R}, K={K}, fill={FILL}, "
                    f"segseed={segseed}, upsample_factor={upsample_factor} "
                    f"(train={APPLY_UPSAMPLE_TRAIN}, val={APPLY_UPSAMPLE_VAL}) ====="
                )

                for split_seed in SPLIT_SEEDS:
                    print(f"\n--- split seed {split_seed:02d} ---")

                    # split CSV에서 train/val/test record 리스트 로드
                    train_records, val_records, test_records = load_split_records(split_seed)

                    # split별 upsampling 적용 여부 반영
                    up_train = int(upsample_factor) if APPLY_UPSAMPLE_TRAIN else 1
                    up_val   = int(upsample_factor) if APPLY_UPSAMPLE_VAL   else 1
                    up_test  = 1  # test는 항상 고정

                    # split별 tag 생성
                    # - train/val/test에 서로 다른 upsample_factor가 적용될 수 있으므로 각각 별도 tag
                    tag_train = _cache_tag(METHOD, R, K, FILL, segseed, up_train)
                    tag_val   = _cache_tag(METHOD, R, K, FILL, segseed, up_val)
                    tag_test  = _cache_tag(METHOD, R, K, FILL, segseed, up_test)

                    # 캐시에서 fixed records 로드 시도
                    fixed_train = load_fixed_cache(tag_train, split_seed, "train", METHOD, R, K, FILL, up_train)
                    fixed_val   = load_fixed_cache(tag_val,   split_seed, "val",   METHOD, R, K, FILL, up_val)
                    fixed_test  = load_fixed_cache(tag_test,  split_seed, "test",  METHOD, R, K, FILL, up_test)

                    # 캐시가 없거나 무효면 Ray로 생성 후 저장
                    if fixed_train is None:
                        fixed_train = build_fixed_records_with_ray(train_records, METHOD, R, K, FILL, segseed, up_train)
                        save_fixed_cache(tag_train, split_seed, "train", fixed_train, METHOD, R, K, FILL, up_train)

                    if fixed_val is None:
                        fixed_val = build_fixed_records_with_ray(val_records, METHOD, R, K, FILL, segseed, up_val)
                        save_fixed_cache(tag_val, split_seed, "val", fixed_val, METHOD, R, K, FILL, up_val)

                    if fixed_test is None:
                        fixed_test = build_fixed_records_with_ray(test_records, METHOD, R, K, FILL, segseed, up_test)
                        save_fixed_cache(tag_test, split_seed, "test", fixed_test, METHOD, R, K, FILL, up_test)

                    # 학습에 쓰는 fixed_train 기반으로 pos_weight 계산
                    if USE_DYNAMIC_POS_WEIGHT:
                        pos_weight_val, pos_cnt, neg_cnt, pos_ratio = compute_pos_weight_from_fixed_train(fixed_train)
                    else:
                        # dynamic을 끄면 pos_weight는 고정값(여기서는 10.0), 카운트는 -1로 표시
                        pos_weight_val, pos_cnt, neg_cnt, pos_ratio = 10.0, -1, -1, float("nan")

                    # 학습 데이터 분포/pos_weight 로깅
                    print(
                        f"[INFO] fixed_train size={len(fixed_train)} | pos={pos_cnt} neg={neg_cnt} "
                        f"| pos_ratio={pos_ratio if pos_ratio==pos_ratio else float('nan'):.4f} "
                        f"| pos_weight={pos_weight_val:.4f} (dynamic={int(USE_DYNAMIC_POS_WEIGHT)})"
                    )

                    # 모델 조합 순회
                    for enc_name, head_name in MODEL_CONFIGS:
                        # 실험 결과 디렉토리 구조
                        # - config(샘플링)+upsample_factor 폴더 아래
                        # - encoder+head
                        # - seed(split_seed)
                        # - segseed
                        exp_dir = os.path.join(
                            result_dir_root,
                            f"{enc_name}+{head_name}",
                            f"seed{split_seed:02d}",
                            f"segseed{segseed:02d}",
                        )
                        os.makedirs(exp_dir, exist_ok=True)

                        # -------------------------
                        # DataLoader 결정론 시드 생성
                        # -------------------------
                        # train_loader shuffle이 매번 같은 순서를 갖도록 generator seed를 고정한다.
                        # seed에는 cfg_id/split_seed/segseed/upsample 설정과 모델 이름을 포함해
                        # 서로 다른 실험 조합에서는 다른 shuffle이 되도록 만든다.
                        loader_seed_train = _stable_seed(
                            "train_loader", SEED, cfg_id, split_seed, segseed,
                            int(upsample_factor), int(APPLY_UPSAMPLE_TRAIN), int(APPLY_UPSAMPLE_VAL),
                            enc_name, head_name
                        )

                        # -------------------------
                        # DataLoader 생성
                        # -------------------------
                        train_loader = DataLoader(
                            MILBagDatasetFixed(fixed_train),
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=8,
                            generator=_make_generator(loader_seed_train),
                            worker_init_fn=_worker_init_fn,
                        )

                        # val/test는 shuffle=False (평가 안정성)
                        val_loader   = DataLoader(
                            MILBagDatasetFixed(fixed_val),
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=8,
                            worker_init_fn=_worker_init_fn,
                        )
                        test_loader  = DataLoader(
                            MILBagDatasetFixed(fixed_test),
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=8,
                            worker_init_fn=_worker_init_fn,
                        )

                        # -------------------------
                        # 모델 초기화 전 시드 재고정
                        # -------------------------
                        # 모델 파라미터 초기화 등 학습 관련 난수를 고정하기 위한 재시딩
                        random.seed(SEED)
                        np.random.seed(SEED)
                        torch.manual_seed(SEED)

                        # -------------------------
                        # 모델 생성
                        # -------------------------
                        model = build_model(
                            selected_leads=SELECTED_LEADS,
                            enc_name=enc_name,
                            head_name=head_name,
                            hidden_dim=HIDDEN_DIM,
                            meta_dim=META_DIM,
                            dropout=DROPOUT,
                            feature_cols=FEATURE_COLS,
                        )

                        # -------------------------
                        # 학습 및 best 모델 저장
                        # -------------------------
                        val_best_auc = train(
                            model,
                            train_loader,
                            val_loader,
                            save_dir=exp_dir,
                            pos_weight=pos_weight_val,
                        )

                        # -------------------------
                        # best 모델 로드 후 test 평가
                        # -------------------------
                        model.load_state_dict(torch.load(os.path.join(exp_dir, "best.pt"), map_location=DEVICE))
                        test_metrics = evaluate(model, test_loader, out_dir=exp_dir)

                        # -------------------------
                        # per_run_summary.csv에 기록할 row 구성
                        # -------------------------
                        row = {
                            "config_id": cfg_id,
                            "method": METHOD,
                            "radius": R,
                            "seg_per_record": K,
                            "fill": FILL,
                            "upsample_factor": int(upsample_factor),
                            "upsample_train": int(APPLY_UPSAMPLE_TRAIN),
                            "upsample_val": int(APPLY_UPSAMPLE_VAL),
                            "split_seed": split_seed,
                            "seg_seed": segseed,
                            "encoder": enc_name,
                            "head": head_name,
                            "val_best_auc": val_best_auc,
                            "test_auc": test_metrics["auc"],
                            "test_ap": test_metrics["ap"],
                            "test_acc": test_metrics["acc"],
                            "test_f1": test_metrics["f1"],
                            "test_precision": test_metrics["precision"],
                            "test_recall": test_metrics["recall"],

                            # 실제 적용된 split별 effective upsample 값 기록
                            "up_train_effective": int(up_train),
                            "up_val_effective": int(up_val),
                            "up_test_effective": int(up_test),

                            # fixed record 개수(upsampling 적용 결과 포함)
                            "n_train_records_fixed": int(len(fixed_train)),
                            "n_val_records_fixed": int(len(fixed_val)),
                            "n_test_records_fixed": int(len(fixed_test)),

                            # pos_weight 관련 기록 (재현 및 분석 용이)
                            "pos_weight": float(pos_weight_val),
                            "train_pos_count": int(pos_cnt) if pos_cnt >= 0 else -1,
                            "train_neg_count": int(neg_cnt) if neg_cnt >= 0 else -1,
                            "train_pos_ratio": float(pos_ratio) if pos_ratio == pos_ratio else float("nan"),
                            "pos_weight_cap": float(POS_WEIGHT_CAP),
                            "pos_weight_fallback": float(POS_WEIGHT_FALLBACK),
                            "use_dynamic_pos_weight": int(USE_DYNAMIC_POS_WEIGHT),
                        }

                        # -------------------------
                        # CSV 기록 및 주기적 집계
                        # -------------------------
                        append_per_run_row(row)

                        run_counter += 1
                        if run_counter % AGG_EVERY == 0:
                            recompute_aggregates_from_per_run()

                        # -------------------------
                        # 콘솔 로그 출력
                        # -------------------------
                        print(
                            f"[cfg{cfg_id:02d}] [{METHOD} R{R} K{K} F{FILL}] "
                            f"UP={int(upsample_factor)} (train={int(APPLY_UPSAMPLE_TRAIN)}, val={int(APPLY_UPSAMPLE_VAL)}) "
                            f"split{split_seed:02d} segseed{segseed:02d} {enc_name}+{head_name} → "
                            f"val_auc={val_best_auc:.4f}, test_auc={test_metrics['auc']:.4f}, pos_w={pos_weight_val:.2f}"
                        )

                        # -------------------------
                        # 메모리 정리
                        # -------------------------
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    # 전체 실험 종료 후 마지막으로 집계 파일 재계산
    recompute_aggregates_from_per_run()

    # 저장된 파일 경로 안내
    print("\nSaved summaries:")
    print(" -", PER_RUN_CSV)
    print(" -", SEGSEED_AGG_CSV)
    print(" -", OVERALL_CSV)
    print(" -", SUMMARY_CSV)


# =========================
# 23) 엔트리포인트
# =========================
if __name__ == "__main__":
    main()
