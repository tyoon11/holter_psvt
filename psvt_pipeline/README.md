
---

# PSVT Multimodal MIL Pipeline (Holter HDF5 기반)

이 저장소는 **24시간 Holter ECG를 10초 단위 세그먼트로 분할한 표준 HDF5 데이터셋**을 사용하여
**PSVT 여부를 이진 분류**하는 **Multimodal MIL (ECG bag + metadata)** 학습/평가 파이프라인이다.

본 파이프라인은 다음을 특징으로 한다.

* **record 단위 MIL** (segment bag)
* **S-beat 중심 segment sampling**
* **Ray 기반 병렬 스캔 + JSONL 캐시**
* **positive record upsampling**
* **dynamic pos_weight 기반 불균형 대응**
* **완전 결정론적 재현성 설계**

---

## 1. 데이터셋 요구사항

### 1.1 HDF5 구조 (필수)

각 HDF5 파일은 다음 구조를 **반드시 만족**해야 한다.

```
<root>
 └─ ECG
     └─ segments
         └─ seg_000000
             ├─ signal
             │   ├─ II
             │   ├─ V1
             │   └─ V5
             └─ beat_annotation
                 └─ symbol
         └─ seg_000001
         └─ ...
```

#### 핵심 규칙

* `ECG/segments/seg_xxxxxx` 형태의 segment key
* `signal`은 **lead별 dataset group**
* `beat_annotation/symbol`은

  * 문자열 `"S"` 또는
  * byte string `b"S"`
    를 포함해야 **supraventricular segment**로 인식됨
* 한 record에서 `"S"` segment가 **2개 미만이면 해당 record는 제외됨**

---

### 1.2 Metadata CSV (`h5_metadata_summary.csv`)

`CSV_PATH`로 지정된 CSV는 다음 컬럼을 **반드시 포함**해야 한다.

#### 필수 컬럼

* `File` : HDF5 파일 경로
* `Label` : 0 또는 1 (PSVT 여부)

#### Feature 컬럼

```text
Age, Gender, DurationHours, QRScomplexes,
SupraventricularBeats, VentricularBeats, NoisePercentage,
AverageRate, MaximumRate, MinimumRate,
TachycardiaBeats, TachycardiaPercentage,
SV_Isolated, SV_Couplets, SV_Runs, SV_TotalBeats, SV_BigeminalCycles,
V_Isolated, V_Couplets, V_Runs, V_TotalBeats
```

* `Gender`는 문자열이어도 무관 (내부에서 LabelEncoder 적용)
* feature들은 **전체 데이터 기준 z-score 정규화**됨

---

### 1.3 Split CSV (`seed_splits/*.csv`)

각 split 파일은 다음 컬럼을 포함해야 한다.

```text
PID, file_path, split, label
```

* `split` ∈ `{train, val, test}`
* **patient-wise split**이 이미 반영된 상태여야 함
* `PID` 기준 중복 제거 후 사용됨

---

## 2. 주요 개념 설명

### 2.1 MIL Bag 구성

* 하나의 HDF5 파일 = **하나의 record**
* record에서 **K개의 segment**를 선택 → 하나의 bag
* bag shape: `(K, C, T)`

  * `C`: 선택된 ECG lead 수
  * `T`: `SEGMENT_LENGTH` (기본 1250)

---

### 2.2 Segment Sampling (`scan_record`)

* sampling은 **record 단위**로 수행
* 핵심 흐름:

  1. `ECG/segments`의 모든 key를 숫자 기준 정렬
  2. `"S"` symbol 포함 segment만 `s_keys_all`로 추출
  3. `select_keys()` 호출하여 K개 선택

#### 결정론 보장

sampling RNG seed는 다음 값으로 고정된다.

```text
(SEED, segment_seed, file_name, method, radius, K, fill, upsample_index)
```

→ 같은 설정이면 **항상 동일한 segment 조합**

---

### 2.3 Upsampling 방식

* **positive record (`label == 1`)만** upsampling
* `UPSAMPLE_FACTORS = [1, 5, 10]`이면:

  * 동일 record에서 서로 다른 bag을 최대 10개 생성
* `upsample_index`가 seed에 포함되어 bag 다양성 보장

#### split별 제어

```python
APPLY_UPSAMPLE_TRAIN = True
APPLY_UPSAMPLE_VAL   = True
# test는 항상 False
```

---

### 2.4 캐시 구조

* scan 결과는 JSONL로 저장
* 캐시 분리 기준:

```text
(method, radius, K, fill, segseed, upsample_factor, split_seed, split)
```

* 캐시 파일명 예시:

```
base_R1_K60_Fduplicate_SS42_UP5_split98_train.jsonl
```

* `FORCE_RESCAN=True`면 캐시 무시하고 재생성

---

### 2.5 Dynamic pos_weight

* 학습용 `fixed_train` 기준으로 계산

```text
pos_weight = (# negative) / (# positive)
```

* 상한/예외 처리:

```python
POS_WEIGHT_CAP = 20.0
POS_WEIGHT_FALLBACK = 15.0
```

* BCEWithLogitsLoss에 그대로 적용

---

## 3. 실행 방법

### 3.1 환경 준비

```bash
conda create -n psvt python=3.9
conda activate psvt

pip install torch ray h5py numpy pandas scikit-learn tqdm
```

(GPU 사용 시 CUDA 버전에 맞는 PyTorch 설치 필요)

---

### 3.2 경로 설정 확인

`pipeline.py` 상단에서 반드시 확인:

```python
CSV_PATH
BASE_DIR
SPLIT_DIR
CACHE_DIR
RESULT_ROOT
```

---

### 3.3 실행

```bash
python pipeline.py
```

실행 시 다음이 자동 수행된다.

1. split CSV 로드
2. Ray 기반 segment scan
3. fixed bag 캐시 생성/재사용
4. 모델 학습 + early stopping
5. test 평가
6. CSV 결과 누적 및 집계

---

## 4. 결과물 구조

### 4.1 결과 디렉토리

```
RESULT_ROOT/
 └─ base_R1_K60_Fduplicate_UP5/
     └─ resnet+abmil/
         └─ seed98/
             └─ segseed42/
                 ├─ best.pt
                 └─ eval.txt
```

---

### 4.2 CSV 결과

| 파일                        | 설명                           |
| ------------------------- | ---------------------------- |
| `per_run_summary.csv`     | 모든 실험 run 결과 (1 row = 1 run) |
| `segseed_agg_summary.csv` | segseed 반복 평균/CI             |
| `overall_summary.csv`     | split × segseed 전체 평균        |
| `summary.csv`             | 위 3개를 합친 통합 파일               |


