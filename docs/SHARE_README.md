# SNUH 24시간 Holter 데이터셋 (PSVT / TOF / LongQT)

서울대병원 24시간 Holter ECG 원본, 이를 변환한 HDF5(v2), 라벨, 변환 코드를 한 곳에 모은 패키지입니다.

> ⚠️ **환자 식별정보가 들어 있습니다.** 파일명 끝 토큰(`PID`)은 병원 등록번호이고,
> `.json` 리포트와 h5 attr 에 나이·성별·검사일시가 있습니다. IRB 승인 범위 안에서만 쓰고,
> 외부로 다시 옮기지 마세요.

---

## 1. 폴더 구성

위치: `/home/coder/workspace/Holter_TOF/holter_total` (`tools/make_share.py` 로 생성,
원래 데이터는 각자 자리에 그대로 있음)

```
holter_total/
├─ README.md                    ← 이 문서
├─ raw/                         원본 (PID 정정본)
│  ├─ nas1_Holter_PSVT/
│  ├─ nas1_Holter_TOF_250917/
│  └─ nas1_Holter_LQT_260615/   (하위 addition_260729/ 포함)
├─ h5/                          변환된 HDF5 (schema v2), record 1개 = 파일 1개
│  ├─ <record_name>.h5
│  └─ conversion_log.csv        record 별 변환 성공/실패 기록
├─ labels/
│  ├─ splits.csv                ★ 학습에 바로 쓰는 표 (manifest + split + 라벨)
│  ├─ splits_summary.txt        split × 과제별 양성/음성 수, 교란 점검
│  ├─ manifest.csv / .parquet   h5 전체의 메타 1행/record (split·라벨 제외)
│  ├─ duplicates.csv            신호가 같은 중복 record 묶음과 keep 여부
│  ├─ clinical_data_psvt.csv    PSVT 임상 라벨 원본 (Label 1/0)
│  ├─ clinical_data_tof.csv     TOF 임상 정보 원본
│  └─ psvt_labeling.csv         PSVT 하위유형(AVNRT/AVRT) 라벨 — 아직 학습에 안 씀
└─ code/holter_psvt/            변환·전처리·학습 코드 (git 저장소 스냅샷)
```

처음 쓴다면 **`labels/splits.csv` + `h5/`** 두 가지만 있으면 됩니다. `raw/` 는 재변환하거나
h5 에 없는 정보를 확인할 때만 필요합니다.

---

## 2. 원본 (`raw/`)

record 하나는 같은 이름(stem)의 파일 4개로 이루어집니다.

| 확장자 | 내용 | 필수 |
|---|---|---|
| `.hea` | WFDB 헤더 (fs=125 Hz, 3채널, 시작 일시) | ✅ |
| `.SIG` | 신호 (MARS export, WFDB 표준 `.dat` 아님) | ✅ |
| `.ANN` | beat 주석 (WFDB annotation, 심볼 N/S/V/…) | 없으면 `has_beats=False` |
| `.json` | 벤더 Holter 리포트 (환자정보, HR, beat 통계) | 없으면 `has_report=False` |

- 파일명 규칙: `<디스크·날짜 등>_<번호>_<PID>` (예: `10_50_2247355`). **끝 토큰이 PID** 입니다.
- 확장자 대소문자를 그대로 두세요 (`.SIG`, `.ANN`). 리눅스에서 wfdb 가 구분합니다.
- **NAS 원본이 아니라 `fix_pid.py` 를 적용한 복사본입니다.** 원본 `.hea`/`.json` 은 안에 적힌
  record 이름·PID 가 파일명과 다른 경우가 있어서, 파일명 기준으로 고쳐 두었습니다.
- **채널 순서는 `V5, V1, II`** 입니다 (`.hea` 채널 설명은 전부 `"MARS export"` 라 이름이 없음.
  전 코호트에서 확인, 2026-09-17). h5 에서는 `II, V1, V5` 로 재배열되어 있습니다.
- 같은 Holter 가 이름만 바꿔 여러 번 내보내진 경우가 있습니다 → `labels/duplicates.csv` 참고.

---

## 3. HDF5 (`h5/`, schema v2)

24시간 전체를 빠르게 읽도록 신호를 dataset 하나에 연속 저장한 구조입니다.
record 하나 ≈ 63 MB.

```
/  root attrs : schema_version, record_name, cohort, pid, age, gender,
│               fs=125, n_samples, n_seg, seg_len=1250, duration_h,
│               sig_name='["II","V1","V5"]', scale='[s_II,s_V1,s_V5]',
│               has_beats, has_fiducial, has_report,
│               (리포트) hr_min/avg/max, vb_*, sb_*, NoisePercentage, …
├─ signal         (n_samples, 3) int16      mV = signal * scale[lead]
├─ beat/          sample(절대 인덱스) · symbol · subtype · chan · num · aux_code · seg_offset
├─ seg/           quality (n_seg,5,3) · fiducial_feat (n_seg,19) · similarity (n_seg,2,3)
└─ meta/          adc_gain · baseline · adc_res · adc_zero
```

읽는 법:

```python
import sys; sys.path.insert(0, "code/holter_psvt")
from h5_converter.schema_v2 import RecordV2

with RecordV2("h5/10_50_2247355.h5") as r:
    r.fs, r.n_seg, r.sig_name        # 125.0, 8634, ['II','V1','V5']
    x = r.segment(42)                # (1250, 3) float32, mV — 10초 구간
    x = r.window(0, 125 * 3600)      # 첫 1시간
    samples, symbols = r.beats_in(100, 110)
```

주의할 점:

- 기본 변환은 dummy 모드라 **`seg/fiducial_feat`, `seg/similarity` 는 전부 NaN** 이고 `/fid` 가 없습니다.
  `seg/quality` 는 항상 계산되어 있습니다.
- `age` 는 문자열이며 `.json` 이 없으면 `"-1"` 입니다.
- 리포트 attr 은 `.json` 에 그 필드가 있을 때만 생기므로 record 마다 attr 목록이 다를 수 있습니다.
- 필드 전체 정의와 v1 대비 변경점: `code/holter_psvt/h5_converter/SCHEMA_V2.md`

---

## 4. 라벨 (`labels/splits.csv`)

`manifest.csv` 의 모든 열에 아래 열이 추가된 표입니다. `path` 열은 이 폴더의
`h5/<record>.h5` 절대경로로 바꿔 두었습니다. 폴더를 다른 곳으로 옮기면 `path` 만 고치면 됩니다.

| 열 | 뜻 |
|---|---|
| `split` | `train` / `val` / `test` — **환자(PID) 단위** 70/10/20, 과제별 양성 비율 층화 |
| `eligible` | 학습 적격 = 중복 아님(`dup_keep`) ∧ 12시간 이상 ∧ 무신호 구간 ≤ 20% |
| `y_psvt` | `clinical_data_psvt.csv` 의 `Label` (1/0). CSV 에 없으면 NaN |
| `y_tof` | TOF 코호트 폴더 소속이면 1, 아니면 0 |
| `y_lqt` | LongQT 코호트 폴더 소속이면 1, 아니면 0 |

- **`y_tof`, `y_lqt` 는 진단이 아니라 코호트(폴더) 소속입니다.** 장비·시기 차이만으로 맞힐 수 있으니
  해석에 주의하세요.
- 같은 신호가 다른 PID 에 붙은 record(`dup_pid_conflict=True`)는 모든 라벨이 NaN 입니다.
- 규모 (2026-09-22 기준): PSVT 라벨 3,822 record / 양성 환자 184명, LongQT·TOF 6,451 record /
  양성 환자 79명·674명. 정확한 수는 `splits_summary.txt` 를 보세요.
- **소아 코호트입니다.** 환자의 83%가 20세 미만이고, TOF 양성만 나이가 높습니다(중앙값 21세).
  나이로 교란을 점검할 때 성인 기준 구간을 쓰면 결론이 달라집니다.

```python
import pandas as pd
df = pd.read_csv("labels/splits.csv", dtype={"pid": str}, low_memory=False)
psvt = df[df.eligible & df.y_psvt.notna()]
train, test = psvt[psvt.split == "train"], psvt[psvt.split == "test"]
```

---

## 5. 코드 (`code/holter_psvt/`)

| 폴더 | 내용 |
|---|---|
| `h5_converter/` | 원본 → h5 변환 (`fix_pid.py`, `convert_to_h5.py`, `schema_v2.py`) |
| `tools/` | 대량 변환·중복 탐지·manifest·split 생성 등 |
| `holter_encoder/` | 24h SSL 인코더 (Stage A/B) 와 probe 평가 |
| `psvt_pipeline/` | 초기 MIL 기반 PSVT 분류 (v1 h5 기준) |

설치: `pip install -r code/holter_psvt/requirements.txt` (torch 는 CUDA 버전에 맞춰 따로)

### h5 · 라벨을 다시 만드는 순서

`raw/` 는 이미 `fix_pid.py` 가 적용된 상태라 바로 변환할 수 있습니다.
**출력은 이 폴더 밖(본인 작업 디렉토리)에 두세요.** 공유본은 덮어쓰지 않습니다.

```bash
cd code/holter_psvt
RAW=/home/coder/workspace/Holter_TOF/holter_total/raw
LAB=/home/coder/workspace/Holter_TOF/holter_total/labels
OUT=<본인 작업 디렉토리>

python tools/find_duplicates.py --raw $RAW --out $OUT/duplicates --workers 16
python tools/run_conversion.py --raw $RAW/nas1_Holter_PSVT $RAW/nas1_Holter_TOF_250917 \
       $RAW/nas1_Holter_LQT_260615 --out $OUT/h5 --cpus 32
python tools/build_manifest.py --src $OUT/h5 --out $OUT/manifest --duplicates $OUT/duplicates.csv \
       --clinical psvt=$LAB/clinical_data_psvt.csv tof=$LAB/clinical_data_tof.csv
python tools/make_splits.py --manifest $OUT/manifest.csv --out $OUT/splits     # seed 42
```

`run_conversion.py` 는 출력에 이미 있는 record 를 건너뛰므로 중단 후 다시 실행하면 이어집니다.
NAS(`Holter_TOF/`) 읽기는 약 86 MB/s 라, 많이 읽을 거면 `raw/` 를 로컬 디스크로 먼저 복사하세요.
모델 학습·평가 절차는 `docs/METHOD.md`, 실험 결과와 교란 분석은 `docs/EXPERIMENTS.md` 에 있습니다.

---

## 6. 포함하지 않은 것

| 항목 | 이유 |
|---|---|
| v1 h5 (`Holter_TOF/holter_h5`, `nas1_Holter_PSVT/h5`) | v2 로 대체됨. 세그먼트당 dataset 구조라 24h 읽기가 느림 |
| `denoised_3_lead`, `10s_segment_final`, `remove_silent_bandpass_notch` | 예전 실험용 전처리 산출물 |
| `segmeta/`, `tokens_a/`, `runs/` (체크포인트·임베딩) | 코드로 다시 만들 수 있는 학습 중간 산출물 |

---

문의: tyoon0110_2@snuh-bmilab.ai.kr · 작성 2026-10-02
