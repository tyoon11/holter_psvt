# Holter PSVT

24시간 **Holter ECG**로부터 **PSVT(발작성 상심실성 빈맥, Paroxysmal Supraventricular Tachycardia)** 여부를 이진 분류하는 end-to-end 파이프라인입니다.

전체 흐름은 크게 두 단계로 나뉩니다.

```
 raw Holter ECG (.dat/.hea/.json)
            │
            ▼
   ┌───────────────────┐
   │   h5_converter     │   raw → 표준 HDF5 변환 (10초 세그먼트 단위)
   └───────────────────┘
            │  .h5
            ▼
   ┌───────────────────┐
   │   psvt_pipeline    │   Multimodal MIL 학습/평가 (ECG bag + metadata)
   └───────────────────┘
            │
            ▼
   PSVT 이진 분류 결과 + 성능 리포트
```

---

## 📁 저장소 구조

```
holter_psvt/
├─ README.md                  # (현재 문서) 프로젝트 전체 개요
├─ .gitignore
│
├─ h5_converter/              # 1단계: raw Holter ECG → 표준 HDF5 변환
│  ├─ README.md               #   변환 파이프라인 상세 문서
│  ├─ fix_pid.py              #   .hea/.json 내부 PID·record명 정정 (사전 작업)
│  ├─ utils.py                #   신호 품질/fiducial/유효 레코드 유틸
│  ├─ create_h5_structure.py  #   HDF5 저장 구조 정의
│  ├─ convert_to_h5.py        #   raw → .h5 변환 (Ray 병렬)
│  └─ h5_test.ipynb           #   생성된 .h5 구조 탐색·시각화
│
└─ psvt_pipeline/             # 2단계: HDF5 기반 PSVT 분류 학습/평가
   ├─ README.md               #   MIL 파이프라인 상세 문서
   ├─ pipeline.py             #   메인 학습/평가 파이프라인 (표준 H5 버전)
   ├─ gn_pipeline.py          #   GN 데이터셋 버전 파이프라인
   ├─ sampling.py             #   S-우선 segment 샘플링 로직
   ├─ models.py               #   1D 인코더 + MIL Head + Multimodal 분류기
   └─ requirements.txt        #   학습 파이프라인 의존성
```

---

## 🚀 빠른 시작

### 1단계 — HDF5 변환 (`h5_converter`)

```bash
pip install pandas numpy h5py matplotlib neurokit2 dtw wfdb ray

cd h5_converter
python fix_pid.py          # raw .hea/.json 내부 PID·record명 정정 (필수 사전 작업)
python convert_to_h5.py    # raw → .h5 변환 (Ray 병렬 처리)
```

자세한 옵션(`use_dummy_fiducial`, `use_dummy_similarity`, 이미 변환된 파일 건너뛰기 등)과
HDF5 저장 구조는 [h5_converter/README.md](h5_converter/README.md)를 참고하세요.

### 2단계 — PSVT 분류 학습/평가 (`psvt_pipeline`)

```bash
conda create -n psvt python=3.9 && conda activate psvt
pip install -r psvt_pipeline/requirements.txt   # GPU 사용 시 CUDA에 맞는 torch 별도 설치

cd psvt_pipeline
# pipeline.py 상단의 CSV_PATH / BASE_DIR / SPLIT_DIR / CACHE_DIR / RESULT_ROOT 경로 확인·수정
python pipeline.py
```

주요 개념(record 단위 MIL, S-beat 중심 샘플링, upsampling, dynamic `pos_weight`,
결정론적 재현성, 결과물 구조 등)은 [psvt_pipeline/README.md](psvt_pipeline/README.md)를 참고하세요.

---

## ⚠️ 경로 설정 안내

`psvt_pipeline/pipeline.py`, `gn_pipeline.py` 상단에는 개발 환경 기준의 절대 경로
(`/home/coder/workspace/...`)가 하드코딩되어 있습니다. 실행 전 본인 환경에 맞게
아래 변수를 수정해야 합니다.

| 변수 | 설명 |
| --- | --- |
| `CSV_PATH` | `h5_metadata_summary.csv` 경로 |
| `BASE_DIR` | HDF5 데이터 루트 |
| `SPLIT_DIR` | patient-wise split CSV 디렉토리 (`seed_splits/*.csv`) |
| `CACHE_DIR` | segment scan 캐시 보관소 |
| `RESULT_ROOT` | 실험 결과 저장 루트 |

---

## 📝 주요 특징

- **record 단위 MIL** — 하나의 HDF5(=record)에서 K개 segment를 뽑아 하나의 bag 구성
- **S-beat 중심 샘플링** — supraventricular(`"S"`) segment 우선 선택 후 이웃/fill 정책으로 보충
- **Ray 기반 병렬 처리** — HDF5 변환과 segment scan 모두 병렬화
- **완전 결정론적 재현성** — seed 조합으로 동일 설정 시 항상 동일한 segment 조합 보장
- **불균형 대응** — positive record upsampling + dynamic `pos_weight`

---

## 📌 참고

- 데이터(`.h5`), 학습 산출물(`*.pt`, `results*/`, `seed_caches*/`), 발표 자료(`*.pptx`)는
  `.gitignore`에 의해 버전 관리에서 제외됩니다.
