# HDF5 저장 구조 v1 → v2

24시간 전체를 입력으로 쓰는 학습에서 v1 구조가 병목이라 저장 구조를 바꿨다.
**정보는 하나도 버리지 않는다.** v1의 모든 필드가 v2에 대응된다.

## 왜 바꾸는가

실측(24h record 1개, `tools/inspect_data.py`):

| | v1 | v2 (예상) |
|---|---|---|
| dataset 수 | **141,271** | **15 이하** |
| group 수 | 62,798 | 4 |
| attribute 수 | 157,006 | ~60 |
| 파일 크기 / 신호 크기 | **3.88x** | **1.00x** |
| 24h 전체 읽기 | **13 MB/s** | 디스크 대역폭 그대로 |
| 파일 크기 | 236 MB | **65 MB** |

HDF5 자체가 느린 게 아니라 **dataset 하나당 고정 오버헤드**(열기 + B-tree 링크 조회 +
object header 파싱, 개당 수십~수백 µs)가 8,640번 곱해지는 것이 문제다.

---

## v1 구조 (현재)

```
/  [dataset_version, created_by, created_at, file_name]
├─ patient/  [pid, age, gender]
└─ ECG/
   ├─ segments/  [seg_len]          ← ⚠ 이름과 달리 "세그먼트 개수"가 들어있다
   │  ├─ 0/                          ← 세그먼트마다 group 8개 + dataset 18개 + attr 21개
   │  │  ├─ signal/
   │  │  │  ├─ II   (1250,) float16
   │  │  │  ├─ V1   (1250,) float16
   │  │  │  └─ V5   (1250,) float16
   │  │  ├─ beat_annotation/
   │  │  │  ├─ sample   (n_b,) int16     ← 세그먼트 기준 상대 인덱스 0..1249
   │  │  │  ├─ symbol   (n_b,) utf8
   │  │  │  ├─ subtype  (n_b,) int16
   │  │  │  ├─ chan     (n_b,) int16
   │  │  │  ├─ num      (n_b,) int16
   │  │  │  └─ aux_note (n_b,) utf8
   │  │  ├─ fiducial_point/  [extraction_method]
   │  │  │  ├─ fsample  (k,) int16       ← 세그먼트 기준 상대
   │  │  │  └─ fiducial (k,) utf8
   │  │  ├─ fiducial_feature/            ← 값이 attr 19개로 흩어져 있다
   │  │  │  [p_amp q_amp r_amp s_amp t_amp p_dur pr_seg qrs_dur st_seg t_dur
   │  │  │   pr_int qt_int qtc_baz qtc_frid rr_int tp_seg p_axis r_axis t_axis]
   │  │  └─ signal_quality/
   │  │     ├─ nan_ratio (3,) float16
   │  │     ├─ amplitude/
   │  │     │  ├─ amp_mean     (3,) float16
   │  │     │  ├─ amp_std      (3,) float16
   │  │     │  ├─ amp_skewness (3,) float16
   │  │     │  └─ amp_kurtosis (3,) float16
   │  │     └─ beat_similarity/
   │  │        ├─ bs_correlation (3,) float16
   │  │        └─ bs_dtw         (3,) float16
   │  ├─ 1/ … (동일 구조가 8,640번 반복)
   │  └─ …
   ├─ metadata/  [record_name, n_sig, fs, sig_len, base_time, base_date, dtype]
   │  ├─ sig_name (3,) utf8   = ["V5","V1","II"]   ← ⚠ 문서의 II→V1→V5 와 역순
   │  ├─ fmt, units (3,) utf8
   │  ├─ adc_gain (3,) float16
   │  └─ baseline, adc_res, adc_zero (3,) int16
   └─ annotation/  [ann_len, NoisePercentage, AFAFLPercentage]
      └─ beat_count/
         ├─ VentricularBeat/  [total, Isolated, Couplets, BigeminalCycles]
         │  └─ Runs/  [count, TotalBeats, LongestRunBeats, LongestRunBPM,
         │             LongestRunTimestamp, FastestRunBeats, FastestRunBPM,
         │             FastestRunTimestamp]
         ├─ SupraventricularBeat/  (동일 구조)
         ├─ PacedBeats/  [total]
         ├─ BBBeats/  [total]
         ├─ JunctionalBeats/  [total]
         └─ AberrantBeats/  [total]
```

---

## v2 구조 (신규)

아래는 현재 코드(`tools/run_conversion.py`)로 실제 변환한 파일에서 뽑은 구조다.
24h record(10,350,000 sample, 8,280 세그먼트, beat 약 10만 개) 기준 크기를 함께 적었다.

```
/  root attrs
│  ── 식별 ──
│  schema_version="2.0", record_name, cohort, raw_path, source="SNUH",
│  created_at, created_by
│  ── 신호 ──
│  fs=125.0, n_sig=3, n_samples, n_seg, seg_len=1250, duration_h, dtype="int16"
│  sig_len                                                         ← .hea 원래 샘플 수 (n_samples 는 10초 단위로 자른 길이)
│  sig_name='["II","V1","V5"]'   scale='[s_II, s_V1, s_V5]'        ← JSON 문자열
│  base_date, base_time                                           ← .hea
│  ── 환자 ──
│  pid (파일명 끝 토큰), age, gender                                ← age 는 문자열, .json 없으면 "-1"
│  ── 상태 플래그 ──
│  has_beats, has_fiducial, has_report
│  ── 사전 / 배열 열 이름 ──
│  symbol_table, aux_vocab, fiducial_feature_names, quality_names, similarity_names
│  ── .json 리포트 (has_report=True 일 때만) ──
│  report_pid, report_age, report_gender, report_duration, hookup_date, hookup_time
│  ann_len, NoisePercentage, AFAFLPercentage                      ← 문자열 ("< 1", "Unknown")
│  hr_min, hr_min_ts, hr_avg, hr_max, hr_max_ts                    ← JSON 에 있을 때만
│  tachy_beats, tachy_pct, brady_beats, brady_pct                  ← "Unknown" 은 nan
│  vb_total, vb_Isolated, vb_Couplets, vb_BigeminalCycles,
│  vb_run_count, vb_run_TotalBeats, vb_run_{Longest,Fastest}Run{Beats,BPM,Timestamp}
│  sb_… (상심실성, 동일 12개)
│  PacedBeats_total, BBBeats_total, JunctionalBeats_total, AberrantBeats_total
│
├─ signal            (n_samples, 3)    int16    contiguous   ≈ 62 MB
│
├─ beat/                                                       (has_beats=True 일 때만)
│  ├─ sample         (n_beats,)        int32    record 기준 절대 인덱스
│  ├─ symbol         (n_beats,)        uint8    symbol_table[code]
│  ├─ subtype        (n_beats,)        int8
│  ├─ chan           (n_beats,)        int8
│  ├─ num            (n_beats,)        int8
│  ├─ aux_code       (n_beats,)        uint16   aux_vocab[code]
│  └─ seg_offset     (n_seg+1,)        int32    seg i = sample[off[i]:off[i+1]]   ≈ 1 MB 합계
│
├─ fid/                                                        (has_fiducial=True 일 때만)
│  ├─ sample, label, seg_offset
│
├─ seg/
│  ├─ fiducial_feat  (n_seg, 19)       float16
│  ├─ quality        (n_seg, 5, 3)     float16  [nan_ratio, amp_mean, amp_std, amp_skewness, amp_kurtosis]
│  └─ similarity     (n_seg, 2, 3)     float16  [bs_correlation, bs_dtw]
│
└─ meta/  attrs: fmt, units (JSON 문자열)
   ├─ adc_gain       (3,)              float32
   └─ baseline, adc_res, adc_zero (3,) int32
```

record 하나 ≈ **63 MB**, dataset 18개, group 3~4개.

### 입력 상태에 따라 달라지는 것

| 상황 | 결과 |
|---|---|
| `.ANN` 없음 | `has_beats=False`, `/beat` 없음 |
| `.json` 없음 | `has_report=False`, 리포트 attr 전부 없음, `age="-1"`, `gender=""` |
| 기본(dummy) 모드 | `has_fiducial=False`, `/fid` 없음, `seg/fiducial_feat`·`seg/similarity` 는 **전부 NaN** |
| `--real-fiducial` | `/fid` 생성, `seg/fiducial_feat` 채워짐 |
| `--real-similarity` | `seg/similarity` 채워짐 (dtw-python 필요) |

`seg/quality` 는 모드와 무관하게 항상 계산된다.
리포트 attr 은 `.json` 에 그 필드가 있을 때만 생기므로 record 마다 attr 목록이 다를 수 있다.

### 코호트

`cohort` 는 `--raw` 로 준 디렉토리 아래 첫 디렉토리 이름이다.
`--raw /home/coder/workspace/data/raw` 이면 `nas1_Holter_PSVT`, `nas1_Holter_TOF_250917`,
`nas1_Holter_LQT_260615` (하위 `addition_260729` 포함) 중 하나가 된다.

---

## 필드 대응표

| v1 위치 | v2 위치 | 비고 |
|---|---|---|
| `ECG/segments/{i}/signal/{lead}` float16 | `/signal[i*1250:(i+1)*1250, j]` int16 | dtype·레이아웃 변경 |
| `…/beat_annotation/sample` int16 (상대) | `/beat/sample` int32 (**절대**) | `i*seg_len + sample` |
| `…/beat_annotation/symbol` utf8 | `/beat/symbol` uint8 | `symbol_table[code]` |
| `…/beat_annotation/{subtype,chan,num}` int16 | `/beat/{subtype,chan,num}` int8 | 값 범위상 int8 충분 |
| `…/beat_annotation/aux_note` utf8 | `/beat/aux_code` uint16 | `aux_vocab[code]` |
| `…/fiducial_point/fsample` int16 (상대) | `/fid/sample` int32 (**절대**) | |
| `…/fiducial_point/fiducial` utf8 | `/fid/label` uint16 | `fiducial_vocab[code]` |
| `…/fiducial_point.attrs[extraction_method]` | (생략) | 레코드 내 상수 |
| `…/fiducial_feature.attrs[19개]` | `/seg/fiducial_feat[i, :]` | attr → 배열 열 |
| `…/signal_quality/nan_ratio` | `/seg/quality[i, 0, :]` | |
| `…/signal_quality/amplitude/amp_*` | `/seg/quality[i, 1:5, :]` | |
| `…/signal_quality/beat_similarity/bs_*` | `/seg/similarity[i, :, :]` | |
| `patient.attrs[pid,age,gender]` | root attrs 동일 이름 | |
| `ECG/metadata.attrs[*]` | root attrs 동일 이름 | |
| `ECG/metadata/sig_name` | root attr `sig_name` | **II,V1,V5 로 정규화** |
| `ECG/metadata/{adc_gain,baseline,adc_res,adc_zero}` | `/meta/*` | lead 순열 적용 |
| `ECG/metadata/{fmt,units}` | `/meta.attrs[fmt,units]` | lead 순열 적용 |
| `ECG/annotation.attrs[*]` | root attrs 동일 이름 | |
| `…/beat_count/VentricularBeat.attrs[X]` | root attr `vb_X` | |
| `…/VentricularBeat/Runs.attrs[X]` | root attr `vb_run_X` | |
| `…/SupraventricularBeat/**` | root attr `sb_*` | |
| `…/{Paced,BB,Junctional,Aberrant}Beats.attrs[total]` | root attr `*_total` | |
| — | root attr `n_samples`, `duration_h`, `scale` | 신규 |
| — | root attr `has_beats/has_fiducial/has_report` | 신규, 필터용 |
| — | root attr `cohort`, `raw_path` | 신규, 코호트 구분·원본 추적 |
| .json `HeartRates/*`, `PatientInfo/*` | root attr `hr_*`, `tachy_*`, `brady_*`, `hookup_*`, `report_*` | v1 은 읽지 않았음 |
| — | `/beat/seg_offset`, `/fid/seg_offset` | 신규, CSR 인덱스 |

**v1에만 있고 v2에 없는 필드는 `extraction_method` 하나뿐**이고, 레코드 내에서 값이
같아 의미가 없어 뺐다. 필요하면 root attr로 되살릴 수 있다.

---

## 의미가 바뀌는 것 (주의)

### 1. `seg_len`의 뜻이 바뀐다
- v1: `ECG/segments.attrs["seg_len"]` = **세그먼트 개수** (이름과 반대다.
  `convert_to_h5.py`에서 `seg_len=data["segment_cnt"]`로 넘긴다)
- v2: `seg_len` = **세그먼트당 샘플 수**(1250), 개수는 `n_seg`로 분리

### 2. lead 순서가 정규화된다
- v1: `.hea` 기록 순서 그대로 = `["V5","V1","II"]` (README 문서와 역순)
- v2: 항상 `["II","V1","V5"]`

신호는 `signal/{이름}`으로 저장돼 있어 이름으로 읽으면 안전하지만,
`nan_ratio`·`adc_gain` 같은 `(3,)` 배열은 `.hea` 순서를 따른다. **신호만 재배열하고
이 배열들을 그대로 두면 조용히 어긋난다.** repack은 `metadata/sig_name`을 기준으로
같은 순열을 적용한다. `h5py`의 `group.keys()`는 알파벳 정렬이라 물리 순서로 쓰면 안 된다.

### 3. dtype이 float16 → int16 + scale
복원: `x_mV = signal.astype(np.float32) * scale[lead]`

크기는 같지만(2 byte) int16은 ±32767을 lead 최대 진폭에 균등 배분한다.
float16은 지수 표현이라 진폭이 클수록 간격이 벌어진다(진폭 10 근처에서 0.0078,
1 근처에서 0.00098). 관측된 신호 범위가 ±37까지 가므로 int16이 유리하다.

### 4. beat/fiducial 인덱스가 절대 좌표가 된다
v1은 세그먼트 기준 0..1249. v2는 record 기준 절대값이라 세그먼트 경계와 무관하게
구간 질의가 된다. 세그먼트별 접근은 `seg_offset`으로 그대로 가능하다.

---

## 읽는 법

```python
from h5_converter.schema_v2 import RecordV2

with RecordV2("REC.h5") as r:
    r.fs, r.n_samples, r.n_seg, r.sig_name      # 125.0, 10792500, 8634, ['II','V1','V5']
    x = r.window(0, 1250)          # (1250, 3) float32, mV 단위
    x = r.segment(42)              # 42번 세그먼트
    x = r.window(0, 1250, mv=False)  # 원시 int16
    samples, symbols = r.beats_in(100, 110)   # 세그먼트 100~109의 beat
```

`/signal`이 contiguous·무압축이라 `RecordV2`는 `dataset.id.get_offset()`으로 파일 내
바이트 오프셋을 얻어 **직접 memmap**한다. `.npy` memmap과 동일한 경로이므로

- OS page cache가 그대로 적용된다
- DataLoader worker에서 HDF5 전역 락(기본 빌드는 thread-safe가 아니라 내부 직렬화)과
  GIL을 거치지 않는다

압축을 켜면 이 경로가 막히므로, 로컬 NVMe에 둘 거면 무압축이 맞다.

---

## 생성 / 변환

```bash
# 앞으로 변환되는 레코드 — convert_to_h5.py 의 USE_V2 = True (기본값)
python convert_to_h5.py

# 이미 만들어진 v1 파일 변환 (원본은 읽기만 한다)
python tools/repack_to_v2.py --src <v1 디렉토리> --dst <v2 디렉토리> --workers 32

# 신호만 있는 flat h5 도 같은 포맷으로 (lead 순서는 profile_records.py 로 먼저 확인)
python tools/repack_to_v2.py --src <flat 디렉토리> --dst <v2 디렉토리> --leads II,V1,V5

# 전체를 훑어 manifest 하나로
python tools/build_manifest.py --src <v2 디렉토리> --out <v2 디렉토리>/manifest \
    --clinical psvt=clinical_data_psvt.csv tof=clinical_data_tof.csv

# 회귀 테스트 (22개 항목)
python -m h5_converter.test_schema_v2
```
