# 24시간 Holter 인코더 — 설계와 학습

24시간 Holter ECG 한 건을 통째로 읽어 표현(embedding)을 만드는 자기지도 인코더.
목표는 **라벨 없이 대량의 Holter 로 사전학습한 뒤, 적은 라벨로 여러 질환 태스크를
푸는 것**이다. 현재 downstream 은 LongQT / TOF / PSVT 세 가지.

> 이 문서는 진행에 따라 갱신한다. 마지막 갱신과 변경 이력은 맨 아래 참고.
> 성능 병목을 어떻게 찾아 없앴는지는 [PERFORMANCE.md](PERFORMANCE.md),
> 저장 포맷의 세부는 [../h5_converter/SCHEMA_V2.md](../h5_converter/SCHEMA_V2.md).

---

## 1. 전체 흐름

```
원본 (.hea / .SIG / .ANN / .json)          MARS export, 3채널, 125 Hz, 약 24시간
   │  convert_to_h5  (+ fix_pid)
   ▼
v2 h5   /signal (n_samples, 3) int16 + beat/fiducial/품질/리포트          6,763 record, 404 GB
   │  build_manifest → make_splits
   ▼
splits.csv   환자 단위 70/10/20, 태스크 라벨, 적격 표시                    적격 6,454 record
   │  prep_meta   (품질·정규화·beat 요약·signal offset 을 묶음 파일로)
   ▼
Stage A   10초 구간으로 CNN stem 사전학습                                 stem.pt
   │  cache_tokens
   ▼
토큰 캐시  record 당 (n_seg, 256) float16                                  약 30 GB
   │  Stage B   24시간 토큰열로 backbone 사전학습
   ▼
encoder.pt   stem + backbone + time-of-day
   │  embed → probe
   ▼
downstream 평가 (환자 단위 AUROC, 교란 기준선 대비)
```

---

## 2. 데이터

### 코호트

| 코호트 | record | 내용 |
|---|---|---|
| `nas1_Holter_PSVT` | 4,463 | PSVT 의심/확진 및 대조. 소아 비중 큼 (나이 중앙값 11) |
| `nas1_Holter_TOF_250917` | 1,995 | 활로4징 교정 후 추적. 성인 중심 (나이 중앙값 22) |
| `nas1_Holter_LQT_260615` | 305 | QT 연장 증후군 (나이 중앙값 12) |

총 6,763 record / 156,360 시간 / 환자 3,833명. record 당 평균 23.4시간.

### 신호

- 3채널, 125 Hz. 파일의 채널 순서는 **V5, V1, II** (원본 `.hea` 에 이름이 없어 확인 후 확정).
  v2 는 이를 **II, V1, V5** 로 재배열해 저장한다.
- `.ANN` 의 beat 주석(정상/상심실성/심실성 등)과 `.json` 의 판독 리포트가 함께 들어있다.
  주석 보유 5,840 record, 리포트 보유 6,682 record.
- **시각은 `.json` 의 `hookup_time` 기준.** `.hea` 의 시각은 MARS 내보내기 시각이라
  촬영 시작과 일치하지 않는다(일치율 0%).

### 전처리

- record·채널마다 `seg/quality` 의 amp_std 중앙값으로 나눠 진폭을 맞춘다.
  전극 부착과 이득이 record 마다 달라서, 형태 학습에 진폭 차이가 섞이지 않게 한다.
- 정규화 후 ±20 으로 자른다(일부 record 에 ±37 mV 아티팩트가 있다).
- 품질 불량 세그먼트(결측 존재 또는 채널 무신호)는 학습에서 빼고, Stage B 손실에서도 제외한다.

### split 과 라벨

환자(PID) 단위 70/10/20. 층화 기준은 (소속 코호트, PSVT 라벨 상태, 촬영 횟수 1/2/3+).
촬영 횟수를 넣은 이유는 다회 촬영 환자가 한쪽에 몰리면 test 집단의 성격이 달라지기 때문이다.

| 태스크 | 양성 | 음성 | 전체 유병률 | test 양성 환자 |
|---|---|---|---|---|
| LongQT | LQT 코호트 | 나머지 전부 | 4.34% | **14명** |
| TOF | TOF 코호트 | 나머지 전부 | 29.2% | 133명 |
| PSVT | `clinical_data_psvt` Label=1 | Label=0 | 9.08% | 38명 |

적격 조건: 신호 중복 아님, 12시간 이상, 무신호 세그먼트 20% 이하.
같은 신호에 다른 PID 가 붙은 6 record 는 SSL 에만 쓰고 모든 라벨을 비웠다.

---

## 3. 모델

### 3.1 BeatCNNStem — 10초를 토큰 하나로

```
(N, 3, 1250)                     10초 구간, 125 Hz
  Conv1d k=15 s=2 → 32ch
  ResBlock1D ×6, stride 2        32 → 64 → 96 → 128 → 192 → 256 → 256
(N, 256, 10)                     프레임 10개 (약 1.1초 간격)
  attention pooling (프레임 축)
(N, 256)                         토큰 1개
```

프레임을 그대로 두는 이유는 Stage A 의 파형 복원에 쓰기 위해서다.
토큰은 Stage B 로 넘어간다. 파라미터 약 2.69M.

### 3.2 HierarchicalS4 — 24시간 문맥

토큰열 `(B, 256, 8640)` 을 세 해상도로 훑고 되돌아오는 U-Net.

```
L0  8640 토큰 (10초)   블록 2개 ─────────────────┐ skip
  ↓ Conv1d stride 6
L1  1440 토큰 (1분)    블록 4개 ───────┐ skip     │
  ↓ Conv1d stride 10
L2   144 토큰 (10분)   블록 4개        │          │
  ↑ ConvTranspose ×10                 │          │
L1' 1440 토큰          블록 2개 ───────┘          │
  ↑ ConvTranspose ×6                             │
L0' 8640 토큰          블록 2개 ─────────────────┘
```

- **L0** 박동 형태와 연속 리듬, **L1** PSVT run·AF 같은 에피소드, **L2** 하루 주기.
- skip 연결이 있어 저해상도를 거치며 국소 정보가 사라지지 않는다.
- 길이가 60(=6×10)의 배수가 아니면 패딩 후 복원한다.

블록은 세 종류 중 고른다.

| `--block` / `--mid-block` | 성질 | 비고 |
|---|---|---|
| `s4` | S4D. 시간 불변 필터, FFT 컨볼루션 | 순수 PyTorch. 24h 도 빠름 |
| `mamba` | 양방향 선택적 SSM. Δ·B·C 가 입력에 따라 변함 | 드문 사건 구간을 골라 유지. `mamba-ssm` 필요 |
| `attn` (mid 전용) | 10분 해상도 self-attention (144 토큰) | attention 가중치로 참고 시간대를 볼 수 있음 |

S4 는 모든 위치에 같은 필터를 적용하므로 "여기는 중요하니 오래 기억" 을 내용 기반으로
할 수 없다. Mamba 는 가능하다. 24시간 중 수십 초짜리 PSVT run 을 다루는 이 과제에서는
후자가 유리할 수 있어 **같은 토큰 캐시로 셋을 비교**한다.

### 3.3 time-of-day

토큰마다 "하루 중 몇 시인가" 를 sin/cos 6쌍으로 넣는다. 심박은 circadian 구조가 강해서
(수면 중 서맥, 주간 빈맥) 절대 시각이 유용한 사전 정보다. `hookup_time` 이 없는 record 는
해당 임베딩이 0 이 된다.

---

## 4. 학습

### 왜 두 단계인가

24시간은 10초 세그먼트 8,640개다. 원신호부터 끝까지 한 번에 학습하면 한 샘플이 62 MB 라
배치를 키울 수 없다. **짧은 구간을 배우는 단계**와 **하루 흐름을 배우는 단계**를 나누면,
2단계는 1단계 결과를 미리 계산해 두고 쓰므로 **같은 캐시로 백본만 바꿔 비교**할 수 있다.

### Stage A — stem 사전학습

train 환자 record 에서 무작위 10초 구간을 뽑는다(품질 통과 구간만).

| 과제 | 내용 | 가중치 |
|---|---|---|
| 파형 복원 | 구간 30% 를 가리고(0.2~1초 구간들) 원파형 복원. 가린 곳 1.0, 나머지 0.1 | 1.0 |
| beat 요약 | 그 10초의 정상/상심실성/심실성/전체 박동 수(log1p)와 심박수 회귀 | 0.3 |

beat 과제는 `.ANN` 에서 공짜로 얻는 정답이라 **부정맥 관련 표현이 빨리 자리잡게** 한다.
주석이 없는 record 는 이 항을 마스크로 제외한다.

주요 기본값: `d_model 256`, `batch 64 record × segs-per-record 32`(= GPU 당 2,048 세그먼트),
`lr 1e-3` cosine + warmup 1,000, `weight decay 0.05`, bf16, 30,000 step.

### 토큰 캐시

Stage A stem 으로 **전 split** 을 인코딩해 `(n_seg, 256) float16` 으로 저장한다.
record 당 62 MB → 4.4 MB. Stage B 는 원신호를 건드리지 않아 IO 부담이 거의 없다.
downstream 평가도 같은 토큰을 쓴다.

### Stage B — backbone 사전학습

캐시 토큰 24시간(8,640개)을 입력으로 받는다.

- **1~6분 구간(6~36 토큰)으로 토큰 15% 를 가리고**, 가려진 자리의 **원래 stem 토큰**을 맞힌다
  (layer norm 정규화 후 smooth L1, 타깃은 stop-grad).
- 타깃이 고정된 stem 출력이라 표현이 한 점으로 무너지는 문제(collapse)가 없다.
- 패딩과 품질 불량 토큰은 손실에서 제외한다.

주요 기본값: `crop 8640`, `batch 8`(GPU 당), `lr 5e-4` cosine + warmup 500,
`mask-ratio 0.15`, `span 6~36`, `dropout 0.1`, 20,000 step.

끝나면 `encoder.pt` 를 만든다. `HolterEncoder` 의 state_dict 형식이며 stem(Stage A) +
backbone·tod(Stage B) 를 담는다. attention pool 은 사전학습되지 않으므로 downstream 에서
학습하거나 평균 풀링을 쓴다.

---

## 5. 평가

인코더를 **고정**하고 record 임베딩(품질 통과 토큰의 평균)을 뽑아 선형 분류기만 얹는다.

- train 라벨로 학습 → val 로 정규화 강도 선택 → test 보고.
- record 는 한 환자 안에서 상관되므로 **환자 단위**로도 집계하고, 신뢰구간은
  **환자 부트스트랩**으로 낸다.
- 세 특징을 나란히 비교한다.

| 특징 | 뜻 |
|---|---|
| `demo` | 나이 + 평균 HR + 성별. **교란 기준선** |
| `enc` | 인코더 임베딩 |
| `enc+demo` | 둘 다. `demo` 대비 상승폭이 인코더의 순수 기여분 |

**TOF 는 `demo` 만으로 test AUROC 0.884** 가 나온다(코호트가 성인 중심이라).
그래서 `enc` 단독 수치로 질환 학습을 주장할 수 없다.
LongQT 는 test 양성 환자가 14명뿐이라 `--cv`(SSL 이 보지 않은 val+test 환자로 5-fold)를
근거로 삼는다. `--fractions` 로 라벨 10%/25%/100% 도 함께 본다.

---

## 6. 재현 명령

```bash
OUT=/home/coder/workspace/data/holter_v2
RUN=/home/coder/workspace/data/runs

python -m holter_encoder.prep_meta --splits $OUT/splits.csv --out $OUT/segmeta --workers 32

torchrun --nproc_per_node 4 -m holter_encoder.train_stage_a --splits $OUT/splits.csv \
    --out $RUN/stage_a --meta-dir $OUT/segmeta --gpus 0,1,2,3 \
    --batch 64 --segs-per-record 32 --workers 12

torchrun --nproc_per_node 4 -m holter_encoder.cache_tokens --splits $OUT/splits.csv \
    --stem $RUN/stage_a/stem.pt --out $OUT/tokens_a --meta-dir $OUT/segmeta --gpus 0,1,2,3

# 백본 비교: GPU 하나씩 배정해 동시에
python -m holter_encoder.train_stage_b --splits $OUT/splits.csv --tokens $OUT/tokens_a \
    --stem $RUN/stage_a/stem.pt --out $RUN/stage_b_s4 --block s4 --gpus 0 --batch 32 &
python -m holter_encoder.train_stage_b --splits $OUT/splits.csv --tokens $OUT/tokens_a \
    --stem $RUN/stage_a/stem.pt --out $RUN/stage_b_mamba --block mamba --gpus 1 --batch 32 &
python -m holter_encoder.train_stage_b --splits $OUT/splits.csv --tokens $OUT/tokens_a \
    --stem $RUN/stage_a/stem.pt --out $RUN/stage_b_mamba_attn --block mamba --mid-block attn \
    --gpus 2 --batch 32 &
wait

for B in s4 mamba mamba_attn; do
  python -m holter_encoder.embed --splits $OUT/splits.csv --tokens $OUT/tokens_a \
      --encoder $RUN/stage_b_$B/encoder.pt --out $RUN/stage_b_$B/emb.npz --gpus 0
done
python -m holter_encoder.probe --splits $OUT/splits.csv --out $RUN/probe --cv \
    --emb $RUN/stage_b_s4/emb.npz $RUN/stage_b_mamba/emb.npz $RUN/stage_b_mamba_attn/emb.npz
```

---

## 7. 진행 현황

| 단계 | 상태 | 결과 |
|---|---|---|
| v2 변환 | 완료 | 6,763/6,769 성공, 1시간 38분, 404 GB. 실패 6건은 길이 0 원본 |
| manifest / split | 완료 | 적격 6,454 record, 환자 3,826. 유병률 split 간 차이 0.6%p 이내 |
| Stage A | 완료 | 30,000 step, 약 58분, 70,000+ seg/s. val `rec` 1.43 → **1.14**, `beat` **0.0102** |
| 토큰 캐시 | 완료 | record 당 0.6초 |
| Stage B (s4) | 진행 중 | 91 record/s, 20,000 step ≈ 2시간. 손실 0.39 → 0.29 (80 step) |
| Stage B (mamba / mamba+attn) | 예정 | |
| embed / probe | 예정 | |

### 아직 정하지 않은 것

- **백본**: s4 / mamba / mamba+attn 중 무엇을 쓸지. downstream 결과로 정한다.
- **Stage A·B 학습량**: val 곡선이 평평해지는 지점을 보고 줄이거나 늘린다.
- **fiducial/유사도**: 현재 변환은 이 둘을 계산하지 않아 `seg/fiducial_feat`,
  `seg/similarity` 가 NaN 이다. 필요해지면 `--real-fiducial` 로 다시 변환해야 한다.
- **LongQT 표본**: test 양성 환자 14명. CV 로 보완하지만 근본적으로 부족하다.

### 알려진 한계

- LongQT·TOF 는 코호트 소속으로 라벨을 정의했다. 코호트마다 나이·평균 HR 분포가 달라
  (TOF 22세/HR 75, PSVT 11세/HR 96) 모델이 질환 대신 인구학적 차이를 배울 수 있다.
  반드시 `demo` 기준선과 함께 보고한다.
- 같은 신호가 다른 PID 로 들어간 묶음이 3개 있다. 원인 확인 필요.
- lead 이름은 원본에 없어 확인 후 확정한 값이다. 채널 **순서**는 전 코호트에서 동일하다.

---

## 변경 이력

- 2026-09-17 최초 작성. Stage A 완료·토큰 캐시 완료·Stage B 진행 중 시점.
