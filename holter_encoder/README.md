# holter_encoder — 24시간 Holter SSL 인코더

```
원신호 (n_samples, 3)  →  BeatCNNStem (10초 → 토큰)  →  HierarchicalS4 (10초/1분/10분 U-Net)  →  토큰 · record 임베딩
```

## 학습 단계

| 단계 | 스크립트 | 입력 | 학습 과제 |
|---|---|---|---|
| A | `train_stage_a` | 무작위 10초 구간 | 가린 파형 복원 + 구간 beat 요약 회귀 |
| 캐시 | `cache_tokens` | 전 split record | Stage A stem 으로 토큰화 (`.npy`) |
| B | `train_stage_b` | 캐시 토큰 24h (8640) | 1~6분 구간 가림 → 원래 토큰 예측 |

SSL 은 `splits.csv` 의 **train** 만 쓴다. val 은 손실 추적용. test 는 사용하지 않는다.

## 실행

```bash
OUT=/home/coder/workspace/data/holter_v2
RUN=/home/coder/workspace/data/runs

# Stage A (4 GPU)
torchrun --nproc_per_node 4 -m holter_encoder.train_stage_a --splits $OUT/splits.csv --out $RUN/stage_a

# 토큰 캐시
torchrun --nproc_per_node 4 -m holter_encoder.cache_tokens \
    --splits $OUT/splits.csv --stem $RUN/stage_a/stem.pt --out $OUT/tokens_a

# Stage B — 백본 비교 (같은 토큰 캐시 사용)
torchrun --nproc_per_node 4 -m holter_encoder.train_stage_b --splits $OUT/splits.csv \
    --tokens $OUT/tokens_a --stem $RUN/stage_a/stem.pt --out $RUN/stage_b_s4 --block s4
torchrun --nproc_per_node 4 -m holter_encoder.train_stage_b --splits $OUT/splits.csv \
    --tokens $OUT/tokens_a --stem $RUN/stage_a/stem.pt --out $RUN/stage_b_mamba --block mamba
torchrun --nproc_per_node 4 -m holter_encoder.train_stage_b --splits $OUT/splits.csv \
    --tokens $OUT/tokens_a --stem $RUN/stage_a/stem.pt --out $RUN/stage_b_mamba_attn --block mamba --mid-block attn
```

같은 `--out` 으로 다시 실행하면 `last.pt` 에서 이어서 학습한다.
각 Stage B 는 `encoder.pt` 를 남긴다 (`HolterEncoder` state_dict + config).

## GPU 지정

공용 서버라 쓸 GPU 를 고를 수 있다. 모든 학습/캐시 스크립트가 `--gpus` 를 받는다.

```bash
python -m holter_encoder.train_stage_a --gpus 2 ...                  # 단일 GPU
torchrun --nproc_per_node 2 -m holter_encoder.train_stage_a --gpus 1,3 ...   # rank 마다 하나씩
```

torchrun 과 함께 쓰면 rank 마다 목록에서 하나씩 맡는다 (`--nproc_per_node` 와 개수를 맞출 것).
`CUDA_VISIBLE_DEVICES` 를 직접 써도 된다.

## 백본 선택

| 옵션 | 특성 | 커널 필요 |
|---|---|---|
| `--block s4` | S4D. 시간 불변 필터. FFT 컨볼루션이라 24h 도 빠르다 | 없음 |
| `--block mamba` | 전 해상도 양방향 선택적 SSM. Δ·B·C 가 입력에 따라 변해 드문 사건을 골라 유지 | mamba-ssm 필요 |
| `--block s4 --mid-block mamba` | 10분 해상도(24h=144 토큰)에서만 Mamba. 토큰이 적어 **커널 없이도 쓸 만하다** | 없음 |
| `--mid-block attn` | 10분 해상도 self-attention. attention 가중치로 참고 시간대를 볼 수 있다 | 없음 |

합성 데이터 CPU 처리량 참고: s4 약 120, s4+mid mamba 58, 전 해상도 mamba 5.7 record/s.
커널 없이 전 해상도 Mamba 로 24h 를 학습하는 것은 현실적이지 않다.

### mamba-ssm 설치 (사내 프록시 환경)

`pip install mamba-ssm` 은 setup.py 가 GitHub 릴리스에서 wheel 을 받으려다
`SSL: CERTIFICATE_VERIFY_FAILED (self-signed certificate in certificate chain)` 로 실패한다.
pip 자체는 통과하므로 아래 순서로 우회한다.

```bash
# 1) pip 으로 wheel URL 을 직접 설치 (pip 의 TLS 를 타므로 우회됨)
#    오류 메시지에 찍힌 "Guessing wheel URL" 주소를 그대로 쓴다
pip install --no-build-isolation   "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu13torch2.11cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"

# 2) 404 면 그 조합의 wheel 이 없는 것 → 직접 컴파일 (nvcc 필요, 10~40분)
nvcc --version
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE   pip install --no-build-isolation causal-conv1d mamba-ssm

# 3) setup.py 의 다운로드를 살리려면 사내 CA 를 urllib 에도 알려준다
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE

# 확인
python -m holter_encoder.check_mamba --gpus 0
```

torch 2.11+cu130 처럼 최신 조합은 미리 빌드된 wheel 이 없을 수 있다. 그때는 2) 또는
`mamba-ssm==2.2.4` 처럼 낮은 버전을 시도한다. 끝내 안 되면 `--block s4 --mid-block mamba`
로 진행한다.

## downstream 에서 인코더 불러오기

```python
import torch
from holter_encoder.model import HolterEncoder
ck = torch.load("stage_b_mamba/encoder.pt"); c = ck["config"]
enc = HolterEncoder(d_model=c["d_model"], d_state=c["d_state"], depths=tuple(c["depths"]),
                    pool_factors=tuple(c["pool_factors"]), block=c["block"],
                    mid_block=c["mid_block"], mamba_d_state=c["mamba_d_state"])
enc.load_state_dict(ck["state_dict"], strict=False)   # attention pool 은 사전학습되지 않음
```

## downstream 평가

```bash
# 1) 인코더 고정 → record 임베딩
python -m holter_encoder.embed --splits $OUT/splits.csv --tokens $OUT/tokens_a \
    --encoder $RUN/stage_b_mamba/encoder.pt --out $RUN/stage_b_mamba/emb.npz --gpus 0

# 2) 선형 probe (백본 여러 개를 한 번에 비교 가능)
python -m holter_encoder.probe --splits $OUT/splits.csv --out $RUN/probe \
    --emb $RUN/stage_b_s4/emb.npz $RUN/stage_b_mamba/emb.npz --cv
```

세 가지 특징을 나란히 본다.

| 특징 | 뜻 |
|---|---|
| `demo` | 나이 + 평균HR + 성별. **교란 기준선** — 인코더는 이걸 넘어야 의미가 있다 |
| `enc` | 인코더 임베딩 |
| `enc+demo` | 둘 다. `demo` 대비 상승폭이 인코더의 순수 기여분 |

- record 는 한 환자 안에서 상관되므로 **환자 단위**로도 집계하고, 신뢰구간은 **환자 부트스트랩**으로 낸다.
- `--fractions` 로 train 라벨의 10%/25%/100% 결과를 함께 낸다. SSL 의 이점은 라벨이 적을 때 드러난다.
- `--cv` 는 SSL 이 보지 않은 val+test 환자만 모아 환자 단위 5-fold 로 평가한다.
  **LongQT 처럼 test 양성 환자가 14명뿐인 태스크는 단일 test 점추정이 크게 흔들리므로 이쪽을 근거로 삼는다**
  (합성 검증: 신호를 심어둔 태스크에서 단일 test 0.41 vs CV 0.741 [0.604-0.844]).
- `--features stem` 으로 돌리면 Stage A stem 만의 표현과 비교되어 Stage B backbone 의 기여를 분리할 수 있다.

## 주의

- time-of-day 는 `.json` 의 `hookup_time` 기준이다. `.hea` 시각은 MARS 내보내기 시각이라 쓰지 않는다.
- TOF 태스크는 나이·HR 만으로 AUROC 0.88 이 나온다 (`splits_summary.txt` 교란 기준선). 인코더 성능은 이 기준선 대비로 해석한다.
- LongQT test 양성 환자는 14명이다. 환자 단위 집계와 환자 부트스트랩 CI 로 보고한다.
