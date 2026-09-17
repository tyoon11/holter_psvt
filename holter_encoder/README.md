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

## 백본 선택

| `--block` | 특성 |
|---|---|
| `s4` | S4D. 시간 불변 필터. 순수 PyTorch, 설치 불필요 |
| `mamba` | 양방향 선택적 SSM. Δ·B·C 가 입력에 따라 변해 드문 사건 구간을 선택적으로 유지. `mamba-ssm` 필요 |
| `--mid-block attn` | 10분 해상도(144 토큰)를 self-attention 으로. attention 가중치로 참고한 시간대를 볼 수 있음 |

Mamba 사용 전 확인:

```bash
pip install causal-conv1d mamba-ssm --no-build-isolation
python -m holter_encoder.check_mamba      # 설치, 참조 구현과의 일치, 속도
```

`mamba-ssm` 이 없으면 순수 PyTorch 참조 구현으로 동작하지만 24h 길이에서는 매우 느리다.

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

## 주의

- time-of-day 는 `.json` 의 `hookup_time` 기준이다. `.hea` 시각은 MARS 내보내기 시각이라 쓰지 않는다.
- TOF 태스크는 나이·HR 만으로 AUROC 0.88 이 나온다 (`splits_summary.txt` 교란 기준선). 인코더 성능은 이 기준선 대비로 해석한다.
- LongQT test 양성 환자는 14명이다. 환자 단위 집계와 환자 부트스트랩 CI 로 보고한다.
