# Stage A 학습 속도: 병목을 찾아 없앤 과정

4×A6000 에서 SSL 사전학습(Stage A)이 **551 seg/s** 로 시작해 여러 단계를 거쳐 개선됐다.
매번 "느리다" 는 같았지만 원인은 네 번 모두 달랐다. 어떻게 원인을 좁혔고 무엇을 고쳤는지
남긴다. 같은 증상이 다시 나오면 이 순서대로 보면 된다.

측정 환경: RTX A6000 4장(48 GB), CPU 128코어, 데이터 `/home/coder/workspace/data/holter_v2`
(v2 h5 6,763개, 404 GB), 학습 대상 4,514 record.

---

## 한눈에 보기

| 시점 | 처리량 | GPU util | 무엇이 문제였나 | 어떻게 찾았나 |
|---|---|---|---|---|
| 시작 | 551 seg/s | 0~100% 들쭉날쭉 | 샘플마다 h5 를 열고 메타 1.3 MB 재독 | 메모리·util·처리량 조합 해석 |
| 1차 수정 후 | 2,896 seg/s | 10~14% | 여전히 IO 대기 | 동일 |
| 2차 수정 후 | 6,071 seg/s | 10~23% | record 여는 비용이 상한 | `io_bench` 의 `record-open/s` |
| 3차 수정 후 | 6,661 seg/s | 14~15% | **로더가 아니라 학습 루프** | `bench_loader` 로 로더만 분리 측정 |
| 4차 수정 후 | 서버 확인 대기 | — | `span_mask` 의 GPU 동기화 | 코드 검토 + 규모별 시간 측정 |

처리량 단위는 초당 10초 세그먼트 수. Stage A 는 GPU 당 배치 × `segs-per-record` 개를
한 스텝에 쓴다 (예: 64 × 32 = 2,048, 4 GPU 면 8,192).

---

## 1. 증상 읽는 법

첫 로그에서 세 숫자를 같이 봤다.

```
step 200/30000  rec 4.5187  beat 0.0289  551 seg/s
GPU 메모리 1452MiB / 49140MiB,  util 0% / 100% / 100% / 100%
```

- **메모리가 3%** → 배치가 작아서 느린 게 아니다. 모델도 GPU 를 거의 안 쓰고 있다.
- **util 이 GPU 마다 다름** → 한 GPU 가 데이터를 기다리면 DDP 가 모두를 세운다.
- **551 seg/s × 7.5 KB ≈ 4 MB/s** → 디스크 대역폭 문제로 보기엔 너무 작다.

결론: 신호를 읽는 양이 아니라 **읽는 방식**이 문제다.

---

## 2. 첫 번째 원인 — 샘플마다 h5 를 열었다

Stage A 는 학습마다 무작위 record 의 무작위 10초 구간을 쓴다. 당시 구현은
`__getitem__` 에서 record 를 고르고 h5 를 열었다. 열어둔 핸들 캐시는 64개인데 record 는
4,514개라 **적중률이 1.4%** 였다. 사실상 매 샘플마다

- h5 파일 열기
- `seg/quality` 읽기 (약 250 KB)
- beat 주석 읽기 (약 1 MB)

즉 **7.5 KB 를 쓰려고 1.3 MB 를 읽었다.**

### 고친 방법

`holter_encoder/prep_meta.py` 로 record 당 한 번만 계산해 사이드카 파일에 저장한다.

- 품질 마스크, 채널별 정규화 계수, 세그먼트별 beat 요약
- **`/signal` 의 파일 내 바이트 offset**

offset 을 저장했기 때문에 학습 중에는 **h5py 를 아예 열지 않고 numpy memmap 만** 쓴다.
더불어 `--segs-per-record` 로 한 번 연 record 에서 여러 구간을 뽑아 여는 비용을 나눠 갚게 했다.

결과: 551 → **2,896 seg/s**.

> 같이 넣은 것: 정규화 후 진폭 `--clip 20`. 일부 record 에 ±37 mV 같은 아티팩트가 있어
> 복원 손실이 3.7~7.5 로 출렁였는데, clip 이후 1.8~2.3 으로 안정됐다. 속도와는 무관하다.

---

## 3. 두 번째 원인 — 메타 파일을 여는 비용

여전히 util 10~14% 였다. 이번엔 추측 대신 저장소를 직접 쟀다 (`tools/io_bench.py`).

```
[랜덤 10초] threads   1   639 seg/s   639 record-open/s   4.8 MB/s
[랜덤 10초] threads   8   614 seg/s   614 record-open/s   4.6 MB/s
[랜덤 10초] threads  32   431 seg/s   431 record-open/s   3.2 MB/s
[연속 16개] threads   1  8723 seg/s   545 record-open/s  65.4 MB/s
[연속 16개] threads  64  8793 seg/s   550 record-open/s  65.9 MB/s
```

핵심은 **`record-open/s` 가 스레드 수와 무관하게 545~639 로 고정**이라는 점이다.
디스크 지연이 원인이면 동시 읽기를 늘릴 때 올라가야 한다. 전혀 안 올랐고 읽는 양도
초당 65 MB 뿐이었다. 즉 **디스크가 아니라 파일 하나를 여는 파이썬 비용**(`np.load` 의
zip 파싱)이 상한이었고, GIL 때문에 스레드로도 풀리지 않았다.

> 주의: 같은 출력의 "순차 4~7 GB/s" 는 디스크 성능이 아니다. 같은 record 를 반복해서
> 읽어 **페이지 캐시**에 올라간 값이다. 콜드 값은 첫 줄의 37 MB/s 에 가깝다.

### 고친 방법

record 별 메타를 **세 개의 큰 파일로 묶었다** (`prep_meta --pack`).

```
pack_index.npz   record → 행 위치, offset, shape, scale, norm …
pack_valid.npy   모든 record 의 품질 마스크를 이어붙인 것
pack_beats.npy   모든 record 의 세그먼트별 beat 요약
```

워커는 시작할 때 한 번만 memmap 하고, 이후 record 열기는 **배열 슬라이스**다.
페이지 캐시도 워커끼리 공유된다.

결과: record 여는 속도 3,435 → **43,334/s** (로컬 측정). 저장소 측정도 달라졌다.

```
[랜덤 10초] threads 1   7,202 seg/s    54 MB/s
[연속 16개] threads 8  59,950 seg/s   450 MB/s
```

디스크는 필요치(8,192 seg/s)의 5~7배를 낼 수 있었다. 학습은 6,071 → **6,661 seg/s**.

### 같이 넣은 것

- **연속 블록 읽기**: 한 record 에서 K 개를 흩어 읽는 대신 연속 구간을 한 번에 읽는다.
  읽기 횟수가 1/K 이 된다 (로컬: 랜덤 3,113 → 연속 31,221 seg/s).
- **int16 그대로 전송**: 워커는 원값과 채널별 계수만 보내고 정규화·clip 은 GPU 가 한다.
  전송량 절반, 변환 연산도 CPU → GPU.
- **스레드 과구독 방지**: 프로세스마다 numpy/OpenBLAS 가 스레드를 띄우지 않게 제한하고,
  시작할 때 `CPU N코어 / 프로세스 M개` 를 출력해 초과 시 경고한다.

---

## 4. 세 번째 원인 — 로더가 아니라 학습 루프였다

CPU 는 128코어에 프로세스 52개로 여유가 있는데도 6,661 seg/s, util 15% 였다.
이번엔 **로더만 떼어** 쟀다 (`tools/bench_loader.py`).

```
[1] 아이템 하나 구성요소별
    record 추첨 0.05 ms / record 열기 0.06 ms / 신호 32개 읽기 1.56 ms / beat 0.15 ms
[3] DataLoader
    workers  0    48,451 seg/s
    workers  4   166,515 seg/s
    workers 12   395,891 seg/s
    workers 32   259,567 seg/s
```

**로더만으로 395,891 seg/s.** 학습에서 본 6,661 seg/s 의 60배다. 데이터는 넘치게
공급되고 있었다. 병목은 학습 루프 안에 있었다.

> 부수 수확: 워커 32개는 12개보다 느리다(259,567 vs 395,891). 워커를 무작정 늘리면
> 손해다.

### 진짜 원인: 숨은 GPU 동기화

마스킹 함수가 이렇게 생겼었다.

```python
while int(mask[b].sum()) < target:      # ← 매 반복 GPU → CPU 동기화
    ...
    mask[b, st:st + span] = True        # ← 작은 CUDA 연산
```

`int(tensor.sum())` 은 값을 읽기 위해 **GPU 가 끝날 때까지 기다린다.** Stage A 는
GPU 당 배치가 2,048 이고 샘플마다 구간이 5~6개라 **스텝당 1만 번 넘게 GPU 를 세웠다.**

### 고친 방법

시작점과 길이를 한 번에 뽑고 `+1/-1` 누적합으로 구간을 칠한다. 파이썬 반복도 동기화도 없다.
구간 수는 겹침을 감안해 기대 커버리지가 목표 비율이 되도록 정한다.

```
n = log(1 - ratio) / log(1 - E[len] / L)
```

검증: Stage A 규모(2048×1250)에서 CPU 77.8 → **1.3 ms**, 가림 비율 0.301(목표 0.30),
연속 구간 형태·`valid` 구간 보존·재현성 확인. **서버에서의 최종 처리량은 확인 대기 중이다.**

---

## 5. 진단 도구

| 도구 | 답하는 질문 |
|---|---|
| `tools/io_bench.py` | 저장소가 초당 몇 세그먼트를 줄 수 있나. 지연이 문제인가 대역폭이 문제인가 |
| `tools/bench_loader.py` | 로더가 GPU 를 먹여 살릴 수 있나. 아이템 하나에 무엇이 비싼가 |
| 학습 로그 첫 줄들 | 스텝당 세그먼트 수, DataLoader 재시작 주기, CPU 코어 대비 프로세스 수 |
| `nvidia-smi` | 메모리와 util. 둘 다 낮으면 GPU 밖이 문제 |

---

## 6. 다음에 느릴 때 볼 순서

1. **메모리와 util 을 같이 본다.** 둘 다 낮으면 GPU 밖(입력 또는 호스트 코드)이다.
2. **저장소를 잰다** (`io_bench`). `record-open/s` 가 동시성과 무관하면 여는 비용,
   `MB/s` 가 천장이면 대역폭이다. 순차 수치는 페이지 캐시를 의심한다.
3. **로더만 잰다** (`bench_loader`). 로더가 충분히 빠르면 범인은 학습 루프다.
4. **학습 루프에서 동기화를 찾는다.** `.item()`, `int(tensor)`, `float(tensor)`,
   `.cpu()`, `print(tensor)`, `tensor` 를 조건문에 쓰기 — 전부 GPU 를 세운다.
   배치 안의 샘플마다 도는 파이썬 반복문도 같은 이유로 위험하다.
5. **그래도 낮으면** `torch.profiler` 로 스텝을 쪼갠다.

---

## 7. 현재 권장 설정

```bash
python -m holter_encoder.prep_meta --splits $OUT/splits.csv --out $OUT/segmeta --workers 32

torchrun --nproc_per_node 4 -m holter_encoder.train_stage_a \
    --splits $OUT/splits.csv --out $RUN/stage_a --meta-dir $OUT/segmeta --gpus 0,1,2,3 \
    --batch 64 --segs-per-record 32 --workers 12
```

- `--meta-dir` 를 빠뜨리면 1번 문제로 되돌아간다 (학습 시작 시 경고가 뜬다).
- `--workers` 는 12 근처가 최적이었다. 늘린다고 좋아지지 않는다.
- `--batch` 는 record 수다. 실제 세그먼트 수는 `batch × segs-per-record`.
