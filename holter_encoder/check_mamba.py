# -*- coding: utf-8 -*-
"""
check_mamba.py — 서버(GPU)에서 mamba_ssm 설치와 참조 구현 일치를 확인한다.

  python -m holter_encoder.check_mamba

  1) mamba_ssm import / CUDA 커널 동작
  2) 참조 구현(MambaRef)과 파라미터 이름·모양이 같은지
  3) 같은 가중치에서 두 구현의 출력이 같은지 (float32)
  4) L=8640 (24h 토큰) 에서 속도 비교
"""

import argparse
import os
import time

import torch

from .mamba import HAS_MAMBA_SSM, MAMBA_SSM_SOURCE, MAMBA_SSM_VERSION, MambaRef, make_mamba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default=None, help='쓸 GPU 번호, 예: "0"')
    a = ap.parse_args()
    if a.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpus
    if not torch.cuda.is_available():
        print("CUDA 없음 — GPU 서버에서 실행하세요."); return
    if not HAS_MAMBA_SSM:
        print("mamba_ssm import 실패. 설치 방법은 holter_encoder/README.md 참고.\n"
              "  사내 프록시면 CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE 로 직접 컴파일"); return
    print(f"[import] {MAMBA_SSM_SOURCE}  (mamba_ssm {MAMBA_SSM_VERSION}, torch {torch.__version__})")
    dev = torch.device("cuda")
    torch.manual_seed(0)
    fast = make_mamba(256, 16, 4, 2).to(dev).float()
    if isinstance(fast, MambaRef):
        print("  ** 빠른 커널을 못 잡았습니다 (참조 구현 반환) **"); return
    ref = MambaRef(256, 16, 4, 2).to(dev).float()
    fn = {k: tuple(v.shape) for k, v in fast.state_dict().items()}
    rn = {k: tuple(v.shape) for k, v in ref.state_dict().items()}
    print(f"[이름·모양] {'일치' if fn == rn else '불일치'}")
    if fn != rn:
        print("  mamba_ssm 에만:", sorted(set(fn) - set(rn)))
        print("  참조에만     :", sorted(set(rn) - set(fn)))
        print("  모양 다름    :", [k for k in fn if k in rn and fn[k] != rn[k]])
        print("  → 이름이 다르면 CPU 참조 구현과 체크포인트를 주고받을 수 없습니다. "
              "GPU 에서만 쓰면 학습에는 지장 없습니다.")
        return
    ref.load_state_dict(fast.state_dict())
    x = torch.randn(2, 512, 256, device=dev)
    with torch.no_grad():
        d = (fast(x) - ref(x)).abs().max().item()
    print(f"[출력] 최대 차이 {d:.2e}  {'OK' if d < 1e-3 else '확인 필요'}")
    # 속도·메모리: 학습에서 중요한 것은 backward 와 peak memory 다.
    # 첫 호출에는 커널 초기화가 섞이므로 워밍업 후 여러 번 잰다.
    def bench(m, B=4, L=8640, reps=3, backward=True):
        x = torch.randn(B, L, 256, device=dev, requires_grad=backward)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):                                  # 워밍업
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = m(x)
            if backward:
                y.float().pow(2).mean().backward()
            m.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(reps):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = m(x)
            if backward:
                y.float().pow(2).mean().backward()
            m.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return (time.time() - t) / reps, torch.cuda.max_memory_allocated() / 2**30

    print("[속도·메모리] B=4, L=8640 (24h), d_model=256, bf16")
    for name, m, bw in (("mamba_ssm fwd", fast, False), ("참조     fwd", ref, False),
                        ("mamba_ssm fwd+bwd", fast, True), ("참조     fwd+bwd", ref, True)):
        try:
            sec, mem = bench(m, backward=bw)
            print(f"  {name:<20s} {sec*1000:8.0f} ms   peak {mem:5.1f} GiB")
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  {name:<20s}   OOM — 이 구성으로는 학습 불가")
    print("\n참조 구현은 (B, L, d_inner, d_state) 중간 텐서를 통째로 들고 있어 backward 에서 메모리가 커진다.\n"
          "위 fwd+bwd 수치를 보고 --block mamba (전 해상도) 를 쓸지, --mid-block mamba (10분 해상도만) 로\n"
          "갈지 정한다. 층이 여러 개 쌓이면 실제 학습 메모리는 이보다 더 든다.")


if __name__ == "__main__":
    main()
