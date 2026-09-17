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

from .mamba import HAS_MAMBA_SSM, MambaRef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default=None, help='쓸 GPU 번호, 예: "0"')
    a = ap.parse_args()
    if a.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpus
    if not torch.cuda.is_available():
        print("CUDA 없음 — GPU 서버에서 실행하세요."); return
    if not HAS_MAMBA_SSM:
        print("mamba_ssm import 실패.\n  pip install causal-conv1d mamba-ssm --no-build-isolation\n"
              "  (torch 와 CUDA 버전에 맞는 wheel 이 필요합니다. 실패 시 로그를 확인)"); return
    from mamba_ssm import Mamba
    dev = torch.device("cuda")
    torch.manual_seed(0)
    fast = Mamba(d_model=256, d_state=16, d_conv=4, expand=2).to(dev).float()
    ref = MambaRef(256, 16, 4, 2).to(dev).float()
    fn = {k: tuple(v.shape) for k, v in fast.state_dict().items()}
    rn = {k: tuple(v.shape) for k, v in ref.state_dict().items()}
    print(f"[이름·모양] {'일치' if fn == rn else '불일치'}")
    if fn != rn:
        print("  mamba_ssm 에만:", sorted(set(fn) - set(rn)))
        print("  참조에만     :", sorted(set(rn) - set(fn)))
        print("  모양 다름    :", [k for k in fn if k in rn and fn[k] != rn[k]])
        return
    ref.load_state_dict(fast.state_dict())
    x = torch.randn(2, 512, 256, device=dev)
    with torch.no_grad():
        d = (fast(x) - ref(x)).abs().max().item()
    print(f"[출력] 최대 차이 {d:.2e}  {'OK' if d < 1e-3 else '확인 필요'}")
    x = torch.randn(4, 8640, 256, device=dev)
    for name, m in (("mamba_ssm", fast), ("참조", ref)):
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            m(x)
        torch.cuda.synchronize()
        print(f"[속도] {name:<9s} B=4 L=8640 forward {time.time() - t:.2f}s")


if __name__ == "__main__":
    main()
