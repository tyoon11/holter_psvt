# -*- coding: utf-8 -*-
"""
mamba.py — 양방향 Mamba 블록과 저해상도 attention 블록 (HierarchicalS4 의 교체 블록).

왜 Mamba 인가
  S4D 는 시간 불변(LTI)이라 모든 위치에 같은 필터를 적용한다. Mamba 는 Δ(갱신량), B, C 가
  입력에 따라 달라지는 선택적 SSM 이라, 24시간 중 드물게 나타나는 PSVT run·이소성 박동처럼
  "중요한 구간은 붙잡고 나머지는 흘려보내는" 동작을 내용 기반으로 할 수 있다.
  백본은 원파형이 아니라 10초 토큰을 다루므로 선택성이 유리한 영역이다.
  (원파형 수준에서는 LTI 가 나을 수 있어 CNN stem 은 그대로 둔다.)

구현
  - CUDA 에서 mamba_ssm 이 설치돼 있으면 mamba_ssm.Mamba (선택적 스캔 CUDA 커널)를 쓴다.
  - 없으면 같은 파라미터 이름/모양의 순수 PyTorch 참조 구현(순차 스캔)을 쓴다.
    CPU 테스트·정합성 확인용이며 L=8640 학습에는 느리다.
    가중치 이름이 같아 두 구현 사이에 state_dict 를 그대로 옮길 수 있다.
  - Mamba 는 인과적이므로 정방향 + 뒤집은 역방향 두 믹서를 더해 양방향으로 쓴다.

MidAttentionBlock
  계층형 U-Net 의 최저 해상도(기본 10분, 24h=144 토큰)에 두는 self-attention.
  144² 라 비용이 작고, attention 가중치로 "어느 시간대를 참고했는지" 직접 볼 수 있다.
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# mamba_ssm 은 버전에 따라 Mamba(v1) 의 위치가 달라진다. 하나만 보고 조용히 느린 참조
# 구현으로 떨어지지 않도록 후보 경로를 모두 시도하고, 무엇을 썼는지 남긴다.
_FastMamba, MAMBA_SSM_SOURCE = None, None
for _mod, _name in (("mamba_ssm", "Mamba"),
                    ("mamba_ssm.modules.mamba_simple", "Mamba"),
                    ("mamba_ssm.modules.mamba2", "Mamba2")):
    try:
        _FastMamba = getattr(__import__(_mod, fromlist=[_name]), _name)
        MAMBA_SSM_SOURCE = f"{_mod}.{_name}"
        break
    except Exception:                                  # ImportError, CUDA 커널 로드 실패 등
        continue
HAS_MAMBA_SSM = _FastMamba is not None
MAMBA_SSM_VERSION = None
try:
    import mamba_ssm as _ms
    MAMBA_SSM_VERSION = getattr(_ms, "__version__", "?")
except Exception:
    pass

_warned = False


class MambaRef(nn.Module):
    """mamba_ssm.Mamba(v1) 과 파라미터 이름·모양이 같은 순수 PyTorch 구현. 입력 (B, L, d)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 dt_min=1e-3, dt_max=1e-1, dt_init_floor=1e-4):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # Δ 초기화: softplus(bias) 가 [dt_min, dt_max] 로그균등이 되게
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        for p in (self.A_log, self.D):
            p._optim = {"weight_decay": 0.0}

    def forward(self, h, return_dt=False):
        B, L, _ = h.shape
        x, z = self.in_proj(h).chunk(2, dim=-1)
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :L].transpose(1, 2))
        dt, Bm, Cm = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                             # (B, L, Di)
        A = -torch.exp(self.A_log.float())                            # (Di, N)
        xf, dtf, Bf, Cf = x.float(), dt.float(), Bm.float(), Cm.float()
        dA = torch.exp(dtf.unsqueeze(-1) * A)                         # (B, L, Di, N)
        dBx = dtf.unsqueeze(-1) * Bf.unsqueeze(2) * xf.unsqueeze(-1)   # (B, L, Di, N)
        s = torch.zeros(B, self.d_inner, self.d_state, device=h.device, dtype=torch.float32)
        ys = []
        for t in range(L):
            s = dA[:, t] * s + dBx[:, t]
            ys.append((s * Cf[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, 1) + xf * self.D.float()
        y = (y * F.silu(z.float())).to(h.dtype)
        out = self.out_proj(y)
        return (out, dt) if return_dt else out


def make_mamba(d_model, d_state=16, d_conv=4, expand=2, prefer_fast=True):
    """CUDA + mamba_ssm 이면 빠른 커널, 아니면 참조 구현."""
    global _warned
    if prefer_fast and HAS_MAMBA_SSM and torch.cuda.is_available():
        return _FastMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    if not _warned and torch.cuda.is_available():
        why = ("mamba_ssm import 실패" if not HAS_MAMBA_SSM else "prefer_fast=False")
        warnings.warn(f"순수 PyTorch 참조 Mamba 를 사용합니다 ({why}). L=8640 학습은 매우 느립니다. "
                      "설치 확인: python -m holter_encoder.check_mamba")
        _warned = True
    return MambaRef(d_model, d_state, d_conv, expand)


class BiMamba(nn.Module):
    """정방향 + 역방향(뒤집어 넣고 다시 뒤집음) Mamba 의 합. 입출력 (B, L, d)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, prefer_fast=True):
        super().__init__()
        self.fwd = make_mamba(d_model, d_state, d_conv, expand, prefer_fast)
        self.bwd = make_mamba(d_model, d_state, d_conv, expand, prefer_fast)

    def forward(self, h):
        return self.fwd(h) + self.bwd(h.flip(1)).flip(1)


class MambaBlock(nn.Module):
    """S4Block 과 같은 틀: x + Mixer(Norm(x)), x + FF(Norm(x)). 입출력 (B, H, L)."""

    def __init__(self, d_model, d_state=16, dropout=0.1, ff_mult=2, expand=2, prefer_fast=True, **_):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = BiMamba(d_model, d_state, expand=expand, prefer_fast=prefer_fast)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_mult * d_model), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(ff_mult * d_model, d_model))
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = h + self.drop1(self.mixer(self.norm1(h)))
        h = h + self.drop2(self.ff(self.norm2(h)))
        return h.transpose(1, 2)


class MidAttentionBlock(nn.Module):
    """저해상도 self-attention 블록. 입출력 (B, H, L). 마지막 attention 가중치를 보관한다."""

    def __init__(self, d_model, n_heads=8, dropout=0.1, ff_mult=2, **_):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_mult * d_model), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(ff_mult * d_model, d_model))
        self.drop = nn.Dropout(dropout)
        self.last_weights = None

    def forward(self, x):
        h = x.transpose(1, 2)
        q = self.norm1(h)
        a, w = self.attn(q, q, q, need_weights=not self.training, average_attn_weights=True)
        if w is not None:
            self.last_weights = w.detach()
        h = h + self.drop(a)
        h = h + self.drop(self.ff(self.norm2(h)))
        return h.transpose(1, 2)
