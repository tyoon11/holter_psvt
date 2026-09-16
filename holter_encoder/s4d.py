# -*- coding: utf-8 -*-
"""
s4d.py — S4D (diagonal state space) 레이어, 순수 PyTorch 구현.

병원 폐쇄망에서 mamba-ssm / causal-conv1d 같은 nvcc 컴파일 의존성을 피하기 위해
커스텀 CUDA 커널 없이 FFT convolution만으로 구현했다. torch 외 의존성 없음.

수식 (Gu et al., "On the Parameterization and Initialization of Diagonal SSMs"):
    연속 상태공간   x'(t) = A x(t) + B u(t),  y(t) = C x(t) + D u(t)
    A 는 대각(diagonal) 이고 켤레쌍이므로 절반(N/2)만 저장한 뒤 2*Re(...) 로 복원한다.
    ZOH 이산화:  Abar = exp(dt*A),  Bbar = (exp(dt*A) - 1) / A * B
    convolution kernel:  K[l] = 2 * Re( sum_n C_n * Bbar_n * Abar_n^l )
    B 는 1로 흡수하고 C 에 학습을 맡긴다 (S4D 표준).

핵심 성질: kernel 길이 L 을 한 번에 만들어 FFT 로 곱하므로 시퀀스 길이에 대해
O(L log L). 8640 토큰(24h @ 10초)은 물론 43200 토큰도 문제없다.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# S4D kernel
# -----------------------------------------------------------------------------
class S4DKernel(nn.Module):
    """길이 L 의 convolution kernel (H, L) 을 생성한다.

    Args:
        d_model: 채널 수 H. 채널마다 독립적인 SSM 을 둔다(depthwise).
        d_state: 상태 차원 N. 켤레쌍이므로 실제 파라미터는 N//2 개.
        dt_min/dt_max: 이산화 스텝 dt 의 초기 로그균등 분포 범위.
            dt 가 작을수록 긴 시간상수(= 더 먼 과거를 기억)를 갖는다.
    """

    def __init__(self, d_model, d_state=64, dt_min=1e-3, dt_max=1e-1, lr=None):
        super().__init__()
        H, N = d_model, d_state // 2

        # dt: 채널마다 로그균등 초기화 → 다양한 시간 스케일을 동시에 커버
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)

        # S4D-Lin 초기화: A = -1/2 + i*pi*n
        #   실수부는 감쇠(안정성), 허수부는 진동 주파수를 담당한다.
        A_real = torch.full((H, N), 0.5)  # 항상 exp() 를 씌워 양수로 유지 → Re(A) < 0 보장
        A_imag = math.pi * torch.arange(N).repeat(H, 1).float()

        # C: 복소수 (H, N). 실수 2채널로 저장해 optimizer 호환성을 확보한다.
        C = torch.randn(H, N, 2) * (0.5 ** 0.5)

        self.register("log_dt", log_dt, lr)
        self.register("log_A_real", torch.log(A_real), lr)
        self.register("A_imag", A_imag, lr)
        self.C = nn.Parameter(C)

    def register(self, name, tensor, lr=None):
        """SSM 내부 파라미터(dt, A)는 weight decay 를 걸면 안 되고 보통 낮은 LR 을 쓴다.
        optimizer 가 param_group 을 나눌 수 있도록 _optim 메타데이터를 붙여둔다."""
        self.register_parameter(name, nn.Parameter(tensor))
        optim = {"weight_decay": 0.0}
        if lr is not None:
            optim["lr"] = lr
        setattr(getattr(self, name), "_optim", optim)

    def forward(self, L):
        dt = torch.exp(self.log_dt)                       # (H,)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N), Re(A) < 0
        C = torch.view_as_complex(self.C.float())         # (H, N)

        dtA = A * dt.unsqueeze(-1)                        # (H, N)
        # Bbar 를 C 에 흡수: C * (exp(dtA) - 1) / A
        Cb = C * (torch.exp(dtA) - 1.0) / A               # (H, N)
        # Abar^l = exp(dtA * l)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)   # (H, N, L)
        K = 2.0 * torch.einsum("hn,hnl->hl", Cb, torch.exp(K)).real
        return K                                          # (H, L)


# -----------------------------------------------------------------------------
# S4D layer (단방향 / 양방향)
# -----------------------------------------------------------------------------
class S4D(nn.Module):
    """입력 (B, H, L) → 출력 (B, H, L).

    bidirectional=True 면 정방향/역방향 커널을 각각 두고 하나의 비인과(non-causal)
    커널로 합친다. 인코더이므로 미래를 봐도 되며, 실제로 부정맥 에피소드의 시작점을
    잡으려면 뒤쪽 문맥이 필요하다.
    """

    def __init__(self, d_model, d_state=64, dropout=0.0, bidirectional=True,
                 transposed=True, **kernel_args):
        super().__init__()
        self.h = d_model
        self.bidirectional = bidirectional
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(d_model))  # skip(=feedthrough) 항
        channels = 2 if bidirectional else 1
        self.kernel = nn.ModuleList(
            [S4DKernel(d_model, d_state=d_state, **kernel_args) for _ in range(channels)]
        )

        self.activation = nn.GELU()
        self.dropout = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()
        # GLU: 출력 게이팅. S4 원논문 구성과 동일하며 학습 안정성에 기여한다.
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        k0 = self.kernel[0](L)                       # (H, L)
        if self.bidirectional:
            k1 = self.kernel[1](L)
            # 비인과 커널 구성: [정방향 k0 | 0] + [0 | 뒤집은 k1]
            #   → 길이 2L 원형(circular) 컨볼루션에서 음수 lag 를 담당한다.
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))
            n_fft = 2 * L
        else:
            k = k0
            n_fft = 2 * L

        k_f = torch.fft.rfft(k.float(), n=n_fft)                 # (H, n_fft//2+1)
        u_f = torch.fft.rfft(u.float(), n=n_fft)                 # (B, H, n_fft//2+1)
        y = torch.fft.irfft(u_f * k_f, n=n_fft)[..., :L]         # (B, H, L)

        y = y + u * self.D.unsqueeze(-1)
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y


# -----------------------------------------------------------------------------
# Residual block
# -----------------------------------------------------------------------------
class S4Block(nn.Module):
    """pre-norm residual 블록: x + FF(S4D(Norm(x))).

    입출력 모두 (B, H, L). LayerNorm 은 채널 축에 걸어야 하므로 transpose 한다.
    """

    def __init__(self, d_model, d_state=64, dropout=0.1, bidirectional=True,
                 ff_mult=2, **kernel_args):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.s4 = S4D(d_model, d_state=d_state, dropout=dropout,
                      bidirectional=bidirectional, **kernel_args)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, H, L)
        z = self.norm1(x.transpose(-1, -2)).transpose(-1, -2)
        x = x + self.drop1(self.s4(z))

        z = self.norm2(x.transpose(-1, -2))
        x = x + self.drop2(self.ff(z)).transpose(-1, -2)
        return x


def s4_param_groups(model, ssm_lr=1e-3):
    """S4 내부 파라미터(dt, A)는 낮은 LR + weight decay 0 으로 분리한다.
    이걸 안 하면 학습이 불안정해지거나 장기 기억이 무너진다 (S4 논문 권장사항)."""
    ssm, normal = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (ssm if hasattr(p, "_optim") else normal).append(p)
    return [
        {"params": normal},
        {"params": ssm, "lr": ssm_lr, "weight_decay": 0.0},
    ]
