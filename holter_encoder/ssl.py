# -*- coding: utf-8 -*-
"""
ssl.py — 자기지도 학습 모델/손실.

Stage A (StageAModel): 10초 세그먼트 단위 stem 사전학습
  - 입력 파형의 구간 30% 를 0 으로 가리고(0.2~1초 구간들) stem 프레임으로 원파형 복원
    손실은 가린 구간 가중 1.0, 나머지 0.1
  - 보조: stem 토큰으로 구간 beat 요약(정상/상심실성/심실성/전체 박동 수의 log1p, 심박수)
    회귀. beat 주석(.ANN)이 없는 record 는 마스크로 제외
    주석 기반 보조 과제가 수렴을 빠르게 하고 부정맥 관련 표현을 끌어올린다.

Stage B (StageBModel): 캐시 토큰 시퀀스에 대한 계층형 S4 사전학습
  - 1~6분(6~36 토큰) 구간들로 토큰 15% 를 학습 가능한 mask 토큰으로 바꾸고,
    backbone 출력으로 원래 stem 토큰(layer norm 정규화, stop-grad)을 회귀 (smooth L1)
  - 타깃이 고정된 Stage A stem 의 출력이라 표현 붕괴(collapse)가 일어나지 않는다
  - 패딩과 품질 불량 세그먼트는 손실에서 제외
  - 파라미터 이름(backbone., tod.)은 model.HolterEncoder 와 같아 그대로 옮겨 쓸 수 있다
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_BEAT_TARGETS
from .model import BeatCNNStem, HierarchicalS4, TimeOfDayEmbedding


def span_mask(B, L, ratio, span_min, span_max, device, valid=None, generator=None):
    """(B, L) bool, True = 가림. 구간 여러 개를 겹쳐 대략 ratio 만큼 덮는다.

    예전 구현은 샘플마다 while 문으로 구간을 추가하면서 int(mask[b].sum()) 으로 진행도를
    확인했다. 이 한 줄이 반복마다 GPU→CPU 동기화를 일으켜, 배치 2,048 이면 스텝당 1만 번
    동기화가 발생했다 (서버에서 로더는 395,891 seg/s 를 내는데 학습은 6,661 seg/s).
    지금은 시작점·길이를 한 번에 뽑고 +1/-1 누적합으로 구간을 칠한다. 동기화도 파이썬
    반복도 없다.

    구간 수는 겹침을 감안해 기대 커버리지가 ratio 가 되도록 정한다:
      coverage = 1 - (1 - E[len]/L)^n  →  n = log(1-ratio) / log(1 - E[len]/L)
    """
    mean_len = (span_min + span_max) / 2.0
    q = max(1e-6, min(1 - 1e-6, mean_len / L))
    n_spans = max(1, int(math.ceil(math.log(max(1e-6, 1 - ratio)) / math.log(1 - q))))

    dev = device if (generator is None or generator.device.type == device.type) else torch.device("cpu")
    starts = torch.randint(0, L, (B, n_spans), device=dev, generator=generator)
    lens = torch.randint(span_min, span_max + 1, (B, n_spans), device=dev, generator=generator)
    ends = (starts + lens).clamp_(max=L)
    diff = torch.zeros(B, L + 1, dtype=torch.int32, device=dev)
    ones = torch.ones_like(starts, dtype=torch.int32)
    diff.scatter_add_(1, starts, ones)
    diff.scatter_add_(1, ends, -ones)
    mask = diff.cumsum(1)[:, :L] > 0
    mask = mask.to(device)
    if valid is not None:
        mask &= valid
    return mask


class StemDecoder(nn.Module):
    """stem 프레임 (N, d, 10) → 파형 (N, C, 1250). 5배씩 세 번 업샘플."""

    def __init__(self, d_model, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(d_model, 128, 5, stride=5), nn.GELU(),
            nn.ConvTranspose1d(128, 64, 5, stride=5), nn.GELU(),
            nn.ConvTranspose1d(64, 32, 5, stride=5), nn.GELU(),
            nn.Conv1d(32, out_channels, 7, padding=3),
        )

    def forward(self, frames):
        return self.net(frames)


class StageAModel(nn.Module):
    def __init__(self, in_channels=3, d_model=256, dropout=0.0):
        super().__init__()
        self.stem = BeatCNNStem(in_channels, d_model, dropout=dropout)
        self.decoder = StemDecoder(d_model, in_channels)
        self.beat_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                       nn.Linear(d_model, N_BEAT_TARGETS))

    def forward(self, x, beat, beat_mask, mask_ratio=0.3, span=(25, 125), generator=None):
        N, C, L = x.shape
        m = span_mask(N, L, mask_ratio, span[0], span[1], x.device, generator=generator)
        x_in = x.masked_fill(m[:, None, :], 0.0)
        tok, frames = self.stem(x_in)
        rec = self.decoder(frames)[..., :L]
        w = torch.where(m, 1.0, 0.1)[:, None, :]
        loss_rec = ((rec.float() - x) ** 2 * w).sum() / (w.sum() * C)
        pred = self.beat_head(tok).float()
        per = F.smooth_l1_loss(pred, beat, reduction="none").mean(1)
        loss_beat = (per * beat_mask).sum() / beat_mask.sum().clamp(min=1.0)
        return loss_rec, loss_beat


class StageBModel(nn.Module):
    def __init__(self, d_model=256, d_state=64, depths=(2, 4, 4, 2, 2),
                 pool_factors=(6, 10), dropout=0.1, block="s4", mid_block=None, mamba_d_state=16):
        super().__init__()
        self.backbone = HierarchicalS4(d_model, d_state, depths, pool_factors, dropout,
                                       block=block, mid_block=mid_block, mamba_d_state=mamba_d_state)
        self.tod = TimeOfDayEmbedding(d_model)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def encode(self, tokens, tod):
        x = tokens + self.tod(tod)
        out, _ = self.backbone(x.transpose(1, 2))
        return out.transpose(1, 2)

    def forward(self, tokens, tod, pad, seg_valid, mask_ratio=0.15, span=(6, 36), generator=None):
        B, L, d = tokens.shape
        m = span_mask(B, L, mask_ratio, span[0], span[1], tokens.device,
                      valid=~pad, generator=generator)
        x = torch.where(m.unsqueeze(-1), self.mask_token.to(tokens.dtype), tokens)
        x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        out = self.encode(x, tod)
        pred = self.head(out).float()
        target = F.layer_norm(tokens.float(), (d,)).detach()
        use = m & seg_valid
        n = use.sum().clamp(min=1)
        loss = F.smooth_l1_loss(pred[use], target[use], reduction="sum") / (n * d)
        return loss, int(use.sum())
