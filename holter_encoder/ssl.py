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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_BEAT_TARGETS
from .model import BeatCNNStem, HierarchicalS4, TimeOfDayEmbedding


def span_mask(B, L, ratio, span_min, span_max, device, valid=None, generator=None):
    """(B, L) bool, True = 가림. valid(True=가릴 수 있음) 안에서 ratio 만큼 구간으로 채운다."""
    mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    for b in range(B):
        n_valid = int(valid[b].sum()) if valid is not None else L
        target = int(round(ratio * n_valid))
        tries = 0
        while int(mask[b].sum()) < target and tries < 1000:
            tries += 1
            span = int(torch.randint(span_min, span_max + 1, (1,), generator=generator))
            st = int(torch.randint(0, max(1, L - span + 1), (1,), generator=generator))
            mask[b, st:st + span] = True
        if valid is not None:
            mask[b] &= valid[b]
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
