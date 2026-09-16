# -*- coding: utf-8 -*-
"""
model.py — 24시간 Holter 인코더.

3단 계층 구조:
  (1) BeatCNNStem      10초 세그먼트 내부(1250 sample) → 프레임 10개 → 토큰 1개
  (2) HierarchicalS4   토큰 8640개(=24h) 를 10초 / 1분 / 10분 해상도로 U-Net 처리
  (3) AttentionPool    토큰 시퀀스 → record 임베딩 1개

출력은 두 가지이며 용도가 다르다.
  - tokens  (B, S, d): 에피소드 localization, 토큰 단위 SSL/downstream
  - record  (B, d)   : record 단위 분류(PSVT 등), retrieval

설계 근거:
  - S4 를 원신호(10.8M step)에 직접 걸지 않는다. CNN 으로 1250:1 토큰화한 뒤 건다.
  - local/global 을 별도 브랜치로 두지 않고 해상도 피라미드 하나로 처리한다.
    L0(10초)=QRS 형태·연속 박동, L1(1분)=PSVT run·AF 에피소드, L2(10분)=circadian.
  - skip connection 으로 국소 정보가 저해상도 단계를 거치며 소실되지 않게 한다.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .s4d import S4Block


# =============================================================================
# (1) 세그먼트 내부 인코더
# =============================================================================
class ResBlock1D(nn.Module):
    """stride 로 시간축을 줄이는 pre-activation residual block."""

    def __init__(self, c_in, c_out, kernel_size=7, stride=2, dropout=0.0):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.short = (
            nn.Identity() if (c_in == c_out and stride == 1)
            else nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                               nn.BatchNorm1d(c_out))
        )

    def forward(self, x):
        y = F.gelu(self.bn1(self.conv1(x)))
        y = self.drop(y)
        y = self.bn2(self.conv2(y))
        return F.gelu(y + self.short(x))


class BeatCNNStem(nn.Module):
    """(B*S, C, seg_len) → frames (B*S, d, F), token (B*S, d).

    stride 2 를 7번 적용해 1250 → 10 프레임(약 1초/프레임)으로 줄인다.
    프레임은 local head(파형 복원, beat 예측)가 쓰고, attention pool 한 토큰은
    global backbone 이 쓴다.
    """

    def __init__(self, in_channels=3, d_model=256, widths=(32, 64, 96, 128, 192, 256),
                 dropout=0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, widths[0], 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.GELU(),
        )
        blocks, c = [], widths[0]
        for w in widths[1:]:
            blocks.append(ResBlock1D(c, w, stride=2, dropout=dropout))
            c = w
        blocks.append(ResBlock1D(c, d_model, stride=2, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

        # 프레임 → 토큰 attention pooling. 세그먼트 안에서 이상 박동이 있는
        # 프레임에 가중치를 몰아줄 수 있게 평균 대신 attention 을 쓴다.
        self.attn = nn.Sequential(
            nn.Conv1d(d_model, d_model // 4, 1),
            nn.Tanh(),
            nn.Conv1d(d_model // 4, 1, 1),
        )
        self.d_model = d_model

    def forward(self, x):
        # x: (N, C, seg_len)
        f = self.blocks(self.stem(x))            # (N, d, F)
        w = torch.softmax(self.attn(f), dim=-1)  # (N, 1, F)
        tok = (f * w).sum(-1)                    # (N, d)
        return tok, f


# =============================================================================
# (2) 계층형 S4 backbone
# =============================================================================
class TimeOfDayEmbedding(nn.Module):
    """토큰마다 '하루 중 언제인가'를 sin/cos 로 주입한다.

    심박은 circadian 구조가 매우 강해서(수면 중 서맥, 주간 활동성 빈맥) 절대 시각이
    유용한 사전 정보다. base_datetime 은 h5 metadata 에 이미 들어있다.
    """

    def __init__(self, d_model, n_freq=6):
        super().__init__()
        self.n_freq = n_freq
        self.proj = nn.Linear(2 * n_freq, d_model)

    def forward(self, tod):
        # tod: (B, S) — 0..1 로 정규화된 하루 중 시각
        k = torch.arange(1, self.n_freq + 1, device=tod.device, dtype=tod.dtype)
        ang = 2 * math.pi * tod.unsqueeze(-1) * k       # (B, S, n_freq)
        return self.proj(torch.cat([ang.sin(), ang.cos()], dim=-1))


class HierarchicalS4(nn.Module):
    """토큰 시퀀스 (B, d, S) 를 다해상도로 처리한다.

    S=8640 기준:
        L0  8640 tok (10초)  ─ depths[0] blocks ─┐
          ↓ stride 6                            │ skip
        L1  1440 tok (1분)   ─ depths[1] blocks ─┼─┐
          ↓ stride 10                           │ │ skip
        L2   144 tok (10분)  ─ depths[2] blocks  │ │
          ↑ upsample ×10                        │ │
        L1' 1440 tok         ─ depths[3] blocks ─┘ │
          ↑ upsample ×6                            │
        L0' 8640 tok         ─ depths[4] blocks ───┘
    """

    def __init__(self, d_model=256, d_state=64, depths=(2, 4, 4, 2, 2),
                 pool_factors=(6, 10), dropout=0.1, bidirectional=True):
        super().__init__()
        self.pool_factors = pool_factors
        mk = lambda n: nn.ModuleList(
            [S4Block(d_model, d_state=d_state, dropout=dropout,
                     bidirectional=bidirectional) for _ in range(n)]
        )
        self.enc0, self.enc1, self.mid, self.dec1, self.dec0 = (mk(n) for n in depths)

        p0, p1 = pool_factors
        self.down0 = nn.Conv1d(d_model, d_model, p0, stride=p0)
        self.down1 = nn.Conv1d(d_model, d_model, p1, stride=p1)
        self.up1 = nn.ConvTranspose1d(d_model, d_model, p1, stride=p1)
        self.up0 = nn.ConvTranspose1d(d_model, d_model, p0, stride=p0)
        self.norm_out = nn.LayerNorm(d_model)

    @staticmethod
    def _run(blocks, x):
        for b in blocks:
            x = b(x)
        return x

    @staticmethod
    def _match(x, ref):
        """pool/unpool 로 길이가 어긋날 때 ref 길이에 맞춘다 (S 가 배수가 아닐 때)."""
        if x.size(-1) == ref.size(-1):
            return x
        if x.size(-1) > ref.size(-1):
            return x[..., : ref.size(-1)]
        return F.pad(x, (0, ref.size(-1) - x.size(-1)))

    def forward(self, x):
        # x: (B, d, S)
        # pool_factors 의 곱(=최저해상도 1토큰에 필요한 최소 길이)의 배수로 맞춘다.
        # 짧은 record 나 배수가 아닌 길이에서 down/up 이 깨지는 것을 막는다.
        S = x.size(-1)
        p0, p1 = self.pool_factors
        unit = p0 * p1
        S_pad = max(unit, -(-S // unit) * unit)
        if S_pad != S:
            x = F.pad(x, (0, S_pad - S), mode="replicate")

        h0 = self._run(self.enc0, x)
        h1 = self._run(self.enc1, self.down0(h0))
        h2 = self._run(self.mid, self.down1(h1))

        u1 = self._match(self.up1(h2), h1)
        h1d = self._run(self.dec1, u1 + h1)

        u0 = self._match(self.up0(h1d), h0)
        h0d = self._run(self.dec0, u0 + h0)

        out = self.norm_out(h0d.transpose(-1, -2)).transpose(-1, -2)
        if S_pad != S:                       # 패딩 구간 제거, 원래 길이로 복원
            out = out[..., :S]
            h0 = h0[..., :S]
        return out, {"L0": h0, "L1": h1, "L2": h2}


class AttentionPool(nn.Module):
    """토큰 시퀀스 → record 임베딩. padding mask 를 존중한다."""

    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, key_padding_mask=None):
        # tokens: (B, S, d)
        q = self.q.expand(tokens.size(0), -1, -1)
        out, _ = self.attn(q, tokens, tokens, key_padding_mask=key_padding_mask)
        return self.norm(out.squeeze(1))


# =============================================================================
# (3) 전체 인코더
# =============================================================================
class HolterEncoder(nn.Module):
    """24시간 Holter → 토큰 시퀀스 + record 임베딩.

    입력은 두 형태를 모두 받는다.
      - segments: (B, S, C, seg_len)  원신호. stem 부터 돌린다 (Stage A/C)
      - tokens:   (B, S, d)           미리 계산한 stem 출력 (Stage B, 훨씬 빠름)
    """

    def __init__(self, in_channels=3, d_model=256, d_state=64,
                 depths=(2, 4, 4, 2, 2), pool_factors=(6, 10),
                 dropout=0.1, seg_chunk=512, use_time_of_day=True):
        super().__init__()
        self.stem = BeatCNNStem(in_channels, d_model, dropout=dropout)
        self.backbone = HierarchicalS4(d_model, d_state, depths, pool_factors, dropout)
        self.pool = AttentionPool(d_model)
        self.tod = TimeOfDayEmbedding(d_model) if use_time_of_day else None
        self.d_model = d_model
        # stem 을 한 번에 8640 세그먼트 돌리면 activation 이 터진다.
        # seg_chunk 단위로 쪼개고 학습 시에는 gradient checkpointing 을 건다.
        self.seg_chunk = seg_chunk

    def encode_segments(self, segments, checkpoint=True):
        """(B, S, C, L) → tokens (B, S, d).  청크 단위 + gradient checkpointing."""
        B, S, C, L = segments.shape
        flat = segments.reshape(B * S, C, L)
        outs = []
        for i in range(0, flat.size(0), self.seg_chunk):
            chunk = flat[i: i + self.seg_chunk]
            if checkpoint and self.training and torch.is_grad_enabled():
                tok, _ = torch.utils.checkpoint.checkpoint(
                    self.stem, chunk, use_reentrant=False)
            else:
                tok, _ = self.stem(chunk)
            outs.append(tok)
        return torch.cat(outs, 0).reshape(B, S, -1)

    def forward(self, segments=None, tokens=None, tod=None, mask=None,
                checkpoint=True):
        """
        Args:
            segments: (B, S, C, seg_len) 원신호 — tokens 와 택일
            tokens:   (B, S, d) 사전계산 토큰 — tokens 와 택일
            tod:      (B, S) 0..1 하루 중 시각. None 이면 생략
            mask:     (B, S) bool, True = 유효하지 않은(padding) 토큰
        Returns:
            dict(tokens=(B,S,d), record=(B,d), stem_tokens=(B,S,d), scales=...)
        """
        assert (segments is None) ^ (tokens is None), "segments 또는 tokens 중 하나만"
        if tokens is None:
            tokens = self.encode_segments(segments, checkpoint=checkpoint)
        stem_tokens = tokens

        x = tokens
        if self.tod is not None and tod is not None:
            x = x + self.tod(tod)

        out, scales = self.backbone(x.transpose(1, 2))      # (B, d, S)
        out = out.transpose(1, 2)                           # (B, S, d)
        rec = self.pool(out, key_padding_mask=mask)
        return {"tokens": out, "record": rec,
                "stem_tokens": stem_tokens, "scales": scales}


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_encoder(size="base", **kw):
    """size 프리셋. record 수가 적으면 small 부터 시작할 것."""
    presets = {
        "small": dict(d_model=128, d_state=64, depths=(2, 2, 2, 2, 2)),
        "base":  dict(d_model=256, d_state=64, depths=(2, 4, 4, 2, 2)),
        "large": dict(d_model=384, d_state=64, depths=(3, 6, 6, 3, 3)),
    }
    cfg = dict(presets[size])
    cfg.update(kw)
    return HolterEncoder(**cfg)
