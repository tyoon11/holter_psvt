# models.py
# =============================================================================
# 다양한 1D 인코더(세그먼트 단위 feature 추출) + MIL Head(세그먼트 집계) +
# 메타데이터 결합을 통한 최종 Multimodal MIL 이진 분류 모델을 구성하는 모듈.
#
# 핵심 데이터 흐름(“확실히 코드에서 보이는 사실”)
# 1) 입력 bag: (B, N, C, L)
#    - B: batch size
#    - N: bag 안의 segment 개수(K)
#    - C: ECG 채널 수(= selected_leads 길이)
#    - L: segment length(예: 1250)
# 2) encoder: (B*N, C, L) -> (B*N, H)  (H = hidden_dim)
# 3) mil_head: (B, N, H) -> (B, H) + (B, N) weights(또는 None)
# 4) meta_mlp: (B, feature_dim) -> (B, meta_dim)
# 5) final: concat((B,H),(B,meta_dim)) -> (B,) logits
#
# 사용 예:
#   from models import build_model
#   model = build_model(selected_leads=["II","V1"],
#                       enc_name="resnet", head_name="abmil",
#                       hidden_dim=32, meta_dim=16, dropout=0.3,
#                       feature_cols=FEATURE_COLS)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1) Segment Encoders
# -----------------------------------------------------------------------------
# "세그먼트 내부" 시계열 신호 (C,L)로부터 하나의 고정 길이 벡터(H)를 뽑는 모듈들.
# 모든 encoder는 최종 출력 차원을 (B*N, H)로 맞추도록 설계되어 있다.
# =============================================================================


# ──────────────────────────────────────────────────────────────────────────────
# 1-1) Basic CNN + Temporal Attention (기본 인코더)
# ──────────────────────────────────────────────────────────────────────────────
class SegmentEncoder(nn.Module):
    """
    입력:  x = (B*N, C, L)
    출력:  z = (B*N, H)

    구조:
    - Conv1d 3개로 다운샘플링하면서 특징 추출
      (stride=2를 3번 적용 → time length는 대략 1/8 수준으로 감소)
    - temporal_attn:
      * Conv1d(H->H, k=1) + Tanh + Conv1d(H->1, k=1)
      * 각 time step의 중요도를 score로 만들고 softmax로 정규화
    - 가중합(pooling)으로 (B*N, H) 벡터 생성
    """
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        H = hidden_dim

        # CNN feature extractor: (C,L) -> (H,L')
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, H, 25, stride=2, padding=12),
            nn.ReLU(),
            nn.Conv1d(H, H, 25, stride=2, padding=12),
            nn.ReLU(),
            nn.Conv1d(H, H, 25, stride=2, padding=12),
            nn.ReLU(),
        )

        # Temporal attention scorer: (H,L') -> (1,L')
        self.temporal_attn = nn.Sequential(
            nn.Conv1d(H, H, 1),
            nn.Tanh(),
            nn.Conv1d(H, 1, 1),
        )

    def forward(self, x):  # x: (B*N, C, L)
        h = self.cnn(x)  # (B*N, H, L')
        # attention weights over time dimension (dim=2)
        w = F.softmax(self.temporal_attn(h), dim=2)  # (B*N, 1, L')
        # weighted sum pooling → (B*N, H)
        z = torch.sum(w * h, dim=2)
        return z


# ──────────────────────────────────────────────────────────────────────────────
# 1-2) ResNet1D Encoder
# ──────────────────────────────────────────────────────────────────────────────
class ResBlock1D(nn.Module):
    """
    1D ResNet block.
    - conv-bn-relu -> conv-bn -> residual add -> relu
    - stride s로 다운샘플링 가능
    - 채널 수가 다르거나 stride!=1이면 1x1 conv로 downsample branch를 맞춘다.
    """
    def __init__(self, c_in, c_out, k=7, s=1):
        super().__init__()
        p = k // 2

        self.conv1 = nn.Conv1d(c_in, c_out, k, stride=s, padding=p)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(c_out, c_out, k, stride=1, padding=p)
        self.bn2 = nn.BatchNorm1d(c_out)

        # residual branch 맞추기(채널/stride mismatch 해결)
        self.down = nn.Conv1d(c_in, c_out, 1, stride=s) if (s != 1 or c_in != c_out) else None

    def forward(self, x):
        # identity/residual branch
        iden = x if self.down is None else self.down(x)

        # main branch
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # residual add + relu
        out = self.relu(out + iden)
        return out


class SegmentEncoder_ResNet1D(nn.Module):
    """
    입력:  (B*N, C, L)
    출력:  (B*N, H)

    구조:
    - stem: conv7 stride2
    - residual blocks:
      layer1: stride1
      layer2: stride2 (downsample)
      layer3: stride2 (downsample)
    - 마지막에 temporal attention으로 time dimension을 가중합
    """
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        H = hidden_dim

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, H, 7, stride=2, padding=3),
            nn.BatchNorm1d(H),
            nn.ReLU(inplace=True),
        )

        self.layer1 = ResBlock1D(H, H, k=7, s=1)
        self.layer2 = ResBlock1D(H, H, k=7, s=2)
        self.layer3 = ResBlock1D(H, H, k=7, s=2)

        self.temporal_attn = nn.Sequential(
            nn.Conv1d(H, H, 1),
            nn.Tanh(),
            nn.Conv1d(H, 1, 1),
        )

    def forward(self, x):  # (B*N, C, L)
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        w = F.softmax(self.temporal_attn(h), dim=2)  # (B*N,1,L')
        z = torch.sum(w * h, dim=2)  # (B*N,H)
        return z


# ──────────────────────────────────────────────────────────────────────────────
# 1-3) InceptionTime-style Encoder
# ──────────────────────────────────────────────────────────────────────────────
class InceptionBlock1D(nn.Module):
    """
    InceptionTime 스타일 1D block.
    - 서로 다른 커널 크기(9,19,39)의 conv branch 3개
    - maxpool + 1x1 conv branch 1개
    - concat 후 BN+ReLU

    주의(코드로부터 확실):
    - c_out은 4로 나누어떨어진다는 가정이 주석에 있고,
      실제로 c_out//4를 사용하므로 4의 배수가 아니면 채널 손실/불일치가 날 수 있다.
    """
    def __init__(self, c_in, c_out):
        super().__init__()
        self.b1 = nn.Conv1d(c_in, c_out // 4, 9, padding=4)
        self.b2 = nn.Conv1d(c_in, c_out // 4, 19, padding=9)
        self.b3 = nn.Conv1d(c_in, c_out // 4, 39, padding=19)

        self.b4 = nn.MaxPool1d(3, stride=1, padding=1)
        self.b4c = nn.Conv1d(c_in, c_out // 4, 1)

        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4c(self.b4(x))
        y = torch.cat([y1, y2, y3, y4], dim=1)  # channel concat
        return self.act(self.bn(y))


class SegmentEncoder_InceptionTime(nn.Module):
    """
    입력:  (B*N, C, L)
    출력:  (B*N, H)

    구조:
    - stem 1x1 conv로 채널을 H로 맞춤
    - inception block 3개
    - temporal attention으로 time 축 가중합
    """
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        H = hidden_dim
        self.stem = nn.Conv1d(in_channels, H, 1)
        self.inc1 = InceptionBlock1D(H, H)
        self.inc2 = InceptionBlock1D(H, H)
        self.inc3 = InceptionBlock1D(H, H)

        self.temporal_attn = nn.Sequential(
            nn.Conv1d(H, H, 1),
            nn.Tanh(),
            nn.Conv1d(H, 1, 1),
        )

    def forward(self, x):
        h = self.stem(x)
        h = self.inc1(h)
        h = self.inc2(h)
        h = self.inc3(h)
        w = F.softmax(self.temporal_attn(h), dim=2)
        z = torch.sum(w * h, dim=2)
        return z


# ──────────────────────────────────────────────────────────────────────────────
# 1-4) Transformer1D Encoder (경량)
# ──────────────────────────────────────────────────────────────────────────────
class SegmentEncoder_Transformer1D(nn.Module):
    """
    입력:  (B*N, C, L)
    출력:  (B*N, H)

    구조:
    - Conv1d로 (C,L) -> (H,L) 투영 후 (B*N, L, H)로 transpose
    - TransformerEncoder로 time token 간 상호작용 반영
    - Linear 기반 temporal attention으로 (B*N,H)로 요약

    주의(코드로부터 확실):
    - TransformerEncoderLayer에 batch_first=True 이므로 입력은 (B, L, H) 형태여야 한다.
    """
    def __init__(self, in_channels: int, hidden_dim: int, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        H = hidden_dim

        self.proj = nn.Conv1d(in_channels, H, 7, padding=3)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=nhead,
            dim_feedforward=H * 4,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        # (B*N, L, H) -> (B*N, L, 1)
        self.temporal_attn = nn.Sequential(
            nn.Linear(H, H),
            nn.Tanh(),
            nn.Linear(H, 1),
        )

    def forward(self, x):  # (B*N, C, L)
        h = self.proj(x).transpose(1, 2)  # (B*N, L, H)
        h = self.enc(h)                  # (B*N, L, H)
        w = F.softmax(self.temporal_attn(h), dim=1)  # (B*N, L, 1)
        z = torch.sum(w * h, dim=1)                  # (B*N, H)
        return z


# ──────────────────────────────────────────────────────────────────────────────
# 1-5) BiLSTM Encoder
# ──────────────────────────────────────────────────────────────────────────────
class SegmentEncoder_BiLSTM(nn.Module):
    """
    입력:  (B*N, C, L)
    출력:  (B*N, H)

    구조:
    - (B*N, C, L) -> (B*N, L, C) 로 변환해 time-major sequence로 처리
    - Linear로 C -> H 차원 투영
    - BiLSTM으로 (B*N, L, 2H) 생성
    - attention으로 time weighted sum -> (B*N, 2H)
    - out Linear로 (B*N, H)로 축소(다른 인코더들과 output dim 일치)
    """
    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        H = hidden_dim

        self.input_proj = nn.Linear(in_channels, H)

        self.rnn = nn.LSTM(
            H,
            H,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.attn = nn.Sequential(
            nn.Linear(2 * H, H),
            nn.Tanh(),
            nn.Linear(H, 1),
        )

        # 2H -> H로 맞춤
        self.out = nn.Linear(2 * H, H)

    def forward(self, x):  # (B*N, C, L)
        x = x.transpose(1, 2)   # (B*N, L, C)
        x = self.input_proj(x)  # (B*N, L, H)
        o, _ = self.rnn(x)      # (B*N, L, 2H)

        # attn(o): (B*N, L, 1) → squeeze(-1) → (B*N, L)
        w = F.softmax(self.attn(o).squeeze(-1), dim=1)

        # weighted sum: (B*N, 2H)
        h = torch.sum(o * w.unsqueeze(-1), dim=1)

        z = self.out(h)  # (B*N, H)
        return z


# ──────────────────────────────────────────────────────────────────────────────
# 1-6) MLP-Mixer 1D Encoder (patch token 기반)
# ──────────────────────────────────────────────────────────────────────────────
class _MLP(nn.Module):
    """MLP-Mixer 내부에서 token/channel mixing에 사용하는 2-layer MLP 블록"""
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


class SegmentEncoder_MLPMixer1D(nn.Module):
    """
    입력:  (B*N, C, L)
    출력:  (B*N, H)

    동작:
    1) patch 단위로 time axis를 쪼갬 (unfold)
       - patch=16이면 L을 16씩 끊어 P개의 patch token 생성
    2) 각 token의 feature는 (C*patch)
    3) Linear embed로 (C*patch) -> H
    4) Mixer blocks:
       - token mixing: LayerNorm + MLP (여기 구현은 (B*N, P, H)에서 각 token에 동일 MLP)
       - channel mixing: LayerNorm + MLP
       (코드상 token/channel mixing이 “전형적인” mixer처럼 축을 바꾸는 형태가 아니라,
        동일 dim에 대해 residual MLP를 적용하는 형태로 구현되어 있음은 코드로 확인 가능)
    5) token 방향 평균 풀링으로 (B*N, H)

    padding:
    - L % patch != 0이면 오른쪽에 pad해서 patch로 딱 나눠지게 만든다.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        patch: int = 16,
        depth: int = 6,
        token_mlp_dim: int = 256,
        channel_mlp_dim: int = 256,
    ):
        super().__init__()
        self.patch = patch

        # patch token의 raw feature: C*patch → hidden_dim
        self.embed = nn.Linear(in_channels * patch, hidden_dim)

        # depth개의 residual mixer block
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(hidden_dim),
                        "token": _MLP(hidden_dim, token_mlp_dim),
                        "ln2": nn.LayerNorm(hidden_dim),
                        "channel": _MLP(hidden_dim, channel_mlp_dim),
                    }
                )
                for _ in range(depth)
            ]
        )

        # token(P) 방향으로 평균 풀링하기 위해 (B*N, H, P)로 바꾼 뒤 AdaptiveAvgPool1d(1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # x: (B*N, C, L)
        Bn, C, L = x.shape

        # patch 크기로 딱 나눠지도록 오른쪽 pad
        if L % self.patch != 0:
            pad = self.patch - (L % self.patch)
            x = F.pad(x, (0, pad))
            L = x.shape[-1]

        # unfold로 patch token 생성
        # x.unfold(2, patch, patch) -> (B*N, C, P, patch)
        x = x.unfold(dimension=2, size=self.patch, step=self.patch)

        # (B*N, C, P, patch) -> (B*N, P, C, patch) -> (B*N, P, C*patch)
        x = x.permute(0, 2, 1, 3).contiguous().view(Bn, -1, C * self.patch)

        # embedding: (B*N, P, C*patch) -> (B*N, P, H)
        x = self.embed(x)

        # residual mixer blocks
        for blk in self.blocks:
            y = blk["ln1"](x)
            x = x + blk["token"](y)    # token mixing (residual)
            y = blk["ln2"](x)
            x = x + blk["channel"](y)  # channel mixing (residual)

        # (B*N, P, H) -> (B*N, H, P) -> pool -> (B*N, H)
        h = x.transpose(1, 2)
        z = self.pool(h).squeeze(-1)
        return z


# =============================================================================
# 2) MIL Heads
# -----------------------------------------------------------------------------
# 입력:  seg_feats = (B, N, H)  (각 세그먼트의 feature 벡터)
# 출력:  bag_repr = (B, H) + weights (B, N) 또는 None
# =============================================================================

class SimpleAttentionMIL(nn.Module):
    """
    단순 attention MIL.
    - attn MLP로 각 instance의 score를 산출한 뒤 softmax로 정규화
    - 가중합으로 bag representation 생성
    """
    def __init__(self, h_in: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(h_in, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, seg_feats):  # (B, N, H)
        scores = self.attn(seg_feats).squeeze(-1)  # (B, N)
        w = F.softmax(scores, dim=1)               # (B, N)
        bag = torch.sum(w.unsqueeze(-1) * seg_feats, dim=1)  # (B, H)
        return bag, w


class GatedAttentionMIL(nn.Module):
    """
    ABMIL (Ilse et al.)의 gated attention.
    - tanh branch와 sigmoid branch를 곱해 gating 효과를 만든 뒤
      linear로 score 산출, softmax로 가중치 생성.
    """
    def __init__(self, h_in: int):
        super().__init__()
        self.att_a = nn.Linear(h_in, h_in)
        self.att_b = nn.Linear(h_in, h_in)
        self.att = nn.Linear(h_in, 1)

    def forward(self, seg_feats):  # (B, N, H)
        A = torch.tanh(self.att_a(seg_feats)) * torch.sigmoid(self.att_b(seg_feats))  # (B,N,H)
        scores = self.att(A).squeeze(-1)  # (B, N)
        w = F.softmax(scores, dim=1)      # (B, N)
        bag = torch.sum(w.unsqueeze(-1) * seg_feats, dim=1)  # (B, H)
        return bag, w


class DSMILLite(nn.Module):
    """
    DSMIL-lite 형태의 top-k pooling.
    - instance classifier로 각 segment의 logit/prob를 만들고,
    - prob 기준 top-k segment의 feature를 평균내어 bag representation 생성.

    반환:
    - bag: (B,H)
    - probs: (B,N)  (여기서는 attention weight 대신 “중요도” 대용으로 사용 가능)
    """
    def __init__(self, h_in: int, k: int = 5):
        super().__init__()
        self.inst_clf = nn.Linear(h_in, 1)
        self.k = k

    def forward(self, seg_feats):  # (B, N, H)
        logits = self.inst_clf(seg_feats).squeeze(-1)  # (B, N)
        probs = torch.sigmoid(logits)                  # (B, N)

        k = min(self.k, probs.size(1))
        topk_vals, topk_idx = torch.topk(probs, k=k, dim=1)  # (B, k)

        # topk_idx로 seg_feats에서 top-k feature를 gather
        topk_seg = torch.gather(
            seg_feats,
            1,
            topk_idx.unsqueeze(-1).expand(-1, -1, seg_feats.size(-1)),
        )  # (B, k, H)

        bag = topk_seg.mean(dim=1)  # (B, H)
        return bag, probs


# ──────────────────────────────────────────────────────────────────────────────
# SetTransformer MIL Head
# ──────────────────────────────────────────────────────────────────────────────
class SAB(nn.Module):
    """
    SetTransformer의 Self-Attention Block.
    - MultiheadAttention + residual + LayerNorm
    - FFN + residual + LayerNorm
    """
    def __init__(self, h, nhead=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(h, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(h, 4 * h),
            nn.ReLU(),
            nn.Linear(4 * h, h),
        )
        self.ln1 = nn.LayerNorm(h)
        self.ln2 = nn.LayerNorm(h)

    def forward(self, X):  # (B, N, H)
        y, _ = self.mha(X, X, X)  # self-attention
        X = self.ln1(X + y)
        y = self.ff(X)
        return self.ln2(X + y)


class PMA(nn.Module):
    """
    Pooling by Multihead Attention.
    - 학습 가능한 seed vector S(1,m,H)가 query가 되어 set X를 요약한다.
    """
    def __init__(self, h, m=1, nhead=4):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, m, h))
        self.mha = nn.MultiheadAttention(h, nhead, batch_first=True)

    def forward(self, X):  # (B, N, H)
        S = self.S.expand(X.size(0), -1, -1)  # (B, m, H)
        Y, _ = self.mha(S, X, X)              # (B, m, H)
        return Y


class SetTransformerMIL(nn.Module):
    """
    SAB 2개 + PMA로 set을 1개 벡터로 요약하는 MIL head.

    반환:
    - (B,H) bag representation
    - attention weight은 반환하지 않음(None)
    """
    def __init__(self, h_in: int, nhead: int = 4):
        super().__init__()
        self.sab1 = SAB(h_in, nhead)
        self.sab2 = SAB(h_in, nhead)
        self.pma = PMA(h_in, m=1, nhead=nhead)

    def forward(self, seg_feats):  # (B, N, H)
        H = self.sab1(seg_feats)
        H = self.sab2(H)
        Y = self.pma(H).squeeze(1)  # (B, H)
        return Y, None


# =============================================================================
# 3) Time-aware MIL Head (TimeMIL)
# -----------------------------------------------------------------------------
# segment feats에 positional encoding을 더해 시간 순서를 반영한 뒤,
# (옵션) temporal transformer encoder를 통해 interaction을 섞고,
# gated attention으로 pooling.
# =============================================================================
class TimeMIL(nn.Module):
    """
    입력:  seg_feats = (B, N, H)
    출력:  bag = (B, H), w = (B, N)

    구성 요소:
    - sinusoidal positional encoding: 길이 N에 대해 (N,H) 생성
    - pe_scale: positional encoding 크기를 조절하는 학습 파라미터(스칼라)
    - temporal encoder(옵션):
      * nlayers>0이면 TransformerEncoder 사용
      * nlayers==0이면 temporal=None으로 pass
    - gated attention scorer로 score -> softmax weight -> weighted sum pooling

    top-k 옵션:
    - k가 설정되면 score 기준 상위 k개만 남기고 나머지는 -inf로 마스킹하여 weight=0에 가깝게 만든다.
    """
    def __init__(self, h_in: int, nhead: int = 4, nlayers: int = 1, k: int = None, dropout: float = 0.0):
        super().__init__()
        self.h = h_in
        self.k = k

        # positional encoding에 곱해줄 learnable scalar
        self.pe_scale = nn.Parameter(torch.ones(1))

        if nlayers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=h_in,
                nhead=nhead,
                dim_feedforward=4 * h_in,
                batch_first=True,
                dropout=dropout,
                activation="relu",
            )
            self.temporal = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        else:
            self.temporal = None

        # gated attention scorer
        self.att_a = nn.Linear(h_in, h_in)
        self.att_b = nn.Linear(h_in, h_in)
        self.att_s = nn.Linear(h_in, 1)

    @staticmethod
    def _sinusoidal_pe(length: int, dim: int, device: torch.device):
        """
        (length, dim) sinusoidal positional encoding 생성.
        - 짝수 차원: sin
        - 홀수 차원: cos
        """
        pos = torch.arange(length, device=device).float().unsqueeze(1)  # (N,1)
        i = torch.arange(dim, device=device).float().unsqueeze(0)       # (1,H)
        angle_rates = 1.0 / torch.pow(10000, (2 * (i // 2)) / dim)
        angles = pos * angle_rates

        pe = torch.zeros(length, dim, device=device)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        return pe  # (N,H)

    def forward(self, seg_feats):  # (B, N, H)
        B, N, H = seg_feats.shape
        device = seg_feats.device

        # 1) positional encoding 주입
        pe = self._sinusoidal_pe(N, H, device) * self.pe_scale  # (N,H)
        X = seg_feats + pe.unsqueeze(0)                          # (B,N,H)

        # 2) optional temporal transformer
        if self.temporal is not None:
            X = self.temporal(X)

        # 3) gated attention scores
        A = torch.tanh(self.att_a(X)) * torch.sigmoid(self.att_b(X))  # (B,N,H)
        scores = self.att_s(A).squeeze(-1)                             # (B,N)

        # 4) optional top-k masking
        if self.k is not None and self.k > 0 and self.k < N:
            topk_vals, topk_idx = torch.topk(scores, k=self.k, dim=1)
            mask = torch.full_like(scores, float("-inf"))
            mask.scatter_(1, topk_idx, 0.0)
            scores = scores + mask

        # 5) softmax weights + weighted sum pooling
        w = F.softmax(scores, dim=1)                 # (B,N)
        bag = torch.sum(w.unsqueeze(-1) * X, dim=1)  # (B,H)
        return bag, w


# =============================================================================
# 4) Multimodal MIL Core (encoder + head + meta + final clf)
# =============================================================================
class MultimodalMILCore(nn.Module):
    """
    전체 모델 본체.

    encoder:
      - 입력: (B*N, C, L)
      - 출력: (B*N, H)

    mil_head:
      - 입력: (B, N, H)
      - 출력: bag_repr (B, H) 및 attention weight (B, N) 또는 None

    meta_mlp:
      - 입력: meta (B, feature_dim)
      - 출력: meta_repr (B, meta_dim)

    final:
      - 입력: concat((B,H), (B,meta_dim))
      - 출력: logits (B,)  (binary classification)
    """
    def __init__(
        self,
        encoder: nn.Module,
        mil_head: nn.Module,
        feature_dim: int,
        hidden_dim: int,
        meta_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = encoder
        self.mil_head = mil_head

        # metadata를 작은 MLP로 임베딩
        self.meta_mlp = nn.Sequential(
            nn.Linear(feature_dim, meta_dim),
            nn.ReLU(),
        )

        # 최종 분류기: (H + meta_dim) -> 1
        self.final = nn.Sequential(
            nn.Linear(hidden_dim + meta_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, bag, meta):
        """
        bag : (B, N, C, L)
        meta: (B, feature_dim)

        반환:
        - out: (B,) logits
        - attn_w: (B,N) or None
        """
        B, N, C, L = bag.shape

        # encoder 입력을 (B*N, C, L)로 reshape
        x = bag.view(B * N, C, L)

        # (B*N, H) -> (B, N, H)
        seg = self.encoder(x).view(B, N, -1)

        # MIL pooling
        bag_repr, attn_w = self.mil_head(seg)

        # metadata embedding
        meta_repr = self.meta_mlp(meta)

        # concat 후 최종 logits
        out = self.final(torch.cat([bag_repr, meta_repr], dim=1)).squeeze(-1)
        return out, attn_w


# =============================================================================
# 5) Builder functions: 이름으로 encoder/head 선택
# =============================================================================
def _make_encoder(enc_name: str, in_channels: int, hidden_dim: int) -> nn.Module:
    """
    enc_name 문자열에 따라 encoder 인스턴스를 생성한다.
    - 지원 목록은 아래 if 분기에서 결정된다.
    - 정의되지 않은 이름이면 ValueError 발생.
    """
    name = (enc_name or "basic").lower()

    if name in ["basic", "cnn", "temporal"]:
        return SegmentEncoder(in_channels, hidden_dim)

    if name in ["resnet", "resnet1d"]:
        return SegmentEncoder_ResNet1D(in_channels, hidden_dim)

    if name in ["inception", "inceptiontime"]:
        return SegmentEncoder_InceptionTime(in_channels, hidden_dim)

    if name in ["transformer", "tfn"]:
        return SegmentEncoder_Transformer1D(in_channels, hidden_dim)

    # 추가된 encoder들
    if name in ["bilstm", "lstm", "gru"]:
        # 주의(코드로부터 확실):
        # - "gru"라고 써도 실제로는 SegmentEncoder_BiLSTM(LSTM 기반)을 반환한다.
        # - GRU를 진짜로 쓰려면 별도 구현/분기가 필요하다.
        return SegmentEncoder_BiLSTM(in_channels, hidden_dim)

    if name in ["mixer", "mlpmixer", "mlp-mixer"]:
        return SegmentEncoder_MLPMixer1D(in_channels, hidden_dim)

    raise ValueError(f"Unknown encoder: {enc_name}")


def _make_head(head_name: str, hidden_dim: int) -> nn.Module:
    """
    head_name 문자열에 따라 MIL head 인스턴스를 생성한다.

    현재 코드 특징(확실):
    - 함수 시작 시 DEBUG print가 수행된다.
      => 학습 루프에서 head를 만들 때마다 표준출력에 찍힌다.
    """
    print(f"[DEBUG _make_head] Received head_name={head_name} (type: {type(head_name)})")
    name = (head_name or "simple").lower()
    print(f"[DEBUG _make_head] Normalized name={name}")

    if name in ["simple", "attn", "attention"]:
        print(f"[DEBUG _make_head] Using SimpleAttentionMIL")
        return SimpleAttentionMIL(hidden_dim)

    if name in ["abmil", "gated", "gatedattention"]:
        print(f"[DEBUG _make_head] Using GatedAttentionMIL")
        return GatedAttentionMIL(hidden_dim)

    if name in ["dsmil", "topk"]:
        return DSMILLite(hidden_dim, k=5)

    if name in ["set", "settransformer"]:
        return SetTransformerMIL(hidden_dim)

    if name in ["timemil", "time", "temporal"]:
        # 기본 설정: nlayers=1 transformer encoder, top-k 비활성
        return TimeMIL(hidden_dim, nhead=4, nlayers=1, k=None, dropout=0.0)

    raise ValueError(f"Unknown head: {head_name}")


def build_model(
    selected_leads,
    enc_name: str = "resnet",
    head_name: str = "abmil",
    hidden_dim: int = 32,
    meta_dim: int = 16,
    dropout: float = 0.3,
    feature_cols=None,
) -> nn.Module:
    """
    외부에서 쓰는 모델 빌더.

    입력
    - selected_leads: 예) ["II", "V1", "V5"]
      => in_channels = len(selected_leads) 로 encoder 입력 채널 수를 결정
    - enc_name/head_name: encoder/head 선택 키
    - hidden_dim: encoder 출력 및 mil_head 입력 차원 H
    - meta_dim: meta_mlp 출력 차원
    - dropout: final classifier의 dropout 비율
    - feature_cols: metadata 컬럼 리스트
      => feature_dim = len(feature_cols)

    출력
    - MultimodalMILCore 인스턴스
    """
    if feature_cols is None:
        raise ValueError("feature_cols must be provided (list of meta feature names).")

    in_channels = len(selected_leads)
    feature_dim = len(feature_cols)

    encoder = _make_encoder(enc_name, in_channels, hidden_dim)
    head = _make_head(head_name, hidden_dim)

    return MultimodalMILCore(
        encoder=encoder,
        mil_head=head,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        meta_dim=meta_dim,
        dropout=dropout,
    )


# =============================================================================
# 6) 공개 API 목록(__all__)
# =============================================================================
__all__ = [
    "build_model",
    # encoders
    "SegmentEncoder",
    "SegmentEncoder_ResNet1D",
    "SegmentEncoder_InceptionTime",
    "SegmentEncoder_Transformer1D",
    # heads
    "SimpleAttentionMIL",
    "GatedAttentionMIL",
    "DSMILLite",
    "SetTransformerMIL",
    "TimeMIL",
    # core
    "MultimodalMILCore",
]
