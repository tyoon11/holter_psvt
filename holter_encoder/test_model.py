# -*- coding: utf-8 -*-
"""test_model.py — S4D 수치 정확성 + 인코더 shape/메모리 검증.

    python -m holter_encoder.test_model          (repo 루트에서)
"""

import time

import torch
import torch.nn as nn

from .s4d import S4D, S4Block, s4_param_groups
from .model import HolterEncoder, build_encoder, count_params

OK, FAIL = "  [OK]", "  [FAIL]"


def _bare(s4):
    """D / activation / output_linear 를 무력화해 순수 convolution 만 남긴다."""
    with torch.no_grad():
        s4.D.zero_()
    s4.activation = nn.Identity()
    s4.output_linear = nn.Identity()
    return s4


def test_fft_conv_matches_naive():
    """단방향 S4D 가 커널과의 naive causal convolution 과 일치하는지."""
    torch.manual_seed(0)
    H, L = 4, 32
    s4 = _bare(S4D(H, d_state=16, bidirectional=False)).eval()
    u = torch.randn(2, H, L)

    with torch.no_grad():
        y = s4(u)
        k = s4.kernel[0](L)                       # (H, L)
        naive = torch.zeros_like(u)
        for n in range(L):
            for m in range(n + 1):
                naive[:, :, n] += u[:, :, m] * k[:, n - m]

    err = (y - naive).abs().max().item()
    print(f"{OK if err < 1e-4 else FAIL} FFT conv == naive causal conv  (max err {err:.2e})")
    return err < 1e-4


def test_causality():
    """단방향은 미래에 의존하지 않고, 양방향은 의존해야 한다."""
    torch.manual_seed(0)
    H, L, t = 4, 32, 10
    results = {}
    for bidir in (False, True):
        s4 = S4D(H, d_state=16, bidirectional=bidir).eval()
        u = torch.randn(1, H, L)
        u2 = u.clone()
        u2[:, :, t + 1:] += 5.0                   # 미래만 크게 흔든다
        with torch.no_grad():
            d = (s4(u)[:, :, :t + 1] - s4(u2)[:, :, :t + 1]).abs().max().item()
        results[bidir] = d

    ok = results[False] < 1e-4 and results[True] > 1e-2
    print(f"{OK if ok else FAIL} 인과성  단방향 과거영향={results[False]:.2e} (≈0 이어야) / "
          f"양방향={results[True]:.2e} (>0 이어야)")
    return ok


def test_long_range_memory():
    """길이 8640 에서 먼 과거 입력이 출력까지 전파되는지 (장기 문맥 sanity)."""
    torch.manual_seed(0)
    H, L = 8, 8640
    s4 = S4D(H, d_state=64, bidirectional=True).eval()
    u = torch.zeros(1, H, L)
    u[:, :, 0] = 1.0                              # 맨 앞에만 임펄스
    with torch.no_grad():
        y = s4(u)[0].abs().mean(0)                # (L,)
    far = y[L // 2:].max().item()
    ok = far > 1e-6
    print(f"{OK if ok else FAIL} 장기 기억  L={L}, 후반부 최대 응답={far:.2e} (>0 이어야)")
    return ok


def test_encoder_shapes():
    """토큰 경로 / 원신호 경로 모두 shape 가 맞는지."""
    torch.manual_seed(0)
    B, S, C, SEG, D = 2, 60, 3, 1250, 128
    enc = build_encoder("small", in_channels=C, pool_factors=(6, 10), seg_chunk=32).eval()

    tok = torch.randn(B, S, D)
    tod = torch.rand(B, S)
    with torch.no_grad():
        out = enc(tokens=tok, tod=tod)
    ok1 = out["tokens"].shape == (B, S, D) and out["record"].shape == (B, D)
    print(f"{OK if ok1 else FAIL} 토큰 경로  tokens={tuple(out['tokens'].shape)} "
          f"record={tuple(out['record'].shape)} "
          f"scales={ {k: tuple(v.shape) for k, v in out['scales'].items()} }")

    seg = torch.randn(B, S, C, SEG)
    with torch.no_grad():
        out2 = enc(segments=seg, tod=tod, checkpoint=False)
    ok2 = out2["tokens"].shape == (B, S, D) and out2["stem_tokens"].shape == (B, S, D)
    print(f"{OK if ok2 else FAIL} 원신호 경로  stem_tokens={tuple(out2['stem_tokens'].shape)} "
          f"tokens={tuple(out2['tokens'].shape)}")
    return ok1 and ok2


def test_non_multiple_length():
    """S 가 pool_factors 의 배수가 아닐 때도 깨지지 않는지 (짧은 record 대비)."""
    torch.manual_seed(0)
    enc = build_encoder("small").eval()
    oks = []
    for S in (60, 61, 77, 143):
        with torch.no_grad():
            out = enc(tokens=torch.randn(1, S, 128))
        oks.append(out["tokens"].shape == (1, S, 128))
    print(f"{OK if all(oks) else FAIL} 비배수 길이  S={[60, 61, 77, 143]} → {oks}")
    return all(oks)


def test_backward():
    """gradient 가 stem 까지 흐르는지 + checkpointing 동작 확인.

    주의: record 는 마지막이 LayerNorm 이므로 record.sum() 을 손실로 쓰면 안 된다.
    초기화 시 gamma=1 이라 특징 축 합이 항상 0 이고 gradient 가 해석적으로 0 이 된다.
    실제 downstream 처럼 선형 head 를 통과시킨다.
    """
    torch.manual_seed(0)
    enc = build_encoder("small", seg_chunk=16).train()
    head = nn.Linear(enc.d_model, 1)
    seg = torch.randn(1, 24, 3, 1250)
    out = enc(segments=seg, tod=torch.rand(1, 24), checkpoint=True)
    loss = nn.functional.binary_cross_entropy_with_logits(
        head(out["record"]).squeeze(-1), torch.ones(1))
    loss.backward()

    first = enc.stem.stem[0].weight
    g = first.grad
    ok = g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    print(f"{OK if ok else FAIL} backward  stem 첫 conv grad norm={g.norm().item():.3e}")

    # 깊이에 따른 gradient 크기 — S4 U-Net 이 깊어 소실/폭주를 눈으로 확인한다
    probe = [("stem.stem.0", enc.stem.stem[0].weight),
             ("stem.blocks.-1", enc.stem.blocks[-1].conv2.weight),
             ("backbone.enc0.0", enc.backbone.enc0[0].ff[0].weight),
             ("backbone.mid.0", enc.backbone.mid[0].ff[0].weight),
             ("backbone.dec0.-1", enc.backbone.dec0[-1].ff[0].weight),
             ("pool.q", enc.pool.q)]
    line = "  ".join(f"{n}={p.grad.norm().item():.1e}" for n, p in probe if p.grad is not None)
    print(f"       깊이별 grad norm: {line}")
    finite = all(torch.isfinite(p.grad).all() for _, p in probe if p.grad is not None)
    ok = ok and finite

    groups = s4_param_groups(enc)
    n_ssm = sum(p.numel() for p in groups[1]["params"])
    print(f"       param group 분리: 일반 {sum(p.numel() for p in groups[0]['params']):,} / "
          f"SSM(dt,A) {n_ssm:,}")
    return ok and n_ssm > 0


def bench_full_day():
    """실제 크기(S=8640)에서 토큰 경로 forward 시간과 파라미터 수."""
    print("\n  --- 실제 규모 (S=8640 = 24h @ 10초) ---")
    for size in ("small", "base"):
        enc = build_encoder(size).eval()
        d = enc.d_model
        x = torch.randn(1, 8640, d)
        t0 = time.time()
        with torch.no_grad():
            out = enc(tokens=x, tod=torch.rand(1, 8640))
        dt = time.time() - t0
        print(f"  {size:6s} d_model={d:3d}  총 {count_params(enc):>11,} "
              f"(stem {count_params(enc.stem):>9,} / backbone {count_params(enc.backbone):>10,})"
              f"  CPU forward {dt:.2f}s  → {tuple(out['tokens'].shape)}")


if __name__ == "__main__":
    torch.set_num_threads(4)
    print("=" * 70)
    print("S4D 수치 검증")
    print("=" * 70)
    r = [test_fft_conv_matches_naive(), test_causality(), test_long_range_memory()]
    print("\n" + "=" * 70)
    print("인코더 검증")
    print("=" * 70)
    r += [test_encoder_shapes(), test_non_multiple_length(), test_backward()]
    bench_full_day()
    print("\n" + ("전체 통과" if all(r) else f"실패 {r.count(False)}건"))
