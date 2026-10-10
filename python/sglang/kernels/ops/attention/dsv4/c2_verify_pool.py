"""Ratio-2 verify pooling + RMSNorm, with a separate race-free ring update.

All partner reads finish before any ring writes. This is necessary even for
static verify: six tokens may wrap a small ring or revisit rejected positions.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _c2_verify_pool_norm_kernel(
    KV,
    Score,
    Pos,
    RawLoc,
    OutLoc,
    Req,
    StateKV,
    StateScore,
    Weight,
    Latent,
    GroupPos,
    Slots,
    KV_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    STATE_KV_STRIDE: tl.constexpr,
    STATE_SCORE_STRIDE: tl.constexpr,
    PAD_ROW: tl.constexpr,
    RING: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    pos = tl.load(Pos + row)
    req = tl.load(Req + row)
    raw = tl.load(RawLoc + row)
    prev_req = tl.load(Req + row - 1, row > 0, -1)
    prev_pos = tl.load(Pos + row - 1, row > 0, -2)
    prev_raw = tl.load(RawLoc + row - 1, row > 0, 0)
    in_batch = (row > 0) & (req == prev_req) & (pos == prev_pos + 1)
    in_batch = in_batch & (raw != 0) & (prev_raw != 0)
    state_row = tl.where(
        (raw == 0) | (pos == 0), PAD_ROW, req * RING + (pos - 1) % RING
    )
    kv = tl.load(KV + row * KV_STRIDE + col, col < D, 0)
    score = tl.load(Score + row * SCORE_STRIDE + col, col < D, 0)
    carried_kv = tl.load(
        StateKV + state_row * STATE_KV_STRIDE + col, (col < D) & ~in_batch, 0
    ).to(tl.float32)
    carried_score = tl.load(
        StateScore + state_row * STATE_SCORE_STRIDE + col, (col < D) & ~in_batch, 0
    ).to(tl.float32)
    batch_kv = tl.load(KV + (row - 1) * KV_STRIDE + col, (col < D) & in_batch, 0)
    batch_score = tl.load(
        Score + (row - 1) * SCORE_STRIDE + col, (col < D) & in_batch, 0
    )
    p_kv = tl.where(in_batch, batch_kv, carried_kv)
    p_score = tl.where(in_batch, batch_score, carried_score)
    maximum = tl.maximum(p_score, score)
    e0 = libdevice.exp(p_score - maximum)
    e1 = libdevice.exp(score - maximum)
    denom = e0 + e1
    # Match pool_pairs: separate FP32 products, sum, then BF16 before RMSNorm.
    pooled = p_kv * tl.div_rn(e0, denom) + kv * tl.div_rn(e1, denom)
    rounded = pooled.to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(rounded * rounded, 0) / D + EPS)
    weight = tl.load(Weight + col, col < D, 0).to(tl.float32)
    latent = (rounded * inv) * weight
    tl.store(Latent + row * D + col, latent, col < D)
    tl.store(GroupPos + row, pos - pos % 2)
    loc = tl.load(OutLoc + row)
    tl.store(Slots + row, tl.maximum(loc, 0))


@triton.jit
def _c2_verify_ring_update_kernel(
    KV,
    Score,
    Pos,
    RawLoc,
    Req,
    StateKV,
    StateScore,
    N: tl.constexpr,
    KV_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    STATE_KV_STRIDE: tl.constexpr,
    STATE_SCORE_STRIDE: tl.constexpr,
    RING: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    req = tl.load(Req + row)
    pos = tl.load(Pos + row)
    raw = tl.load(RawLoc + row)
    next_req = tl.load(Req + row + RING, row + RING < N, -1)
    next_raw = tl.load(RawLoc + row + RING, row + RING < N, 0)
    # Request-major consecutive rows: only the last RING rows may write.
    keep = (raw != 0) & ((row + RING >= N) | (req != next_req) | (next_raw == 0))
    state_row = req * RING + pos % RING
    kv = tl.load(KV + row * KV_STRIDE + col, (col < D) & keep, 0)
    score = tl.load(Score + row * SCORE_STRIDE + col, (col < D) & keep, 0)
    tl.store(StateKV + state_row * STATE_KV_STRIDE + col, kv, (col < D) & keep)
    tl.store(StateScore + state_row * STATE_SCORE_STRIDE + col, score, (col < D) & keep)


def c2_verify_pool_norm(
    kv,
    score,
    pos,
    raw_out_loc,
    out_loc,
    req,
    state_kv,
    state_score,
    weight,
    eps: float,
    *,
    ring_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return BF16 pre-RoPE latent, group positions and nonnegative cache slots.

    Requires request-major consecutive positions and a unique group per live
    request. Graph padding uses raw_out_loc == 0 and never writes the live ring or
    the spare zero row. State halves may be strided views of interleaved storage.
    """
    assert kv.ndim == score.ndim == 2 and kv.shape == score.shape
    assert kv.dtype == score.dtype == torch.float32
    assert kv.stride(1) == score.stride(1) == 1
    n, d = kv.shape
    assert d in (128, 512) and ring_size > 0
    assert state_kv.shape == state_score.shape and state_kv.shape[1] == d
    assert state_kv.stride(1) == state_score.stride(1) == 1
    assert state_kv.dtype == state_score.dtype == torch.float32
    assert weight.shape == (d,) and weight.is_contiguous()
    assert weight.dtype in (torch.bfloat16, torch.float32)
    tensors = (score, pos, raw_out_loc, out_loc, req, state_kv, state_score, weight)
    assert kv.is_cuda and all(t.device == kv.device for t in tensors)
    assert all(
        t.ndim == 1 and t.numel() == n and t.is_contiguous()
        for t in (pos, raw_out_loc, out_loc, req)
    )
    latent = torch.empty((n, d), device=kv.device, dtype=torch.bfloat16)
    group_pos = torch.empty_like(pos)
    slots = torch.empty_like(out_loc)
    if n:
        layout = dict(
            KV_STRIDE=kv.stride(0),
            SCORE_STRIDE=score.stride(0),
            STATE_KV_STRIDE=state_kv.stride(0),
            STATE_SCORE_STRIDE=state_score.stride(0),
            RING=ring_size,
            D=d,
            BLOCK=triton.next_power_of_2(d),
            num_warps=4,
        )
        _c2_verify_pool_norm_kernel[(n,)](
            kv,
            score,
            pos,
            raw_out_loc,
            out_loc,
            req,
            state_kv,
            state_score,
            weight,
            latent,
            group_pos,
            slots,
            PAD_ROW=state_kv.shape[0] - 1,
            EPS=eps,
            enable_fp_fusion=False,
            **layout,
        )
        _c2_verify_ring_update_kernel[(n,)](
            kv,
            score,
            pos,
            raw_out_loc,
            req,
            state_kv,
            state_score,
            N=n,
            **layout,
        )
    return latent, group_pos, slots
