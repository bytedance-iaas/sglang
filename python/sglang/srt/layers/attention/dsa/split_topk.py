"""Dispatch-level split-topk for the SM90 Q8KV8 sparse-prefill kernel.

Motivation (wave quantization at small s_q): the kernel launches a 1-D grid
of ``(h_q / 64) * s_q`` CTAs, one resident CTA per SM (its ~215 KB smem plan
caps occupancy at 1).  At the GLM-5.2 3.5k-il production shape
(s_q ~= 437/rank, h_q = 64) that is 437 CTAs = 3.31 waves on a 132-SM H200,
a measured +21% wave-quantization tail (0.533 vs 0.439 us/token).  Splitting
``topk = 2048`` into two 1024 halves and running the two half-calls
**concurrently on separate CUDA streams** doubles the resident CTA count
(874 CTAs = 6.62 waves, ~+6% tail) without touching the CUDA kernel.  The
same two calls issued back-to-back on ONE stream serialize completely and
keep the original 3.31-wave tail per call, so the overlap is the whole win;
``overlap=False`` exists for A/B measurement only.

The two halves are merged with a standard streaming-softmax LSE merge in a
single Triton kernel.

LSE semantics (must mirror ``kernel.cuh`` epilogue, see lines ~752-753, and
the fp32 reference in ``test/registered/jit/test_sparse_mla_q8kv8_prefill_sm90.py``):

    lse        = ln(sum_i exp(score_i - max)) + max   (natural log; scores
                 already include ``sm_scale``), so exp(lse) is the true
                 unnormalized softmax denominator.
    max_logits = max_i score_i

    Rows with NO valid (non ``-1``) index in a call:
        out = 0, max_logits = -inf, lse = **+inf**  (note: plus infinity).

The +inf empty marker is unambiguous: any row with >= 1 valid token has a
denominator in [1, topk] (the max term contributes exactly 1), hence a
finite lse.  The combine kernel uses ``lse == +inf`` to give empty halves
zero weight and to reproduce the kernel's own (0, -inf, +inf) triple when
both halves are empty.

Index-slicing constraint: the JIT entry (``entry.cuh``) computes
``stride_indices_s_q = h_kv * topk`` from the *passed* topk rather than the
tensor's stride, so ``indices[:, :, :half]`` views cannot be passed
directly; both halves are packed into one contiguous ``[2, s_q, h_kv,
half]`` buffer with a single strided copy.

The ``-1`` sentinel remains safe in both kernel dispatch paths at
``topk = half``: with ``topk_length`` the load is predicated off for
``t < 0``; without it, ``-1`` maps to ``pad_base + slot`` with
``pad_base = s_kv - half``, which stays inside the ``topk`` trailing pad
rows the production integration appends (``half < topk``), and the slot is
masked (-inf) out of the softmax either way.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton LSE-merge kernel
# ---------------------------------------------------------------------------


@triton.jit
def _split2_lse_combine_kernel(
    o1_ptr,  # [s_q * h_q, D_V] bf16 (contiguous view)
    o2_ptr,  # [s_q * h_q, D_V] bf16
    lse1_ptr,  # [s_q * h_q] f32
    lse2_ptr,  # [s_q * h_q] f32
    ml1_ptr,  # [s_q * h_q] f32
    ml2_ptr,  # [s_q * h_q] f32
    out_ptr,  # [s_q * h_q, D_V] bf16
    ml_out_ptr,  # [s_q * h_q] f32
    lse_out_ptr,  # [s_q * h_q] f32
    D_V: tl.constexpr,
):
    """One program per (query row, head): o = w1*o1 + w2*o2 with
    wi = exp(lse_i - logaddexp(lse1, lse2)), fp32 math, bf16 store.

    Empty-half guards (lse == +inf marks a half with zero denominator):
      * one empty half  -> its weight is exactly 0, output = other half;
      * both empty      -> (0, -inf, +inf), bit-matching the kernel's own
        no-valid-token epilogue.
    """
    pid = tl.program_id(0)
    lse1 = tl.load(lse1_ptr + pid)
    lse2 = tl.load(lse2_ptr + pid)

    inf = float("inf")
    e1 = lse1 == inf
    e2 = lse2 == inf
    both_empty = e1 & e2

    # Map the +inf empty marker to -inf so it drops out of the logsumexp.
    l1 = tl.where(e1, -inf, lse1)
    l2 = tl.where(e2, -inf, lse2)
    m = tl.maximum(l1, l2)
    # -inf - (-inf) = nan; keep the exp arguments finite when both empty.
    m_safe = tl.where(both_empty, 0.0, m)
    a1 = tl.exp(l1 - m_safe)  # exp(-inf - finite) = 0 for an empty half
    a2 = tl.exp(l2 - m_safe)
    denom = a1 + a2
    denom_safe = tl.where(both_empty, 1.0, denom)
    w1 = a1 / denom_safe
    w2 = a2 / denom_safe

    offs = tl.arange(0, D_V)
    o1 = tl.load(o1_ptr + pid * D_V + offs).to(tl.float32)
    o2 = tl.load(o2_ptr + pid * D_V + offs).to(tl.float32)
    # where() on the products (not just the weights): bulletproof against a
    # non-zero payload in an empty half's out rows (the kernel writes 0, but
    # nothing downstream should depend on that).
    o = tl.where(e1, 0.0, w1 * o1) + tl.where(e2, 0.0, w2 * o2)
    tl.store(out_ptr + pid * D_V + offs, o.to(tl.bfloat16))

    ml1 = tl.load(ml1_ptr + pid)
    ml2 = tl.load(ml2_ptr + pid)
    # max(-inf, -inf) = -inf reproduces the kernel's empty-row max_logits.
    tl.store(ml_out_ptr + pid, tl.maximum(ml1, ml2))
    lse_c = tl.where(both_empty, inf, m + tl.log(denom_safe))
    tl.store(lse_out_ptr + pid, lse_c)


def combine_split2(
    o1: torch.Tensor,
    ml1: torch.Tensor,
    lse1: torch.Tensor,
    o2: torch.Tensor,
    ml2: torch.Tensor,
    lse2: torch.Tensor,
    out: torch.Tensor,
    ml_out: torch.Tensor,
    lse_out: torch.Tensor,
) -> None:
    """LSE-merge two half-topk kernel results into (out, ml_out, lse_out).

    All tensors must be contiguous; o*/out are [s_q, h_q, d_v] bf16, the
    rest are [s_q, h_q] f32.
    """
    s_q, h_q, d_v = o1.shape
    for t in (o1, ml1, lse1, o2, ml2, lse2, out, ml_out, lse_out):
        if not t.is_contiguous():
            raise ValueError("combine_split2 requires contiguous tensors")
    _split2_lse_combine_kernel[(s_q * h_q,)](
        o1,
        o2,
        lse1,
        lse2,
        ml1,
        ml2,
        out,
        ml_out,
        lse_out,
        D_V=d_v,
        # Tuned at the production shape (s_q=437, h=64, d_v=512, H200):
        # 24.1 us vs a 22.7 us same-traffic torch.add roofline (86.6 MB);
        # num_warps 4/8 regress to 27.7/39.8 us.
        num_warps=2,
    )


# ---------------------------------------------------------------------------
# Scratch buffers (caller-owned, reusable across calls / layers)
# ---------------------------------------------------------------------------


class Split2Buffers(msgspec.Struct):
    """Persistent scratch for :func:`sparse_mla_q8kv8_prefill_fwd_split2`.

    Capacity is the allocated s_q; calls with a smaller s_q use leading
    slices (all remain contiguous).  ``tl_halves`` is only needed when
    ``topk_length`` is passed.
    """

    idx_halves: torch.Tensor  # [2, s_q, h_kv, topk // 2] int32
    o_halves: torch.Tensor  # [2, s_q, h_q, d_v] bf16
    ml_halves: torch.Tensor  # [2, s_q, h_q] f32
    lse_halves: torch.Tensor  # [2, s_q, h_q] f32
    out: torch.Tensor  # [s_q, h_q, d_v] bf16
    ml_out: torch.Tensor  # [s_q, h_q] f32
    lse_out: torch.Tensor  # [s_q, h_q] f32
    tl_halves: Optional[torch.Tensor] = None  # [2, s_q] int32

    @classmethod
    def allocate(
        cls,
        s_q: int,
        h_q: int,
        h_kv: int,
        half_topk: int,
        d_v: int,
        device: torch.device,
        with_topk_length: bool = False,
    ) -> Split2Buffers:
        return cls(
            idx_halves=torch.empty(
                (2, s_q, h_kv, half_topk), dtype=torch.int32, device=device
            ),
            o_halves=torch.empty(
                (2, s_q, h_q, d_v), dtype=torch.bfloat16, device=device
            ),
            ml_halves=torch.empty((2, s_q, h_q), dtype=torch.float32, device=device),
            lse_halves=torch.empty((2, s_q, h_q), dtype=torch.float32, device=device),
            out=torch.empty((s_q, h_q, d_v), dtype=torch.bfloat16, device=device),
            ml_out=torch.empty((s_q, h_q), dtype=torch.float32, device=device),
            lse_out=torch.empty((s_q, h_q), dtype=torch.float32, device=device),
            tl_halves=(
                torch.empty((2, s_q), dtype=torch.int32, device=device)
                if with_topk_length
                else None
            ),
        )


# ---------------------------------------------------------------------------
# Side-stream / event cache (per device)
# ---------------------------------------------------------------------------

_side_stream_cache: dict[
    int, tuple[torch.cuda.Stream, torch.cuda.Event, torch.cuda.Event]
] = {}


def _get_side_stream(
    device: torch.device,
) -> tuple[torch.cuda.Stream, torch.cuda.Event, torch.cuda.Event]:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    entry = _side_stream_cache.get(idx)
    if entry is None:
        entry = (
            torch.cuda.Stream(device=device),
            torch.cuda.Event(),
            torch.cuda.Event(),
        )
        _side_stream_cache[idx] = entry
    return entry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sparse_mla_q8kv8_prefill_fwd_split2(
    q: torch.Tensor,  # [s_q, h_q, d_qk], float8_e4m3fn
    kv: torch.Tensor,  # [s_kv, h_kv, d_qk], float8_e4m3fn
    indices: torch.Tensor,  # [s_q, h_kv, topk], int32, contiguous
    sm_scale: float,
    q_scale: torch.Tensor,  # scalar tensor on GPU, float32
    kv_scale: torch.Tensor,  # scalar tensor on GPU, float32
    d_v: int = 512,
    topk_length: Optional[torch.Tensor] = None,  # [s_q], int32
    *,
    out_bufs: Optional[Split2Buffers] = None,
    overlap: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the Q8KV8 sparse prefill as two half-topk calls + an LSE merge.

    Same contract as ``sparse_mla_q8kv8_prefill_fwd`` (no ``attn_sink``
    support: the sink term enters only the output denominator, once, so a
    naive per-half merge would double-count it; production passes None).

    ``overlap=True`` (default) runs the second half on a cached side stream
    (fork/join via cached CUDA events) so the two 1-CTA-per-SM grids fill
    waves together — the entire point of the split.  ``overlap=False``
    issues both on the current stream (for A/B measurement; expected to be
    *slower* than a single full-topk call).

    ``topk_length`` (production backscan, min 1 per row) splits as
      half1 = clamp(topk_length, 1, half); half2 = clamp(topk_length - half, 1, half)
    A pad-only half (its clamped length lands on a ``-1`` slot) contributes
    zero and is detected in the merge via its lse == +inf marker.

    Returns ``(out, max_logits, lse)`` views into ``out_bufs`` (allocated
    fresh when not supplied), with the exact single-call semantics:
    ``lse = logaddexp(lse1, lse2)``, ``max_logits = max(ml1, ml2)``, and
    all-pad rows produce (0, -inf, +inf).
    """
    from sglang.jit_kernel.sparse_mla_q8kv8_prefill_sm90 import (
        sparse_mla_q8kv8_prefill_fwd,
    )

    s_q, h_q, _ = q.shape
    h_kv = kv.shape[1]
    topk = indices.shape[2]
    if topk % 256 != 0:
        raise ValueError(
            f"split2 requires topk % 256 == 0 (each half must stay a multiple "
            f"of 128), got topk={topk}"
        )
    if not indices.is_contiguous():
        raise ValueError("split2 requires contiguous indices")
    half = topk // 2
    device = q.device

    bufs = out_bufs
    if bufs is None:
        bufs = Split2Buffers.allocate(
            s_q, h_q, h_kv, half, d_v, device, topk_length is not None
        )
    else:
        if (
            bufs.idx_halves.shape[1] < s_q
            or bufs.idx_halves.shape[3] != half
            or bufs.o_halves.shape[2] != h_q
        ):
            raise ValueError(
                f"out_bufs capacity mismatch: need s_q>={s_q}, half={half}, "
                f"h_q={h_q}; got idx_halves {tuple(bufs.idx_halves.shape)}, "
                f"o_halves {tuple(bufs.o_halves.shape)}"
            )
        if topk_length is not None and bufs.tl_halves is None:
            raise ValueError("out_bufs must carry tl_halves when topk_length is used")

    # Pack BOTH contiguous index halves with ONE strided copy (the JIT entry
    # derives the s_q stride from topk, so sliced views cannot be passed).
    idx_halves = bufs.idx_halves[:, :s_q]
    idx_halves.copy_(indices.view(s_q, h_kv, 2, half).permute(2, 0, 1, 3))
    idx1, idx2 = idx_halves[0], idx_halves[1]

    tl1 = tl2 = None
    if topk_length is not None:
        tl_halves = bufs.tl_halves[:, :s_q]
        tl1, tl2 = tl_halves[0], tl_halves[1]
        torch.clamp(topk_length, 1, half, out=tl1)
        torch.sub(topk_length, half, out=tl2)
        tl2.clamp_(1, half)

    o1, o2 = bufs.o_halves[0, :s_q], bufs.o_halves[1, :s_q]
    ml1, ml2 = bufs.ml_halves[0, :s_q], bufs.ml_halves[1, :s_q]
    lse1, lse2 = bufs.lse_halves[0, :s_q], bufs.lse_halves[1, :s_q]
    out = bufs.out[:s_q]
    ml_out = bufs.ml_out[:s_q]
    lse_out = bufs.lse_out[:s_q]

    def _half_call(idx, tl_half, o, ml, lse):
        sparse_mla_q8kv8_prefill_fwd(
            q=q,
            kv=kv,
            indices=idx,
            sm_scale=sm_scale,
            q_scale=q_scale,
            kv_scale=kv_scale,
            d_v=d_v,
            attn_sink=None,
            topk_length=tl_half,
            out=o,
            max_logits=ml,
            lse=lse,
        )

    if overlap:
        side_stream, ev_fork, ev_join = _get_side_stream(device)
        cur_stream = torch.cuda.current_stream(device)
        # Fork AFTER the index/topk_length prep ops so the side stream sees
        # their results.
        ev_fork.record(cur_stream)
        side_stream.wait_event(ev_fork)
        with torch.cuda.stream(side_stream):
            _half_call(idx2, tl2, o2, ml2, lse2)
        _half_call(idx1, tl1, o1, ml1, lse1)
        ev_join.record(side_stream)
        cur_stream.wait_event(ev_join)
    else:
        _half_call(idx1, tl1, o1, ml1, lse1)
        _half_call(idx2, tl2, o2, ml2, lse2)

    combine_split2(o1, ml1, lse1, o2, ml2, lse2, out, ml_out, lse_out)
    return out, ml_out, lse_out
