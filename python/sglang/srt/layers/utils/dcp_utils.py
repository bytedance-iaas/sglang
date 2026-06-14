"""Decode Context Parallel (DCP) helpers.

Ported from upstream PR #14194 (Decode CP for MLA models). Provides:

- ``_correct_attn_cp_out_kernel`` Triton kernel that, given each rank's local
  attention output ``[B, H, D]`` and the all-gathered LSE table ``[N, B, H]``,
  computes the merged final LSE and the per-rank correction factor, and
  rewrites the output as ``correction * local_out`` into a transposed
  ``[H, B, D]`` buffer ready for ``reduce_scatter_along_dim(out, dim=0)``.
- ``correct_attn_out`` Python wrapper around the kernel.
- ``cp_lse_ag_out_rs`` higher-level helper: all-gather LSE → correct → reduce
  scatter back, returning the merged attention output for the caller's rank.

These are all no-ops when ``cp_group.world_size == 1`` so they are safe to
call unconditionally from DSv4 attention forward.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.distributed.parallel_state import GroupCoordinator


class CPTritonContext:
    """Cache compiled Triton kernels for repeated DCP calls.

    The decode/verify path may call ``correct_attn_out`` once per layer per
    forward; recompiling Triton metadata each call wastes cuda graph capture
    time. We hold a single ``triton.runtime.Autotuner`` style trampoline that
    forwards positional+keyword args to the cached kernel handle.
    """

    def __init__(self) -> None:
        self._compiled = None

    def call_kernel(self, kernel, grid, *args, **kwargs):
        # Triton's @jit kernels are themselves callable objects, so we just
        # delegate. The ctx exists primarily to give Phase 3d a hook point if
        # we later need to swap in autotuner / persistent kernel handles.
        return kernel[grid](*args, **kwargs)


@triton.jit
def _correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    new_outputs_stride_H,
    new_outputs_stride_B,
    new_outputs_stride_D,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    N_ROUNDED: tl.constexpr,
):
    """Apply all-gathered LSEs to correct each rank's attention output.

    Args:
        outputs_ptr: Pointer to local attention output ``[B, H, D]``.
        new_output_ptr: Pointer to merged output ``[H, B, D]`` (written).
        lses_ptr: Pointer to all-gathered LSE table ``[N, B, H]``.
        vlse_ptr: Pointer to merged final LSE ``[B, H]`` (written).
        lse_idx: This rank's index inside the DCP group (``cp_rank``).
    """
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)

    # Use int32 for offsets where possible to reduce register pressure.
    b_i32 = batch_idx.to(tl.int32)
    h_i32 = head_idx.to(tl.int32)

    # Vectorized load of LSE values: shape = [N].
    num_n_offsets = tl.arange(0, N_ROUNDED)
    lse_offsets = (
        num_n_offsets * lses_stride_N + b_i32 * lses_stride_B + h_i32 * lses_stride_H
    )

    # Online softmax over LSEs from all ranks: log2-sum-exp, normalised by
    # the max for numerical stability.
    lse = tl.load(lses_ptr + lse_offsets)
    neg_inf = float("-inf")
    lse = tl.where((lse != lse) | (lse == float("inf")), neg_inf, lse)

    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == neg_inf, 0.0, lse_max)
    lse = lse - lse_max
    lse_exp = tl.exp2(lse)
    lse_acc = tl.sum(lse_exp, axis=0)
    final_lse = tl.log2(lse_acc) + lse_max

    # Correction factor for this rank.
    lse_offset = lse_idx * lses_stride_N + b_i32 * lses_stride_B + h_i32 * lses_stride_H
    local_lse = tl.load(lses_ptr + lse_offset)
    lse_diff = local_lse - final_lse
    lse_diff = tl.where(
        (lse_diff != lse_diff) | (lse_diff == float("inf")),
        neg_inf,
        lse_diff,
    )
    factor = tl.exp2(lse_diff)

    tl.store(vlse_ptr + b_i32 * lses_stride_B + h_i32 * lses_stride_H, final_lse)

    # Apply correction factor to local output and store into transposed
    # ``[H, B, D]`` buffer (which is then reduce-scattered along dim=0).
    d_offsets = tl.arange(0, HEAD_DIM)
    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )
    new_output_offsets = (
        head_idx * new_outputs_stride_H
        + batch_idx * new_outputs_stride_B
        + d_offsets * new_outputs_stride_D
    )

    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor
    tl.store(new_output_ptr + new_output_offsets, output)


def correct_attn_out(
    out: torch.Tensor,
    lses: torch.Tensor,
    cp_rank: int,
    ctx: Optional[CPTritonContext],
    new_output: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply LSE correction and produce the transposed output buffer.

    Args:
        out: Local attention output ``[B, H, D]`` (or ``[B, 1, H, D]``).
        lses: All-gathered LSE table ``[N, B, H]``.
        cp_rank: Index of the current rank inside the DCP group.
        ctx: Optional Triton context cache.
        new_output: Pre-allocated ``[H, B, D]`` float32 output buffer.

    Returns:
        ``(new_output, final_lse)`` where ``final_lse`` has the same B/H
        strides as ``lses`` so it can be fed into a follow-up correction
        without an extra contiguous copy.
    """
    if ctx is None:
        ctx = CPTritonContext()

    # Normalise to 3-D views.
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    assert out.ndim == 3, (
        f"expected out [B, H, D] or [B, 1, H, D], got {tuple(out.shape)}"
    )

    if lses.ndim == 4 and lses.shape[-1] == 1:
        lses = lses.squeeze(-1)
    if lses.ndim == 4 and lses.shape[1] == 1:
        lses = lses.squeeze(1)
    assert lses.ndim == 3, (
        f"expected lses [N, B, H] (optionally with a 1-sized extra dim), "
        f"got {tuple(lses.shape)}"
    )

    B, H, D = out.shape
    N = lses.shape[0]

    o_sB, o_sH, o_sD = out.stride()
    l_sN, l_sB, l_sH = lses.stride()
    no_sH, no_sB, no_sD = new_output.stride()

    # Allocate the merged LSE with the same B/H strides as ``lses`` so writes
    # land correctly even when ``lses`` is a non-contiguous view.
    lse = torch.empty_strided(
        (B, H), (l_sB, l_sH), device=lses.device, dtype=lses.dtype
    )

    grid = (B, H, 1)
    ctx.call_kernel(
        _correct_attn_cp_out_kernel,
        grid,
        out,
        new_output,
        lses,
        lse,
        o_sB,
        o_sH,
        o_sD,
        l_sN,
        l_sB,
        l_sH,
        no_sH,
        no_sB,
        no_sD,
        cp_rank,
        HEAD_DIM=D,
        N_ROUNDED=N,
    )
    return new_output, lse


def cp_lse_ag_out_rs(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: Optional[CPTritonContext] = None,
) -> torch.Tensor:
    """All-gather LSE, correct local outputs, then reduce-scatter back.

    Args:
        cp_attn_out: This rank's local attention output ``[B, H, D]``.
        cp_attn_lse: This rank's local LSE ``[B, H]``.
        cp_group: The DCP ``GroupCoordinator`` (returned by
            ``get_dcp_group()``).
        ctx: Optional Triton context cache.

    Returns:
        The cross-rank-merged attention output for this rank's slice of the
        head dimension, shape ``[H // world_size, B, D]``. When
        ``world_size == 1`` simply returns ``cp_attn_out`` unchanged.
    """
    if cp_group.world_size == 1:
        return cp_attn_out

    if ctx is None:
        ctx = CPTritonContext()

    # The kernel expects an [H, B, D] output buffer; we'll reduce-scatter
    # along dim=0 (heads) afterwards.
    new_output = cp_attn_out.new_empty(
        cp_attn_out.transpose(0, 1).shape, dtype=torch.float32
    )
    # Clone before the collective so CUDA graph replay does not keep
    # FlashMLA's LSE storage alive longer than necessary.
    cp_attn_lse = cp_attn_lse.to(torch.float32).clone()

    lses = cp_group.all_gather(cp_attn_lse, dim=0).view(
        (cp_group.world_size,) + cp_attn_lse.shape
    )
    out, _ = correct_attn_out(
        cp_attn_out, lses, cp_group.rank_in_group, ctx, new_output
    )
    out = cp_group.reduce_scatter_along_dim(out, dim=0)
    return out.to(cp_attn_out.dtype)
