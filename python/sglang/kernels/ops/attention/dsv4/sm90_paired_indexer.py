"""32-head paired WGMMA scoring with prefix-bounded writes."""

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit

from .sm90_fp4_query_pack import pack_queries


@cache_once
def _jit_module():
    return load_jit(
        "sm90_fp4_paired_indexer",
        cuda_files=["sm90_fp4_grouped_indexer/entry.cuh"],
        cuda_wrappers=[("dispatch", "sm90_fp4_paired_indexer_dispatch")],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-DNDEBUG",
            "-DCUTE_USE_PACKED_TUPLE=1",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "--use_fast_math",
            "--ftz=false",
        ],
        extra_dependencies=["cutlass"],
    )


def paired_logits(
    q, weights, mapping, req, lens, table, page, ratio, width, group_size=6
):
    rows = q.shape[0]
    assert rows % 2 == 0 and group_size in (4, 6)
    assert q.is_contiguous() and weights.is_contiguous()
    assert q.dtype == weights.dtype == torch.bfloat16 and q.shape[1:] == (32, 128)
    paired = q.view(rows // 2, 64, 128)
    fallback = torch.empty((rows,), device=q.device, dtype=torch.int32)
    payload, scales = pack_queries(paired, clear_flags=fallback)
    payload = payload.view(rows // 2, 64, 64).view(torch.uint8)
    scales = scales.view(rows // 2, 64)
    weight = weights.view(rows // 2, 64)
    lens = lens.to(torch.int64)
    stride = (width + 3) // 4 * 4
    out = torch.empty((rows, stride), device=q.device, dtype=torch.float32)
    _jit_module().dispatch(
        payload,
        scales,
        weight,
        mapping,
        req,
        lens,
        table,
        out[:, :width],
        fallback,
        rows // 2,
        width,
        group_size // 2,
        page,
        ratio,
        payload.stride(0),
        payload.stride(1),
        scales.stride(0),
        weight.stride(0),
        mapping.stride(0),
        table.stride(0),
        stride,
        torch._C._cuda_getCurrentRawStream(q.device.index),
    )
    from .sm90_length_aware_indexer import prefix_logits

    return prefix_logits(
        q,
        weights,
        mapping,
        req,
        lens,
        table,
        page,
        ratio,
        width,
        _fallback=fallback,
        _out=out,
    )


@triton.jit
def _paired_candidate_lengths(
    VISIBLE, LENGTHS, FALLBACK, ROWS: tl.constexpr, LIMIT: tl.constexpr
):
    row = tl.program_id(0) * 256 + tl.arange(0, 256)
    n = tl.load(VISIBLE + row, row < ROWS, 0)
    use = n <= LIMIT
    tl.store(LENGTHS + row, tl.where(use, n, 0).to(tl.int64), row < ROWS)
    tl.store(FALLBACK + row, ~use, row < ROWS)


@triton.jit
def _paired_compact_scores(
    SCORES,
    OUT,
    LENS,
    VISIBLE,
    BLOCKS,
    PREFIX,
    FALLBACK,
    SSTR: tl.constexpr,
    OSTR: tl.constexpr,
    BSTR: tl.constexpr,
    BS: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(FALLBACK + row) != 0:
        return
    n = tl.load(LENS + row)
    vn = tl.load(VISIBLE + row)
    prefix = tl.load(PREFIX + row)
    for tile in range(tl.program_id(1), tl.cdiv(n, 256), tl.num_programs(1)):
        col = tile * 256 + tl.arange(0, 256)
        pos = col
        if prefix == 0:
            block = tl.load(BLOCKS + row * BSTR + col // BS, col < n, -1)
            pos = block * BS + col % BS
        valid = (col < n) & (pos >= 0) & (pos < vn)
        score = tl.load(SCORES + row * SSTR + pos, valid, -float("inf"))
        tl.store(OUT + row * OSTR + col, score, col < OSTR)


@triton.jit
def _paired_mask_scores(SCORES, MASK, LENS, SSTR: tl.constexpr, MSTR: tl.constexpr):
    row = tl.program_id(0)
    n = tl.load(LENS + row)
    for tile in range(tl.program_id(1), tl.cdiv(n, 256), tl.num_programs(1)):
        col = tile * 256 + tl.arange(0, 256)
        valid = (col < n) & tl.load(MASK + row * MSTR + col, col < n, False)
        score = tl.load(SCORES + row * SSTR + col, valid, -float("inf"))
        tl.store(SCORES + row * SSTR + col, score, col < SSTR)


def paired_prefix_logits(
    q,
    weights,
    mapping,
    req,
    lens,
    table,
    page,
    ratio,
    width,
    mask=None,
    *,
    candidates=None,
    visible=None,
    group_size=6,
):
    """Score FP4-rounded queries against paged index keys on SM90.

    ``q`` is contiguous BF16 [rows, 32, 128] after FP4 quantization and
    dequantization; arbitrary BF16 queries are not supported. Each consecutive
    group of four or six rows must belong to the same request. ``weights`` is
    contiguous BF16 [rows, 32], with scaling already included.

    Return FP32 [rows, ceil(width / 4) * 4]. Only each row's ``lens`` prefix
    is defined for consumers; the capacity tail must never be read by TopK.
    With compact candidates, ``lens`` bounds candidate positions, ``visible``
    bounds the original prefix, and output columns follow ``candidates.blocks``.
    Unsupported scale ranges use the original BF16 scoring implementation.
    """
    rows = q.shape[0]
    if candidates is None:
        scores = paired_logits(
            q, weights, mapping, req, lens, table, page, ratio, width, group_size
        )
        if mask is not None:
            _paired_mask_scores[(rows, 16)](
                scores, mask, lens, scores.stride(0), mask.stride(0), num_warps=4
            )
        return scores
    # Above this bound, compact scoring avoids scanning a much larger prefix.
    # The dispatch decision stays on the GPU and follows CUDA graph replay lengths.
    limit = min(candidates.width, 3 * width)
    native_lengths = torch.empty((rows,), device=q.device, dtype=torch.int64)
    fallback = torch.empty((rows,), device=q.device, dtype=torch.int32)
    _paired_candidate_lengths[(triton.cdiv(rows, 256),)](
        visible, native_lengths, fallback, rows, limit, num_warps=4
    )
    dense = paired_logits(
        q, weights, mapping, req, native_lengths, table, page, ratio, limit, group_size
    )
    stride = triton.cdiv(width, 4) * 4
    out = torch.empty((rows, stride), device=q.device, dtype=torch.float32)
    _paired_compact_scores[(rows, 16)](
        dense,
        out,
        lens,
        visible,
        candidates.blocks,
        candidates.is_prefix,
        fallback,
        dense.stride(0),
        stride,
        candidates.blocks.stride(0),
        candidates.block_size,
        num_warps=4,
    )
    from .sm90_length_aware_indexer import prefix_logits

    return prefix_logits(
        q,
        weights,
        mapping,
        req,
        lens,
        table,
        page,
        ratio,
        width,
        candidates=candidates,
        visible=visible,
        _fallback=fallback,
        _out=out,
    )
