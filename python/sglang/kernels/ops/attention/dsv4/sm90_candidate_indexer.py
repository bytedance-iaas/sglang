"""Candidate compaction and same-layer query pairing with dense logits output.

Both paths use the BF16 scoring arithmetic of ``fp4_indexer``. They deliberately
keep the original logical column order and fully initialize unreachable columns
to -inf so the caller can retain its original torch.topk, including cutoff ties.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .fp4_indexer import (
    INDEX_HEAD_DIM,
    _fp4_index_k_tile,
    _fp4_index_logits_from_k,
    _fp4_index_logits_tile,
)


@triton.jit
def _compact_logits(
    Q,
    W,
    SLOTS,
    LENS,
    TABLE,
    BLOCKS,
    MASK,
    OUT,
    WIDTH: tl.constexpr,
    COUNT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    QB: tl.constexpr,
    QH: tl.constexpr,
    WB: tl.constexpr,
    BB: tl.constexpr,
    BC: tl.constexpr,
    MB: tl.constexpr,
    MC: tl.constexpr,
    HAS_MASK: tl.constexpr,
    H: tl.constexpr,
):
    row = tl.program_id(0)
    compact_col = tl.program_id(1) * 64 + tl.arange(0, 64)
    block = tl.load(
        BLOCKS + row * BB + (compact_col // BLOCK_SIZE) * BC,
        compact_col < COUNT * BLOCK_SIZE,
        -1,
    ).to(tl.int64)
    col = block * BLOCK_SIZE + compact_col % BLOCK_SIZE
    visible = tl.load(LENS + row)
    in_bounds = (block >= 0) & (col >= 0) & (col < WIDTH)
    valid = in_bounds & (col < visible)
    if HAS_MASK:
        valid = valid & tl.load(MASK + row * MB + col * MC, in_bounds, False)
    if tl.sum(valid.to(tl.int32), 0) > 0:
        slot = tl.load(SLOTS + row * WIDTH + col, valid, 0).to(tl.int64)
        score = _fp4_index_logits_tile(
            Q,
            W,
            TABLE,
            row,
            slot,
            valid,
            PAGE_SIZE,
            TABLE_STRIDE,
            QB,
            QH,
            WB,
            H,
            64,
        )
        # Blocks are unique within a row. No atomics or cross-CTA reductions.
        tl.store(OUT + row * WIDTH + col, score, valid)


def _check_inputs(q, weights, slots, lens, table, candidate_mask):
    assert q.is_cuda and q.dtype == torch.bfloat16
    assert q.ndim == 3 and q.shape[-1] == INDEX_HEAD_DIM
    rows = q.shape[0]
    assert weights.shape == q.shape[:2]
    assert slots.ndim == 2 and slots.shape[0] == rows
    assert slots.dtype in (torch.int32, torch.int64)
    assert lens.shape == (rows,) and lens.dtype in (torch.int32, torch.int64)
    assert table.dtype == torch.uint8 and table.ndim == 2 and table.stride(1) == 1
    for tensor in (weights, slots, lens, table):
        assert tensor.device == q.device
    if candidate_mask is not None:
        assert candidate_mask.shape == slots.shape
        assert candidate_mask.dtype == torch.bool and candidate_mask.device == q.device


def compact_candidate_logits(
    q: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
    blocks: torch.Tensor,
    block_size: int = 8,
    *,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score unique candidate block ids and scatter into dense FP32 [B, L].

    ``blocks`` is [B, K] int32/int64 with unique nonnegative block ids per row;
    -1 pads underfilled rows. Source publication supplies this device tensor.
    ``candidate_mask`` optionally restricts individual positions within blocks.
    All shape decisions are static; block ids, masks, lengths and pages remain
    live CUDA Graph inputs. Initialization and scatter are part of this call.
    """
    _check_inputs(q, weights, slots, lens, table, candidate_mask)
    assert 0 < block_size <= 64 and 64 % block_size == 0
    assert blocks.ndim == 2 and blocks.shape[0] == q.shape[0]
    assert blocks.dtype in (torch.int32, torch.int64) and blocks.device == q.device
    assert page_size > 0
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    slots = slots.contiguous()
    lens = lens.contiguous()
    rows, width = slots.shape
    out = torch.full((rows, width), -torch.inf, dtype=torch.float32, device=q.device)
    if rows and width and blocks.shape[1]:
        _compact_logits[(rows, triton.cdiv(blocks.shape[1] * block_size, 64))](
            q,
            weights,
            slots,
            lens,
            table,
            blocks,
            candidate_mask,
            out,
            width,
            blocks.shape[1],
            block_size,
            page_size,
            table.stride(0),
            q.stride(0),
            q.stride(1),
            weights.stride(0),
            blocks.stride(0),
            blocks.stride(1),
            candidate_mask.stride(0) if candidate_mask is not None else 0,
            candidate_mask.stride(1) if candidate_mask is not None else 0,
            candidate_mask is not None,
            q.shape[1],
            num_warps=4,
        )
    return out


@triton.jit
def _paired_logits(
    Q,
    W,
    SLOTS,
    LENS,
    TABLE,
    REQ,
    MASK,
    OUT,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    QB: tl.constexpr,
    QH: tl.constexpr,
    WB: tl.constexpr,
    RB: tl.constexpr,
    MB: tl.constexpr,
    MC: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    row0 = tl.program_id(0) * 2
    row1 = tl.minimum(row0 + 1, ROWS - 1)
    col = tl.program_id(1) * 64 + tl.arange(0, 64)
    visible0 = tl.load(LENS + row0)
    visible1 = tl.load(LENS + row1)
    valid0 = (col < WIDTH) & (col < visible0)
    valid1 = (row0 + 1 < ROWS) & (col < WIDTH) & (col < visible1)
    if HAS_MASK:
        valid0 = valid0 & tl.load(MASK + row0 * MB + col * MC, col < WIDTH, False)
        valid1 = valid1 & tl.load(MASK + row1 * MB + col * MC, col < WIDTH, False)
    if tl.sum((valid0 | valid1).to(tl.int32), 0) > 0:
        slot0 = tl.load(SLOTS + row0 * WIDTH + col, valid0, 0).to(tl.int64)
        slot1 = tl.load(SLOTS + row1 * WIDTH + col, valid1, 0).to(tl.int64)
        same_request = tl.load(REQ + row0 * RB) == tl.load(REQ + row1 * RB)
        # Guard on device: ragged verify rows may cross a request boundary, and
        # replay may change request ids or mappings. Neither may corrupt scores.
        same_slots = tl.sum((valid0 & valid1 & (slot0 != slot1)).to(tl.int32), 0) == 0
        if (row0 + 1 < ROWS) & same_request & same_slots:
            shared_slot = tl.where(valid0, slot0, slot1)
            low, high = _fp4_index_k_tile(
                TABLE, shared_slot, valid0 | valid1, PAGE_SIZE, TABLE_STRIDE, 64
            )
            # Keep H32 dots/reductions and their BF16 rounding unchanged. An
            # unrolled pair keeps both queries' intermediates live and reduces
            # occupancy on SM90. Store each result before processing the next
            # query, retaining only the shared decoded K across iterations.
            for qi in tl.range(0, 2, loop_unroll_factor=1, num_stages=1):
                valid = tl.where(qi == 0, valid0, valid1)
                score = _fp4_index_logits_from_k(
                    Q, W, row0 + qi, valid, low, high, QB, QH, WB, 32, 64
                )
                tl.store(OUT + (row0 + qi) * WIDTH + col, score, col < WIDTH)
        else:
            for qi in tl.range(0, 2, loop_unroll_factor=1, num_stages=1):
                row = tl.minimum(row0 + qi, ROWS - 1)
                valid = tl.where(qi == 0, valid0, valid1)
                slot = tl.where(qi == 0, slot0, slot1)
                score = tl.full((64,), -float("inf"), tl.float32)
                if tl.sum(valid.to(tl.int32), 0) > 0:
                    score = _fp4_index_logits_tile(
                        Q,
                        W,
                        TABLE,
                        row,
                        slot,
                        valid,
                        PAGE_SIZE,
                        TABLE_STRIDE,
                        QB,
                        QH,
                        WB,
                        32,
                        64,
                    )
                tl.store(
                    OUT + (row0 + qi) * WIDTH + col,
                    score,
                    (row0 + qi < ROWS) & (col < WIDTH),
                )
    else:
        tl.store(OUT + row0 * WIDTH + col, -float("inf"), col < WIDTH)
        tl.store(
            OUT + (row0 + 1) * WIDTH + col,
            -float("inf"),
            (row0 + 1 < ROWS) & (col < WIDTH),
        )


def paired_query_logits(
    q: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
    request_ids: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reuse K loading/dequantization for adjacent H32 same-layer queries.

    Each row keeps its own causal length, candidate mask, Q and head weights.
    Different requests, conflicting visible mappings and odd final rows use
    independent scoring. No K cache is shared between layers. Request ids and
    the overlap check are evaluated on every graph replay, without a host read.
    """
    _check_inputs(q, weights, slots, lens, table, candidate_mask)
    assert q.shape[1] == 32 and page_size > 0
    assert request_ids.shape == (q.shape[0],)
    assert request_ids.dtype in (torch.int32, torch.int64)
    assert request_ids.device == q.device
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    slots = slots.contiguous()
    lens = lens.contiguous()
    rows, width = slots.shape
    out = torch.empty((rows, width), dtype=torch.float32, device=q.device)
    if rows and width:
        _paired_logits[(triton.cdiv(rows, 2), triton.cdiv(width, 64))](
            q,
            weights,
            slots,
            lens,
            table,
            request_ids,
            candidate_mask,
            out,
            rows,
            width,
            page_size,
            table.stride(0),
            q.stride(0),
            q.stride(1),
            weights.stride(0),
            request_ids.stride(0),
            candidate_mask.stride(0) if candidate_mask is not None else 0,
            candidate_mask.stride(1) if candidate_mask is not None else 0,
            candidate_mask is not None,
            num_warps=4,
        )
    return out
