from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.dsv4.v41_indexer.types import CandidateMetadata


@dataclass
class CandidateMasks(CandidateMetadata):
    mask: Optional[torch.Tensor] = None  # decode: [rows, width] bool
    request_masks: Optional[List[torch.Tensor]] = None  # prefill: [rows_b, lc_b] each
    # Optional unique source-selected block ids, with -1 for unreachable blocks.
    # Reused by SM90 consumers; never compact the logits supplied to torch.topk.
    block_indices: Optional[torch.Tensor] = None
    block_size: int = 0

    def tail(self, rows_per_request: List[int]) -> CandidateMasks:
        assert self.request_masks is not None
        return CandidateMasks(
            request_masks=[
                mask[-rows:] if rows else mask[:0]
                for mask, rows in zip(self.request_masks, rows_per_request, strict=True)
            ]
        )


@dataclass
class CandidateBlocks(CandidateMetadata):
    """Sorted SM90 candidate block IDs; the device lengths refresh on replay.

    Full masks are materialized only for fallback consumers. Do not cache them:
    blocks may have been updated by graph replay since the last conversion.
    """

    blocks: torch.Tensor  # [rows, topk_blocks] int32, -1 pads invalid blocks
    lengths: torch.Tensor  # [rows] int32, selected block capacity in positions
    is_prefix: torch.Tensor  # [rows] int32, selected blocks are consecutive from 0
    width: int  # original source mask width, in compressed positions
    block_size: int


def published_masks(candidate) -> CandidateMasks:
    if isinstance(candidate, CandidateBlocks):
        from sglang.kernels.ops.attention.dsv4.sm90_length_aware_indexer import (
            materialize_candidate_mask,
        )

        return CandidateMasks(mask=materialize_candidate_mask(candidate))
    assert isinstance(candidate, CandidateMasks), "candidate masks missing"
    return candidate


def mask_topk_scores(
    scores: torch.Tensor,
    indices: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Keep masked indexer scores out of attention even when top-k underfills."""
    columns = indices.to(torch.int64)
    if offsets is not None:
        columns = columns - offsets[:, None]
    selected_scores = scores.gather(1, columns.clamp(0, scores.shape[1] - 1))
    valid = (
        (columns >= 0) & (columns < scores.shape[1]) & (selected_scores > -torch.inf)
    )
    return indices.masked_fill(~valid, -1)


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
    *,
    return_block_indices: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Level one of the two-level top-k: a bool mask over positions keeping the
    topk_blocks best-scoring blocks per query. Unreachable positions are already -inf
    in logits, so an all -inf block means not reachable yet; the block holding the
    query's newest position is always kept."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    mask = keep.repeat_interleave(block_size, dim=-1)[..., :width]
    if return_block_indices:
        blocks = top.indices.masked_fill(~(top.values > -torch.inf), -1).to(torch.int32)
        return mask, blocks
    return mask


def should_move_sm90_candidate_mask(width: int) -> bool:
    """Width policy; the caller separately controls the candidate-mask flag.

    Use the static logits width, including CUDA Graph padding. Reading live
    lengths on the host would synchronize and freeze a replay-dependent choice.
    Even short live sequences can benefit from fusing a large padded post-mask.
    """
    from sglang.srt.environ import envs

    return width >= envs.SGLANG_DSV41_SM90_CANDIDATE_MASK_MIN_WIDTH.get()


def sm90_index_logits_with_candidates(
    q: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
    *,
    indexer,
    candidate_metadata: Optional[CandidateMetadata],
    request_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[CandidateMetadata]]:
    """SM90 source publication and consumer masking, before the original TopK.

    The source always scores the full context. Consumers keep its per-query
    mask, including distinct rows of one request under speculative verification.
    Short logits widths retain dense scoring followed by the candidate mask.
    Optional compaction reuses the source's block TopK and scatters scores back
    into the original dense layout. Query pairing reuses K only within this
    layer, with device-side request/mapping checks. Compact consumers take
    priority when both experimental paths are enabled. TopK stays unchanged.
    """
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_logits_decode
    from sglang.srt.environ import envs

    consume = None
    if indexer.uses_candidates and not indexer.is_candidate_source:
        consume = published_masks(candidate_metadata).mask
        assert torch.is_tensor(consume) and consume.shape[0] == q.shape[0], (
            "candidate mask missing for decode"
        )
        consume = consume[:, : slots.shape[1]]
    move_mask = (
        consume is not None
        and envs.SGLANG_DSV41_SM90_CANDIDATE_MASK.get()
        and should_move_sm90_candidate_mask(slots.shape[1])
    )
    compact_enabled = (
        envs.SGLANG_DSV41_SM90_CANDIDATE_MASK.get()
        and envs.SGLANG_DSV41_SM90_CANDIDATE_COMPACT.get()
        and should_move_sm90_candidate_mask(slots.shape[1])
    )
    compact_blocks = (
        published_masks(candidate_metadata).block_indices
        if consume is not None and compact_enabled
        else None
    )
    use_compact = (
        compact_blocks is not None
        and candidate_metadata.block_size in (1, 2, 4, 8, 16, 32, 64)
        and compact_blocks.shape[1] * candidate_metadata.block_size < slots.shape[1]
    )
    if use_compact:
        from sglang.kernels.ops.attention.dsv4.sm90_candidate_indexer import (
            compact_candidate_logits,
        )

        scores = compact_candidate_logits(
            q,
            weights,
            slots,
            lens,
            table,
            page_size,
            compact_blocks,
            candidate_metadata.block_size,
            candidate_mask=consume,
        )
    elif (
        envs.SGLANG_DSV41_SM90_QUERY_K_REUSE.get()
        and request_ids is not None
        and q.shape[0] > 1
        and q.shape[1] == 32
    ):
        from sglang.kernels.ops.attention.dsv4.sm90_candidate_indexer import (
            paired_query_logits,
        )

        scores = paired_query_logits(
            q,
            weights,
            slots,
            lens,
            table,
            page_size,
            request_ids,
            candidate_mask=consume if move_mask else None,
        )
    else:
        scores = fp4_index_logits_decode(
            q,
            weights,
            slots,
            lens,
            table,
            page_size,
            candidate_mask=consume if move_mask else None,
        )
    if indexer.is_candidate_source:
        selected = select_candidate_blocks(
            scores,
            lens[:, None],
            topk_blocks=indexer.candidate_topk_blocks,
            block_size=indexer.candidate_block_size,
            return_block_indices=compact_enabled,
        )
        if compact_enabled:
            mask, blocks = selected
            candidate_metadata = CandidateMasks(
                mask=mask, block_indices=blocks, block_size=indexer.candidate_block_size
            )
        else:
            candidate_metadata = CandidateMasks(mask=selected)
    elif consume is not None and not move_mask:
        # Dense oracle for A/B, including the original choice among ties.
        scores = scores.masked_fill(~consume, -torch.inf)
    return scores, candidate_metadata
