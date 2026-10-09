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
) -> torch.Tensor:
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
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]
