"""Request-local attention state carried between DeepSeek V4 PP stages."""

import torch


def remap_sparse_slots(
    slots: torch.Tensor,
    source_pages: torch.Tensor,
    target_pages: torch.Tensor,
    slots_per_page: int,
    num_pages: torch.Tensor,
) -> torch.Tensor:
    """Translate physical slots through each query's logical page table.

    Radix eviction and HiCache reload may allocate different physical pages on
    each stage. Even shared prefixes can have different sharing on each stage,
    so a single global physical-to-physical map is insufficient.
    """
    if slots.numel() == 0:
        return slots.clone()
    logical_ids = torch.arange(source_pages.shape[-1], device=source_pages.device)
    active_pages = logical_ids < num_pages.unsqueeze(-1)
    searchable_pages = source_pages.masked_fill(
        ~active_pages, torch.iinfo(source_pages.dtype).max
    )
    sorted_pages, logical_pages = searchable_pages.sort(dim=-1)
    physical_pages = slots.clamp_min(0) // slots_per_page
    matches = torch.searchsorted(
        sorted_pages.contiguous(), physical_pages.contiguous()
    ).clamp_max(source_pages.shape[-1] - 1)
    logical = logical_pages.gather(-1, matches)
    remapped = (
        target_pages.gather(-1, logical) * slots_per_page + slots % slots_per_page
    )
    valid = (slots >= 0) & (sorted_pages.gather(-1, matches) == physical_pages)
    return torch.where(valid, remapped, -1).to(slots.dtype)


def export_candidate_metadata(tensors: dict, candidate, compress_ratio) -> None:
    """Carry logical candidates; the receiving stage rebuilds physical addresses."""
    from .candidate_indexer import CandidateBlocks, CandidateMasks
    from .v41_indexer.dense_blocks import BlockIds
    from .v41_indexer.sparse_table import _SparsePrefillTable, _SparseTable

    if candidate is None:
        return
    tensors["pp_candidate_ratio"] = compress_ratio
    if isinstance(candidate, CandidateMasks):
        tensors["pp_candidate_kind"] = "masks"
        if candidate.mask is not None:
            tensors["pp_candidate_mask"] = candidate.mask
        if candidate.request_masks is not None:
            tensors["pp_candidate_count"] = len(candidate.request_masks)
            for index, mask in enumerate(candidate.request_masks):
                tensors[f"pp_candidate_{index}"] = mask
    elif isinstance(candidate, CandidateBlocks):
        tensors["pp_candidate_kind"] = "sm90_blocks"
        for key in ("blocks", "lengths", "is_prefix", "width", "block_size"):
            tensors[f"pp_candidate_{key}"] = getattr(candidate, key)
    elif isinstance(candidate, BlockIds):
        tensors["pp_candidate_kind"] = "dense_blocks"
        tensors["pp_candidate_blocks"] = candidate.blocks
        tensors["pp_candidate_rows_per_request"] = candidate.rows_per_request
        if candidate.decode_mask is not None:
            tensors["pp_candidate_decode_mask"] = candidate.decode_mask
    elif isinstance(candidate, _SparseTable):
        torch.cuda.current_stream().wait_event(candidate.ready)
        tensors["pp_candidate_kind"] = "sparse_decode"
        tensors["pp_candidate_blocks"] = candidate.blocks
        tensors["pp_candidate_valid_lens"] = candidate.valid_lens
        if isinstance(candidate, _SparsePrefillTable):
            tensors["pp_candidate_kind"] = "sparse_prefill"
            for key in (
                "compress_lens",
                "request_ids",
                "page_size",
                "rows_per_request",
            ):
                tensors[f"pp_candidate_{key}"] = getattr(candidate, key)
    else:
        raise TypeError(
            f"Unsupported PP candidate metadata: {type(candidate).__name__}"
        )


def restore_candidate_metadata(tensors: dict, metadata, forward_batch):
    from .candidate_indexer import CandidateBlocks, CandidateMasks
    from .v41_indexer.dense_blocks import BlockIds

    kind = tensors.get("pp_candidate_kind")
    if kind is None:
        return None
    if kind == "masks":
        count = tensors.get("pp_candidate_count")
        return CandidateMasks(
            mask=tensors.get("pp_candidate_mask"),
            request_masks=(
                [tensors[f"pp_candidate_{i}"] for i in range(count)]
                if count is not None
                else None
            ),
        )
    if kind == "sm90_blocks":
        return CandidateBlocks(
            **{
                key: tensors[f"pp_candidate_{key}"]
                for key in ("blocks", "lengths", "is_prefix", "width", "block_size")
            }
        )
    if kind == "dense_blocks":
        return BlockIds(
            blocks=tensors["pp_candidate_blocks"],
            rows_per_request=tensors["pp_candidate_rows_per_request"],
            decode_mask=tensors.get("pp_candidate_decode_mask"),
        )

    from sglang.kernels.ops.attention.dsv4.candidate_table import (
        build_sparse_indexer_schedule,
        sort_candidate_blocks,
    )

    from .metadata import expand_index_page_table
    from .v41_indexer.sparse_table import _build_prefill_table, _SparseTable

    blocks = tensors["pp_candidate_blocks"]
    ratio = tensors["pp_candidate_ratio"]
    if kind == "sparse_prefill":
        page_size = tensors["pp_candidate_page_size"]
        core = metadata.core_metadata
        page_table = expand_index_page_table(
            core.page_table[: blocks.shape[0]],
            full_page_size=core.page_size,
            compress_ratio=ratio,
            index_page_size=page_size,
        ).contiguous()
        return _build_prefill_table(
            blocks=blocks,
            compress_lens=tensors["pp_candidate_compress_lens"],
            page_table=page_table,
            page_size=page_size,
            request_ids=tensors["pp_candidate_request_ids"],
            rows_per_request=tensors["pp_candidate_rows_per_request"],
            q_dtype=torch.int8,
            valid_lens=tensors["pp_candidate_valid_lens"],
        )
    if kind != "sparse_decode":
        raise ValueError(f"Unknown PP candidate metadata kind: {kind}")
    index = getattr(metadata, f"c{ratio}_indexer_metadata")
    lens = index.compressed_seq_lens.reshape(-1)
    phys_blocks = sort_candidate_blocks(
        blocks, lens, index.page_table, index.compressed_page_size
    )
    rows = blocks.shape[0]
    requests = forward_batch.batch_size
    request_ids = torch.arange(
        requests, dtype=torch.int32, device=blocks.device
    ).repeat_interleave(rows // requests)
    schedule = build_sparse_indexer_schedule(
        blocks,
        lens,
        index.page_table,
        index.compressed_page_size,
        torch.int8,
        request_ids,
    )
    ready = torch.cuda.Event()
    ready.record()
    return _SparseTable(
        blocks, schedule, phys_blocks, tensors["pp_candidate_valid_lens"], ready
    )
