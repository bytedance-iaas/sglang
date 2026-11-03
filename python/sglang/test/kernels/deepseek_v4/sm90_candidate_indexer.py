"""Synthetic paged FP4 inputs shared by SM90 candidate tests and benchmarks."""

import torch


def make_candidate_case(
    rows: int,
    width: int,
    heads: int = 32,
    *,
    page_size: int = 64,
    block_size: int = 8,
    candidate_blocks: int | None = None,
    density: float = 0.125,
    pattern: str = "scattered",
    mixed_lengths: bool = False,
    seed: int = 0,
) -> dict[str, torch.Tensor | int]:
    """Use random physical pages, signed FP4-grid queries and BF16 weights.

    The newest visible block is retained, just as in source publication. With
    mixed lengths the rows cover zero through full width. Cache rows include
    padding so their physical stride is not the logical page byte count.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    pages_per_row = max(1, (width + page_size - 1) // page_size)
    num_pages = max(1, rows * pages_per_row)
    storage = torch.empty(
        (num_pages, page_size * 68 + 16), dtype=torch.uint8, device="cuda"
    )
    table = storage[:, : page_size * 68]
    table[:, : page_size * 64] = torch.randint(
        0,
        256,
        (num_pages, page_size * 64),
        generator=generator,
        dtype=torch.uint8,
        device="cuda",
    )
    table[:, page_size * 64 :] = torch.randint(
        122,
        132,
        (num_pages, page_size * 4),
        generator=generator,
        dtype=torch.uint8,
        device="cuda",
    )

    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.bfloat16,
        device="cuda",
    )
    codes = torch.randint(
        0, 16, (rows, heads, 4, 32), generator=generator, device="cuda"
    )
    scales = torch.randint(
        -4, 5, (rows, heads, 4, 1), generator=generator, device="cuda"
    )
    q = torch.ldexp(values[codes].float(), scales).to(torch.bfloat16)
    q = q.reshape(rows, heads, 128)
    weights = (
        torch.randn(
            (rows, heads), generator=generator, dtype=torch.bfloat16, device="cuda"
        )
        * (128 * heads) ** -0.5
    )
    lens = torch.full((rows,), width, dtype=torch.int64, device="cuda")
    if mixed_lengths and rows:
        lens = torch.linspace(0, width, rows, device="cuda").to(torch.int64)

    pages = torch.randperm(
        rows * pages_per_row, generator=generator, device="cuda"
    ).reshape(rows, pages_per_row)
    columns = torch.arange(width, device="cuda")
    slots = pages[:, columns // page_size] * page_size + columns % page_size

    num_blocks = (width + block_size - 1) // block_size
    blocks = torch.arange(num_blocks, device="cuda")
    if pattern in ("all", "none"):
        mask = torch.full(
            (rows, width), pattern == "all", dtype=torch.bool, device="cuda"
        )
    elif pattern == "alternating":
        mask = ((blocks % 2) == 0).expand(rows, -1)
        mask = mask.repeat_interleave(block_size, dim=-1)[:, :width]
    else:
        assert pattern in ("scattered", "clustered"), pattern
        if pattern == "scattered":
            keys = torch.rand((rows, num_blocks), generator=generator, device="cuda")
        else:
            keys = -blocks.float().expand(rows, -1).clone()
        visible_blocks = (lens + block_size - 1) // block_size
        keys.masked_fill_(blocks[None, :] >= visible_blocks[:, None], -torch.inf)
        keys.masked_fill_(
            (lens[:, None] > 0)
            & (blocks[None, :] == (lens[:, None] - 1) // block_size),
            torch.inf,
        )
        count = min(
            num_blocks,
            candidate_blocks
            if candidate_blocks is not None
            else max(1, int(num_blocks * density)),
        )
        top = keys.topk(count, dim=-1)
        kept = torch.zeros_like(keys, dtype=torch.bool)
        kept.scatter_(1, top.indices, top.values > -torch.inf)
        mask = kept.repeat_interleave(block_size, dim=-1)[:, :width]
    return dict(
        q=q,
        weights=weights,
        slots=slots,
        lens=lens,
        table=table,
        page_size=page_size,
        candidate_mask=mask,
    )


def select_candidate_topk(scores, slots, lens, topk, *, filter_candidates=True):
    """The production SM90 post-score selection, including underfilled rows.

    Keep dense width and logical ordering to exercise the original torch.topk
    tie behavior. Return both logical positions and physical cache slots.
    """
    from sglang.srt.layers.attention.dsv4.candidate_indexer import mask_topk_scores

    width = scores.shape[1]
    indices = scores.topk(min(topk, width), dim=-1, sorted=False).indices
    if filter_candidates:
        indices = mask_topk_scores(scores, indices)
        indices = indices.masked_fill(indices < 0, width)
    indices = indices.sort(dim=-1).values
    valid = indices < lens[:, None]
    raw = torch.where(valid, indices, -1).to(torch.int32)
    physical = torch.where(valid, slots.gather(1, indices.clamp_max(width - 1)), -1).to(
        torch.int32
    )
    return raw, physical
