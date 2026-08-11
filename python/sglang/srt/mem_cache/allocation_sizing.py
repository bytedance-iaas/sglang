from __future__ import annotations

from typing import Optional

from sglang.srt.server_args import ServerArgs


def get_alloc_len_per_decode(
    server_args: ServerArgs,
    *,
    max_draft_tokens: Optional[int] = None,
) -> int:
    if server_args.speculative_algorithm is None:
        return 1

    spec_steps = int(server_args.speculative_num_steps or 1)
    spec_topk = int(server_args.speculative_eagle_topk or 1)
    spec_tokens = (
        max_draft_tokens
        if max_draft_tokens is not None
        else server_args.max_speculative_num_draft_tokens
    )
    spec_tokens = int(spec_tokens or 0)
    page_size = int(server_args.page_size)

    if page_size == 1 or spec_topk == 1:
        return max(spec_steps * spec_topk, spec_tokens)

    num_new_pages_per_topk = (
        (page_size - 1) + spec_steps + page_size - 1
    ) // page_size
    return max(num_new_pages_per_topk * page_size * spec_topk, spec_tokens)


def get_alloc_reserve_per_decode(
    server_args: ServerArgs,
    *,
    max_draft_tokens: Optional[int] = None,
) -> int:
    """Maximum per-request decode reservation, including the double buffer.

    The second window absorbs the lag between ``kv_allocated_len`` and
    ``kv_committed_len`` in result-based speculative decoding.
    """

    return 2 * get_alloc_len_per_decode(
        server_args, max_draft_tokens=max_draft_tokens
    )


def get_req_to_token_extra_context_len(server_args: ServerArgs) -> int:
    """Headroom required beyond the model context in each request row."""

    if server_args.speculative_algorithm is None:
        return 4
    max_draft_tokens = int(server_args.max_speculative_num_draft_tokens or 0)
    return max(
        4 + max_draft_tokens,
        get_alloc_reserve_per_decode(
            server_args, max_draft_tokens=max_draft_tokens
        )
        + int(server_args.page_size)
        - 1,
    )
