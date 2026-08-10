from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.speculative.dspark_components.dspark_draft import (
    make_next_draft_input,
)

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2


def build_dspark_disagg_draft_input(
    batch: ScheduleBatch,
    server_args: ServerArgs,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> DFlashDraftInputV2:
    """Build the first DSpark decode state for a completed PD prefill.

    This fork intentionally runs DSpark without overlap scheduling.  Keep the
    upstream PD contract (last target token plus committed sequence length), but
    reject the newer FutureMap relay protocol instead of silently constructing
    an unusable draft state.
    """

    del server_args, future_map
    if batch.enable_overlap:
        raise RuntimeError(
            "DSpark PD decode requires overlap scheduling to be disabled in "
            "this fork."
        )

    local_bs = int(batch.seq_lens.numel())
    if last_tokens_tensor.numel() != local_bs:
        raise RuntimeError(
            "Invalid DSpark PD draft input: "
            f"local_bs={local_bs}, bonus_tokens_shape="
            f"{tuple(last_tokens_tensor.shape)}, new_seq_lens_shape="
            f"{tuple(batch.seq_lens.shape)}."
        )
    if last_tokens_tensor.device != batch.seq_lens.device:
        raise RuntimeError(
            "Invalid DSpark PD draft input device placement: "
            f"bonus_tokens_device={last_tokens_tensor.device}, "
            f"new_seq_lens_device={batch.seq_lens.device}."
        )

    return make_next_draft_input(
        bonus_tokens=last_tokens_tensor.to(dtype=torch.int64),
        new_seq_lens=batch.seq_lens.to(dtype=torch.int64),
    )
