"""Token-sharded residual region for DSV4.1 SM90 MegaMoE decode.

Attention consumes gathered H-wide inputs and reduce-scatters its row-parallel
projection. The 4H-wide mHC residual stays local across decoder layers. This is
separate from the generic LayerNorm SP path, which does not support this model.
"""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DSV41TokenParallel:
    group: Any
    rows: int

    @property
    def local_rows(self) -> int:
        return self.rows // self.group.world_size

    def local(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == self.rows
        return x.narrow(
            0, self.group.rank_in_group * self.local_rows, self.local_rows
        ).contiguous()

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == self.local_rows
        out = x.new_empty((self.rows, *x.shape[1:]))
        self.group.all_gather_into_tensor(out, x.contiguous())
        return out

    def reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == self.rows
        # SGLang's custom TP all-reduce accumulates in FP32, then rounds once.
        # NCCL BF16 reduce-scatter would round at intermediate reduction steps.
        # Preserve that accumulation precision while changing token ownership.
        reduced_input = x.float().contiguous()
        out = reduced_input.new_empty((self.local_rows, *x.shape[1:]))
        self.group.reduce_scatter_tensor(out, reduced_input)
        return out.to(x.dtype)


def can_use_dsv41_token_parallel(
    *,
    enabled,
    model_type,
    sm90,
    cuda,
    decode,
    capture_hidden,
    capture_dspark,
    hc_pre_from_prev,
    pp_size,
    tp_size,
    attn_tp_size,
    attn_dp_size,
    attn_cp_size,
    moe_ep_size,
    moe_tp_size,
    megamoe,
    other_sp,
    rows,
) -> bool:
    # Only the validated single-node TP8/EP8 decode layout opts in. Ragged eager
    # inputs, prefill/CP, hidden-state capture and other models keep their path.
    return bool(
        enabled
        and model_type == "deepseek_v41"
        and sm90
        and cuda
        and decode
        and not capture_hidden
        and not capture_dspark
        and hc_pre_from_prev
        and pp_size == attn_dp_size == attn_cp_size == moe_tp_size == 1
        and tp_size == attn_tp_size == moe_ep_size == 8
        and megamoe
        and not other_sp
        and rows > 0
        and rows % tp_size == 0
    )
