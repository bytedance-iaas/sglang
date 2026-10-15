"""Token-sharded residual region for DSV4.1 SM90 MegaMoE decode.

Attention consumes gathered H-wide inputs and reduce-scatters its row-parallel
projection. The 4H-wide mHC residual stays local across decoder layers. This is
separate from the generic LayerNorm SP path, which does not support this model.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import attn_cp_reduce_scatter_tensor
from sglang.srt.runtime_context import get_parallel


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


def dsv41_cp_reduce_scatter(
    hidden_states: torch.Tensor,
    pre_reduce: Optional[torch.Tensor] = None,
    deferred_moe=None,
):
    attn_dp_size = get_parallel().attn_dp_size
    attn_tp_size = get_parallel().attn_tp_size
    assert attn_dp_size == 1 and attn_tp_size == 1
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    if deferred_moe is not None:
        assert pre_reduce is not None
        group = get_parallel().attn_cp_group
        ca_comm = group.ca_comm
        local_tokens = deferred_moe.expert_weights.shape[0] // cp_size + int(
            cp_rank < deferred_moe.expert_weights.shape[0] % cp_size
        )
        can_use_fused = (
            ca_comm is not None
            and not ca_comm.disabled
            and ca_comm.obj.push is not None
            and deferred_moe.gemm2_out.is_contiguous()
            and deferred_moe.expanded_idx_to_permuted_idx.is_contiguous()
            and deferred_moe.expert_weights.is_contiguous()
            and pre_reduce.is_contiguous()
            and local_tokens * pre_reduce.shape[1] * pre_reduce.element_size()
            <= ca_comm.max_push_size
        )
        if (
            envs.SGLANG_DSV41_MOE_FINALIZE_REDUCE_SCATTER_STRICT.get()
            and not can_use_fused
        ):
            raise RuntimeError(
                "Strict fused MoE finalize+RS rejected fallback: "
                f"message_bytes={local_tokens * pre_reduce.shape[1] * pre_reduce.element_size()}, "
                f"max_push_size={ca_comm.max_push_size if ca_comm is not None else -1}"
            )
        if can_use_fused:
            from sglang.kernels.ops.communication.moe_finalize_reduce_scatter import (
                moe_finalize_reduce_scatter,
            )

            return moe_finalize_reduce_scatter(
                ca_comm.obj,
                deferred_moe.gemm2_out,
                deferred_moe.expanded_idx_to_permuted_idx,
                deferred_moe.expert_weights,
                pre_reduce,
            )

        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            finalize_flashinfer_trtllm_deferred_output,
        )

        hidden_states = finalize_flashinfer_trtllm_deferred_output(
            deferred_moe,
            pre_reduce,
        )
        pre_reduce = None

    input_hidden_states = hidden_states
    hidden_states = hidden_states.tensor_split(cp_size)[cp_rank]
    if pre_reduce is not None:
        group = get_parallel().attn_cp_group
        ca_comm = group.ca_comm
        can_use_custom = (
            ca_comm is not None
            and not ca_comm.disabled
            and ca_comm.obj.push is not None
            and input_hidden_states.is_contiguous()
            and pre_reduce.is_contiguous()
            and input_hidden_states.shape == pre_reduce.shape
            and input_hidden_states.dtype == pre_reduce.dtype == torch.bfloat16
            and hidden_states.nbytes <= ca_comm.max_push_size
        )
        if can_use_custom:
            from sglang.kernels.ops.communication import nvlink_comm

            nvlink_comm.reduce_scatter_push(
                ca_comm.obj,
                input_hidden_states,
                hidden_states,
                pre_reduce=pre_reduce,
            )
            return hidden_states
        input_hidden_states.add_(pre_reduce)
    attn_cp_reduce_scatter_tensor(hidden_states, input_hidden_states)
    return hidden_states
