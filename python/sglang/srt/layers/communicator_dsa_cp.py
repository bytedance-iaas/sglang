# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from functools import partial
from typing import Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
)
from sglang.srt.layers.communicator import (
    CommunicateContext,
    CommunicateSimpleFn,
    CommunicateSummableTensorPairFn,
    CommunicateWithAllReduceAndLayerNormFn,
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    attn_cp_reduce_scatter_tensor,
    get_local_dp_buffer,
)
from sglang.srt.layers.utils.cp_utils import mla_use_prefill_cp
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.runtime_context import (
    get_parallel,
    max_prefill_buffer_tokens,
)


def dsa_enable_prefill_cp():
    # After using cp, the communication mode of this part changes.
    # The three parts of prepare_attn, prepare_mlp, and postprocess_layer
    # no longer require additional communication for reduce, scatter, etc.
    return is_dsa_enable_prefill_cp()


def maybe_prefetch_next_full_attention_kv(
    forward_batch: ForwardBatch,
    next_full_attention_layer_id: Optional[int],
) -> None:
    """Prefetch (owner-broadcast) the next layer's DSA KV under layer split.

    No-op unless the current batch runs DSA prefill-CP and the active KV pool is
    a layer-sharded pool exposing ``prefetch_kv_buffer`` (i.e.
    ``LayerSplitDSATokenToKVPool``). Kicking the broadcast off one layer ahead
    overlaps it with the current layer's attention compute.
    """
    if next_full_attention_layer_id is None or not dsa_use_prefill_cp(forward_batch):
        return

    prefetch_kv_buffer = getattr(get_token_to_kv_pool(), "prefetch_kv_buffer", None)
    if prefetch_kv_buffer is not None:
        prefetch_kv_buffer(next_full_attention_layer_id)


def dsa_cp_gather_hidden_states(hidden_states: torch.Tensor):
    attn_dp_size = get_parallel().attn_dp_size
    attn_tp_size = get_parallel().attn_tp_size
    assert attn_dp_size == 1 and attn_tp_size == 1
    hidden_states, local_hidden_states = (
        get_local_dp_buffer(get_parallel().attn_cp_group),
        hidden_states,
    )
    attn_cp_all_gather_into_tensor(hidden_states, local_hidden_states)
    return hidden_states


def dsa_cp_fused_ag_shared_experts_eligible(
    mlp,
    forward_batch: ForwardBatch,
    hidden_states: torch.Tensor,
) -> bool:
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.attention.dsa.utils import (
        is_dsa_prefill_cp_round_robin_split,
    )
    from sglang.srt.model_executor.runner import get_is_capture_mode

    if not envs.SGLANG_DSV41_CP_AG_SHARED_GEMM.get() or get_is_capture_mode():
        return False
    if not dsa_use_prefill_cp(forward_batch):
        return False
    if not is_dsa_prefill_cp_round_robin_split():
        return False
    parallel = get_parallel()
    if parallel.attn_dp_size != 1 or parallel.attn_tp_size != 1:
        return False
    comm = get_tp_group().torch_symm_mem_comm
    if comm is None or comm.disabled or comm.world_size != parallel.attn_cp_size:
        return False
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
        return False
    if hidden_states.shape[0] <= 0:
        return False
    if hidden_states.shape[0] * parallel.attn_cp_size > max_prefill_buffer_tokens():
        return False
    metadata = getattr(forward_batch, "attn_cp_metadata", None)
    per_rank_tokens = getattr(metadata, "per_rank_actual_token", None)
    if per_rank_tokens is not None:
        if len(set(per_rank_tokens)) != 1:
            return False
    elif sum(forward_batch.extend_seq_lens_cpu) % parallel.attn_cp_size != 0:
        return False
    if (
        getattr(mlp, "num_fused_shared_experts", 0) != 0
        or getattr(mlp, "_shared_expert_tp1", False)
        or not getattr(mlp, "shared_experts_is_fp8", False)
        or getattr(mlp, "shared_experts_weight_block_size", None) != [128, 128]
    ):
        return False
    shared = getattr(mlp, "shared_experts", None)
    gate_up = getattr(shared, "gate_up_proj", None)
    return (
        gate_up is not None
        and hasattr(gate_up, "weight")
        and hasattr(gate_up, "weight_scale_inv")
    )


def dsa_cp_fused_ag_shared_experts(
    mlp,
    forward_batch: ForwardBatch,
    hidden_states: torch.Tensor,
):
    """Gather the CP shard while computing shared-expert gate/up."""
    if not dsa_cp_fused_ag_shared_experts_eligible(mlp, forward_batch, hidden_states):
        return dsa_cp_gather_hidden_states(hidden_states), None

    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.device_communicators.symm_mem_kernels import (
        maybe_fused_ag_shared_experts,
    )

    comm = get_tp_group().torch_symm_mem_comm
    comm.set_use_cp_fused_ag(True)
    try:
        gathered, gate_up = maybe_fused_ag_shared_experts(
            hidden_states, mlp.shared_experts.gate_up_proj
        )
    finally:
        comm.set_use_cp_fused_ag(False)
    if gathered is None or gate_up is None:
        return dsa_cp_gather_hidden_states(hidden_states), None
    shared_output = mlp.shared_experts(
        gathered,
        precomputed_gate_up=gate_up,
    )
    return gathered, shared_output


def dsa_cp_reduce_scatter_hidden_states(
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
        num_tokens = deferred_moe.expert_weights.shape[0]
        local_tokens = num_tokens // cp_size + int(cp_rank < num_tokens % cp_size)
        message_bytes = local_tokens * pre_reduce.shape[1] * pre_reduce.element_size()
        can_use_fused = (
            ca_comm is not None
            and not ca_comm.disabled
            and ca_comm.obj.push is not None
            and deferred_moe.gemm2_out.is_contiguous()
            and deferred_moe.expanded_idx_to_permuted_idx.is_contiguous()
            and deferred_moe.expert_weights.is_contiguous()
            and pre_reduce.is_contiguous()
            and message_bytes <= ca_comm.max_push_size
        )
        if (
            envs.SGLANG_DSV41_MOE_FINALIZE_REDUCE_SCATTER_STRICT.get()
            and not can_use_fused
        ):
            raise RuntimeError(
                "Strict fused MoE finalize+RS rejected fallback: "
                f"message_bytes={message_bytes}, "
                f"max_push_size={ca_comm.max_push_size if ca_comm else -1}"
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
    input_hidden_states = hidden_states
    hidden_states = hidden_states.tensor_split(cp_size)[cp_rank]
    attn_cp_reduce_scatter_tensor(hidden_states, input_hidden_states)
    return hidden_states


class DSACPLayerCommunicator(LayerCommunicator):
    def __init__(
        self,
        layer_scatter_modes: LayerScatterModes,
        input_layernorm: torch.nn.Module,
        post_attention_layernorm: torch.nn.Module,
        # Reduce scatter requires skipping all-reduce in model code after MoE/MLP, so only enable for models which have that implemented. Remove flag once done for all models that use LayerCommunicator.
        allow_reduce_scatter: bool = False,
        is_last_layer: bool = False,
        qkv_latent_func: Optional[Callable] = None,
    ):
        super().__init__(
            layer_scatter_modes,
            input_layernorm,
            post_attention_layernorm,
            allow_reduce_scatter,
            is_last_layer,
            qkv_latent_func,
        )

    def _post_init_communicate(self):
        # SCATTERED in attn tp is different from SCATTERED in global tp when dp_size > 1
        if self.layer_scatter_modes.mlp_mode != ScatterMode.SCATTERED:
            assert (
                self._context.attn_dp_size == 1
            ), "dp_size should be 1 when moe_runner_backend is none"
        self._communicate_simple_fn = DSACPCommunicateSimpleFn.get_fn(
            input_mode=ScatterMode.SCATTERED,
            output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )
        self._communicate_with_all_reduce_and_layer_norm_fn = DSACPCommunicateWithAllReduceAndLayerNormFn.get_fn(
            hidden_states_input_mode=ScatterMode.SCATTERED,
            residual_input_mode=ScatterMode.SCATTERED,
            hidden_states_output_mode=self.layer_scatter_modes.mlp_mode,  # SCATTERED, FULL
            residual_output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )
        self._communicate_summable_tensor_pair_fn = DSACPCommunicateSummableTensorPairFn.get_fn(
            hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,  # SCATTERED, FULL
            residual_input_mode=ScatterMode.SCATTERED,
            output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )


class DSACPCommunicateSimpleFn(CommunicateSimpleFn):
    @staticmethod
    def get_fn(
        input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        if context.is_same_group_size(input_mode, output_mode):
            return DSACPCommunicateSimpleFn._trivial

        raise NotImplementedError(f"{input_mode=} {output_mode=}")


class DSACPCommunicateWithAllReduceAndLayerNormFn(
    CommunicateWithAllReduceAndLayerNormFn
):
    """Besides communication, needs to
    1. All reduce in tp_attn_group on hidden_states
    2. Apply layer norm
    """

    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        hidden_states_output_mode: ScatterMode,
        residual_output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        assert hidden_states_input_mode == ScatterMode.SCATTERED
        assert residual_input_mode == ScatterMode.SCATTERED
        assert residual_output_mode == ScatterMode.SCATTERED
        if hidden_states_output_mode == ScatterMode.SCATTERED:
            return DSACPCommunicateWithAllReduceAndLayerNormFn._simple

        if hidden_states_output_mode == ScatterMode.FULL:
            return partial(
                DSACPCommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
                residual_input_mode=residual_input_mode,
            )

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {hidden_states_output_mode=} {residual_output_mode=}"
        )

    @staticmethod
    def _gather_hidden_states_and_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        residual_input_mode,
    ):
        if hidden_states.shape[0] != 0:
            hidden_states, residual = layernorm(hidden_states, residual)
        # for prefill: attn tp scattered -> full
        # for decode: attn tp full -> full
        if dsa_use_prefill_cp(forward_batch) or mla_use_prefill_cp(forward_batch):
            hidden_states = dsa_cp_gather_hidden_states(hidden_states)
        return hidden_states, residual


class DSACPCommunicateSummableTensorPairFn(CommunicateSummableTensorPairFn):
    """It is allowed to make (hidden_states, residual) := (hidden_states + residual, None) if needed."""

    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        # Check exact enum match first: even if group sizes happen to be equal
        # (e.g. tp_size == attn_cp_size makes FULL and SCATTERED both size 1),
        # FULL and SCATTERED have different data layouts under CP and require
        # an explicit scatter operation.
        if (
            (hidden_states_input_mode == ScatterMode.FULL)
            and (residual_input_mode == ScatterMode.SCATTERED)
            and (output_mode == ScatterMode.SCATTERED)
        ):
            return DSACPCommunicateSummableTensorPairFn._scatter_hidden_states

        if context.is_same_group_size(
            hidden_states_input_mode, output_mode
        ) and context.is_same_group_size(residual_input_mode, output_mode):
            return DSACPCommunicateSummableTensorPairFn._trivial

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {output_mode=}"
        )

    @staticmethod
    def _scatter_hidden_states(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        allow_reduce_scatter: bool = False,
    ):
        # for prefill: full -> attn tp scattered
        # for decode: full -> attn tp full
        if dsa_use_prefill_cp(forward_batch) or mla_use_prefill_cp(forward_batch):
            hidden_states = dsa_cp_reduce_scatter_hidden_states(hidden_states)
        return hidden_states, residual
