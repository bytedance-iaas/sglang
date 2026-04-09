from __future__ import annotations

import enum
import logging
from enum import Enum
from typing import TYPE_CHECKING

import torch
from compressed_tensors import CompressionFormat

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW4A16Int4DynamicMoEMethod,
)
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    WNA16_SUPPORTED_BITS,
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.gptq import gptq_marlin_moe_repack
from sglang.srt.layers.quantization.marlin_utils import marlin_moe_permute_scales
from sglang.srt.layers.quantization.utils import replace_parameter
from sglang.srt.utils import get_bool_env_var, is_cuda, is_hip, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )


__all__ = [
    "CompressedTensorsWNA16MoE",
    "CompressedTensorsWNA16TritonMoE",
    "NPUCompressedTensorsW4A16Int4DynamicMoE",
]

_is_hip = is_hip()
_is_cuda = is_cuda()
_LOW_LATENCY_PROFILE_LOG = get_bool_env_var("SGLANG_DEEPEP_LOW_LATENCY_PROFILE_LOG")

logger = logging.getLogger(__name__)

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
if _use_aiter:
    pass

if _is_cuda:
    import triton
    import triton.language as tl

if _is_cuda:

    @triton.jit
    def _compact_deepep_ll_hidden_states_kernel(
        input_ptr,
        output_ptr,
        prefix_ptr,
        masked_m_ptr,
        stride_input_expert,
        stride_input_token,
        stride_input_hidden,
        stride_output_token,
        stride_output_hidden,
        hidden_size,
        BLOCK_H: tl.constexpr,
    ):
        block_h = tl.program_id(0)
        token_id = tl.program_id(1)
        expert_id = tl.program_id(2)

        token_num_cur_expert = tl.load(masked_m_ptr + expert_id)
        if token_id >= token_num_cur_expert:
            return

        dst_token = tl.load(prefix_ptr + expert_id) + token_id
        offsets_h = block_h * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = offsets_h < hidden_size

        input_ptr = (
            input_ptr
            + expert_id * stride_input_expert
            + token_id * stride_input_token
            + offsets_h * stride_input_hidden
        )
        output_ptr = (
            output_ptr + dst_token * stride_output_token + offsets_h * stride_output_hidden
        )

        data = tl.load(input_ptr, mask=mask_h, other=0.0)
        tl.store(output_ptr, data, mask=mask_h)

    @triton.jit
    def _expand_deepep_ll_hidden_states_kernel(
        input_ptr,
        output_ptr,
        prefix_ptr,
        masked_m_ptr,
        stride_input_token,
        stride_input_hidden,
        stride_output_expert,
        stride_output_token,
        stride_output_hidden,
        hidden_size,
        BLOCK_H: tl.constexpr,
    ):
        block_h = tl.program_id(0)
        token_id = tl.program_id(1)
        expert_id = tl.program_id(2)

        token_num_cur_expert = tl.load(masked_m_ptr + expert_id)
        offsets_h = block_h * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = offsets_h < hidden_size
        dst_ptr = (
            output_ptr
            + expert_id * stride_output_expert
            + token_id * stride_output_token
            + offsets_h * stride_output_hidden
        )

        if token_id < token_num_cur_expert:
            src_token = tl.load(prefix_ptr + expert_id) + token_id
            src_ptr = (
                input_ptr
                + src_token * stride_input_token
                + offsets_h * stride_input_hidden
            )
            data = tl.load(src_ptr, mask=mask_h, other=0.0)
        else:
            data = tl.zeros((BLOCK_H,), dtype=dst_ptr.dtype.element_ty)

        tl.store(dst_ptr, data, mask=mask_h)

    @triton.jit
    def _build_deepep_ll_prefix_sum_kernel(
        masked_m_ptr,
        prefix_ptr,
        num_experts,
    ):
        expert_id = tl.program_id(0)
        if expert_id > num_experts:
            return

        running = tl.zeros((), dtype=tl.int32)
        idx = 0
        while idx < expert_id:
            running += tl.load(masked_m_ptr + idx)
            idx += 1
        tl.store(prefix_ptr + expert_id, running)

    @triton.jit
    def _build_deepep_ll_marlin_layout_kernel(
        aligned_prefix_ptr,
        prefix_ptr,
        masked_m_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        size_m,
        BLOCK_M: tl.constexpr,
    ):
        local_block_id = tl.program_id(0)
        expert_id = tl.program_id(1)

        aligned_start = tl.load(aligned_prefix_ptr + expert_id)
        aligned_end = tl.load(aligned_prefix_ptr + expert_id + 1)
        aligned_count = aligned_end - aligned_start
        block_offset = local_block_id * BLOCK_M
        if block_offset >= aligned_count:
            return

        real_start = tl.load(prefix_ptr + expert_id)
        real_count = tl.load(masked_m_ptr + expert_id)
        block_start = aligned_start + block_offset
        lanes = tl.arange(0, BLOCK_M)
        token_offsets = block_offset + lanes
        token_ids = tl.where(token_offsets < real_count, real_start + token_offsets, size_m)
        tl.store(sorted_token_ids_ptr + block_start + lanes, token_ids)
        tl.store(expert_ids_ptr + (block_start // BLOCK_M), expert_id)

    @triton.jit
    def _build_deepep_ll_aligned_prefix_kernel(
        aligned_prefix_ptr,
        num_tokens_post_padded_ptr,
        masked_m_ptr,
        num_experts,
        BLOCK_M: tl.constexpr,
    ):
        expert_id = tl.program_id(0)
        if expert_id >= num_experts:
            return

        running = tl.zeros((), dtype=tl.int32)
        idx = 0
        while idx <= expert_id:
            count = tl.load(masked_m_ptr + idx)
            running += ((count + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
            idx += 1

        if expert_id == 0:
            tl.store(aligned_prefix_ptr, 0)
        tl.store(aligned_prefix_ptr + expert_id + 1, running)
        if expert_id == num_experts - 1:
            tl.store(num_tokens_post_padded_ptr, running)

class GPTQMarlinState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


class CompressedTensorsWNA16MoE(CompressedTensorsMoEScheme):

    def __init__(self, quant_config: CompressedTensorsConfig, num_gpu_experts=-1):
        self.quant_config = quant_config
        config = self.quant_config.target_scheme_map["Linear"].get("weights")
        self.num_bits = config.num_bits
        self.packed_factor = 32 // config.num_bits
        self.strategy = config.strategy
        self.group_size = config.group_size
        self.actorder = config.actorder
        assert config.symmetric, "Only symmetric quantization is supported for MoE"

        if not (
            self.quant_config.quant_format == CompressionFormat.pack_quantized.value
            and self.num_bits in WNA16_SUPPORTED_BITS
        ):
            raise ValueError(
                "For Fused MoE layers, only ",
                f"{CompressionFormat.pack_quantized.value} ",
                "is supported for the following bits: ",
                f"{WNA16_SUPPORTED_BITS}",
            )
        self.num_gpu_experts = num_gpu_experts

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        extra_weight_attrs.update(
            {"is_transposed": True, "quant_method": self.strategy}
        )
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.packed_factor,
                2 * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.packed_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # In the case where we have actorder/g_idx,
        # we do not partition the w2 scales
        load_full_w2 = self.actorder and self.group_size != -1

        if load_full_w2:
            w2_scales_size = intermediate_size_per_partition * layer.moe_tp_size
        else:
            w2_scales_size = intermediate_size_per_partition

        self.is_k_full = (not self.actorder) or layer.moe_tp_size == 1

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            num_groups_w2 = w2_scales_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        w13_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.ones(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"load_full_w2": load_full_w2})

        w2_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)
        w13_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )

        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None
        layer.marlin_state = GPTQMarlinState.REPACK

        if not hasattr(layer, "_original_shapes"):
            layer._original_shapes = {}

        # Force record: these are the target GPTQ shapes for rollback.
        layer._original_shapes["w13_weight_packed"] = tuple(w13_weight.shape)
        layer._original_shapes["w2_weight_packed"] = tuple(w2_weight.shape)

        # Also record the shapes of the scales.
        layer._original_shapes["w2_weight_scale"] = tuple(w2_scale.shape)
        layer._original_shapes["w13_weight_scale"] = tuple(w13_scale.shape)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        # Skip if the layer is already converted to Marlin format to prevent double-packing.
        if getattr(layer, "is_marlin_converted", False):
            return

        if not hasattr(layer, "_original_shapes"):
            layer._original_shapes = {}

        def replace_tensor(name, new_t):
            target_attr = getattr(layer, name)

            # Only save if the key doesn't exist to prevent overwriting with Marlin shapes.
            if name not in layer._original_shapes:
                # This is a safety check; `create_weights` usually handles this already.
                layer._original_shapes[name] = tuple(target_attr.shape)

            # It is important to use resize_() here since it ensures
            # the same buffer is reused
            target_attr.resize_(new_t.shape)
            target_attr.copy_(new_t)
            del new_t

        num_experts = layer.w13_weight_g_idx.shape[0]
        device = layer.w13_weight_g_idx.device

        # when running models with grouped act order,
        # resort to g_idx values provided in checkpoint
        if self.actorder == "group":
            w13_g_idx_sort_indices = torch.empty_like(layer.w13_weight_g_idx)
            w2_g_idx_sort_indices = torch.empty_like(layer.w2_weight_g_idx)
            w13_sorted_g_idx = torch.empty_like(layer.w13_weight_g_idx)
            w2_sorted_g_idx = torch.empty_like(layer.w2_weight_g_idx)

            for e in range(num_experts):
                w13_g_idx_sort_indices[e] = torch.argsort(layer.w13_weight_g_idx[e]).to(
                    torch.int32
                )
                w2_g_idx_sort_indices[e] = torch.argsort(layer.w2_weight_g_idx[e]).to(
                    torch.int32
                )
                w13_sorted_g_idx[e] = layer.w13_weight_g_idx[e][
                    w13_g_idx_sort_indices[e]
                ]
                w2_sorted_g_idx[e] = layer.w2_weight_g_idx[e][w2_g_idx_sort_indices[e]]

            replace_parameter(layer, "w13_weight_g_idx", w13_sorted_g_idx)
            replace_parameter(layer, "w2_weight_g_idx", w2_sorted_g_idx)
            replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
            replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)

        else:
            layer.w13_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w13_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )

        marlin_w13_qweight = gptq_marlin_moe_repack(
            layer.w13_weight_packed,
            layer.w13_g_idx_sort_indices,
            layer.w13_weight_packed.shape[1] * self.packed_factor,
            layer.w13_weight_packed.shape[2],
            self.num_bits,
        )
        replace_tensor("w13_weight_packed", marlin_w13_qweight)
        marlin_w2_qweight = gptq_marlin_moe_repack(
            layer.w2_weight_packed,
            layer.w2_g_idx_sort_indices,
            layer.w2_weight_packed.shape[1] * self.packed_factor,
            layer.w2_weight_packed.shape[2],
            self.num_bits,
        )
        replace_tensor("w2_weight_packed", marlin_w2_qweight)
        # Repack scales
        marlin_w13_scales = marlin_moe_permute_scales(
            layer.w13_weight_scale,
            layer.w13_weight_packed.shape[2],
            layer.w13_weight_scale.shape[2],
            self.group_size,
        )
        replace_tensor("w13_weight_scale", marlin_w13_scales)

        marlin_w2_scales = marlin_moe_permute_scales(
            layer.w2_weight_scale,
            layer.w2_weight_scale.shape[1]
            * (self.group_size if self.group_size != -1 else self.packed_factor),
            layer.w2_weight_scale.shape[2],
            self.group_size,
        )
        replace_tensor("w2_weight_scale", marlin_w2_scales)

        layer.is_marlin_converted = True

    def restore_weights_before_loading(self, layer: torch.nn.Module):
        """Forcibly resize parameters back to their original shapes (e.g., GPTQ format) before loading weights."""

        if not hasattr(layer, "_original_shapes"):
            return

        for name, orig_shape in layer._original_shapes.items():
            param = getattr(layer, name, None)

            if param is not None and param.shape != orig_shape:
                param.resize_(orig_shape)

        layer.is_marlin_converted = False

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def _get_deepep_local_expert_mapping(self, layer: torch.nn.Module) -> torch.Tensor:
        mapping = getattr(layer, "_deepep_local_expert_mapping", None)
        if mapping is not None:
            return mapping

        num_fused_shared_experts = getattr(layer, "num_fused_shared_experts", 0)
        num_global_routed_experts = layer.num_experts - num_fused_shared_experts
        num_local_routed_experts = layer.num_local_experts - num_fused_shared_experts
        start_idx = layer.moe_ep_rank * num_local_routed_experts
        end_idx = (layer.moe_ep_rank + 1) * num_local_routed_experts

        mapping = torch.full(
            (layer.num_experts,),
            -1,
            dtype=torch.int32,
            device=layer.w13_weight_packed.device,
        )
        mapping[start_idx:end_idx] = torch.arange(
            0, num_local_routed_experts, dtype=torch.int32, device=mapping.device
        )

        if num_fused_shared_experts > 0:
            mapping[num_global_routed_experts:] = torch.arange(
                num_local_routed_experts,
                num_local_routed_experts + num_fused_shared_experts,
                dtype=torch.int32,
                device=mapping.device,
            )

        layer._deepep_local_expert_mapping = mapping
        return mapping

    def _fused_marlin_deepep_ll(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        prefix_sum: torch.Tensor,
        masked_m: torch.Tensor,
    ) -> torch.Tensor:
        from sgl_kernel import silu_and_mul
        from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm

        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            get_scalar_type,
        )
        from sglang.srt.layers.quantization.marlin_utils import (
            should_use_atomic_add_reduce,
        )

        assert hidden_states.ndim == 2
        assert prefix_sum.ndim == 1
        assert masked_m.ndim == 1

        m, k = hidden_states.shape
        if m == 0:
            return hidden_states

        e = layer.w13_weight_packed.shape[0]
        n = layer.w2_weight_packed.shape[1] * 16
        topk = 1
        capture_active = (
            hidden_states.is_cuda and torch.cuda.is_current_stream_capturing()
        )
        profile_enabled = _LOW_LATENCY_PROFILE_LOG and hidden_states.is_cuda and not capture_active

        for block_size_m in [8, 16, 32, 48, 64]:
            if m * topk / e / block_size_m < 0.9:
                break

        marlin_runtime = getattr(layer, "_deepep_ll_marlin_runtime", None)
        device = hidden_states.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_blocks_per_expert = triton.cdiv(m, block_size_m)
        max_active_experts_upper_bound = min(e, m)
        max_num_m_blocks = max(1, max_active_experts_upper_bound * max_blocks_per_expert)
        max_num_tokens_padded = max_num_m_blocks * block_size_m
        workspace_size = (max(2 * n, k) // 64) * max(1, max_num_m_blocks)
        workspace_size = min(workspace_size, sms * 4)
        workspace_size = max(workspace_size, 128)
        intermediate13_size = m * max(2 * n, k)
        if (
            marlin_runtime is None
            or marlin_runtime["device"] != device
            or marlin_runtime["dtype"] != hidden_states.dtype
            or marlin_runtime["buffer_tokens"] < m
            or marlin_runtime["block_size_m"] != block_size_m
            or marlin_runtime["workspace"].numel() < workspace_size
            or marlin_runtime["intermediate_cache2"].shape[1] != n
            or marlin_runtime["intermediate_cache13"].numel() < intermediate13_size
            or marlin_runtime["sorted_token_ids"].numel() < max_num_tokens_padded
            or marlin_runtime["expert_ids"].numel() < max_num_m_blocks
        ):
            workspace = torch.zeros(
                workspace_size, dtype=torch.int, device=device, requires_grad=False
            )
            intermediate_cache2 = torch.empty(
                (m, n),
                device=device,
                dtype=hidden_states.dtype,
            )
            intermediate_cache13 = torch.empty(
                (intermediate13_size,),
                device=device,
                dtype=hidden_states.dtype,
            )
            sorted_token_ids = torch.empty(
                (max_num_tokens_padded,), dtype=torch.int32, device=device
            )
            expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
            num_tokens_post_padded = torch.empty((1,), dtype=torch.int32, device=device)
            topk_weights = torch.ones((m, 1), dtype=hidden_states.dtype, device=device)
            masked_m_int = torch.empty((e,), dtype=torch.int32, device=device)
            aligned_prefix = torch.empty((e + 1,), dtype=torch.int32, device=device)
            marlin_runtime = {
                "device": device,
                "dtype": hidden_states.dtype,
                "buffer_tokens": m,
                "block_size_m": block_size_m,
                "workspace": workspace,
                "intermediate_cache2": intermediate_cache2,
                "intermediate_cache13": intermediate_cache13,
                "sorted_token_ids": sorted_token_ids,
                "expert_ids": expert_ids,
                "num_tokens_post_padded": num_tokens_post_padded,
                "topk_weights": topk_weights,
                "masked_m_int": masked_m_int,
                "aligned_prefix": aligned_prefix,
            }
            layer._deepep_ll_marlin_runtime = marlin_runtime

        workspace = marlin_runtime["workspace"]
        sorted_token_ids = marlin_runtime["sorted_token_ids"]
        expert_ids = marlin_runtime["expert_ids"]
        num_tokens_post_padded = marlin_runtime["num_tokens_post_padded"]
        topk_weights = marlin_runtime["topk_weights"][:m]
        masked_m_int = marlin_runtime["masked_m_int"]
        aligned_prefix = marlin_runtime["aligned_prefix"]

        if masked_m.dtype == torch.int32:
            masked_m_int.copy_(masked_m)
        else:
            masked_m_int.copy_(masked_m.to(dtype=torch.int32))
        _build_deepep_ll_aligned_prefix_kernel[(e,)](
            aligned_prefix,
            num_tokens_post_padded,
            masked_m_int,
            e,
            BLOCK_M=block_size_m,
        )

        layout_grid = (max_blocks_per_expert, e)
        _build_deepep_ll_marlin_layout_kernel[layout_grid](
            aligned_prefix,
            prefix_sum,
            masked_m_int,
            sorted_token_ids,
            expert_ids,
            m,
            BLOCK_M=block_size_m,
        )

        scalar_type = get_scalar_type(self.num_bits, False)
        use_atomic_add_w13 = should_use_atomic_add_reduce(
            m,
            2 * n,
            k,
            hidden_states.device,
            hidden_states.dtype,
        )
        use_atomic_add_w2 = should_use_atomic_add_reduce(
            m,
            k,
            n,
            hidden_states.device,
            hidden_states.dtype,
        )

        intermediate_cache2 = marlin_runtime["intermediate_cache2"][:m]
        intermediate_cache13 = marlin_runtime["intermediate_cache13"]
        intermediate_cache1 = intermediate_cache13[: m * 2 * n].view(-1, 2 * n)
        intermediate_cache3 = intermediate_cache13[: m * k].view(-1, k)
        if profile_enabled:
            w13_start = torch.cuda.Event(enable_timing=True)
            w13_end = torch.cuda.Event(enable_timing=True)
            silu_start = torch.cuda.Event(enable_timing=True)
            silu_end = torch.cuda.Event(enable_timing=True)
            w2_start = torch.cuda.Event(enable_timing=True)
            w2_end = torch.cuda.Event(enable_timing=True)
            w13_start.record()

        intermediate_cache1 = moe_wna16_marlin_gemm(
            hidden_states,
            intermediate_cache1,
            layer.w13_weight_packed,
            None,
            layer.w13_weight_scale,
            None,
            None,
            layer.w13_weight_g_idx,
            layer.w13_g_idx_sort_indices,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            block_size_m,
            1,
            False,
            False,
            scalar_type,
            m,
            2 * n,
            k,
            self.is_k_full,
            use_atomic_add_w13,
            True,
            False,
        )
        if profile_enabled:
            w13_end.record()
            silu_start.record()
        silu_and_mul(intermediate_cache1.view(-1, 2 * n), intermediate_cache2)
        if profile_enabled:
            silu_end.record()
            w2_start.record()

        output = moe_wna16_marlin_gemm(
            intermediate_cache2,
            intermediate_cache3,
            layer.w2_weight_packed,
            None,
            layer.w2_weight_scale,
            None,
            None,
            layer.w2_weight_g_idx,
            layer.w2_g_idx_sort_indices,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            block_size_m,
            1,
            False,
            False,
            scalar_type,
            m,
            k,
            n,
            self.is_k_full,
            use_atomic_add_w2,
            True,
            False,
        )
        if profile_enabled:
            w2_end.record()
            w2_end.synchronize()
            logger.warning(
                "DEEPEP_LL_PROFILE_MARLIN rank=%s m=%s e=%s block_m=%s w13_ms=%.3f silu_ms=%.3f w2_ms=%.3f use_atomic_add_w13=%s use_atomic_add_w2=%s",
                getattr(layer, "moe_ep_rank", -1),
                m,
                e,
                block_size_m,
                w13_start.elapsed_time(w13_end),
                silu_start.elapsed_time(silu_end),
                w2_start.elapsed_time(w2_end),
                use_atomic_add_w13,
                use_atomic_add_w2,
            )
        return output

    def _get_deepep_ll_max_active_tokens(self, dispatch_topk_ids: torch.Tensor) -> int:
        return max(int(dispatch_topk_ids.numel()), 1)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, router_logits = topk_output

        # Get expert_map for EP support
        expert_map = None
        global_num_experts = -1
        if hasattr(layer, "dispatcher") and hasattr(
            layer.dispatcher, "local_expert_mapping"
        ):
            expert_map = layer.dispatcher.local_expert_mapping
            if expert_map is not None:
                global_num_experts = self.moe_runner_config.num_experts

        output = fused_marlin_moe(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            router_logits,
            topk_weights,
            topk_ids,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            g_idx1=layer.w13_weight_g_idx,
            g_idx2=layer.w2_weight_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            num_bits=self.num_bits,
            is_k_full=self.is_k_full,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor,
        )
        return StandardCombineInput(hidden_states=output)

    def apply_deepep_normal(self, layer: torch.nn.Module, dispatch_output):
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )

        hidden_states, _, topk_ids, topk_weights, _ = dispatch_output
        local_expert_mapping = self._get_deepep_local_expert_mapping(layer)

        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."
        if hidden_states.shape[0] == 0:
            return hidden_states

        output = fused_marlin_moe(
            hidden_states,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            topk_weights,
            topk_weights,
            topk_ids,
            global_num_experts=self.moe_runner_config.num_experts,
            expert_map=local_expert_mapping,
            g_idx1=layer.w13_weight_g_idx,
            g_idx2=layer.w2_weight_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            num_bits=self.num_bits,
            is_k_full=self.is_k_full,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor,
        )
        return output

    def apply_deepep_ll(self, layer: torch.nn.Module, dispatch_output):
        hidden_states, hidden_states_scale, dispatch_topk_ids, _, masked_m, _ = (
            dispatch_output
        )

        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."
        assert (
            hidden_states_scale is None
        ), "W4A16 DeepEP low latency currently requires BF16 dispatch."
        assert hidden_states.ndim == 3, "DeepEP low latency expects [E, M, K] inputs."

        num_local_experts, padded_m, hidden_size = hidden_states.shape
        if num_local_experts == 0 or padded_m == 0:
            return torch.zeros_like(hidden_states)

        capture_active = (
            hidden_states.is_cuda and torch.cuda.is_current_stream_capturing()
        )
        if not capture_active and int(masked_m.max().item()) == 0:
            return torch.zeros_like(hidden_states)
        max_active_tokens = int(dispatch_topk_ids.numel())
        if max_active_tokens == 0:
            return torch.zeros_like(hidden_states)
        profile_enabled = _LOW_LATENCY_PROFILE_LOG and hidden_states.is_cuda and not capture_active
        if not capture_active:
            active_tokens = int(masked_m.sum().item())
            if active_tokens == 0:
                return torch.zeros_like(hidden_states)
            token_upper_bound = min(padded_m, max(int(masked_m.max().item()), 1))
            prefix_sum = torch.empty(
                (num_local_experts + 1,),
                dtype=torch.int32,
                device=hidden_states.device,
            )
            compact_hidden_states = torch.empty(
                (active_tokens, hidden_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            expanded_output = hidden_states
        else:
            cached_layout = getattr(layer, "_deepep_ll_marlin_layout", None)
            if (
                cached_layout is None
                or cached_layout["device"] != hidden_states.device
                or cached_layout["num_local_experts"] != num_local_experts
                or cached_layout["padded_m"] != padded_m
                or cached_layout["max_active_tokens"] != max_active_tokens
            ):
                compact_hidden_states = torch.empty(
                    (max_active_tokens, hidden_size),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                prefix_sum = torch.empty(
                    (num_local_experts + 1,),
                    dtype=torch.int32,
                    device=hidden_states.device,
                )
                cached_layout = {
                    "device": hidden_states.device,
                    "num_local_experts": num_local_experts,
                    "padded_m": padded_m,
                    "max_active_tokens": max_active_tokens,
                    "compact_hidden_states": compact_hidden_states,
                    "prefix_sum": prefix_sum,
                }
                layer._deepep_ll_marlin_layout = cached_layout

            prefix_sum = cached_layout["prefix_sum"]
            compact_hidden_states = cached_layout["compact_hidden_states"]
            expanded_output = hidden_states
            active_tokens = max_active_tokens
            token_upper_bound = min(padded_m, max_active_tokens)

        if _is_cuda:
            _build_deepep_ll_prefix_sum_kernel[(num_local_experts + 1,)](
                masked_m,
                prefix_sum,
                num_local_experts,
            )

        if profile_enabled:
            total_start = torch.cuda.Event(enable_timing=True)
            compact_end = torch.cuda.Event(enable_timing=True)
            marlin_end = torch.cuda.Event(enable_timing=True)
            expand_end = torch.cuda.Event(enable_timing=True)
            total_start.record()
        if _is_cuda:
            grid = (
                triton.cdiv(hidden_size, 256),
                token_upper_bound,
                num_local_experts,
            )
            _compact_deepep_ll_hidden_states_kernel[grid](
                hidden_states,
                compact_hidden_states,
                prefix_sum,
                masked_m,
                *hidden_states.stride(),
                *compact_hidden_states.stride(),
                hidden_size,
                BLOCK_H=256,
            )
            if profile_enabled:
                compact_end.record()
        else:
            raise NotImplementedError("DeepEP low latency W4A16 sparse path requires CUDA.")

        flat_output = self._fused_marlin_deepep_ll(
            layer,
            compact_hidden_states,
            prefix_sum,
            masked_m,
        )
        if profile_enabled:
            marlin_end.record()
        if _is_cuda:
            grid = (
                triton.cdiv(hidden_size, 256),
                token_upper_bound,
                num_local_experts,
            )
            _expand_deepep_ll_hidden_states_kernel[grid](
                flat_output,
                expanded_output,
                prefix_sum,
                masked_m,
                *flat_output.stride(),
                *expanded_output.stride(),
                hidden_size,
                BLOCK_H=256,
            )
            if profile_enabled:
                expand_end.record()
                expand_end.synchronize()
                logger.warning(
                    "DEEPEP_LL_PROFILE_COMPUTE rank=%s capture=%s active_tokens=%s token_upper=%s hidden_shape=%s compact_ms=%.3f marlin_ms=%.3f expand_ms=%.3f total_ms=%.3f",
                    getattr(layer, "moe_ep_rank", -1),
                    capture_active,
                    max_active_tokens,
                    token_upper_bound,
                    tuple(hidden_states.shape),
                    total_start.elapsed_time(compact_end),
                    compact_end.elapsed_time(marlin_end),
                    marlin_end.elapsed_time(expand_end),
                    total_start.elapsed_time(expand_end),
                )
            return expanded_output
        raise NotImplementedError("DeepEP low latency W4A16 sparse path requires CUDA.")


class CompressedTensorsWNA16TritonMoE(CompressedTensorsWNA16MoE):
    """ROCm/HIP-compatible W4A16 MoE method using Triton kernels instead of Marlin.

    Inherits weight creation from CompressedTensorsWNA16MoE but converts
    weights to the uint8-packed format expected by the Triton fused MoE kernel
    instead of the Marlin-specific format.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "is_triton_converted", False):
            return

        num_experts = layer.w13_weight_packed.shape[0]

        # Convert w13 weights: [E, K//8, N] int32 -> [E, N, K//2] uint8
        w13 = layer.w13_weight_packed.data
        w13 = w13.transpose(1, 2).contiguous().view(torch.uint8)
        layer.w13_weight_packed = torch.nn.Parameter(w13, requires_grad=False)

        # Convert w2 weights: [E, K//8, N] int32 -> [E, N, K//2] uint8
        w2 = layer.w2_weight_packed.data
        w2 = w2.transpose(1, 2).contiguous().view(torch.uint8)
        layer.w2_weight_packed = torch.nn.Parameter(w2, requires_grad=False)

        # Convert w13 scales: [E, K//group_size, N] -> [E, N, K//group_size]
        w13_scale = layer.w13_weight_scale.data
        w13_scale = w13_scale.transpose(1, 2).contiguous()
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)

        # Convert w2 scales: [E, K//group_size, N] -> [E, N, K//group_size]
        w2_scale = layer.w2_weight_scale.data
        w2_scale = w2_scale.transpose(1, 2).contiguous()
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)

        layer.is_triton_converted = True

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def _get_quant_info(self, layer: torch.nn.Module):
        from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight_packed,
            w2_weight=layer.w2_weight_packed,
            use_int4_w4a16=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            block_shape=[0, self.group_size],
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."
        return self.runner.run(dispatch_output, self._get_quant_info(layer))

    def apply_deepep_normal(self, layer: torch.nn.Module, dispatch_output):
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."
        return self.runner.run(dispatch_output, self._get_quant_info(layer)).hidden_states


class NPUCompressedTensorsW4A16Int4DynamicMoE(CompressedTensorsMoEScheme):

    def __init__(self, quantization_config) -> None:
        self.pack_factor = 8  # weight dtype is int4,  but use int32 to create
        target = (
            "MoEGMM" if "MoEGMM" in quantization_config.target_scheme_map else "Linear"
        )
        if target in quantization_config.target_scheme_map:
            self.group_size = quantization_config.target_scheme_map[target][
                "weights"
            ].group_size
        else:
            self.group_size = 128

        self.kernel = NPUW4A16Int4DynamicMoEMethod()

    # TODO: See if we can merge this method's logic
    # with CompressedTensorsWNA16MoE. Need more models and tests.
    # @OrangeRedeng @TamirBaydasov
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        self.num_experts = num_experts
        if (
            extra_weight_attrs.get(
                "intermediate_size_full", intermediate_size_per_partition
            )
            // intermediate_size_per_partition
            > 1
        ):
            quant_method = FusedMoeWeightScaleSupported.GROUP.value
        else:
            quant_method = FusedMoeWeightScaleSupported.CHANNEL.value
        extra_weight_attrs.update({"quant_method": quant_method})
        # weight
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # scale
        weight_scale_dtype = torch.bfloat16
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # offset
        w13_weight_offset = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_offset", w13_weight_offset)
        set_weight_attrs(w13_weight_offset, extra_weight_attrs)

        w2_weight_offset = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_offset", w2_weight_offset)
        set_weight_attrs(w2_weight_offset, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        return self.kernel.apply(layer, dispatch_output)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        return self.kernel.apply_without_routing_weights(
            layer,
            hidden_states,
            hidden_states_scale,
            group_list_type,
            group_list,
            output_dtype,
        )
