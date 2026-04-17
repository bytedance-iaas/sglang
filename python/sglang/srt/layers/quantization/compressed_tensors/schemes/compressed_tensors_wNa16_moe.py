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
_DEEPEP_LL_GRAPH_DEBUG = get_bool_env_var("SGLANG_DEEPEP_LL_GRAPH_DEBUG")

logger = logging.getLogger(__name__)

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
if _use_aiter:
    pass


def _get_deepep_ll_direct_workspace_size(
    num_experts: int,
    device: torch.device,
) -> int:
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    # Direct Marlin uses one lock stripe array per expert, and the kernel may
    # advance the per-expert lock offset by one extra slot while reducing the
    # last slice. Keep one extra lock per expert on top of the max blocks-per-sm
    # budget used by the kernel config search.
    return max((sms * 4 + 1) * max(num_experts, 1), 128)

if _is_cuda:
    import triton
    import triton.language as tl

if _is_cuda:

    @triton.jit
    def _build_active_expert_ids_kernel(
        masked_m_ptr,
        active_expert_ids_ptr,
        active_expert_counter_ptr,
        num_experts,
        active_expert_capacity,
        BLOCK_E: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
        mask = offs < num_experts
        token_count = tl.load(masked_m_ptr + offs, mask=mask, other=0)
        active = mask & (token_count > 0)
        slots = tl.atomic_add(
            active_expert_counter_ptr + tl.zeros([BLOCK_E], dtype=tl.int32),
            1,
            mask=active,
        )
        store_mask = active & (slots < active_expert_capacity)
        tl.store(active_expert_ids_ptr + slots, offs, mask=store_mask)


def _masked_silu_and_mul_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    masked_m: torch.Tensor,
    token_upper_hint: Optional[int] = None,
):
    from sglang.jit_kernel.masked_silu_and_mul import masked_silu_and_mul

    return masked_silu_and_mul(
        input, output, masked_m,
        token_upper_hint=token_upper_hint,
        use_fp32_accum=True,
    )


def _build_active_expert_ids_fwd(
    masked_m: torch.Tensor,
    active_expert_ids: torch.Tensor,
    active_expert_counter: torch.Tensor,
):
    assert masked_m.is_cuda
    assert active_expert_ids.is_cuda
    assert active_expert_counter.is_cuda
    assert masked_m.ndim == 1
    assert active_expert_ids.ndim == 1
    assert active_expert_counter.numel() == 1

    active_expert_ids.fill_(-1)
    active_expert_counter.zero_()
    num_experts = int(masked_m.numel())
    capacity = int(active_expert_ids.numel())
    if num_experts == 0 or capacity == 0:
        return active_expert_ids

    block_e = 256
    grid = (triton.cdiv(num_experts, block_e),)
    _build_active_expert_ids_kernel[grid](
        masked_m,
        active_expert_ids,
        active_expert_counter,
        num_experts,
        capacity,
        BLOCK_E=block_e,
        num_warps=4,
    )
    return active_expert_ids


def _select_deepep_ll_graph_launch_experts(
    active_tokens: int,
    num_experts: int,
) -> int:
    # CUDA graph replay requires a fixed launch shape. Keep small batches tight,
    # but avoid over-compressing y-dimension parallelism once batch size reaches
    # the 8/16/32+ range.
    if active_tokens <= 1:
        return 1
    if active_tokens <= 2:
        return min(num_experts, 2)
    if active_tokens <= 4:
        return min(num_experts, 4)
    if active_tokens <= 8:
        return min(num_experts, 8)
    if active_tokens <= 16:
        return min(num_experts, 16)
    if active_tokens <= 32:
        return min(num_experts, 24)
    if active_tokens <= 64:
        return min(num_experts, 32)
    if active_tokens <= 128:
        return min(num_experts, 40)
    return num_experts


def _select_deepep_ll_direct_block_size(
    masked_m: torch.Tensor,
    *,
    capture_active: bool,
    expected_m: Optional[int] = None,
    token_upper_hint: Optional[int] = None,
    total_tokens_hint: Optional[int] = None,
    active_experts_hint: Optional[int] = None,
    is_w13_stage: bool = False,
) -> tuple[int, int, int]:
    def _select_three_tier_block(
        token_upper: int,
        active_experts: int,
        total_tokens: int,
        is_w13_stage: bool,
    ) -> int:
        # Three-tier policy for LL decode:
        # - 8: tiny sparse cases, best latency on current H100 traces
        # - 16: default middle ground for moderate token upper bound
        # - 32: only when upper bound is clearly larger and can amortize block cost
        if token_upper <= 8:
            return 8
        if token_upper <= 16:
            return 16
        if token_upper > 32:
            return 32

        padded_16 = ((total_tokens + 15) // 16) * 16
        padded_32 = ((total_tokens + 31) // 32) * 32
        prefer_32 = padded_32 + max(active_experts, 2) < padded_16
        if is_w13_stage:
            prefer_32 = prefer_32 or (token_upper >= 28 and active_experts >= 2)
        else:
            prefer_32 = prefer_32 or (token_upper >= 20 and active_experts >= 2)
        if prefer_32:
            return 32
        return 16

    if capture_active:
        token_upper = int(expected_m or 0)
        if token_upper <= 0:
            token_upper = 8
        return (
            _select_three_tier_block(
                token_upper,
                int(masked_m.numel()),
                token_upper * max(int(masked_m.numel()), 1),
                is_w13_stage,
            ),
            int(masked_m.numel()),
            token_upper,
        )

    if masked_m.numel() == 0:
        return 8, 0, 0

    if (
        token_upper_hint is not None
        and total_tokens_hint is not None
        and active_experts_hint is not None
    ):
        token_upper = max(int(token_upper_hint), 1)
        active_experts = max(int(active_experts_hint), 0)
        total_tokens = max(int(total_tokens_hint), 0)
        if active_experts == 0 or total_tokens == 0:
            return 8, 0, 0
        return (
            _select_three_tier_block(
                token_upper, active_experts, total_tokens, is_w13_stage
            ),
            active_experts,
            token_upper,
        )

    masked_m_host = masked_m.detach().cpu() if masked_m.is_cuda else masked_m
    if masked_m_host.dtype != torch.int32:
        masked_m_host = masked_m_host.to(dtype=torch.int32)

    active = masked_m_host[masked_m_host > 0]
    if active.numel() == 0:
        return 8, 0, 0

    active_experts = int(active.numel())
    token_upper = int(active.max().item())
    total_tokens = int(active.sum().item())
    return (
        _select_three_tier_block(
            token_upper, active_experts, total_tokens, is_w13_stage
        ),
        active_experts,
        token_upper,
    )

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

    def _get_deepep_ll_direct_impl(self):
        try:
            from sglang.jit_kernel.deepep_moe_wna16_marlin_direct import (
                deepep_moe_wna16_marlin_direct_gemm,
            )

            return deepep_moe_wna16_marlin_direct_gemm, True
        except Exception:
            from sglang.jit_kernel.deepep_moe_wna16_marlin import (
                deepep_moe_wna16_marlin,
            )

            return deepep_moe_wna16_marlin, False

    def _get_deepep_ll_direct_runtime(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        *,
        num_experts: int,
        logical_m: int,
        size_k: int,
        size_n: int,
    ) -> dict:
        runtime_cache = getattr(layer, "_deepep_ll_direct_runtime_cache", None)
        if runtime_cache is None:
            runtime_cache = {}
            layer._deepep_ll_direct_runtime_cache = runtime_cache

        runtime_key = (
            hidden_states.device.index,
            str(hidden_states.dtype),
            num_experts,
            size_k,
            size_n,
        )
        runtime = runtime_cache.get(runtime_key)
        workspace_size = _get_deepep_ll_direct_workspace_size(
            num_experts, hidden_states.device
        )
        needs_grow = (
            runtime is None
            or runtime["logical_m_capacity"] < logical_m
            or runtime["workspace_size"] < workspace_size
        )
        if needs_grow:
            workspace_size = _get_deepep_ll_direct_workspace_size(
                num_experts, hidden_states.device
            )
            runtime = {
                "logical_m_capacity": logical_m,
                "workspace_size": workspace_size,
                "tmp1": torch.empty(
                    (num_experts, logical_m, 2 * size_n),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                ),
                "tmp2": torch.empty(
                    (num_experts, logical_m, size_n),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                ),
                "w13_workspace": torch.zeros(
                    workspace_size,
                    dtype=torch.int,
                    device=hidden_states.device,
                    requires_grad=False,
                ),
                "w13_c_tmp": torch.empty(
                    0,
                    dtype=torch.float32,
                    device=hidden_states.device,
                ),
                "w2_workspace": torch.zeros(
                    workspace_size,
                    dtype=torch.int,
                    device=hidden_states.device,
                    requires_grad=False,
                ),
                "w2_c_tmp": torch.empty(
                    0,
                    dtype=torch.float32,
                    device=hidden_states.device,
                ),
                "active_expert_ids": torch.empty(
                    (num_experts,),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
                "active_expert_counter": torch.empty(
                    (1,),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
            }
            runtime_cache[runtime_key] = runtime
            if _DEEPEP_LL_GRAPH_DEBUG:
                logger.warning(
                    "SGLANG_DEEPEP_LL_GRAPH_DEBUG_RUNTIME_CREATE rank=%s layer_id=%s key=%s logical_m_capacity=%s workspace_size=%s tmp1_ptr=%s tmp2_ptr=%s w13_ws_ptr=%s w13_c_tmp_ptr=%s w2_ws_ptr=%s w2_c_tmp_ptr=%s active_ids_ptr=%s active_counter_ptr=%s",
                    getattr(layer, "moe_ep_rank", -1),
                    id(layer),
                    runtime_key,
                    runtime["logical_m_capacity"],
                    runtime["workspace_size"],
                    runtime["tmp1"].data_ptr(),
                    runtime["tmp2"].data_ptr(),
                    runtime["w13_workspace"].data_ptr(),
                    runtime["w13_c_tmp"].data_ptr(),
                    runtime["w2_workspace"].data_ptr(),
                    runtime["w2_c_tmp"].data_ptr(),
                    runtime["active_expert_ids"].data_ptr(),
                    runtime["active_expert_counter"].data_ptr(),
                )
        elif _DEEPEP_LL_GRAPH_DEBUG:
            logger.warning(
                "SGLANG_DEEPEP_LL_GRAPH_DEBUG_RUNTIME_REUSE rank=%s layer_id=%s key=%s logical_m_capacity=%s workspace_size=%s request_logical_m=%s tmp1_ptr=%s tmp2_ptr=%s w13_ws_ptr=%s w13_c_tmp_ptr=%s w2_ws_ptr=%s w2_c_tmp_ptr=%s active_ids_ptr=%s active_counter_ptr=%s",
                getattr(layer, "moe_ep_rank", -1),
                id(layer),
                runtime_key,
                runtime["logical_m_capacity"],
                runtime["workspace_size"],
                logical_m,
                runtime["tmp1"].data_ptr(),
                runtime["tmp2"].data_ptr(),
                runtime["w13_workspace"].data_ptr(),
                runtime["w13_c_tmp"].data_ptr(),
                runtime["w2_workspace"].data_ptr(),
                runtime["w2_c_tmp"].data_ptr(),
                runtime["active_expert_ids"].data_ptr(),
                runtime["active_expert_counter"].data_ptr(),
            )
        return runtime

    def _run_deepep_ll_stageb_direct(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        masked_m: torch.Tensor,
        active_expert_ids: Optional[torch.Tensor],
        active_expert_count: Optional[torch.Tensor],
        launch_experts: int,
        *,
        direct_expected_m: int,
        w13_block_size_m: int,
        w2_block_size_m: int,
        logical_m: int,
        direct_gemm,
        scalar_type,
        stage_profile: Optional[dict] = None,
    ) -> torch.Tensor:
        size_k = hidden_states.shape[2]
        size_n = layer.w2_weight_packed.shape[1] * 16
        runtime = self._get_deepep_ll_direct_runtime(
            layer,
            hidden_states,
            num_experts=hidden_states.shape[0],
            logical_m=logical_m,
            size_k=size_k,
            size_n=size_n,
        )
        tmp1 = runtime["tmp1"][:, :logical_m, :]
        tmp2 = runtime["tmp2"][:, :logical_m, :]

        if stage_profile is not None:
            stage_profile["w13_start"] = torch.cuda.Event(enable_timing=True)
            stage_profile["w13_end"] = torch.cuda.Event(enable_timing=True)
            stage_profile["act_start"] = torch.cuda.Event(enable_timing=True)
            stage_profile["act_end"] = torch.cuda.Event(enable_timing=True)
            stage_profile["w2_start"] = torch.cuda.Event(enable_timing=True)
            stage_profile["w2_end"] = torch.cuda.Event(enable_timing=True)
            stage_profile["w13_start"].record()
        tmp1 = direct_gemm(
            hidden_states[:, :logical_m, :],
            tmp1,
            layer.w13_weight_packed,
            None,
            layer.w13_weight_scale,
            None,
            None,
            layer.w13_weight_g_idx,
            active_expert_ids,
            active_expert_count,
            launch_experts,
            runtime["w13_workspace"],
            runtime["w13_c_tmp"],
            masked_m,
            direct_expected_m,
            w13_block_size_m,
            scalar_type,
            logical_m,
            2 * size_n,
            size_k,
            True,
            is_k_full=self.is_k_full,
            use_atomic_add=False,
            use_fp32_reduce=False,
            is_zp_float=False,
        )

        if stage_profile is not None:
            stage_profile["w13_end"].record()
            stage_profile["act_start"].record()
        _masked_silu_and_mul_fwd(
            tmp1,
            tmp2,
            masked_m,
            token_upper_hint=logical_m,
        )
        if stage_profile is not None:
            stage_profile["act_end"].record()
            stage_profile["w2_start"].record()

        direct_gemm(
            tmp2,
            hidden_states[:, :logical_m, :],
            layer.w2_weight_packed,
            None,
            layer.w2_weight_scale,
            None,
            None,
            layer.w2_weight_g_idx,
            active_expert_ids,
            active_expert_count,
            launch_experts,
            runtime["w2_workspace"],
            runtime["w2_c_tmp"],
            masked_m,
            direct_expected_m,
            w2_block_size_m,
            scalar_type,
            logical_m,
            size_k,
            size_n,
            False,
            is_k_full=self.is_k_full,
            use_atomic_add=False,
            use_fp32_reduce=False,
            is_zp_float=False,
        )
        if stage_profile is not None:
            stage_profile["w2_end"].record()
        return hidden_states

    def _fused_marlin_deepep_ll_direct(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        masked_m: torch.Tensor,
        *,
        expected_m: Optional[int],
        capture_active: bool,
        active_tokens_hint: int,
        token_upper_hint: int,
        active_experts_hint: int,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            get_scalar_type,
        )
        direct_gemm, use_direct = self._get_deepep_ll_direct_impl()

        assert hidden_states.ndim == 3
        assert masked_m.ndim == 1

        num_experts = hidden_states.shape[0]
        active_tokens = max(int(active_tokens_hint), 0)
        if active_tokens == 0:
            return torch.zeros_like(hidden_states)

        w13_block_size_m, active_experts, token_upper = _select_deepep_ll_direct_block_size(
            masked_m,
            capture_active=capture_active,
            expected_m=expected_m,
            token_upper_hint=token_upper_hint,
            total_tokens_hint=active_tokens_hint,
            active_experts_hint=active_experts_hint,
            is_w13_stage=True,
        )
        w2_block_size_m, _, _ = _select_deepep_ll_direct_block_size(
            masked_m,
            capture_active=capture_active,
            expected_m=expected_m,
            token_upper_hint=token_upper_hint,
            total_tokens_hint=active_tokens_hint,
            active_experts_hint=active_experts_hint,
            is_w13_stage=False,
        )
        if capture_active:
            direct_expected_m = token_upper if expected_m is None else int(expected_m)
        else:
            direct_expected_m = token_upper

        profile_enabled = _LOW_LATENCY_PROFILE_LOG and hidden_states.is_cuda
        stage_profile = {} if profile_enabled and use_direct else None
        if profile_enabled:
            marlin_start = torch.cuda.Event(enable_timing=True)
            marlin_end = torch.cuda.Event(enable_timing=True)
            marlin_start.record()

        if use_direct:
            size_m = hidden_states.shape[1]
            logical_m = max(min(direct_expected_m, size_m), 1)
            active_expert_ids = None
            active_expert_count = None
            launch_experts = num_experts
            if active_tokens < num_experts and masked_m.is_cuda:
                runtime = self._get_deepep_ll_direct_runtime(
                    layer,
                    hidden_states,
                    num_experts=num_experts,
                    logical_m=logical_m,
                    size_k=hidden_states.shape[2],
                    size_n=layer.w2_weight_packed.shape[1] * 16,
                )
                active_expert_count = runtime["active_expert_counter"]
                if capture_active:
                    launch_experts = _select_deepep_ll_graph_launch_experts(
                        active_tokens,
                        num_experts,
                    )
                    active_expert_ids = runtime["active_expert_ids"]
                    _build_active_expert_ids_fwd(
                        masked_m,
                        active_expert_ids,
                        active_expert_count,
                    )
                else:
                    active_expert_ids = torch.nonzero(
                        masked_m > 0, as_tuple=False
                    ).flatten()
                    if active_expert_ids.numel() > 0:
                        active_expert_ids = active_expert_ids.to(
                            device=masked_m.device, dtype=torch.int32
                        ).contiguous()
                        launch_experts = int(active_expert_ids.numel())
                        active_expert_count.fill_(launch_experts)
                    else:
                        active_expert_ids = None
                        active_expert_count = None
            output = self._run_deepep_ll_stageb_direct(
                layer,
                hidden_states,
                masked_m,
                active_expert_ids,
                active_expert_count,
                launch_experts,
                direct_expected_m=direct_expected_m,
                w13_block_size_m=w13_block_size_m,
                w2_block_size_m=w2_block_size_m,
                logical_m=logical_m,
                direct_gemm=direct_gemm,
                scalar_type=get_scalar_type(self.num_bits, False),
                stage_profile=stage_profile,
            )
        else:
            runtime_cache = getattr(layer, "_deepep_ll_direct_marlin_runtime", None)
            if runtime_cache is None:
                runtime_cache = {}
                layer._deepep_ll_direct_marlin_runtime = runtime_cache
            output = direct_gemm(
                hidden_states,
                masked_m,
                layer.w13_weight_packed,
                layer.w2_weight_packed,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                layer.w13_weight_g_idx,
                layer.w2_weight_g_idx,
                layer.w13_g_idx_sort_indices,
                layer.w2_g_idx_sort_indices,
                self.num_bits,
                self.is_k_full,
                w13_block_size_m,
                runtime_cache,
            )

        if profile_enabled:
            marlin_end.record()
            marlin_end.synchronize()
            logger.warning(
                "DEEPEP_LL_PROFILE_MARLIN_DIRECT rank=%s active_tokens=%s active_experts=%s launch_experts=%s token_upper=%s num_experts=%s w13_block_m=%s w2_block_m=%s marlin_ms=%.3f",
                getattr(layer, "moe_ep_rank", -1),
                active_tokens,
                active_experts,
                launch_experts if use_direct else num_experts,
                token_upper,
                num_experts,
                w13_block_size_m,
                w2_block_size_m,
                marlin_start.elapsed_time(marlin_end),
            )
            if stage_profile:
                logger.warning(
                    "DEEPEP_LL_PROFILE_MARLIN_DIRECT_STAGE rank=%s active_tokens=%s w13_block_m=%s w2_block_m=%s w13_ms=%.3f act_ms=%.3f w2_ms=%.3f",
                    getattr(layer, "moe_ep_rank", -1),
                    active_tokens,
                    w13_block_size_m,
                    w2_block_size_m,
                    stage_profile["w13_start"].elapsed_time(stage_profile["w13_end"]),
                    stage_profile["act_start"].elapsed_time(stage_profile["act_end"]),
                    stage_profile["w2_start"].elapsed_time(stage_profile["w2_end"]),
                )
        return output

    def _prepare_deepep_ll_direct_run(
        self,
        hidden_states: torch.Tensor,
        dispatch_topk_ids: torch.Tensor,
    ) -> Optional[tuple[bool, int, int, int]]:
        if not _is_cuda:
            raise NotImplementedError("DeepEP low latency W4A16 sparse path requires CUDA.")

        num_local_experts, padded_m, _ = hidden_states.shape
        active_tokens = int(dispatch_topk_ids.numel())
        if num_local_experts == 0 or padded_m == 0 or active_tokens == 0:
            return None

        capture_active = (
            hidden_states.is_cuda and torch.cuda.is_current_stream_capturing()
        )
        token_upper_bound = min(padded_m, active_tokens)
        active_experts_hint = min(num_local_experts, active_tokens)
        return capture_active, active_tokens, token_upper_bound, active_experts_hint

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
        hidden_states, hidden_states_scale, dispatch_topk_ids, _, masked_m, expected_m = (
            dispatch_output
        )

        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."
        assert (
            hidden_states_scale is None
        ), "W4A16 DeepEP low latency currently requires BF16 dispatch."
        assert hidden_states.ndim == 3, "DeepEP low latency expects [E, M, K] inputs."

        prepared = self._prepare_deepep_ll_direct_run(hidden_states, dispatch_topk_ids)
        if prepared is None:
            return torch.zeros_like(hidden_states)
        capture_active, active_tokens, token_upper_bound, active_experts_hint = prepared
        profile_enabled = _LOW_LATENCY_PROFILE_LOG and hidden_states.is_cuda and not capture_active

        if profile_enabled:
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            total_start.record()
        output = self._fused_marlin_deepep_ll_direct(
            layer,
            hidden_states,
            masked_m,
            expected_m=expected_m,
            capture_active=capture_active,
            active_tokens_hint=active_tokens,
            token_upper_hint=token_upper_bound,
            active_experts_hint=active_experts_hint,
        )
        if profile_enabled:
            total_end.record()
            total_end.synchronize()
            logger.warning(
                "DEEPEP_LL_PROFILE_COMPUTE rank=%s capture=%s active_tokens=%s token_upper=%s hidden_shape=%s compact_ms=%.3f marlin_ms=%.3f expand_ms=%.3f total_ms=%.3f",
                getattr(layer, "moe_ep_rank", -1),
                capture_active,
                active_tokens,
                token_upper_bound,
                tuple(hidden_states.shape),
                0.0,
                total_start.elapsed_time(total_end),
                0.0,
                total_start.elapsed_time(total_end),
            )
        return output


class CompressedTensorsWNA16TritonMoE(CompressedTensorsWNA16MoE):
    """ROCm/HIP-compatible W4A16 MoE method using Triton kernels instead of Marlin.

    Inherits weight creation from CompressedTensorsWNA16MoE but converts
    weights to the uint8-packed format expected by the Triton fused MoE kernel
    instead of the Marlin-specific format.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "is_triton_converted", False):
            return

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
