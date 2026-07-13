from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.awq import AWQConfig, AWQLinearMethod
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.gptq import GPTQConfig, GPTQLinearMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
        StandardDispatchOutput,
    )

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = logging.getLogger(__name__)


class W4AFp8Config(QuantizationConfig):
    """Config class for MIXED_PRECISION W4AFp8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = True,
        is_checkpoint_w4afp8_serialized: bool = True,
        linear_activation_scheme: str = "dynamic",
        moe_activation_scheme: str = "static",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: Optional[List[int]] = None,
        group_size: int = 128,
        linear_quant_method: Optional[str] = None,
        linear_quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        self.is_checkpoint_w4afp8_serialized = is_checkpoint_w4afp8_serialized
        if is_checkpoint_w4afp8_serialized:
            logger.warning("Detected w4afp8 checkpoint. Please note that")
        if moe_activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {moe_activation_scheme}")
        self.linear_activation_scheme = linear_activation_scheme
        self.moe_activation_scheme = moe_activation_scheme
        self.ignored_layers = ignored_layers or []
        self.weight_block_size = [128, 128]
        self.group_size = group_size
        self.linear_quant_method = linear_quant_method
        self.linear_quant_config = linear_quant_config

    @classmethod
    def get_name(cls) -> str:
        return "w4afp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> W4AFp8Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        is_checkpoint_w4afp8_serialized = "w4afp8" in quant_method
        linear_activation_scheme = "dynamic"
        moe_activation_scheme = "static"
        weight_block_size = [128, 128]
        group_size = cls.get_from_keys_or(config, ["group_size"], default=32)
        linear_quant_method = None
        linear_quant_config = None
        try:
            linear_quant_config = GPTQConfig.from_config(config)
            linear_quant_method = "gptq"
        except Exception:
            try:
                linear_quant_config = AWQConfig.from_config(config)
                linear_quant_method = "awq"
            except Exception:
                linear_quant_config = None
                linear_quant_method = None
        cfg = cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            is_checkpoint_w4afp8_serialized=is_checkpoint_w4afp8_serialized,
            linear_activation_scheme=linear_activation_scheme,
            moe_activation_scheme=moe_activation_scheme,
            weight_block_size=weight_block_size,
            group_size=group_size,
            linear_quant_method=linear_quant_method,
            linear_quant_config=linear_quant_config,
        )
        quant_format = config.get("format")
        has_ct_schema = bool(config.get("config_groups")) or "compression_config" in config
        cfg.is_compressed_tensors = (
            quant_format in ("compressed-tensors", "compressed_tensors")
            or has_ct_schema
        )
        return cfg

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            if (
                getattr(self, "is_compressed_tensors", False)
                and self.linear_quant_method is None
            ):
                return UnquantizedLinearMethod()
            if self.linear_quant_method == "gptq" and isinstance(
                self.linear_quant_config, GPTQConfig
            ):
                return GPTQLinearMethod(self.linear_quant_config)
            if self.linear_quant_method == "awq" and isinstance(
                self.linear_quant_config, AWQConfig
            ):
                return AWQLinearMethod(self.linear_quant_config)
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return W4AFp8MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


def interleave_scales(scales: torch.Tensor) -> torch.Tensor:
    """Interleave scales in groups of 4 similar to TRT-LLM implementation."""
    s_shape = scales.shape
    # Reshape to separate groups of 4
    alignment = 4 if s_shape[2] % 4 == 0 else 1
    scales_interleaved = scales.reshape(
        s_shape[0], s_shape[1], (s_shape[2] // alignment), alignment
    )
    # Permute dimensions to interleave
    scales_interleaved = scales_interleaved.permute(0, 2, 1, 3)
    # Reshape back to original dimensions but with interleaved values
    scales_interleaved = scales_interleaved.reshape(
        s_shape[0], s_shape[2] // alignment, s_shape[1] * alignment
    )
    return scales_interleaved.contiguous()


class W4AFp8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: W4AFp8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        assert "weight_loader" in extra_weight_attrs

        use_compressed = getattr(self.quant_config, "is_compressed_tensors", False)

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                0,
                dtype=torch.int8,
            )
            if use_compressed
            else torch.empty(
                num_experts,
                intermediate_size_per_partition * 2,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                0,
                dtype=torch.int8,
            )
            if use_compressed
            else torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                0,
                dtype=torch.float32,
            )
            if use_compressed
            else torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.quant_config.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                0,
                dtype=torch.float32,
            )
            if use_compressed
            else torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.quant_config.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        if use_compressed:
            packed_factor = 8
            compat_weight_attrs = dict(extra_weight_attrs)
            compat_weight_attrs.update({"is_transposed": True})

            w13_weight_packed = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size // packed_factor,
                    2 * intermediate_size_per_partition,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_packed", w13_weight_packed)
            set_weight_attrs(w13_weight_packed, compat_weight_attrs)

            w2_weight_packed = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    intermediate_size_per_partition // packed_factor,
                    hidden_size,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_packed", w2_weight_packed)
            set_weight_attrs(w2_weight_packed, compat_weight_attrs)

            w13_weight_scale_ct = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size // self.quant_config.group_size,
                    2 * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale_ct)
            set_weight_attrs(w13_weight_scale_ct, compat_weight_attrs)

            w2_weight_scale_ct = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    intermediate_size_per_partition // self.quant_config.group_size,
                    hidden_size,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale_ct)
            set_weight_attrs(w2_weight_scale_ct, compat_weight_attrs)

            w13_weight_shape = torch.nn.Parameter(
                torch.empty(num_experts, 2), requires_grad=False
            )
            layer.register_parameter("w13_weight_shape", w13_weight_shape)
            set_weight_attrs(w13_weight_shape, compat_weight_attrs)

            w2_weight_shape = torch.nn.Parameter(
                torch.empty(num_experts, 2), requires_grad=False
            )
            layer.register_parameter("w2_weight_shape", w2_weight_shape)
            set_weight_attrs(w2_weight_shape, compat_weight_attrs)

        # Input scales
        w13_input_scale = torch.nn.Parameter(
            torch.ones((num_experts, 2), dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w13_input_scale", w13_input_scale)
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

        # Pre-populate the strides
        device = layer.w13_weight.device

        self.a_strides1 = torch.full(
            (num_experts, 3),
            hidden_size,
            device=device,
            dtype=torch.int64,
        )
        self.c_strides1 = torch.full(
            (num_experts, 3),
            2 * intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.a_strides2 = torch.full(
            (num_experts, 3),
            intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.c_strides2 = torch.full(
            (num_experts, 3),
            hidden_size,
            device=device,
            dtype=torch.int64,
        )
        self.b_strides1 = self.a_strides1
        self.s_strides13 = self.c_strides1
        self.b_strides2 = self.a_strides2
        self.s_strides2 = self.c_strides2

        self.expert_offsets = torch.empty(
            (num_experts + 1), dtype=torch.int32, device=device
        )
        self.problem_sizes1 = torch.empty(
            (num_experts, 3), dtype=torch.int32, device=device
        )
        self.problem_sizes2 = torch.empty(
            (num_experts, 3), dtype=torch.int32, device=device
        )

        return

    def process_weights_after_loading(self, layer: Module) -> None:
        dtype = torch.bfloat16
        device = layer.w2_weight.device

        if (
            hasattr(layer, "w13_weight_packed")
            and layer.w13_weight_packed.numel() > 0
            and layer.w13_weight_packed.ne(0).any().item()
        ):
            packed = layer.w13_weight_packed
            num_experts, packed_k, n = packed.shape
            w13_weight = torch.empty(
                (num_experts, n, packed_k * 4),
                dtype=torch.int8,
                device=packed.device,
            )
            for expert_id in range(num_experts):
                tmp = (
                    packed[expert_id]
                    .transpose(0, 1)
                    .contiguous()
                    .view(torch.uint8)
                    .to(torch.int8)
                )
                w13_weight[expert_id].copy_(tmp)
            layer.w13_weight = Parameter(w13_weight, requires_grad=False)
            if (
                hasattr(layer, "w2_weight_packed")
                and layer.w2_weight_packed.numel() > 0
                and layer.w2_weight_packed.ne(0).any().item()
            ):
                packed = layer.w2_weight_packed
                num_experts, packed_k, n = packed.shape
                w2_weight = torch.empty(
                    (num_experts, n, packed_k * 4),
                    dtype=torch.int8,
                    device=packed.device,
                )
                for expert_id in range(num_experts):
                    tmp = (
                        packed[expert_id]
                        .transpose(0, 1)
                        .contiguous()
                        .view(torch.uint8)
                        .to(torch.int8)
                    )
                    w2_weight[expert_id].copy_(tmp)
                layer.w2_weight = Parameter(w2_weight, requires_grad=False)
            if (
                hasattr(layer, "w13_weight_scale")
                and layer.w13_weight_scale.numel() > 0
                and layer.w13_weight_scale.ne(0).any().item()
            ):
                scales = layer.w13_weight_scale
                num_experts, k_groups, n = scales.shape
                w13_scale = torch.empty(
                    (num_experts, n, k_groups),
                    dtype=scales.dtype,
                    device=scales.device,
                )
                for expert_id in range(num_experts):
                    w13_scale[expert_id].copy_(
                        scales[expert_id].transpose(0, 1).contiguous()
                    )
                layer.w13_weight_scale_inv = Parameter(w13_scale, requires_grad=False)
            if (
                hasattr(layer, "w2_weight_scale")
                and layer.w2_weight_scale.numel() > 0
                and layer.w2_weight_scale.ne(0).any().item()
            ):
                scales = layer.w2_weight_scale
                num_experts, k_groups, n = scales.shape
                w2_scale = torch.empty(
                    (num_experts, n, k_groups),
                    dtype=scales.dtype,
                    device=scales.device,
                )
                for expert_id in range(num_experts):
                    w2_scale[expert_id].copy_(
                        scales[expert_id].transpose(0, 1).contiguous()
                    )
                layer.w2_weight_scale_inv = Parameter(w2_scale, requires_grad=False)
            if hasattr(layer, "w13_weight_scale"):
                layer.w13_weight_scale = Parameter(
                    torch.empty(0, dtype=layer.w13_weight_scale.dtype, device=device),
                    requires_grad=False,
                )
            if hasattr(layer, "w2_weight_scale"):
                layer.w2_weight_scale = Parameter(
                    torch.empty(0, dtype=layer.w2_weight_scale.dtype, device=device),
                    requires_grad=False,
                )
            if hasattr(layer, "w13_weight_shape"):
                layer.w13_weight_shape = Parameter(
                    torch.empty(0, dtype=torch.int32, device=device),
                    requires_grad=False,
                )
            if hasattr(layer, "w2_weight_shape"):
                layer.w2_weight_shape = Parameter(
                    torch.empty(0, dtype=torch.int32, device=device),
                    requires_grad=False,
                )
            layer.w13_weight_packed = Parameter(
                torch.empty(0, dtype=torch.int32, device=device), requires_grad=False
            )
            if hasattr(layer, "w2_weight_packed"):
                layer.w2_weight_packed = Parameter(
                    torch.empty(0, dtype=torch.int32, device=device),
                    requires_grad=False,
                )

        # Interleave w13_weight_scale (gate_up_proj)
        w13_weight_scale = layer.w13_weight_scale_inv.to(dtype)
        w13_weight_scale = interleave_scales(w13_weight_scale)
        layer.w13_weight_scale_inv = Parameter(w13_weight_scale, requires_grad=False)

        # Interleave w2_weight_scale (down_proj)
        w2_weight_scale = layer.w2_weight_scale_inv.to(dtype)
        w2_weight_scale = interleave_scales(w2_weight_scale)
        layer.w2_weight_scale_inv = Parameter(w2_weight_scale, requires_grad=False)

        # Process input scales
        w13_input_scale_max = layer.w13_input_scale.max().to(torch.float32).item()
        new_w13_input_scale = torch.tensor(
            [w13_input_scale_max],
            dtype=torch.float32,
            device=device,
        )
        layer.w13_input_scale = Parameter(new_w13_input_scale, requires_grad=False)

        w2_input_scale_max = layer.w2_input_scale.max().to(torch.float32).item()
        new_w2_input_scale = torch.tensor(
            [w2_input_scale_max], dtype=torch.float32, device=device
        )
        layer.w2_input_scale = Parameter(new_w2_input_scale, requires_grad=False)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        output = cutlass_w4a8_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            topk_weights,
            topk_ids,
            self.a_strides1,
            self.b_strides1,
            self.c_strides1,
            self.a_strides2,
            self.b_strides2,
            self.c_strides2,
            self.s_strides13,
            self.s_strides2,
            self.expert_offsets,
            self.problem_sizes1,
            self.problem_sizes2,
            layer.w13_input_scale,
            layer.w2_input_scale,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor or 1.0,
        )
        return StandardCombineInput(hidden_states=output)

    def apply_deepep_ll(
        self,
        layer: DeepEPMoE,
        dispatch_output: DeepEPLLDispatchOutput,
    ) -> torch.Tensor:

        from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe_deepep_ll

        hidden_states, hidden_scales, topk_ids, _, masked_m, _ = dispatch_output

        output = cutlass_w4a8_moe_deepep_ll(
            hidden_states,
            hidden_scales,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            topk_ids,
            masked_m,
            layer.quant_method.a_strides1,
            layer.quant_method.b_strides1,
            layer.quant_method.c_strides1,
            layer.quant_method.a_strides2,
            layer.quant_method.b_strides2,
            layer.quant_method.c_strides2,
            layer.quant_method.s_strides13,
            layer.quant_method.s_strides2,
            layer.quant_method.expert_offsets,
            layer.quant_method.problem_sizes1,
            layer.quant_method.problem_sizes2,
            layer.w13_input_scale,
            layer.w2_input_scale,
        )

        return output

    def apply_deepep_normal(
        self,
        layer: DeepEPMoE,
        dispatch_output: DeepEPNormalDispatchOutput,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.cutlass_w4a8_moe import (
            cutlass_w4a8_moe_deepep_normal,
        )

        hidden_states, topk_idx, topk_weights = (
            dispatch_output.hidden_states,
            dispatch_output.topk_ids,
            dispatch_output.topk_weights,
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        num_tokens = hidden_states.shape[0]
        if num_tokens > 0:
            return cutlass_w4a8_moe_deepep_normal(
                hidden_states,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale_inv,
                layer.w2_weight_scale_inv,
                topk_weights,
                topk_idx,
                self.a_strides1,
                self.b_strides1,
                self.c_strides1,
                self.a_strides2,
                self.b_strides2,
                self.c_strides2,
                self.s_strides13,
                self.s_strides2,
                self.expert_offsets,
                self.problem_sizes1,
                self.problem_sizes2,
                layer.w13_input_scale,
                layer.w2_input_scale,
            )
        else:
            return hidden_states
