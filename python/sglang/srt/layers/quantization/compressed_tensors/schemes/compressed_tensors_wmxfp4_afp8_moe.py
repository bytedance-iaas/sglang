from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_moe_runner_backend,
)
from sglang.srt.layers.quantization.marlin_utils import check_moe_marlin_supports_layer
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)
from sglang.srt.utils import set_weight_attrs

from .compressed_tensors_scheme import CompressedTensorsMoEScheme

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, StandardDispatchOutput


__all__ = ["CompressedTensorsWMXFP4AFP8MoE"]


class CompressedTensorsWMXFP4AFP8MoE(CompressedTensorsMoEScheme):
    """Compressed-tensors MXFP4 MoE path backed by the Marlin runner.

    This is the GLM-5.2 checkpoint family where routed experts are stored as
    `weight_packed` + `weight_scale` tensors under the compressed-tensors
    config, while attention and shared experts stay in higher precision.
    """

    def __init__(self, quant_config, weight_quant, input_quant):
        del input_quant
        self.quant_config = quant_config
        self.group_size = getattr(weight_quant, "group_size", None)
        if self.group_size != 32:
            raise ValueError(
                "CompressedTensorsWMXFP4AFP8MoE currently supports only "
                f"group_size=32 for Marlin, got {self.group_size!r}."
            )

        scale_dtype = str(getattr(weight_quant, "scale_dtype", "") or "").lower()
        self.scale_dtype = torch.uint8 if "uint8" in scale_dtype else torch.float32

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        layer.params_dtype = params_dtype
        layer.orig_dtype = params_dtype

        weight_attrs = dict(extra_weight_attrs)
        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.GROUP.value

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=self.scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, scale_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=self.scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, scale_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not check_moe_marlin_supports_layer(layer, self.group_size):
            raise RuntimeError(
                "Current MXFP4 compressed-tensors MoE layer is not supported by Marlin."
            )

        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")

        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        prepare_moe_mxfp4_layer_for_marlin(layer)
        layer._mxfp4_backend = "marlin"

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        moe_backend = get_moe_runner_backend()
        if not (moe_backend.is_auto() or moe_backend.is_marlin()):
            raise NotImplementedError(
                "CompressedTensorsWMXFP4AFP8MoE currently supports only the "
                f"marlin MoE runner backend, got {moe_backend}."
            )

        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)

    def get_marlin_quant_info(self, layer: torch.nn.Module):
        from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo

        return MarlinMoeQuantInfo(
            w13_qweight=layer.w13_weight,
            w2_qweight=layer.w2_weight,
            w13_scales=layer.w13_weight_scale,
            w2_scales=layer.w2_weight_scale,
            w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None,
            weight_bits=4,
            is_k_full=True,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        quant_info = self.get_marlin_quant_info(layer)
        return self.runner.run(dispatch_output, quant_info)
