"""MXFP4 MoE scheme: packed E2M1 weights + per-32 UE8M0 scales, FP8 activations.

Loads `mxfp4-pack-quantized` compressed-tensors checkpoints (e.g. GLM-5.2
DataFree-WMXFP4AFP8-GS32) and hands the weights to DeepGEMM's mega-MoE kernel.

Checkpoint layout (per expert, before stacking):
    gate/up_proj.weight_packed  uint8  [I, H//2]    two E2M1 nibbles per byte
    gate/up_proj.weight_scale   uint8  [I, H//32]   E8M0 biased exponent
    down_proj.weight_packed     uint8  [H, I//2]
    down_proj.weight_scale      uint8  [H, I//32]

The nibble order (low nibble = even K index) and the E2M1 encoding match
DeepGEMM's `per_token_cast_to_fp4`, so the packed bytes are handed over
untouched -- `MXFP4PackedCompressor` subclasses `NVFP4PackedCompressor` and
reuses its `pack_fp4_to_uint8`. Only the scales need converting: the kernel
consumes UE8M0 values carried in an fp32 container, not the raw E8M0 bytes.

This scheme only serves the mega-MoE path: SGLang has no SM90 MXFP4 grouped
GEMM to fall back on, so both weight preparation and `apply_weights` fail loudly
rather than silently producing garbage.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    MXFP4_PACK_QUANTIZED_FORMAT,
)
from sglang.srt.layers.quantization.mxfp8_block_convert import _ue8m0_to_fp32
from sglang.srt.utils import is_sm90_supported, is_sm100_supported, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsW4A8Mxfp4MoE"]

# Two E2M1 values per byte, and one E8M0 scale per 32 values along K.
MXFP4_PACK_FACTOR = 2
MXFP4_GROUP_SIZE = 32


class CompressedTensorsW4A8Mxfp4MoE(CompressedTensorsMoEScheme):
    """MXFP4 weights (per-32 E8M0 scales) + dynamic FP8 activations, via mega-MoE."""

    def __init__(
        self,
        quant_config: CompressedTensorsConfig,
        weight_quant,
        input_quant,
    ):
        self.quant_config = quant_config
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.packed_factor = MXFP4_PACK_FACTOR
        self.group_size = weight_quant.group_size

        assert self.quant_config.quant_format == MXFP4_PACK_QUANTIZED_FORMAT, (
            "MXFP4 MoE requires mxfp4-pack-quantized format, got "
            f"{self.quant_config.quant_format}"
        )
        assert (
            weight_quant.num_bits == 4
            and weight_quant.symmetric
            and self.group_size == MXFP4_GROUP_SIZE
        ), (
            "MXFP4 MoE requires symmetric 4-bit group-32 weights, got "
            f"num_bits={weight_quant.num_bits}, symmetric={weight_quant.symmetric}, "
            f"group_size={self.group_size}"
        )

        # `get_min_capability` is never consulted on the MoE path (only
        # `get_linear_scheme` calls `_check_scheme_supported`), so gate here.
        if not (is_sm90_supported() or is_sm100_supported()):
            raise ValueError(
                "MXFP4 MoE requires SM90 (Hopper) or SM100 (Blackwell); "
                "the mega-MoE FP4 kernel is unavailable on this device."
            )

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

        # `num_experts` is already EP-local and `intermediate_size_per_partition`
        # already TP-sharded; both shard sizes must stay group/pack aligned or the
        # loader would narrow at a misaligned offset without complaining.
        assert hidden_size % (self.group_size * self.packed_factor) == 0, (
            f"hidden_size {hidden_size} must be divisible by "
            f"{self.group_size * self.packed_factor}"
        )
        assert (
            intermediate_size_per_partition % (self.group_size * self.packed_factor)
            == 0
        ), (
            f"intermediate_size_per_partition {intermediate_size_per_partition} must "
            f"be divisible by {self.group_size * self.packed_factor}"
        )

        # Packed E2M1 weights, checkpoint (non-transposed) layout.
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.packed_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.packed_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # E8M0 scales stay uint8 here: the checkpoint stores raw biased
        # exponents, and copying them into a float parameter would convert the
        # byte values numerically instead of preserving them.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )

        w13_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)

        # No `w13_weight_shape` / `w2_weight_shape` placeholders here: unlike the
        # int-based pack-quantized formats, MXFP4PackedCompressor only emits
        # weight_packed and weight_scale, so such params would stay uninitialised
        # and still be picked up as expert weights by EPLB rebalancing.

        layer.is_mxfp4_converted = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Rename to the mega-MoE parameter names and decode E8M0 scales.

        `build_mega_moe_experts_weights` reads `w13_weight` / `w2_weight` /
        `w13_weight_scale_inv` / `w2_weight_scale_inv` by hard-coded name, and
        writes back through those Parameters, so they must be registered under
        exactly those names rather than aliased with `setattr`.
        """
        if layer.is_mxfp4_converted:
            return

        # uint8 -> int8 is a pure bit reinterpretation (DeepGEMM's kPackedFP4).
        w13 = layer.w13_weight_packed.data.view(torch.int8)
        w2 = layer.w2_weight_packed.data.view(torch.int8)

        # Shift-based decode keeps the fp32 mantissa exactly zero, which
        # DeepGEMM's `pack_ue8m0_to_int` asserts on. Computing 2**(v-127) in
        # floating point would drift into subnormals for small exponents.
        w13_sf = _ue8m0_to_fp32(layer.w13_weight_scale.data)
        w2_sf = _ue8m0_to_fp32(layer.w2_weight_scale.data)

        for stale in (
            "w13_weight_packed",
            "w2_weight_packed",
            "w13_weight_scale",
            "w2_weight_scale",
        ):
            delattr(layer, stale)

        layer.register_parameter(
            "w13_weight", torch.nn.Parameter(w13, requires_grad=False)
        )
        layer.register_parameter(
            "w2_weight", torch.nn.Parameter(w2, requires_grad=False)
        )
        layer.register_parameter(
            "w13_weight_scale_inv", torch.nn.Parameter(w13_sf, requires_grad=False)
        )
        layer.register_parameter(
            "w2_weight_scale_inv", torch.nn.Parameter(w2_sf, requires_grad=False)
        )

        # FP8 activations are quantized inside the kernel; no loaded scales.
        layer.a13_scale = None
        layer.a2_scale = None

        layer.is_mxfp4_converted = True

        self._build_mega_moe_weights(layer)

    def _build_mega_moe_weights(self, layer: torch.nn.Module) -> None:
        """Hand the weights to DeepGEMM's mega-MoE transform.

        Must run after every in-place weight rewrite and instead of (not before)
        any scale re-layout -- the transform consumes checkpoint-layout per-32
        fp32 scales and does the UE8M0 packing itself.
        """
        from sglang.srt.layers.moe.mega_moe import build_mega_moe_experts_weights
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        backend = get_moe_a2a_backend()
        if not backend.is_megamoe():
            raise ValueError(
                "MXFP4 MoE checkpoints are only supported through the mega-MoE "
                f"kernel, but --moe-a2a-backend is '{backend.value}'. Pass "
                "--moe-a2a-backend megamoe, or set "
                "SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1 to have it auto-configured."
            )

        build_mega_moe_experts_weights(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        return self.apply_weights(layer, dispatch_output)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        # Reaching here means the mega-MoE path declined this batch (e.g. the
        # token count exceeded its cap). There is no SM90 MXFP4 grouped GEMM to
        # fall back on, and the weights are already in mega layout, so failing
        # loudly beats returning wrong numbers.
        raise NotImplementedError(
            "MXFP4 MoE weights are prepared exclusively for the DeepGEMM "
            "mega-MoE kernel; there is no fallback grouped-GEMM path. This "
            "batch bypassed forward_mega_moe -- check "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK against the "
            "current batch size."
        )
