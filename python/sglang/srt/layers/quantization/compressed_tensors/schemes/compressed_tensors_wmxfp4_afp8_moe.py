from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import torch

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsWMXFP4AFP8MoE"]

# DeepGEMM SM90 fp8xfp4 kernel indexes weight scales with 32-element groups
# along K, regardless of the checkpoint's stored group_size.
_DEEPGEMM_GRAN_K = 32

# DeepGEMM's kPackedFP4 corresponds to a signed int8 view of the packed
# 2-fp4-per-byte buffer on CUDA. Matches the non-AIter path in fp8.py.
_FP4_WEIGHT_VIEW_DTYPE = torch.int8


class CompressedTensorsWMXFP4AFP8MoE(CompressedTensorsMoEScheme):
    """MoE scheme for compressed-tensors W4A8 FP4 checkpoints.

    Weight  : 4-bit float (E2M1), packed 2-per-byte along the K dimension,
              per-group scales with group_size divisible by 32 and stored
              in UE8M0 encoding (uint8) on disk.
    Act     : 8-bit float (E4M3), dynamic per-token quant.
    Backend : DeepGEMM (SM90 fp8xfp4 masked + contiguous paths).
    """

    def __init__(self, weight_quant, input_quant):
        self.weight_quant = weight_quant
        self.input_quant = input_quant

        group_size = getattr(weight_quant, "group_size", None)
        if group_size is None or group_size <= 0:
            raise ValueError(
                "CompressedTensorsWMXFP4AFP8MoE requires a positive weight "
                f"group_size, got {group_size!r}"
            )
        if group_size % _DEEPGEMM_GRAN_K != 0:
            raise ValueError(
                f"CompressedTensorsWMXFP4AFP8MoE requires group_size divisible "
                f"by {_DEEPGEMM_GRAN_K}, got {group_size}"
            )
        self.group_size = int(group_size)
        self.expand_ratio = self.group_size // _DEEPGEMM_GRAN_K

        scale_dtype_str = str(getattr(weight_quant, "scale_dtype", "") or "").lower()
        # On-disk scale is UE8M0 packed into uint8 for this checkpoint family.
        self._scale_is_ue8m0_uint8 = "uint8" in scale_dtype_str

    @classmethod
    def get_min_capability(cls) -> int:
        # SM90 (Hopper) fp8xfp4 DeepGEMM kernel.
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

        if hidden_size % self.group_size != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by "
                f"group_size={self.group_size}"
            )
        if intermediate_size_per_partition % self.group_size != 0:
            raise ValueError(
                f"intermediate_size_per_partition={intermediate_size_per_partition} "
                f"must be divisible by group_size={self.group_size}"
            )

        layer.params_dtype = params_dtype

        # Packed FP4 weights: 2 fp4 items per uint8 along K.
        w13_weight_packed = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight_packed)
        set_weight_attrs(w13_weight_packed, extra_weight_attrs)

        w2_weight_packed = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight_packed)
        set_weight_attrs(w2_weight_packed, extra_weight_attrs)

        # Per-group scales (UE8M0 packed into uint8 on disk).
        scale_dtype = torch.uint8 if self._scale_is_ue8m0_uint8 else torch.float32
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Activation is dynamic per-token FP8, no persistent input scale.
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def _prepare_scale_for_deepgemm(
        self,
        scale: torch.Tensor,
        weight_packed: torch.Tensor,
    ):
        """Expand a g=group_size, UE8M0-uint8 scale to the layout expected
        by DeepGEMM (K indexed with gran_k=32, TMA-aligned MN dim).

        Mirrors fp8.py's FP4 branch: repeat_interleave -> transform_sf_into_required_layout
        -> empty_strided(uint8) + UE8M0 re-encode.

        Returns (scale_data_float, e8m0_scale_uint8). scale_data_float is the
        kernel-layout float tensor (attached as .data on the scale Parameter);
        e8m0_scale_uint8 is the TMA-strided uint8 UE8M0 view kernel expects.
        """
        if not self._scale_is_ue8m0_uint8:
            raise NotImplementedError(
                "CompressedTensorsWMXFP4AFP8MoE currently only handles UE8M0 "
                "(uint8) on-disk scales."
            )

        from deep_gemm import transform_sf_into_required_layout

        num_experts, n, k_groups_loaded = scale.shape
        k = weight_packed.shape[2] * 2  # 2 fp4 items per packed byte
        expected_loaded_k_groups = k // self.group_size
        if k_groups_loaded != expected_loaded_k_groups:
            raise ValueError(
                f"Loaded FP4 scale shape mismatch: got last dim={k_groups_loaded}, "
                f"expected {expected_loaded_k_groups} for k={k}, "
                f"group_size={self.group_size}"
            )

        # Decode UE8M0 uint8 exponent to float: scale_float = 2^(uint8 - 127).
        scale_float = torch.pow(
            torch.tensor(2.0, device=scale.device, dtype=torch.float32),
            scale.to(torch.float32) - 127.0,
        )

        runtime_scale = scale_float.repeat_interleave(
            self.expand_ratio, dim=2
        ).contiguous()
        expected_runtime_k_groups = k // _DEEPGEMM_GRAN_K
        if runtime_scale.shape[2] != expected_runtime_k_groups:
            raise ValueError(
                f"Expanded runtime FP4 scale shape mismatch: got last dim="
                f"{runtime_scale.shape[2]}, expected {expected_runtime_k_groups} "
                f"for k={k}, deepgemm_gran_k={_DEEPGEMM_GRAN_K}"
            )

        scale_data = transform_sf_into_required_layout(
            runtime_scale,
            mn=n,
            k=k,
            recipe=(1, _DEEPGEMM_GRAN_K),
            num_groups=num_experts,
            disable_ue8m0_cast=False,
        )

        tma_aligned_n_e8m0 = ((n + 15) // 16) * 16
        e8m0_scale_data = torch.empty_strided(
            scale_data.shape,
            (
                tma_aligned_n_e8m0 * scale_data.shape[2],
                1,
                tma_aligned_n_e8m0,
            ),
            device=scale_data.device,
            dtype=torch.uint8,
        )
        e8m0_scale_data.copy_(
            (torch.floor(torch.log2(scale_data.float())) + 127).to(torch.uint8)
        )
        return scale_data, e8m0_scale_data

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Re-view packed uint8 weights as int8 to satisfy DeepGEMM's kPackedFP4
        # scalar_type check on the SM90 fp8xfp4 kernel.
        w13_data = layer.w13_weight_packed.data.view(_FP4_WEIGHT_VIEW_DTYPE)
        w2_data = layer.w2_weight_packed.data.view(_FP4_WEIGHT_VIEW_DTYPE)
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)
        delattr(layer, "w13_weight_packed")
        delattr(layer, "w2_weight_packed")

        w13_scale_data, w13_scale_e8m0 = self._prepare_scale_for_deepgemm(
            layer.w13_weight_scale.data, layer.w13_weight.data
        )
        w2_scale_data, w2_scale_e8m0 = self._prepare_scale_for_deepgemm(
            layer.w2_weight_scale.data, layer.w2_weight.data
        )

        # DeepGemmMoeQuantInfo consumes the float "packed" scale on w*_scale
        # (kernel-required layout via transform_sf_into_required_layout).
        # The extra UE8M0 uint8 view on w*_scale_e8m0 is only meaningful when
        # DeepGEMM was built with UE8M0 scale-B support; otherwise leave it
        # None and let the kernel take the float sfb path (matches fp8.py's
        # DSV4 FP4 branch behavior).
        layer.w13_weight_scale = torch.nn.Parameter(
            w13_scale_data, requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            w2_scale_data, requires_grad=False
        )
        if deep_gemm_wrapper.DEEPGEMM_FP4_SCALE_B_UE8M0:
            layer.w13_weight_scale.scale_e8m0_data = w13_scale_e8m0
            layer.w2_weight_scale.scale_e8m0_data = w2_scale_e8m0
            layer.w13_weight_scale.format_ue8m0 = True
            layer.w2_weight_scale.format_ue8m0 = True
        else:
            layer.w13_weight_scale.scale_e8m0_data = None
            layer.w2_weight_scale.scale_e8m0_data = None
            layer.w13_weight_scale.format_ue8m0 = False
            layer.w2_weight_scale.format_ue8m0 = False

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.DEEP_GEMM, moe_runner_config)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        if not self.runner.runner_backend.is_deep_gemm():
            raise NotImplementedError(
                f"CompressedTensorsWMXFP4AFP8MoE only supports the DeepGEMM MoE "
                f"runner backend, got {self.runner.runner_backend}."
            )

        # DeepGEMM fp8xfp4 kernels use recipe_a=(1,128)/recipe_b=(1,32) for
        # contiguous, and gran_k_a=128/gran_k_b=32 for masked. The runner
        # picks the recipe from is_fp4_experts=True.
        #
        # NOTE: We intentionally *do not* pass w*_scale_e8m0 here. The SM90
        # fp8xfp4 kernel only accepts uint8 UE8M0 sfb when its
        # scale_b_direct_load gate is on (expected_m <= 16 OR
        # bm32_skew_fast_path). Under CUDA-graph capture with cuda_graph_max_bs
        # up to 64, the gate is not universally satisfied and the kernel
        # then requires float sfb. Passing float scale keeps the path stable
        # across all captured batch sizes; the uint8 view is kept only as an
        # attribute for future opt-in use.
        block_shape: List[int] = [self.group_size, self.group_size]
        quant_info = DeepGemmMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_fp8=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_scale_e8m0=None,
            w2_scale_e8m0=None,
            block_shape=block_shape,
            is_fp4_experts=True,
        )
        return self.runner.run(dispatch_output, quant_info)
