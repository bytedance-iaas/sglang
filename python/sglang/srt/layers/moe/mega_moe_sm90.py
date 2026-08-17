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
"""SM90 FP8/FP4 Mega-MoE forward paths and expert-weight preparation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_common.utils import _device_sm

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from deep_gemm import SymmBuffer

    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


def is_sm90_fp8_mega_moe_available(experts) -> bool:
    if _device_sm != 90:
        return False
    try:
        import deep_gemm
    except ImportError:
        return False
    return (
        hasattr(deep_gemm, "fp8_mega_moe")
        and hasattr(deep_gemm, "mega_moe_pre_dispatch_sm90")
        and getattr(experts, "_mega_moe_sm90_fp8_weights", False)
    )


def is_sm90_fp4_mega_moe_available(experts) -> bool:
    if _device_sm != 90:
        return False
    try:
        import deep_gemm
    except ImportError:
        return False
    native_module = getattr(deep_gemm, "_C", None)
    return (
        hasattr(deep_gemm, "fp8_fp4_mega_moe")
        and hasattr(deep_gemm, "mega_moe_pre_dispatch_sm90")
        and native_module is not None
        and hasattr(native_module, "fp8_fp4_mega_moe_sm90")
        and getattr(experts, "_mega_moe_sm90_fp4_weights", False)
    )


def run_sm90_mega_routed(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    buf: SymmBuffer,
    num_tokens: int,
) -> torch.Tensor:
    import deep_gemm

    use_fp4_weights = getattr(moe.experts, "_mega_moe_sm90_fp4_weights", False)
    if use_fp4_weights and envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS.get():
        raise RuntimeError(
            "SM90 FP4 MegaMoE uses FP8 activations; "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS is unsupported"
        )

    if moe.experts.should_fuse_routed_scaling_factor_in_topk:
        routed_scaling_factor = 1.0
    else:
        routed_scaling_factor = float(moe.routed_scaling_factor)

    deep_gemm.mega_moe_pre_dispatch_sm90(
        hidden_states,
        topk_ids,
        topk_weights,
        buf.x,
        buf.x_sf,
        buf.topk_idx,
        buf.topk_weights,
        num_tokens=num_tokens,
        group_size=128,
        routed_scaling_factor=routed_scaling_factor,
    )

    y = torch.empty(
        (max(num_tokens, 1), moe.config.hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    if use_fp4_weights:
        deep_gemm.fp8_fp4_mega_moe(
            y,
            moe.experts.mega_l1_weights,
            moe.experts.mega_l2_weights,
            buf,
            recipe=(1, 1, 32),
            activation="swiglu",
            activation_clamp=getattr(moe.config, "swiglu_limit", None),
            fast_math=True,
        )
    else:
        deep_gemm.fp8_mega_moe(
            y,
            moe.experts.mega_l1_weights,
            moe.experts.mega_l2_weights,
            buf,
            recipe=(128, 128, 128),
            activation="swiglu",
            activation_clamp=getattr(moe.config, "swiglu_limit", None),
            fast_math=True,
        )
    y = y[:num_tokens]

    return y


def _interleave_l1_weight_only(weight: torch.Tensor, gran: int = 8) -> torch.Tensor:
    num_groups, n, *rest = weight.shape
    half = n // 2
    gate = weight[:, :half].reshape(num_groups, half // gran, gran, *rest)
    up = weight[:, half:].reshape(num_groups, half // gran, gran, *rest)
    return torch.stack([gate, up], dim=2).reshape(num_groups, n, *rest)


def _transform_weights_for_mega_moe_sm90_fp4_compat(
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    l2_weights: tuple[torch.Tensor, torch.Tensor],
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Port the pure-Python layout helper absent from some DeepGEMM wheels."""

    def interleave(tensor: torch.Tensor, granularity: int = 8) -> torch.Tensor:
        num_groups, n, *rest = tensor.shape
        assert (
            n % (2 * granularity) == 0
        ), f"gate/up dimension {n} must be divisible by {2 * granularity}"
        half = n // 2
        gate = tensor[:, :half].reshape(
            num_groups, half // granularity, granularity, *rest
        )
        up = tensor[:, half:].reshape(
            num_groups, half // granularity, granularity, *rest
        )
        return torch.stack([gate, up], dim=2).reshape(num_groups, n, *rest)

    def pack_scale(scale: torch.Tensor) -> torch.Tensor:
        assert scale.dtype == torch.float32, f"unexpected scale dtype {scale.dtype}"
        num_experts, n, k_groups = scale.shape
        assert k_groups % 4 == 0, f"K/32={k_groups} must be a multiple of 4"
        exponent = (scale.view(torch.int32) >> 23).bitwise_and(0xFF).to(torch.uint8)
        return (
            exponent.contiguous()
            .view(num_experts, n, k_groups // 4, 4)
            .view(torch.int32)
            .reshape(num_experts, n, k_groups // 4)
            .contiguous()
        )

    def packed_fp4(weight: torch.Tensor) -> torch.Tensor:
        assert weight.dtype in (
            torch.int8,
            torch.uint8,
        ), f"unexpected FP4 storage dtype {weight.dtype}"
        return weight.contiguous().view(torch.int8)

    l1_weight, l1_scale = l1_weights
    l2_weight, l2_scale = l2_weights
    return (
        (interleave(packed_fp4(l1_weight)), pack_scale(interleave(l1_scale))),
        (packed_fp4(l2_weight), pack_scale(l2_scale)),
    )


def _resolve_sm90_fp4_weight_transform(deep_gemm):
    missing_kernels = [
        symbol
        for symbol in ("fp8_fp4_mega_moe", "mega_moe_pre_dispatch_sm90")
        if not hasattr(deep_gemm, symbol)
    ]
    if missing_kernels:
        raise RuntimeError(
            "SM90 FP4 MegaMoE runtime is incomplete; missing "
            + ", ".join(missing_kernels)
        )
    native_module = getattr(deep_gemm, "_C", None)
    if native_module is None or not hasattr(native_module, "fp8_fp4_mega_moe_sm90"):
        raise RuntimeError(
            "SM90 FP4 MegaMoE runtime is incomplete; missing native "
            "fp8_fp4_mega_moe_sm90"
        )
    native = getattr(deep_gemm, "transform_weights_for_mega_moe_sm90_fp4", None)
    if native is not None:
        return native
    logger.warning_once(
        "DeepGEMM lacks transform_weights_for_mega_moe_sm90_fp4; "
        "using the SGLang compatibility transform"
    )
    return _transform_weights_for_mega_moe_sm90_fp4_compat


def build_sm90_mega_moe_experts_weights(experts) -> None:
    if getattr(experts, "_mega_moe_weights_built", False):
        return

    w13 = experts.w13_weight.data
    w13_sf_fp32 = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    w2_sf_fp32 = experts.w2_weight_scale_inv.data

    assert w13.dtype == torch.float8_e4m3fn
    assert w2.dtype == torch.float8_e4m3fn

    num_groups, n1, k1 = w13.shape
    _, n2, k2 = w2.shape
    scale_group_mn, scale_group_k = 128, 128

    assert k1 % scale_group_k == 0 and k2 % scale_group_k == 0, (
        f"invalid SM90 mega-moe K/group_size: k1={k1}, k2={k2}, "
        f"group_k={scale_group_k}"
    )
    expected_n_groups_1 = (n1 + scale_group_mn - 1) // scale_group_mn
    expected_n_groups_2 = (n2 + scale_group_mn - 1) // scale_group_mn
    expected_k_groups_1 = k1 // scale_group_k
    expected_k_groups_2 = k2 // scale_group_k
    assert w13_sf_fp32.shape[1] == expected_n_groups_1, (
        f"w13 scale N groups mismatch: got {w13_sf_fp32.shape[1]}, "
        f"expected {expected_n_groups_1} (n1={n1}, group_mn={scale_group_mn})"
    )
    assert w2_sf_fp32.shape[1] == expected_n_groups_2, (
        f"w2 scale N groups mismatch: got {w2_sf_fp32.shape[1]}, "
        f"expected {expected_n_groups_2} (n2={n2}, group_mn={scale_group_mn})"
    )
    assert w13_sf_fp32.shape[2] == expected_k_groups_1, (
        f"w13 scale K groups mismatch: got {w13_sf_fp32.shape[2]}, "
        f"expected {expected_k_groups_1} (k1={k1}, group_k={scale_group_k})"
    )
    assert w2_sf_fp32.shape[2] == expected_k_groups_2, (
        f"w2 scale K groups mismatch: got {w2_sf_fp32.shape[2]}, "
        f"expected {expected_k_groups_2} (k2={k2}, group_k={scale_group_k})"
    )

    if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
        w13_interleaved = _interleave_l1_weight_only(w13)
        experts.w13_weight.data = w13_interleaved
        experts.mega_l1_weights = (
            experts.w13_weight.data,
            experts.w13_weight_scale_inv.data,
        )
        experts.mega_l2_weights = (
            experts.w2_weight.data,
            experts.w2_weight_scale_inv.data,
        )
    else:
        import deep_gemm

        w13_sf = deep_gemm.transform_sf_into_required_layout(
            w13_sf_fp32,
            mn=n1,
            k=k1,
            recipe=(128, 128),
            num_groups=num_groups,
            disable_ue8m0_cast=True,
        )
        w2_sf = deep_gemm.transform_sf_into_required_layout(
            w2_sf_fp32,
            mn=n2,
            k=k2,
            recipe=(128, 128),
            num_groups=num_groups,
            disable_ue8m0_cast=True,
        )
        l1_pair, l2_pair = deep_gemm.transform_weights_for_mega_moe_sm90(
            (w13, w13_sf), (w2, w2_sf)
        )
        experts.mega_l1_weights = l1_pair
        experts.mega_l2_weights = l2_pair

    experts._mega_moe_sm90_fp8_weights = True
    experts._mega_moe_weights_built = True


def build_sm90_fp4_mega_moe_experts_weights(experts) -> None:
    """Transform packed E2M1 weights and raw FP32 E8M0 scales for H20."""
    if getattr(experts, "_mega_moe_weights_built", False):
        return

    import deep_gemm

    transform = _resolve_sm90_fp4_weight_transform(deep_gemm)
    l1_pair, l2_pair = transform(
        (experts.w13_weight.data, experts.w13_weight_scale.data),
        (experts.w2_weight.data, experts.w2_weight_scale.data),
    )

    if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
        experts.w13_weight.data = l1_pair[0]
        experts.w13_weight_scale.data = l1_pair[1]
        experts.w2_weight.data = l2_pair[0]
        experts.w2_weight_scale.data = l2_pair[1]
        experts.w13_weight_scale.format_ue8m0 = True
        experts.w2_weight_scale.format_ue8m0 = True
        experts.mega_l1_weights = (
            experts.w13_weight.data,
            experts.w13_weight_scale.data,
        )
        experts.mega_l2_weights = (
            experts.w2_weight.data,
            experts.w2_weight_scale.data,
        )
    else:
        experts.mega_l1_weights = l1_pair
        experts.mega_l2_weights = l2_pair

    experts._mega_moe_sm90_fp4_weights = True
    experts._mega_moe_weights_built = True
