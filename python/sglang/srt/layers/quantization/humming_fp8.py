# SPDX-License-Identifier: Apache-2.0
"""Opt-in SM90 W8A8 GEMM for 32x32 UE8M0 weights.

Packed calls need only Humming tensors and logical shape metadata, not the
checkpoint layout. Owners still retain their original parameters for loading
and direct readers; releasing those parameters is deliberately out of scope.
"""

from typing import Any, NamedTuple

import torch
import torch.nn.functional as F

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.runtime_context import get_exec


class HummingFp8Weight(NamedTuple):
    config: Any
    weight: torch.Tensor
    weight_scale: torch.Tensor
    locks: torch.Tensor
    shape: tuple[int, int]


def _requires_deterministic_gemm() -> bool:
    return (
        is_batch_invariant_mode_enabled()
        or get_exec().deterministic.enable_deterministic_inference
    )


@torch.no_grad()
def pack_humming_fp8_weight(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> HummingFp8Weight:
    """Pack once at weight load (or stacked-weight cache construction)."""
    n, k = weight.shape
    if weight.dtype != torch.float8_e4m3fn or n % 32 or k % 32 or min(n, k) == 0:
        raise ValueError("Humming expects nonempty FP8 E4M3 weights aligned to 32x32")
    if tuple(weight_scale.shape) != (n // 32, k // 32):
        raise ValueError("Humming expects a plain 32x32 block-scale grid")
    try:
        from humming.layer import HummingLayer
    except ImportError as exc:
        raise ImportError(
            "--fp8-gemm-backend=humming requires humming-kernels; "
            "install the version pinned in python/pyproject.toml"
        ) from exc

    padded_k = (k + 127) // 128 * 128
    scale = weight_scale.float()
    if padded_k != k:
        # Group-32 short-K needs physical W and A padding. Padding scales
        # alone gives incorrect results (notably TP8 shared-down, K=288).
        padded_weight = weight.new_zeros((n, padded_k))
        padded_scale = scale.new_ones((n // 32, padded_k // 32))
        padded_weight[:, :k] = weight
        padded_scale[:, : k // 32] = scale
        weight, scale = padded_weight, padded_scale

    with torch.device(weight.device):
        packed = HummingLayer(
            shape_n=n,
            shape_k=padded_k,
            weight_config={"quant_method": "fp8", "weight_block_size": [32, 32]},
            input_config={
                "dtype": "float8e4m3",
                "group_size": 32,
                "scale_dtype": "float32",
            },
            pad_n_to_multiple=256,
            pad_k_to_multiple=128,
            torch_dtype=torch.bfloat16,
        )
    packed.load_from_tensors({"weight": weight, "weight_scale_inv": scale})
    packed.transform()
    return HummingFp8Weight(
        packed.humming_config,
        packed.weight.detach(),
        packed.weight_scale.detach(),
        packed.locks.detach(),
        (n, k),
    )


@torch.no_grad()
def prepare_humming_fp8_linear(layer: torch.nn.Module) -> None:
    layer.humming_fp8_ready = False
    if (
        getattr(layer, "keep_plain_weight_layout", False)
        or layer.orig_dtype != torch.bfloat16
        or layer.weight.dtype != torch.float8_e4m3fn
    ):
        return
    n, k = layer.weight.shape
    if n % 32 or k % 32 or min(n, k) == 0:
        return
    packed = pack_humming_fp8_weight(layer.weight, layer.weight_scale_inv)
    if getattr(layer, "_humming_fp8_shape", (n, k)) != (n, k):
        raise ValueError("Humming FP8 weight reload changed the logical shape")
    for name in ("weight", "weight_scale", "locks"):
        tensor = getattr(packed, name)
        cache_name = "_humming_fp8_" + name
        existing = getattr(layer, cache_name, None)
        if existing is not None:
            if (
                existing.shape != tensor.shape
                or existing.dtype != tensor.dtype
                or existing.device != tensor.device
            ):
                raise ValueError("Humming FP8 weight reload changed the cached layout")
            # Preserve CUDA-graph addresses when reloading checkpoint weights.
            existing.copy_(tensor)
        else:
            layer.register_buffer(cache_name, tensor, persistent=False)
    layer._humming_fp8_config = packed.config
    layer._humming_fp8_shape = packed.shape
    layer.humming_fp8_ready = True


def get_humming_fp8_weight(layer: torch.nn.Module) -> HummingFp8Weight:
    return HummingFp8Weight(
        layer._humming_fp8_config,
        layer._humming_fp8_weight,
        layer._humming_fp8_weight_scale,
        layer._humming_fp8_locks,
        layer._humming_fp8_shape,
    )


def can_use_humming_fp8_linear(layer: torch.nn.Module, x) -> bool:
    if not getattr(layer, "humming_fp8_ready", False):
        return False
    input_scale = None
    if isinstance(x, tuple):
        if len(x) != 2 or not isinstance(x[1], torch.Tensor):
            return False
        x, input_scale = x
        dtype = torch.float8_e4m3fn
    else:
        dtype = torch.bfloat16
    if not isinstance(x, torch.Tensor) or x.dtype != dtype:
        return False
    if input_scale is not None and (
        input_scale.dtype != torch.float32
        or input_scale.ndim < 2
        or input_scale.shape[-1] != x.shape[-1] // 32
        or input_scale.numel() != x.numel() // 32
    ):
        return False
    # CP decode-attention-TP can temporarily slice the original parameters.
    # Its unsharded packed cache must not be used for a different local shape.
    raw = getattr(layer, "weight", None)
    if raw is not None and tuple(raw.shape) != layer._humming_fp8_shape:
        return False
    return x.shape[-1] == layer._humming_fp8_shape[1]


def humming_w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor | HummingFp8Weight,
    block_size: list[int],
    weight_scale: torch.Tensor | None,
    input_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    act_scale_ue8m0: bool = False,
) -> torch.Tensor:
    """Block-FP8 runner with an explicit packed-weight contract.

    Raw tensors retain Triton compatibility for formats/dtypes not packed by the
    owner. A packed call is entirely Humming, with no M limit or raw-weight access.
    """
    if not isinstance(weight, HummingFp8Weight):
        from sglang.srt.layers.quantization.fp8_utils import (
            triton_w8a8_block_fp8_linear,
        )

        return triton_w8a8_block_fp8_linear(
            input, weight, block_size, weight_scale, input_scale, bias, act_scale_ue8m0
        )
    if list(block_size) != [32, 32] or not act_scale_ue8m0:
        raise ValueError("Packed Humming requires 32x32 UE8M0 block FP8")
    n, k = weight.shape
    expected_dtype = torch.bfloat16 if input_scale is None else torch.float8_e4m3fn
    if input.dtype != expected_dtype or input.shape[-1] != k:
        raise ValueError("Packed Humming input dtype or K does not match its layout")
    output_shape = (*input.shape[:-1], n)
    if input.numel() == 0:
        return input.new_empty(output_shape, dtype=torch.bfloat16)

    from humming.forward import humming_forward

    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    input_2d = input.reshape(-1, k).contiguous()
    padded_k = (k + 127) // 128 * 128
    if input_scale is None:
        if padded_k != k:
            input_2d = F.pad(input_2d, (0, padded_k - k))
        # Preserve SGLang group-32 power-of-two scales and FP8 rounding.
        q_input, input_scale = sglang_per_token_group_quant_fp8(
            input_2d, 32, scale_ue8m0=True
        )
    else:
        if (
            input_scale.dtype != torch.float32
            or input_scale.ndim < 2
            or input_scale.shape[-1] != k // 32
            or input_scale.numel() != input.numel() // 32
        ):
            raise ValueError(
                "Packed Humming expects FP32 per-row group-32 input scales"
            )
        q_input = input_2d
        input_scale = input_scale.reshape(-1, k // 32).contiguous()
        if padded_k != k:
            q_padded = q_input.new_zeros((q_input.shape[0], padded_k))
            s_padded = input_scale.new_ones((q_input.shape[0], padded_k // 32))
            q_padded[:, :k] = q_input
            s_padded[:, : k // 32] = input_scale
            q_input, input_scale = q_padded, s_padded
    output = humming_forward(
        weight.config,
        inputs=q_input,
        input_scale=input_scale,
        weight=weight.weight,
        weight_scale=weight.weight_scale,
        locks=weight.locks,
        compute_config={"use_batch_invariant": _requires_deterministic_gemm()},
    )
    if bias is not None:
        output = output + bias
    return output.reshape(output_shape)
