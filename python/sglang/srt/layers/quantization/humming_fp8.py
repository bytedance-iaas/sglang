# SPDX-License-Identifier: Apache-2.0
"""Opt-in SM90 dense W8A8 path for DeepSeek-V4.1's 32x32 UE8M0 weights.

Keep checkpoint tensors intact: wo_a and the stacked DSpark KV projection
read them directly, and unsupported inputs still use the Triton fallback.
Only a derived, non-persistent Humming layout is cached at weight-load time.
"""

import torch

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.runtime_context import get_exec


def _requires_deterministic_gemm() -> bool:
    # Humming's default Stream-K reduction is not bitwise deterministic.
    return (
        is_batch_invariant_mode_enabled()
        or get_exec().deterministic.enable_deterministic_inference
    )


@torch.no_grad()
def prepare_humming_fp8_linear(layer: torch.nn.Module) -> None:
    layer.humming_fp8_ready = False
    if (
        getattr(layer, "keep_plain_weight_layout", False)
        or layer.orig_dtype != torch.bfloat16
        or layer.weight.dtype != torch.float8_e4m3fn
        or _requires_deterministic_gemm()
    ):
        return

    n, k = layer.weight.shape
    if n % 32 or k < 1024 or k % 128:
        # In particular, TP8 shared-down has K=288. Padding both activation
        # and weight is necessary for group-32 Humming and can cost more than
        # the small GEMM; padding scales alone silently produces wrong values.
        return

    try:
        from humming.layer import HummingLayer
    except ImportError as exc:
        raise ImportError(
            "--fp8-gemm-backend=humming requires humming-kernels; "
            "install the version pinned in python/pyproject.toml"
        ) from exc

    with torch.device(layer.weight.device):
        packed = HummingLayer(
            shape_n=n,
            shape_k=k,
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
    packed.load_from_tensors(
        {"weight": layer.weight, "weight_scale_inv": layer.weight_scale_inv.float()}
    )
    packed.transform()

    for name in ("weight", "weight_scale", "locks"):
        tensor = getattr(packed, name).detach()
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
    layer._humming_fp8_config = packed.humming_config
    layer.humming_fp8_ready = True


def can_use_humming_fp8_linear(layer: torch.nn.Module, x) -> bool:
    if not getattr(layer, "humming_fp8_ready", False):
        return False
    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
        # Prequantized tuples retain their existing scale-layout contract.
        return False
    if x.shape[-1] != layer.weight.shape[1]:
        return False
    m = x.numel() // x.shape[-1]
    # Limit the first integration to the measured decode/verify envelope.
    return 0 < m <= 64 and not _requires_deterministic_gemm()


def humming_fp8_linear(
    layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    from humming.forward import humming_forward

    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    # Do not use Humming's default per-token quantizer: preserve the existing
    # group-32 power-of-two scales and FP8 activation rounding.
    q_input, input_scale = sglang_per_token_group_quant_fp8(x_2d, 32, scale_ue8m0=True)
    output = humming_forward(
        layer._humming_fp8_config,
        inputs=q_input,
        input_scale=input_scale,
        weight=layer._humming_fp8_weight,
        weight_scale=layer._humming_fp8_weight_scale,
        locks=layer._humming_fp8_locks,
    )
    if bias is not None:
        output = output + bias
    return output.reshape(*x.shape[:-1], layer.weight.shape[0])
