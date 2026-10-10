"""Opt-in Q-LoRA normalization/quantization with explicit GEMM contracts."""

from functools import partial
from typing import NamedTuple

import torch


class QLoRAQuantSpec(NamedTuple):
    group_size: int
    column_major_scales: bool = False
    scale_ue8m0: bool = False


def q_lora_quant_spec(linear, x: torch.Tensor) -> QLoRAQuantSpec | None:
    """Only select runners which accept these prequantized layouts unchanged.

    Use the resolved callable, not --fp8-gemm-backend: auto selection and small
    weight blocks can select a different runner from the requested backend.
    """
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import (
        deepgemm_w8a8_block_fp8_linear_with_fallback,
        triton_w8a8_block_fp8_linear,
    )
    from sglang.srt.layers.quantization.humming_fp8 import (
        can_use_humming_fp8_linear,
        humming_w8a8_block_fp8_linear,
    )

    method = getattr(linear, "quant_method", None)
    if not isinstance(method, Fp8LinearMethod) or (
        method.use_marlin
        or method.use_mxfp8
        or not method.block_quant
        or (
            method.block_fp8_as_mxfp8
            and getattr(linear, "block_fp8_mxfp8_ready", False)
        )
    ):
        return None
    weight = linear.weight
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        return None
    n, k = weight.shape
    if x.shape[-1] != k:
        return None
    runner = method.w8a8_block_fp8_linear
    keywords = runner.keywords if isinstance(runner, partial) else {}
    runner = runner.func if isinstance(runner, partial) else runner
    block = list(method.weight_block_size)
    if runner is deepgemm_w8a8_block_fp8_linear_with_fallback:
        if (
            block == [128, 128]
            and n % 64 == 0
            and k % 128 == 0
            and not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        ):
            return QLoRAQuantSpec(128, column_major_scales=True)
        return None
    if method.use_humming:
        # Do not turn a raw-weight fallback into a packed-weight call.
        if (
            runner is not humming_w8a8_block_fp8_linear
            or not can_use_humming_fp8_linear(linear, x)
        ):
            return None
    elif runner is humming_w8a8_block_fp8_linear:
        return None
    if runner not in (triton_w8a8_block_fp8_linear, humming_w8a8_block_fp8_linear):
        return None
    if block != [32, 32] or not keywords.get("act_scale_ue8m0", False):
        return None
    # This BF16 GEMM shortcut only accepts unquantized inputs. Passing a tuple
    # would silently switch GEMM implementation at M >= 64.
    if (
        runner is triton_w8a8_block_fp8_linear
        and x.shape[0] >= 64
        and getattr(linear, "_block_fp8_bf16_weight", None) is not None
    ):
        return None
    return QLoRAQuantSpec(32, scale_ue8m0=True)


def try_normalize_q_lora_sm90(q, norm, linear):
    """Return (BF16, (FP8, scale)), or None to retain the existing path."""
    from sglang.srt.environ import envs
    from sglang.srt.utils import get_platform

    if not envs.SGLANG_OPT_DSV41_SM90_Q_LORA_QUANT.get():
        return None
    if not (q.is_cuda and torch.version.cuda and get_platform().is_sm90):
        return None
    if not (
        q.ndim == 2
        and 0 < q.shape[0] <= 384
        and q.shape[1] == 1280
        and q.stride(1) == 1
        and q.dtype == norm.weight.dtype == torch.bfloat16
        and norm.weight.is_contiguous()
        and norm.weight.shape == (q.shape[1],)
        and norm.variance_size_override is None
        and not norm.cast_x_before_out_mul
    ):
        return None
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.runtime_context import get_exec

    if (
        is_batch_invariant_mode_enabled()
        or get_exec().deterministic.enable_deterministic_inference
    ):
        return None
    spec = q_lora_quant_spec(linear, q)
    if spec is None:
        return None
    from sglang.kernels.ops.layernorm.rmsnorm_group_fp8 import rmsnorm_group_fp8

    y, quant, scale = rmsnorm_group_fp8(
        q, norm.weight, norm.variance_epsilon, **spec._asdict()
    )
    return y, (quant, scale)
