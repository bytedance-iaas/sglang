"""Unified CPU(AMX INT8) + GPU(SM90 INT8) MoE path for the asym_gemm backend.

Wraps ``asym_gemm.unified_moe.Layer``: at load time the BF16 w13/w2 masters
are quantized to INT8 once and kept in pinned host memory (row-major for the
GPU TMA reads over PCIe, blocked-VNNI for the CPU AMX kernels). At forward
time experts are dispatched per routed token count: small experts run on the
CPU AMX bucket, large ones on the SM90 INT8 grouped kernel.

This path is opt-in via SGLANG_ASYMGEMM_UNIFIED_MOE=1 (see
``asym_gemm_wrapper.configurer``) and bypasses the MoeRunner permute pipeline
entirely — the unified layer consumes raw (hidden_states, topk_ids,
topk_weights) and does gather/scatter/weighted-reduce internally. The
existing asym_gemm paths are untouched and serve as the fallback whenever a
layer cannot be converted.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import asym_gemm_wrapper

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)

_UNIFIED_LAYER_ATTR = "_unified_asym_gemm_layer"

# One CPU worker pool for the whole process; a Runtime spawns n_threads-1 OS
# threads, so per-MoE-layer pools would oversubscribe the host.
_shared_cpu_runtime = None


def unified_asym_gemm_enabled() -> bool:
    return asym_gemm_wrapper.ASYMGEMM_UNIFIED_MOE


def has_unified_asym_gemm_layer(layer: torch.nn.Module) -> bool:
    return getattr(layer, _UNIFIED_LAYER_ATTR, None) is not None


def _get_shared_cpu_runtime():
    global _shared_cpu_runtime
    if _shared_cpu_runtime is None:
        from asym_gemm.unified_moe import _C

        n_threads = envs.SGLANG_ASYMGEMM_UNIFIED_CPU_THREADS.get()
        _shared_cpu_runtime = _C.Runtime(n_threads)
        logger.info(
            "AsymGEMM unified MoE: shared CPU runtime with %d threads",
            _shared_cpu_runtime.threads,
        )
    return _shared_cpu_runtime


def _check_convertible(layer: torch.nn.Module) -> str | None:
    """Return None if `layer` can use the unified path, else the reason."""
    from asym_gemm.unified_moe.runtime import GRAN_K

    w13 = layer.w13_weight
    w2 = layer.w2_weight

    if not layer.moe_runner_config.is_gated:
        return "MoE is not gated (unified kernel computes SwiGLU)"
    if layer.moe_runner_config.activation != "silu":
        return f"activation {layer.moe_runner_config.activation!r} != 'silu'"
    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return f"weights are {w13.dtype}/{w2.dtype}, need bfloat16"
    if w13.is_cuda or w2.is_cuda:
        return "weights are not host-resident"
    if layer.moe_ep_size > 1:
        return "expert parallelism is not supported yet"
    if getattr(layer, "num_fused_shared_experts", 0):
        return "fused shared experts are not supported yet"

    num_experts, two_inter, hidden = w13.shape
    inter = two_inter // 2
    if w2.shape != (num_experts, hidden, inter):
        return f"unexpected w2 shape {tuple(w2.shape)} for w13 {tuple(w13.shape)}"
    if hidden % GRAN_K != 0 or inter % GRAN_K != 0:
        return (
            f"hidden ({hidden}) and intermediate ({inter}) must be "
            f"multiples of {GRAN_K}"
        )
    return None


def maybe_create_unified_asym_gemm_layer(layer: torch.nn.Module) -> None:
    """Build the unified INT8 MoE layer from the BF16 w13/w2 masters.

    Called from UnquantizedFusedMoEMethod.process_weights_after_loading when
    the asym_gemm backend is active and the unified path is enabled. On any
    unsupported configuration the reason is logged once and the layer keeps
    using the existing asym_gemm path. The BF16 masters are kept so the
    fallback path stays functional.
    """
    if not unified_asym_gemm_enabled():
        return

    reason = _check_convertible(layer)
    if reason is not None:
        logger.warning(
            "AsymGEMM unified MoE: layer %s falls back to the existing "
            "asym_gemm path (%s)",
            getattr(layer, "layer_id", "?"),
            reason,
        )
        return

    from asym_gemm.unified_moe import Layer as UnifiedMoeLayer

    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    inter = w13.shape[1] // 2

    # sglang's silu_and_mul convention: first half of w13 is gate, second up.
    unified_layer = UnifiedMoeLayer.from_bf16(
        gate=w13[:, :inter, :],
        up=w13[:, inter:, :],
        down=w2,
        top_k=layer.top_k,
        cuda_device=torch.cuda.current_device(),
        m_cpu=envs.SGLANG_ASYMGEMM_UNIFIED_M_CPU.get(),
        runtime=_get_shared_cpu_runtime(),
    )
    setattr(layer, _UNIFIED_LAYER_ATTR, unified_layer)
    logger.info(
        "AsymGEMM unified MoE: converted layer %s "
        "(experts=%d, hidden=%d, inter=%d, m_cpu=%d)",
        getattr(layer, "layer_id", "?"),
        w13.shape[0],
        w13.shape[2],
        inter,
        unified_layer.m_cpu,
    )
