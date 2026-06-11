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

    # The default loader's device_loading_context temporarily moves the
    # pinned-CPU masters onto the GPU while process_weights_after_loading
    # runs (and restores them afterwards). The unified layer quantizes from
    # host memory and keeps its own pinned INT8 copies, so take a transient
    # host copy here when needed.
    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    if w13.is_cuda:
        w13 = w13.to("cpu")
    if w2.is_cuda:
        w2 = w2.to("cpu")
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
    _maybe_warm_bf16_fallback_kernels(w13.shape[0], w13.shape[2], inter)


_warmed_fallback_shapes = set()


def _maybe_warm_bf16_fallback_kernels(
    num_experts: int, hidden: int, inter: int
) -> None:
    """Pre-warm the BF16 asym kernels the capture-time fallback will run.

    Piecewise CUDA graph capture replays the existing BF16 asym path (the
    unified kernel cannot be captured). With the unified path serving all
    eager forwards, the BF16 kernels would otherwise first run *inside*
    capture, where the asym JIT ensure-compiled step's stream synchronize is
    illegal. Run that step here, outside capture, for this layer's shapes.
    """
    from sglang.srt.server_args import get_global_server_args

    try:
        if get_global_server_args().disable_piecewise_cuda_graph:
            return
    except ValueError:
        pass  # no global server args (unit tests) — warming is harmless

    from sglang.srt.layers.asym_gemm_wrapper import compile_utils

    # (n, k) of the two grouped GEMMs in AsymGemmBf16RunnerCore._run_masked_gemm
    for n, k in ((2 * inter, hidden), (hidden, inter)):
        key = (n, k, num_experts)
        if key in _warmed_fallback_shapes:
            continue
        _warmed_fallback_shapes.add(key)
        compile_utils._maybe_compile_asym_gemm_one_type_all(
            compile_utils.AsymGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED,
            n,
            k,
            num_experts,
        )


_warned_graph_capture = False


def should_use_unified_asym_gemm_forward(layer: torch.nn.Module) -> bool:
    """Early-return guard for UnquantizedFusedMoEMethod.forward_cuda."""
    if not has_unified_asym_gemm_layer(layer):
        return False
    # The CPU bucket does host-side work and device syncs, which cannot be
    # captured into a CUDA graph. Fall back to the existing asym_gemm path
    # (the BF16 masters are kept for exactly this) instead of crashing.
    if torch.cuda.is_current_stream_capturing():
        global _warned_graph_capture
        if not _warned_graph_capture:
            _warned_graph_capture = True
            logger.warning(
                "AsymGEMM unified MoE cannot run under CUDA graph capture; "
                "using the existing asym_gemm path for captured graphs. "
                "Run with --disable-cuda-graph to use the unified kernel "
                "everywhere."
            )
        return False
    return True


def unified_asym_gemm_forward(
    layer: torch.nn.Module,
    dispatch_output: StandardDispatchOutput,
) -> StandardCombineInput:
    """Run one MoE layer through the unified CPU+GPU INT8 kernel.

    The unified layer consumes raw router output and applies the routing
    weights internally; only routed_scaling_factor is applied here, matching
    post_permute_asym_gemm_to_standard.
    """
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output

    route_w = topk_weights.to(torch.float32)
    # Padding / non-local slots carry expert_id < 0: zero their routing
    # weight and clamp the index so the unified layer's per-expert token
    # lists never see a negative id.
    if topk_ids.dtype != torch.int64:
        topk_ids = topk_ids.to(torch.int64)
    route_w = torch.where(topk_ids < 0, torch.zeros_like(route_w), route_w)
    expert_ids = topk_ids.clamp_min(0)

    x = (
        hidden_states
        if hidden_states.dtype == torch.bfloat16
        else hidden_states.to(torch.bfloat16)
    )

    unified_layer = getattr(layer, _UNIFIED_LAYER_ATTR)
    output = unified_layer.forward(x, expert_ids, route_w)

    routed_scaling_factor = layer.moe_runner_config.routed_scaling_factor
    if routed_scaling_factor is not None:
        output *= routed_scaling_factor
    if output.dtype != hidden_states.dtype:
        output = output.to(hidden_states.dtype)
    return StandardCombineInput(hidden_states=output)
