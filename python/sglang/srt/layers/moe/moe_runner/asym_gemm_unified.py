"""Unified CPU(AMX INT8) + GPU(SM90 INT8) MoE path for the asym_gemm backend.

Wraps ``asym_gemm.unified_moe.Layer``: at load time the BF16 w13/w2 masters
are quantized to INT8 once and kept in pinned host memory (the same row-major
bytes the SM90 INT8 GPU kernel reads over PCIe via TMA and the CPU AMX kernel
consumes directly). At forward time experts are dispatched per routed token
count: small experts run on the CPU AMX bucket, large ones on the SM90 INT8
grouped kernel.

This path is opt-in via SGLANG_ASYMGEMM_UNIFIED_MOE=1 (see
``asym_gemm_wrapper.configurer``) and bypasses the MoeRunner permute pipeline
entirely — the unified layer consumes raw (hidden_states, topk_ids,
topk_weights) and does gather/scatter/weighted-reduce internally. The
existing asym_gemm paths are untouched and serve as the fallback whenever a
layer cannot be converted.
"""

from __future__ import annotations

import logging
import os
import re
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

# One AMX worker pool for the whole process, shared by every converted MoE
# layer (mirrors KTransformers' single class-level CPUInfer). Without this each
# layer would build its own _C.Runtime, multiplying the thread budget by the
# MoE-layer count and oversubscribing the CPU. sglang owns the pool's
# lifecycle and budget; AsymGEMM only consumes the injected runtime.
_SHARED_CPU_RUNTIME = None


def _get_cpu_runtime():
    """Return the process-global AMX runtime, building it once on first use.

    Sized by SGLANG_ASYMGEMM_UNIFIED_CPU_THREADS as a *total* thread budget for
    the shared pool (0 = library default = host hardware concurrency). The
    value is read once, when the first convertible layer is built; all layers
    then share that pool.
    """
    global _SHARED_CPU_RUNTIME
    if _SHARED_CPU_RUNTIME is None:
        from asym_gemm.unified_moe import _C

        threads = envs.SGLANG_ASYMGEMM_UNIFIED_CPU_THREADS.get()
        if os.getenv("ASYMGEMM_NUMA_TP", "0") == "1":
            # NUMA tensor-parallel pool pair: workers bound to node 0 / node
            # 1, matching the node-local expert slab halves the unified layer
            # builds under the same env (asym_gemm.unified_moe.runtime).
            _SHARED_CPU_RUNTIME = _C.Runtime(threads, 0, 1)
        else:
            _SHARED_CPU_RUNTIME = _C.Runtime(threads)
    return _SHARED_CPU_RUNTIME


def unified_asym_gemm_enabled() -> bool:
    return asym_gemm_wrapper.ASYMGEMM_UNIFIED_MOE


def has_unified_asym_gemm_layer(layer: torch.nn.Module) -> bool:
    return getattr(layer, _UNIFIED_LAYER_ATTR, None) is not None


# --------------------------------------------------------------------------- #
# INT8 preload: when an offline INT8 slab covers a layer, its BF16 expert
# master weights are pure waste — they would be allocated (pinned), read from
# the checkpoint, and then immediately released after conversion. These
# helpers let create_weights skip the pinned allocation and the checkpoint
# weights iterator skip the disk read entirely.
# --------------------------------------------------------------------------- #

_MASTERS_SKIPPED_ATTR = "_asym_masters_skipped"

# Expert master weights in checkpoint naming: the per-expert form
# (…layers.N.mlp.experts.E.{gate,up,down}_proj.weight) and the fused form
# (…layers.N.mlp.experts.{gate_up_proj,down_proj}). MTP draft weights
# (mtp.…) are never converted to the unified path and must still load.
_EXPERT_WEIGHT_NAME_RE = re.compile(
    r"(?!mtp\.)"
    r".*\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?:\d+\.(?:gate_proj|up_proj|down_proj)\.weight"
    r"|gate_up_proj|down_proj)$"
)


def _int8_slab_exists(layer_id) -> bool:
    path = envs.SGLANG_ASYMGEMM_UNIFIED_INT8_PATH.get()
    return bool(path) and os.path.exists(
        os.path.join(path, f"layer_{layer_id}.safetensors")
    )


def _int8_preload_active() -> bool:
    """Whether skipping BF16 masters in favor of the offline INT8 slab is on.

    Requires the unified path to actually be usable on this host (the full
    configurer check) and tp_size == 1 — with TP sharding the per-partition
    master shapes would never validate against the full-size slab, and the
    layer would need the masters for its fallback path.
    """
    if not unified_asym_gemm_enabled():
        return False
    if not envs.SGLANG_ASYMGEMM_UNIFIED_INT8_PATH.get():
        return False
    from sglang.srt.server_args import get_global_server_args

    try:
        return get_global_server_args().tp_size == 1
    except ValueError:
        return False


def master_weights_preloaded_as_int8(layer: torch.nn.Module) -> bool:
    """True when `layer`'s expert masters will come from the INT8 slab, so
    create_weights may allocate them unpinned (virtual, never touched) and
    mark the layer as skipped. NOTE: assumes `layer` belongs to the target
    model — MTP/draft models must not call this (their layer_ids collide
    with the target's slab files)."""
    if not _int8_preload_active():
        return False
    layer_id = getattr(layer, "layer_id", None)
    return layer_id is not None and _int8_slab_exists(layer_id)


def mark_master_weights_skipped(layer: torch.nn.Module) -> None:
    setattr(layer, _MASTERS_SKIPPED_ATTR, True)


def make_expert_weight_skip_predicate(quant_config):
    """Build a name-level filter for the checkpoint weights iterator, or None.

    Active only for unquantized (BF16) checkpoints with the INT8 preload on:
    returns a predicate(name) -> bool that is True for expert master weights
    of layers covered by an INT8 slab file — the iterator then never calls
    get_tensor on them, so their bytes are never read from disk.
    """
    if quant_config is not None:
        return None
    if not _int8_preload_active():
        return None

    covered: dict = {}

    def _skip(name: str) -> bool:
        m = _EXPERT_WEIGHT_NAME_RE.match(name)
        if m is None:
            return False
        layer_id = int(m.group("layer"))
        hit = covered.get(layer_id)
        if hit is None:
            hit = covered[layer_id] = _int8_slab_exists(layer_id)
            if hit:
                logger.info(
                    "AsymGEMM unified MoE: skipping checkpoint read of layer "
                    "%d expert master weights (INT8 slab covers them)",
                    layer_id,
                )
        return hit

    return _skip


def _check_convertible(layer: torch.nn.Module) -> str | None:
    """Return None if `layer` is *structurally* eligible for the unified path.

    Dtype-agnostic on purpose: the INT8 fast path (``from_int8``) consumes a
    pre-quantized slab and never touches the BF16/FP8 masters, so it works for a
    BF16 *or* a block-scaled FP8 checkpoint. The BF16 requirement only applies to
    the online-quantization fallback (``from_bf16``) and is enforced by the
    caller, not here.
    """
    from asym_gemm.unified_moe.runtime import GRAN_K

    w13 = layer.w13_weight
    w2 = layer.w2_weight

    if not layer.moe_runner_config.is_gated:
        return "MoE is not gated (unified kernel computes SwiGLU)"
    if layer.moe_runner_config.activation != "silu":
        return f"activation {layer.moe_runner_config.activation!r} != 'silu'"
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


def _maybe_load_int8_slab(
    layer: torch.nn.Module, num_experts: int, hidden: int, inter: int
) -> dict | None:
    """Load pre-quantized INT8 expert weights for ``layer`` from the directory
    named by SGLANG_ASYMGEMM_UNIFIED_INT8_PATH, or return None to quantize
    online.

    The offline artifact (AsymGEMM/scripts/convert_int8_weights.py) is one
    ``layer_{layer_id}.safetensors`` per MoE layer, keyed by the model's decoder
    layer index — which is sglang's ``layer.layer_id``. Every load is validated
    against the live BF16 masters' shapes; any missing file or mismatch returns
    None so the caller transparently falls back to online quantization.
    """
    path = envs.SGLANG_ASYMGEMM_UNIFIED_INT8_PATH.get()
    if not path:
        return None

    layer_id = getattr(layer, "layer_id", None)
    if layer_id is None:
        logger.warning(
            "AsymGEMM unified MoE: SGLANG_ASYMGEMM_UNIFIED_INT8_PATH is set but "
            "the layer has no layer_id; quantizing online."
        )
        return None

    file = os.path.join(path, f"layer_{layer_id}.safetensors")
    if not os.path.exists(file):
        logger.warning(
            "AsymGEMM unified MoE: no INT8 weights for layer %s at %s; "
            "quantizing online.",
            layer_id,
            file,
        )
        return None

    from safetensors.torch import load_file

    tensors = load_file(file)
    expected = {
        "gate_int8": (num_experts, inter, hidden),
        "gate_scales": (num_experts, inter),
        "up_int8": (num_experts, inter, hidden),
        "up_scales": (num_experts, inter),
        "down_int8": (num_experts, hidden, inter),
        "down_scales": (num_experts, hidden),
    }
    slab: dict = {}
    for name, want in expected.items():
        t = tensors.get(name)
        if t is None:
            logger.warning(
                "AsymGEMM unified MoE: INT8 file %s missing tensor %r; "
                "quantizing online.",
                file,
                name,
            )
            return None
        if tuple(t.shape) != want:
            logger.warning(
                "AsymGEMM unified MoE: INT8 file %s tensor %r shape %s != "
                "expected %s (checkpoint/model mismatch); quantizing online.",
                file,
                name,
                tuple(t.shape),
                want,
            )
            return None
        slab[name] = t
    return slab


def maybe_create_unified_asym_gemm_layer(layer: torch.nn.Module) -> None:
    """Build the unified INT8 MoE layer for a converted MoE.

    Called from both UnquantizedFusedMoEMethod (BF16 masters) and Fp8MoEMethod
    (block-scaled FP8 masters) in process_weights_after_loading, when the
    asym_gemm backend is active and the unified path is enabled. BF16 masters
    can be quantized online (from_bf16); FP8 masters require a pre-quantized
    INT8 slab (SGLANG_ASYMGEMM_UNIFIED_INT8_PATH). On any unsupported
    configuration the reason is logged and the layer keeps using the existing
    asym_gemm path (with its masters intact). On success the unified layer
    serves every forward for this layer — eager and captured — so the
    BF16/FP8 masters are released to reclaim their memory.
    """
    if not unified_asym_gemm_enabled():
        return

    # Whether create_weights skipped the pinned master allocation (and the
    # loader skipped their checkpoint bytes) because an INT8 slab covers this
    # layer. If so, the masters hold uninitialized memory — every path that
    # would read them must hard-fail instead of silently computing garbage.
    masters_skipped = bool(getattr(layer, _MASTERS_SKIPPED_ATTR, False))

    reason = _check_convertible(layer)
    if reason is not None:
        if masters_skipped:
            raise RuntimeError(
                f"AsymGEMM unified MoE: layer {getattr(layer, 'layer_id', '?')} "
                f"skipped loading its expert master weights (INT8 slab present) "
                f"but cannot take the unified path: {reason}. Unset "
                "SGLANG_ASYMGEMM_UNIFIED_INT8_PATH or fix the configuration."
            )
        logger.warning(
            "AsymGEMM unified MoE: layer %s falls back to the existing "
            "asym_gemm path (%s)",
            getattr(layer, "layer_id", "?"),
            reason,
        )
        return

    from asym_gemm.unified_moe import Layer as UnifiedMoeLayer

    # Shapes come straight from the (possibly still-on-GPU) BF16 masters — no
    # copy needed. w13 is [G, 2*inter, hidden]; sglang's silu_and_mul convention
    # puts gate in the first half and up in the second.
    num_experts, two_inter, hidden = layer.w13_weight.shape
    inter = two_inter // 2

    # BF16 masters can be quantized online; FP8 (block-scaled) masters cannot —
    # for those the pre-quantized INT8 slab is the only way in.
    masters_are_bf16 = (
        layer.w13_weight.dtype == torch.bfloat16
        and layer.w2_weight.dtype == torch.bfloat16
    )

    # Fast path: load INT8 weights pre-quantized offline, skipping the slow
    # per-expert quantization loop. Falls back to None on any mismatch.
    slab = _maybe_load_int8_slab(layer, num_experts, hidden, inter)
    if slab is None and masters_skipped:
        raise RuntimeError(
            f"AsymGEMM unified MoE: layer {getattr(layer, 'layer_id', '?')} "
            "skipped loading its expert master weights (INT8 slab present) "
            "but the slab failed to load/validate (see warnings above) — "
            "online quantization would quantize uninitialized memory. Fix "
            "the INT8 dir or unset SGLANG_ASYMGEMM_UNIFIED_INT8_PATH."
        )

    # Inject the process-global AMX runtime so every converted MoE layer shares
    # one worker pool (and one total thread budget) instead of building its own.
    if slab is not None:
        unified_layer = UnifiedMoeLayer.from_int8(
            gate_int8=slab["gate_int8"],
            gate_scales=slab["gate_scales"],
            up_int8=slab["up_int8"],
            up_scales=slab["up_scales"],
            down_int8=slab["down_int8"],
            down_scales=slab["down_scales"],
            top_k=layer.top_k,
            cuda_device=torch.cuda.current_device(),
            runtime=_get_cpu_runtime(),
            m_cpu=envs.SGLANG_ASYMGEMM_UNIFIED_M_CPU.get(),
        )
        source = "loaded pre-quantized INT8 for"
    elif not masters_are_bf16:
        # No offline INT8 and the masters aren't BF16 (e.g. a block-scaled FP8
        # checkpoint) — online quantization can't run. Leave the layer on its
        # existing asym_gemm path (BF16 or FP8) instead of failing.
        logger.warning(
            "AsymGEMM unified MoE: layer %s has %s/%s masters and no INT8 "
            "weights; set SGLANG_ASYMGEMM_UNIFIED_INT8_PATH to an offline-"
            "converted INT8 dir to enable the unified path. Falling back.",
            getattr(layer, "layer_id", "?"),
            layer.w13_weight.dtype,
            layer.w2_weight.dtype,
        )
        return
    else:
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
        unified_layer = UnifiedMoeLayer.from_bf16(
            gate=w13[:, :inter, :],
            up=w13[:, inter:, :],
            down=w2,
            top_k=layer.top_k,
            cuda_device=torch.cuda.current_device(),
            runtime=_get_cpu_runtime(),
            m_cpu=envs.SGLANG_ASYMGEMM_UNIFIED_M_CPU.get(),
        )
        source = "quantized"

    setattr(layer, _UNIFIED_LAYER_ATTR, unified_layer)
    logger.info(
        "AsymGEMM unified MoE: %s layer %s "
        "(experts=%d, hidden=%d, inter=%d, m_cpu=%d)",
        source,
        getattr(layer, "layer_id", "?"),
        num_experts,
        hidden,
        inter,
        unified_layer.m_cpu,
    )
    _maybe_init_capturable_decode(unified_layer)
    # The unified layer now serves every forward for this layer (the eager
    # path directly, decode capture via the capturable buffers) — the
    # BF16/FP8 masters can never be needed again. Release them.
    _release_master_weights(layer)


def _maybe_init_capturable_decode(unified_layer) -> None:
    """Pre-allocate the CUDA-graph-capturable decode buffers.

    The capturable path expresses the CPU MoE bucket as a stream-ordered
    cudaLaunchHostFunc host node over fixed pinned buffers, so the decode graph
    captures/replays without --disable-cuda-graph. It is all-CPU for any batch
    size; we pre-allocate for every CUDA-graph capture batch size (server_args
    .cuda_graph_bs, already bounded by --cuda-graph-max-bs). Allocating here
    (outside capture) is required — the host node and buffers must exist before
    any graph records them.

    There is no fallback under capture (the masters are released after
    conversion), so failure to initialize is a hard error when CUDA graphs
    are enabled.
    """
    from sglang.srt.server_args import get_global_server_args

    try:
        server_args = get_global_server_args()
    except ValueError:
        server_args = None
    if server_args is not None and getattr(server_args, "disable_cuda_graph", False):
        return  # CUDA graph disabled — nothing to capture

    batch_sizes = None
    if server_args is not None:
        batch_sizes = getattr(server_args, "cuda_graph_bs", None)
    if not batch_sizes:
        batch_sizes = [1]
    batch_sizes = sorted({int(b) for b in batch_sizes if int(b) >= 1})

    try:
        from asym_gemm.unified_moe.capturable import init_capturable_decode

        init_capturable_decode(unified_layer, batch_sizes)
    except Exception as e:
        raise RuntimeError(
            "AsymGEMM unified MoE: capturable decode init failed and there is "
            "no capture-time fallback (the master weights are released after "
            "INT8 conversion). Pass --disable-cuda-graph or fix the asym_gemm "
            f"build. Underlying error: {e}"
        ) from e


def _release_master_weights(layer: torch.nn.Module) -> None:
    """Release the BF16/FP8 master weights once the INT8 slab owns the layer.

    The unified layer's pinned INT8 slab is the only weight copy any forward
    reads from now on, so keeping the masters would only duplicate every MoE
    expert (BF16: 2 bytes/param of pinned host RAM; FP8: bytes in VRAM plus
    block scales). Replace their storage with empty tensors; the loader's
    device_loading_context restore (a plain .to()) remains a no-op on these.

    For INT8-preloaded layers the masters are untouched virtual placeholders
    (never pinned, never loaded) — dropping them frees address space only,
    and the log says so instead of claiming resident memory was freed.
    """
    freed = 0
    for name in (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale_inv",
        "w2_weight_scale_inv",
    ):
        param = getattr(layer, name, None)
        data = getattr(param, "data", None)
        if data is None or data.numel() == 0:
            continue
        freed += data.numel() * data.element_size()
        param.data = torch.empty(0, dtype=data.dtype, device=data.device)
    if freed:
        if getattr(layer, _MASTERS_SKIPPED_ATTR, False):
            # Nothing real was freed — the placeholders were never loaded or
            # resident. Keep this out of the INFO log.
            logger.debug(
                "AsymGEMM unified MoE: dropped never-loaded master-weight "
                "placeholders for layer %s (%.1f MiB nominal, virtual only)",
                getattr(layer, "layer_id", "?"),
                freed / 2**20,
            )
        else:
            logger.info(
                "AsymGEMM unified MoE: released %.1f MiB of master weights "
                "for layer %s",
                freed / 2**20,
                getattr(layer, "layer_id", "?"),
            )


def should_use_unified_asym_gemm_forward(
    layer: torch.nn.Module, num_tokens: int
) -> bool:
    """Early-return guard for the host method's forward_cuda.

    A converted layer always takes the unified path: eager forwards run the
    imperative kernel, and decode capture records the capturable host-node
    chain (buffers for every cuda_graph_bs entry are pre-allocated at load
    time; piecewise prefill capture is disabled in server_args). There is no
    other path — the masters were released at conversion.
    """
    return has_unified_asym_gemm_layer(layer)


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

    # Under CUDA graph capture, route through the capturable host-node chain
    # (D2H -> cudaLaunchHostFunc(CPU MoE) -> H2D over fixed pinned buffers) so
    # the decode step records into the graph. The imperative forward's host
    # work / device syncs would be illegal mid-capture. Every capture batch
    # size has a pre-allocated buffer (init covers all of cuda_graph_bs and
    # piecewise prefill capture is disabled) — the assert enforces that.
    if torch.cuda.is_current_stream_capturing():
        from asym_gemm.unified_moe.capturable import (
            capturable_decode_forward,
            capturable_decode_supported,
        )

        T = x.shape[0]
        assert capturable_decode_supported(unified_layer, T), (
            f"AsymGEMM unified MoE: capture batch {T} has no pre-allocated "
            "capturable buffer. Buffers cover server_args.cuda_graph_bs at "
            "load time; a capture at any other size (e.g. piecewise prefill "
            "force-enabled) is unsupported — there is no fallback."
        )
        output = capturable_decode_forward(unified_layer, x, expert_ids, route_w)
    else:
        output = unified_layer.forward(x, expert_ids, route_w)

    routed_scaling_factor = layer.moe_runner_config.routed_scaling_factor
    if routed_scaling_factor is not None:
        # Out-of-place: the capturable path returns a fixed per-layer device
        # buffer that must not be mutated in place under graph replay.
        output = output * routed_scaling_factor
    if output.dtype != hidden_states.dtype:
        output = output.to(hidden_states.dtype)
    return StandardCombineInput(hidden_states=output)
