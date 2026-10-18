from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import torch
import triton
import triton.language as tl
from packaging.version import Version

_original_sm90_generator = None
_MIN_DSV41_EP4_TACTICS_FLASHINFER_VERSION = Version("0.6.18")
_DSV41_EP4_PROFILE_IDS = {
    256: (56, 241),
    8192: (20, 153),
}


@triton.jit
def _canonicalize_deferred_map_kernel(
    source,
    expert_ids,
    target,
    tokens: tl.constexpr,
    top_k: tl.constexpr,
    expert_token_stride: tl.constexpr,
    expert_slot_stride: tl.constexpr,
    start_expert: tl.constexpr,
    end_expert: tl.constexpr,
    block: tl.constexpr,
):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    in_range = offset < tokens * top_k
    token = offset // top_k
    slot = offset % top_k
    expert = tl.load(
        expert_ids + token * expert_token_stride + slot * expert_slot_stride,
        mask=in_range,
        other=-1,
    )
    local = in_range & (expert >= start_expert) & (expert < end_expert)
    # CUTLASS stores [slot, token] and only initializes local expert entries.
    row = tl.load(source + slot * tokens + token, mask=local, other=-1)
    tl.store(target + offset, row, mask=in_range)


def canonicalize_deferred_map(
    source: torch.Tensor,
    expert_ids: torch.Tensor,
    ep_rank: int,
    local_experts: int,
) -> torch.Tensor:
    """Return token-major indices, with -1 for every nonlocal route."""
    tokens, top_k = expert_ids.shape
    target = torch.empty_like(source)
    if tokens:
        _canonicalize_deferred_map_kernel[(triton.cdiv(tokens * top_k, 256),)](
            source,
            expert_ids,
            target,
            tokens,
            top_k,
            *expert_ids.stride(),
            ep_rank * local_experts,
            (ep_rank + 1) * local_experts,
            256,
        )
    return target


@cache
def _use_tuned_dsv41_ep4_tactics() -> bool:
    return os.environ.get("SGLANG_DSV41_SM90_MOE_EP4_PREFILL_TACTICS", "0") == "1"


def _replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"FlashInfer CUTLASS MoE overlay expected one source anchor, found {count}"
        )
    return source.replace(old, new)


def _patch_cutlass_moe_header(source: str) -> str:
    source = _replace_once(
        source,
        """namespace tensorrt_llm::kernels::cutlass_kernels {
""",
        """namespace tensorrt_llm::kernels::cutlass_kernels {

// Host-thread launch state for SGLang's deferred-finalize overlay.
inline thread_local bool sgl_defer_finalize = false;
""",
    )
    source = _replace_once(
        source,
        """  configureWsPtrs(workspace_ptr, num_rows, hidden_size, inter_size, num_experts_per_node,
                  experts_per_token, fc1_activation_type, parallelism_config, use_lora,
                  use_deepseek_fp8_block_scale, use_mxfp8_act_scaling, min_latency_mode, use_awq);

  int start_expert""",
        """  configureWsPtrs(workspace_ptr, num_rows, hidden_size, inter_size, num_experts_per_node,
                  experts_per_token, fc1_activation_type, parallelism_config, use_lora,
                  use_deepseek_fp8_block_scale, use_mxfp8_act_scaling, min_latency_mode, use_awq);

  // SGLang deferred-finalize ABI: use the caller-owned oversized output
  // storage for GEMM2 so the routed rows survive this call.
  if (!use_fused_finalize_) {
    fc2_result_ = final_output;
    sgl_defer_finalize = true;
  }

  int start_expert""",
    )
    source = _replace_once(
        source,
        """  if (has_different_output_type_ampere || has_different_output_type_tma_ws) {
    finalizeMoeRoutingKernelLauncher<OutputType, UnfusedGemmOutputType>(""",
        """  if (!sgl_defer_finalize &&
      (has_different_output_type_ampere || has_different_output_type_tma_ws)) {
    finalizeMoeRoutingKernelLauncher<OutputType, UnfusedGemmOutputType>(""",
    )
    source = _replace_once(
        source,
        """  } else if (!using_tma_ws_gemm2) {
    finalizeMoeRoutingKernelLauncher<OutputType, T>(""",
        """  } else if (!sgl_defer_finalize && !using_tma_ws_gemm2) {
    finalizeMoeRoutingKernelLauncher<OutputType, T>(""",
    )
    source = _replace_once(
        source,
        """    sync_check_cuda_error(stream);
  }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode, class Enable>
std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>""",
        """    sync_check_cuda_error(stream);

    if (!use_fused_finalize_) {
      // The Python adapter reserves this tail immediately after the
      // [num_rows * top_k, hidden] BF16 GEMM2 matrix.
      auto* map_out = reinterpret_cast<int*>(
          static_cast<OutputType*>(final_output) + expanded_num_rows * hidden_size);
      TLLM_CUDA_CHECK(cudaMemcpyAsync(
          map_out, unpermuted_row_to_permuted_row,
          expanded_num_rows * sizeof(int), cudaMemcpyDeviceToDevice, stream));
      sgl_defer_finalize = false;
    }
  }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode, class Enable>
std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>""",
    )
    return source


def _write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def gen_sgl_cutlass_fused_moe_sm90_module(use_fast_build: bool = False):
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import JitSpecNvcc

    assert _original_sm90_generator is not None
    original = _original_sm90_generator(use_fast_build)
    source_dir = jit_env.FLASHINFER_CSRC_DIR / "fused_moe/cutlass_backend"
    overlay_dir = jit_env.FLASHINFER_GEN_SRC_DIR / "sglang_cutlass_moe_deferred/sm90"
    header_name = "cutlass_fused_moe_kernels.cuh"
    instantiation_name = "cutlass_fused_moe_instantiation.cu"
    _write_if_changed(
        overlay_dir / header_name,
        _patch_cutlass_moe_header((source_dir / header_name).read_text()),
    )
    _write_if_changed(
        overlay_dir / instantiation_name,
        (source_dir / instantiation_name).read_text(),
    )

    sources = [
        overlay_dir / instantiation_name if path.name == instantiation_name else path
        for path in original.sources
    ]
    return JitSpecNvcc(
        name="sgl_cutlass_fused_moe_90_deferred",
        sources=sources,
        extra_cflags=original.extra_cflags,
        extra_cuda_cflags=original.extra_cuda_cflags,
        extra_ldflags=original.extra_ldflags,
        extra_include_dirs=[overlay_dir, *original.extra_include_dirs],
        is_class=original.is_class,
        needs_device_linking=original.needs_device_linking,
    )


@cache
def install_sm90_deferred_overlay() -> None:
    """Build and cache the SGLang-owned deferred ABI before first MoE launch."""
    import flashinfer.fused_moe.core as fi_core

    global _original_sm90_generator
    original = fi_core.gen_cutlass_fused_moe_sm90_module
    _original_sm90_generator = original
    fi_core.gen_cutlass_fused_moe_sm90_module = gen_sgl_cutlass_fused_moe_sm90_module
    try:
        fi_core.get_cutlass_fused_moe_module.cache_clear()
        fi_core.get_cutlass_fused_moe_module("90")
    finally:
        fi_core.gen_cutlass_fused_moe_sm90_module = original


def cutlass_fused_moe_deferred(
    *,
    num_tokens: int,
    hidden_size: int,
    top_k: int,
    **kwargs,
):
    """Run FlashInfer GEMMs and return SGLang's deferred-finalize triple."""
    from flashinfer.fused_moe import cutlass_fused_moe

    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmDeferredFinalizeOutput,
    )

    if kwargs.get("cluster_size", 1) != 1:
        raise ValueError("The SM90 deferred MoE overlay requires cluster_size=1")
    install_sm90_deferred_overlay()
    if (
        _use_tuned_dsv41_ep4_tactics()
        and hidden_size == 5120
        and top_k == 6
        and kwargs.get("tp_size") == 1
        and kwargs.get("ep_size") == 4
        and kwargs.get("use_wfp4afp8_humming", False)
        and tuple(kwargs["fc1_expert_weights"].shape) == (96, 4608, 2560)
        and tuple(kwargs["fc2_expert_weights"].shape) == (96, 5120, 1152)
        and "profile_ids" not in kwargs
    ):
        import flashinfer

        profile_ids = _DSV41_EP4_PROFILE_IDS.get(kwargs.get("tune_max_num_tokens"))
        if (
            Version(flashinfer.__version__) >= _MIN_DSV41_EP4_TACTICS_FLASHINFER_VERSION
            and profile_ids is not None
        ):
            kwargs["profile_ids"] = list(profile_ids)
    expanded_rows = num_tokens * top_k
    gemm2_numel = expanded_rows * hidden_size
    # The C++ overlay writes int32 routing indices into this BF16 storage tail.
    storage = torch.empty(
        gemm2_numel + expanded_rows * 2,
        dtype=torch.bfloat16,
        device=kwargs["input"].device,
    )
    output_view = storage[: num_tokens * hidden_size].view(num_tokens, hidden_size)
    cutlass_fused_moe(
        output=output_view,
        use_fused_finalize=False,
        **kwargs,
    )
    gemm2_out = storage[:gemm2_numel].view(expanded_rows, hidden_size)
    expanded_idx = canonicalize_deferred_map(
        storage[gemm2_numel:].view(torch.int32),
        kwargs["token_selected_experts"],
        kwargs.get("ep_rank", 0),
        kwargs["fc1_expert_weights"].shape[0],
    )
    return FlashInferTrtllmDeferredFinalizeOutput(
        gemm2_out=gemm2_out,
        expert_weights=kwargs["token_final_scales"],
        expanded_idx_to_permuted_idx=expanded_idx,
        top_k=top_k,
    )
