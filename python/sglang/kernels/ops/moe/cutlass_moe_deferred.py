from __future__ import annotations

from functools import cache
from pathlib import Path

import torch

_original_sm90_generator = None


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

    install_sm90_deferred_overlay()
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
    expanded_idx = storage[gemm2_numel:].view(torch.int32)
    return FlashInferTrtllmDeferredFinalizeOutput(
        gemm2_out=gemm2_out,
        expert_weights=kwargs["token_final_scales"],
        expanded_idx_to_permuted_idx=expanded_idx,
        top_k=top_k,
    )
