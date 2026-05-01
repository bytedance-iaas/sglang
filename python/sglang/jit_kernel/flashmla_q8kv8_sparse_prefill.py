"""JIT-compiled Q8KV8 sparse prefill attention kernel for SM90 (Hopper / H200).

Native FP8 GMMA (`SM90_64x64x32_F32E4M3E4M3_SS_TN` for QK,
`SM90_64x256x32_F32E4M3E4M3_SS_TN` for PV) sparse-MLA prefill kernel
exposed to SGLang via tvm_ffi `load_inline` JIT.

Public API:
    flash_mla_sparse_q8kv8_fwd(q_fp8, kv_fp8, indices, ...)
        -> (out_bf16, max_logits_f32, lse_f32)

Inputs use the same sparse-attention contract as the bf16 reference
kernel (`sgl_kernel.flash_mla.flash_mla_sparse_fwd`):
    q   : [s_q, h_q, d_qk]      float8_e4m3fn
    kv  : [s_kv + 1, h_kv, d_qk] float8_e4m3fn   (+1 row landing pad)
    idx : [s_q, h_kv, topk]     int32
    q_scale, kv_scale : per-tensor f32 scalars on GPU

Output is bf16 already de-quantized in the kernel epilogue.
"""

from __future__ import annotations

import importlib.util
import pathlib
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, override_jit_cuda_arch

if TYPE_CHECKING:
    from tvm_ffi.module import Module


# ---------------------------------------------------------------------------
# CUTLASS include resolution (we depend on CUTLASS/CUTE headers for the
# WGMMA instruction wrappers and SM90 layout atoms).
# ---------------------------------------------------------------------------


def _find_package_root(package: str) -> Optional[pathlib.Path]:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        return None
    return pathlib.Path(spec.origin).resolve().parent


def _resolve_cutlass_include_paths() -> list[str]:
    include_paths: list[str] = []

    flashinfer_root = _find_package_root("flashinfer")
    if flashinfer_root is not None:
        for sub in ("include", "tools/util/include"):
            p = flashinfer_root / "data" / "cutlass" / sub
            if p.exists():
                include_paths.append(str(p))

    deep_gemm_root = _find_package_root("deep_gemm")
    if deep_gemm_root is not None:
        p = deep_gemm_root / "include"
        if p.exists():
            include_paths.append(str(p))

    seen: set[str] = set()
    unique: list[str] = []
    for p in include_paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _q8kv8_cuda_flags() -> list[str]:
    return [
        "-DNDEBUG",
        "-DCUTE_USE_PACKED_TUPLE=1",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "--expt-extended-lambda",
    ]


# ---------------------------------------------------------------------------
# JIT module loader.
#
# `override_jit_cuda_arch(9, 0, "a")` forces sm_90a — the trailing 'a' is
# required because the FP8 WGMMA instructions (`wgmma.mma_async.sync.aligned`
# with E4M3 operands) only exist in the architecture-specific PTX dialect.
# ---------------------------------------------------------------------------


@cache_once
def _jit_flashmla_q8kv8_module() -> Module:
    extra_include_paths = _resolve_cutlass_include_paths()
    if not extra_include_paths:
        raise RuntimeError(
            "Cannot find CUTLASS headers required for Q8KV8 FlashMLA JIT kernel. "
            "Please install flashinfer or deep_gemm with CUTLASS headers."
        )

    with override_jit_cuda_arch(9, 0, "a"):
        return load_jit(
            "flashmla_q8kv8_sparse_prefill",
            cuda_files=[
                "flashmla_q8kv8_sparse_prefill/entry.cuh",
            ],
            cuda_wrappers=[
                # Two C++ entry points exposed by entry.cuh:
                #  - "dispatch"      : optional attn_sink / topk_length
                #  - "dispatch_full" : both tensors required (production path)
                ("dispatch", "sparse_prefill_q8kv8_dispatch"),
                ("dispatch_full", "sparse_prefill_q8kv8_dispatch_full"),
            ],
            extra_include_paths=extra_include_paths,
            extra_cuda_cflags=_q8kv8_cuda_flags(),
        )


# Pre-resolve the C++ entry-point callables once (avoids per-call dict
# lookup in the host hot path).
_resolved_entries: Optional[tuple] = None


def _get_entries() -> tuple:
    global _resolved_entries
    if _resolved_entries is None:
        m = _jit_flashmla_q8kv8_module()
        _resolved_entries = (m["dispatch"], m["dispatch_full"])
    return _resolved_entries


# Raw cudaStream_t pointer fetch (avoids constructing a Python Stream
# wrapper on the per-layer hot path).
_get_current_stream_raw = torch._C._cuda_getCurrentRawStream


# ---------------------------------------------------------------------------
# Output buffer cache.
#
# `out` / `max_logits` / `lse` are kernel-write-only on every call.  The
# downstream output-projection matmul reads `out` on the same default
# stream as the kernel writes it, so single-stream eager execution
# guarantees layer N's consumer completes before layer N+1's kernel
# overwrites the buffer.  Caching avoids 3 fresh `torch.empty` allocations
# per layer (61 layers × per-token-batch).
# ---------------------------------------------------------------------------
_q8kv8_outbuf_cache: dict = {}


def _q8kv8_get_outbufs(s_q: int, h_q: int, d_v: int, device: torch.device):
    key = (device, h_q, d_v)
    entry = _q8kv8_outbuf_cache.get(key)
    if entry is None or entry[0].shape[0] < s_q:
        out = torch.empty(s_q, h_q, d_v, dtype=torch.bfloat16, device=device)
        max_logits = torch.empty(s_q, h_q, dtype=torch.float32, device=device)
        lse = torch.empty(s_q, h_q, dtype=torch.float32, device=device)
        _q8kv8_outbuf_cache[key] = (out, max_logits, lse)
    else:
        out, max_logits, lse = entry
    return out[:s_q], max_logits[:s_q], lse[:s_q]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def flash_mla_sparse_q8kv8_fwd(
    q: torch.Tensor,            # [s_q, h_q, d_qk], float8_e4m3fn
    kv: torch.Tensor,           # [s_kv + 1, h_kv, d_qk], float8_e4m3fn (+1 landing pad)
    indices: torch.Tensor,      # [s_q, h_kv, topk], int32
    sm_scale: float,
    q_scale: torch.Tensor,      # scalar tensor on GPU, float32
    kv_scale: torch.Tensor,     # scalar tensor on GPU, float32
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,    # [h_q], float32
    topk_length: Optional[torch.Tensor] = None,  # [s_q], int32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run Q8KV8 FP8 sparse prefill attention on SM90.

    Returns:
        out:        [s_q, h_q, d_v]  bfloat16
        max_logits: [s_q, h_q]       float32
        lse:        [s_q, h_q]       float32
    """
    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    h_kv = kv.shape[1]
    topk = indices.shape[2]

    out, max_logits, lse = _q8kv8_get_outbufs(s_q, h_q, d_v, q.device)
    dispatch_fn, dispatch_full_fn = _get_entries()
    cuda_stream = _get_current_stream_raw(q.device.index)

    if attn_sink is not None and topk_length is not None:
        # Production path: both auxiliary tensors are always provided
        # by `_forward_flashmla_sparse_q8kv8` in nsa_backend.py.
        dispatch_full_fn(
            q, kv, indices, q_scale, kv_scale, attn_sink, topk_length,
            out, max_logits, lse,
            s_q, s_kv, h_q, h_kv, d_qk, d_v, topk, sm_scale,
            cuda_stream,
        )
    else:
        has_attn_sink = 1 if attn_sink is not None else 0
        has_topk_length = 1 if topk_length is not None else 0
        dispatch_fn(
            q, kv, indices, q_scale, kv_scale,
            out, max_logits, lse,
            s_q, s_kv, h_q, h_kv, d_qk, d_v, topk, sm_scale,
            has_attn_sink, has_topk_length,
            cuda_stream,
        )

    return out, max_logits, lse
