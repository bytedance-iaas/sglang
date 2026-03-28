#!/usr/bin/env python3
"""
Comprehensive KV Cache Quantization Benchmark
==============================================

Compares quantization methods for KV cache compression:
  1. FP8 blockwise  (per-token-group FP8 E4M3, block_size=128)
  2. MXFP4          (block_size=32, UE8M0 scale)            [COMMENTED OUT — needs Blackwell]
  3. NVFP4          (block_size=16, E4M3 scale + global)     [COMMENTED OUT — needs Blackwell]
  4. TurboQuant     (1/2/3/3.5/4-bit, Hadamard + Lloyd-Max centroids)

Evaluates:
  - Accuracy: MSE, relative MSE, max absolute error, cosine similarity
  - Latency:  quantize, dequantize, total round-trip (median of N trials)
  - Tested on synthetic tensors (dense/sparse) AND realistic KV cache shapes
    for DeepSeek-V3, GPT-OSS 120B, and Qwen3 model families.

Usage:
    python benchmark/kernels/bench_kv_cache_quant.py [--warmup 5] [--trials 20] [--csv results.csv]
"""

import argparse
import csv
import importlib.util
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Import TurboQuant directly (avoid full sglang init)
# ──────────────────────────────────────────────────────────────────────────────
_kernels_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "python",
    "sglang",
    "srt",
    "layers",
    "quantization",
    "turboquant_kernels.py",
)
_spec = importlib.util.spec_from_file_location(
    "turboquant_kernels", os.path.abspath(_kernels_path)
)
_tq_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tq_mod)

HadamardTransform = _tq_mod.HadamardTransform
turboquant_quantize = _tq_mod.turboquant_quantize
turboquant_dequantize = _tq_mod.turboquant_dequantize
turboquant_quantize_mixed = _tq_mod.turboquant_quantize_mixed
turboquant_dequantize_mixed = _tq_mod.turboquant_dequantize_mixed
initialize_centroids_cache = _tq_mod.initialize_centroids_cache
parse_bits = _tq_mod.parse_bits
compute_compression_ratio = _tq_mod.compute_compression_ratio
_next_power_of_2 = _tq_mod._next_power_of_2

# ──────────────────────────────────────────────────────────────────────────────
# Monkey-patch HadamardTransform to use fast JIT CUDA kernel
# ──────────────────────────────────────────────────────────────────────────────
# The default _fwht() is a pure-Python while-loop doing log2(d) iterations,
# each launching separate CUDA kernels (~21 launches for dim=128).
# The JIT kernel does it in ONE fused CUDA kernel — ~10-20x faster.
_FAST_HADAMARD_AVAILABLE = False
try:
    from sglang.jit_kernel.hadamard import hadamard_transform as _fast_hadamard

    def _fast_fwht_forward(self, x):
        """Fast forward Hadamard using JIT CUDA kernel."""
        orig_dtype = x.dtype
        shape = x.shape
        d = shape[-1]
        if d < self.padded_dim:
            x = torch.nn.functional.pad(x, (0, self.padded_dim - d))
        x = x * self.signs
        # JIT kernel needs bf16/fp16; TurboQuant works in fp32
        if x.dtype == torch.float32:
            x = x.to(torch.bfloat16)
        x = _fast_hadamard(x.contiguous(), scale=self.scale)
        return x.to(orig_dtype)

    def _fast_fwht_inverse(self, y):
        """Fast inverse Hadamard using JIT CUDA kernel."""
        orig_dtype = y.dtype
        inp = y
        if inp.dtype == torch.float32:
            inp = inp.to(torch.bfloat16)
        x = _fast_hadamard(inp.contiguous(), scale=self.scale)
        x = x.to(orig_dtype)
        x = x * self.signs
        return x[..., :self.dim]

    # Patch the class methods
    HadamardTransform.forward = _fast_fwht_forward
    HadamardTransform.inverse = _fast_fwht_inverse
    _FAST_HADAMARD_AVAILABLE = True
    print("[INFO] Using fast JIT CUDA Hadamard transform (single kernel launch)")
except Exception as e:
    print(f"[WARN] Fast JIT Hadamard not available, using slow Python fallback: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Import FP8 blockwise quantization (Triton-based, works on H200)
# ──────────────────────────────────────────────────────────────────────────────
_fp8_kernel_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "python",
    "sglang",
    "srt",
    "layers",
    "quantization",
    "fp8_kernel.py",
)

# We import the Triton-based per_token_group_quant_fp8 via the module directly
# to avoid pulling in sgl_kernel (AOT compiled).  The Triton fallback is always
# available.
try:
    # Prefer sgl_kernel path if available (fastest)
    from sglang.srt.layers.quantization.fp8_kernel import (
        per_token_group_quant_fp8 as _fp8_quant_raw,
        fp8_dtype,
    )

    def fp8_blockwise_quantize(x: torch.Tensor, group_size: int = 128):
        """Quantize BF16 tensor to FP8 E4M3 with per-token-group scales."""
        x_2d = x.reshape(-1, x.shape[-1])
        assert x_2d.shape[-1] % group_size == 0
        x_q, x_s = _fp8_quant_raw(x_2d.contiguous(), group_size)
        return x_q, x_s

    def fp8_blockwise_dequantize(x_q, x_s, group_size: int = 128):
        """Dequantize FP8 back to float32 using per-group scales."""
        m, k = x_q.shape
        num_groups = k // group_size
        # x_s shape: (m, num_groups) — one scale per group per row
        x_f = x_q.to(torch.float32).reshape(m, num_groups, group_size)
        x_s_expanded = x_s.reshape(m, num_groups, 1)
        return (x_f * x_s_expanded).reshape(m, k)

    FP8_AVAILABLE = True
except Exception as e:
    print(f"[WARN] FP8 blockwise not available: {e}")
    FP8_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Import MXFP4 (needs quark on AMD or Blackwell GPU — COMMENTED OUT)
# ──────────────────────────────────────────────────────────────────────────────
MXFP4_AVAILABLE = False

# Uncomment the block below when running on a Blackwell GPU with quark installed.
# ─── BEGIN MXFP4 ───
# try:
#     from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil
#
#     def mxfp4_quantize(x: torch.Tensor, block_size: int = 32):
#         """Quantize BF16 → MXFP4 (block_size=32, UE8M0 scale)."""
#         x_2d = x.reshape(-1, x.shape[-1]).contiguous()
#         quantized_obj, e8m0_scale = MXFP4QuantizeUtil.quantize(x_2d, block_size)
#         return quantized_obj, e8m0_scale
#
#     def mxfp4_dequantize(quantized_obj, e8m0_scale, dtype=torch.float32, block_size: int = 32):
#         """Dequantize MXFP4 → float."""
#         # quantized_obj is an MXFP4QuantizeUtil instance with original_shape, original_dtype, and packed data
#         result = MXFP4QuantizeUtil.dequantize(
#             quantized_data=quantized_obj._data if hasattr(quantized_obj, '_data') else quantized_obj,
#             dtype=dtype,
#             scale=e8m0_scale,
#             block_sizes=[block_size],
#         )
#         return result
#
#     MXFP4_AVAILABLE = True
# except Exception as e:
#     print(f"[WARN] MXFP4 not available: {e}")
#     MXFP4_AVAILABLE = False
# ─── END MXFP4 ───

# ──────────────────────────────────────────────────────────────────────────────
# Import NVFP4 (needs SM100+ Blackwell — COMMENTED OUT)
# ──────────────────────────────────────────────────────────────────────────────
NVFP4_AVAILABLE = False

# Uncomment the block below when running on a Blackwell GPU (SM >= 10.0).
# ─── BEGIN NVFP4 ───
# try:
#     from sglang.jit_kernel.nvfp4 import scaled_fp4_quant, suggest_nvfp4_global_scale
#
#     FLOAT4_E2M1_MAX = 6.0
#     FLOAT8_E4M3_MAX_VAL = torch.finfo(torch.float8_e4m3fn).max
#     NVFP4_BLOCK_SIZE = 16
#
#     # LUT for dequantizing FP4 E2M1 packed nibbles
#     _E2M1_TO_FLOAT32 = [
#         0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
#         0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
#     ]
#
#     def _cast_from_fp4(packed: torch.Tensor, m: int, n: int) -> torch.Tensor:
#         """Unpack FP4 E2M1 nibble-packed uint8 → float32."""
#         v_lo = (packed & 0xF).to(torch.long)
#         v_hi = ((packed >> 4) & 0xF).to(torch.long)
#         interleaved = torch.stack((v_lo, v_hi), dim=-1).flatten()
#         lut = torch.tensor(_E2M1_TO_FLOAT32, device=packed.device, dtype=torch.float32)
#         return lut[interleaved].reshape(m, n)
#
#     def _recover_swizzled_scales(scale: torch.Tensor, m: int, n: int) -> torch.Tensor:
#         """Undo the swizzle layout on NVFP4 per-block scales."""
#         rounded_m = ((m + 128 - 1) // 128) * 128
#         scale_n = n // NVFP4_BLOCK_SIZE
#         rounded_n = ((scale_n + 4 - 1) // 4) * 4
#         tmp = scale.view(1, rounded_m // 128, rounded_n // 4, 32, 4, 4)
#         tmp = tmp.permute(0, 1, 4, 3, 2, 5)
#         result = tmp.reshape(rounded_m, rounded_n).to(torch.float32)
#         return result[:m, :scale_n]
#
#     def nvfp4_quantize(x: torch.Tensor):
#         """Quantize BF16 → NVFP4 (block_size=16, E4M3 block scale + global scale)."""
#         x_2d = x.reshape(-1, x.shape[-1]).contiguous()
#         if x_2d.dtype not in (torch.float16, torch.bfloat16):
#             x_2d = x_2d.to(torch.bfloat16)
#         m, n = x_2d.shape
#         assert n % NVFP4_BLOCK_SIZE == 0
#         global_scale = suggest_nvfp4_global_scale(x_2d)
#         packed_fp4, block_scale = scaled_fp4_quant(x_2d, global_scale)
#         return packed_fp4, block_scale, global_scale, m, n
#
#     def nvfp4_dequantize(packed_fp4, block_scale, global_scale, m, n):
#         """Dequantize NVFP4 → float32."""
#         # Unpack FP4
#         x_fp4_float = _cast_from_fp4(packed_fp4, m, n)
#         # Recover block scales
#         scales_2d = _recover_swizzled_scales(block_scale, m, n)  # (m, n//16)
#         # scales are in FP8 E4M3 format as stored, but _recover gives float32
#         # Dequant: x_float = fp4_value * scale / global_scale
#         # Actually: x ≈ fp4_val * (scale / global_scale)
#         # From the quantization: scaled_x = x * global_scale / scale
#         #                         fp4_val ≈ round(scaled_x)
#         # So: x ≈ fp4_val * scale / global_scale
#         scales_expanded = scales_2d.repeat_interleave(NVFP4_BLOCK_SIZE, dim=-1)[:, :n]
#         return x_fp4_float * scales_expanded / global_scale.float()
#
#     NVFP4_AVAILABLE = True
# except Exception as e:
#     print(f"[WARN] NVFP4 not available: {e}")
#     NVFP4_AVAILABLE = False
# ─── END NVFP4 ───

# ──────────────────────────────────────────────────────────────────────────────
# Device setup
# ──────────────────────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "CUDA required"
DEVICE = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = False  # deterministic
# Initialize centroids for both "cuda" and "cuda:0" device strings,
# since tensors may report either form.
initialize_centroids_cache(DEVICE)
initialize_centroids_cache(torch.device("cuda"))

# ──────────────────────────────────────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────────────────────────────────────


def generate_tensor(
    shape: Tuple[int, ...],
    distribution: str = "dense",
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """
    Generate test tensors with different distributions.

    Args:
        shape: tensor shape
        distribution: one of
            - "dense":        standard normal N(0, 1)
            - "sparse_80":    80% zeros, 20% normal
            - "kv_normal":    typical KV cache distribution ~ N(0, 0.02) (like post-LayerNorm)
            - "kv_lognormal": log-normal with outliers, mimicking attention key activations
    """
    if distribution == "dense":
        x = torch.randn(shape, dtype=torch.float32, device=device)
    elif distribution == "sparse_80":
        x = torch.randn(shape, dtype=torch.float32, device=device)
        mask = torch.rand(shape, device=device) < 0.8
        x[mask] = 0.0
    elif distribution == "kv_normal":
        # Post-layernorm KV values typically have small variance
        x = torch.randn(shape, dtype=torch.float32, device=device) * 0.02
    elif distribution == "kv_lognormal":
        # Keys can have heavy-tailed distributions with some outlier channels
        x = torch.randn(shape, dtype=torch.float32, device=device) * 0.05
        # Add outlier channels (5% of dimensions have 10x amplitude)
        num_outlier_dims = max(1, shape[-1] // 20)
        outlier_dims = torch.randperm(shape[-1], device=device)[:num_outlier_dims]
        x[..., outlier_dims] *= 10.0
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return x.to(dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Quantization method wrappers (uniform interface)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class QuantResult:
    """Result of a quantization round-trip."""

    method: str
    bits_per_elem: float
    compression_ratio: float
    mse: float
    rel_mse: float
    max_abs_err: float
    cosine_sim: float
    quant_time_us: float
    dequant_time_us: float
    total_time_us: float
    shape: Tuple[int, ...]
    distribution: str
    scenario: str
    extra: Dict[str, Any] = field(default_factory=dict)


class QuantMethod:
    """Base class for quantization methods."""

    name: str
    bits_per_elem: float

    def quantize(self, x: torch.Tensor) -> Any:
        raise NotImplementedError

    def dequantize(self, compressed: Any, original_shape: Tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError

    def compression_ratio(self, head_dim: int) -> float:
        return 16.0 / self.bits_per_elem  # BF16 baseline = 16 bits


class FP8BlockwiseMethod(QuantMethod):
    """FP8 E4M3 per-token-group quantization (group_size=128)."""

    def __init__(self, group_size: int = 128):
        self.name = f"fp8_blockwise_g{group_size}"
        self.group_size = group_size
        # 8 bits per element + 32 bits per group for scale
        self.bits_per_elem = 8.0 + 32.0 / group_size

    def quantize(self, x: torch.Tensor) -> Any:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        # Pad last dim to multiple of group_size if needed
        k = x_2d.shape[-1]
        if k % self.group_size != 0:
            pad = self.group_size - (k % self.group_size)
            x_2d = F.pad(x_2d, (0, pad))
        x_q, x_s = fp8_blockwise_quantize(x_2d, self.group_size)
        return (x_q, x_s, x.shape)

    def dequantize(self, compressed: Any, original_shape: Tuple[int, ...]) -> torch.Tensor:
        x_q, x_s, orig_shape = compressed
        result = fp8_blockwise_dequantize(x_q, x_s, self.group_size)
        k = orig_shape[-1]
        result = result[..., :k]  # remove padding
        return result.reshape(orig_shape)

    def compression_ratio(self, head_dim: int) -> float:
        return 16.0 / self.bits_per_elem


class TurboQuantMethod(QuantMethod):
    """TurboQuant: Hadamard rotation + Lloyd-Max centroid quantization."""

    def __init__(self, bits, mode: str = "mse"):
        self.bits_raw = bits
        self.mode = mode
        is_mixed, bh, bl = parse_bits(bits)
        if is_mixed:
            self.name = f"turboquant_{bits}b_mixed"
        else:
            self.name = f"turboquant_{int(bits)}b_{mode}"
        # Approximate bits per element
        self.bits_per_elem = float(bits)
        self._hadamard_cache: Dict[int, Any] = {}

    def _get_hadamard(self, dim: int, seed: int = 42):
        key = (dim, seed)
        if key not in self._hadamard_cache:
            self._hadamard_cache[key] = HadamardTransform(dim, seed=seed, device=DEVICE)
        return self._hadamard_cache[key]

    def quantize(self, x: torch.Tensor) -> Any:
        x_2d = x.reshape(-1, x.shape[-1])
        dim = x_2d.shape[-1]
        is_mixed, bh, bl = parse_bits(self.bits_raw)
        if is_mixed:
            split = dim // 2
            h_hi = self._get_hadamard(split, seed=42)
            h_lo = self._get_hadamard(dim - split, seed=43)
            q = turboquant_quantize_mixed(x_2d, h_hi, h_lo, bh, bl, split)
            return ("mixed", q, x.shape, dim, split)
        else:
            h = self._get_hadamard(dim, seed=42)
            q = turboquant_quantize(x_2d, h, int(self.bits_raw), self.mode)
            return ("uniform", q, x.shape, dim)

    def dequantize(self, compressed: Any, original_shape: Tuple[int, ...]) -> torch.Tensor:
        if compressed[0] == "mixed":
            _, q, orig_shape, dim, split = compressed
            h_hi = self._get_hadamard(split, seed=42)
            h_lo = self._get_hadamard(dim - split, seed=43)
            result = turboquant_dequantize_mixed(q, h_hi, h_lo, torch.float32)
            return result.reshape(orig_shape)
        else:
            _, q, orig_shape, dim = compressed
            h = self._get_hadamard(dim, seed=42)
            result = turboquant_dequantize(
                q, h, int(self.bits_raw), self.mode, torch.float32
            )
            return result[..., :dim].reshape(orig_shape)

    def compression_ratio(self, head_dim: int) -> float:
        return compute_compression_ratio(head_dim, self.bits_raw, self.mode)


# ──────────────────────────────────────────────────────────────────────────────
# MXFP4 method (COMMENTED OUT — uncomment on Blackwell)
# ──────────────────────────────────────────────────────────────────────────────

# class MXFP4Method(QuantMethod):
#     """MXFP4: block_size=32, UE8M0 scale (no global scale)."""
#
#     def __init__(self):
#         self.name = "mxfp4_b32"
#         # 4 bits per element + 8 bits per 32-element block for scale
#         self.bits_per_elem = 4.0 + 8.0 / 32.0  # = 4.25
#
#     def quantize(self, x: torch.Tensor) -> Any:
#         x_2d = x.reshape(-1, x.shape[-1]).contiguous()
#         k = x_2d.shape[-1]
#         # Pad to multiple of 32
#         if k % 32 != 0:
#             pad = 32 - (k % 32)
#             x_2d = F.pad(x_2d, (0, pad))
#         quantized_obj, e8m0_scale = mxfp4_quantize(x_2d, block_size=32)
#         return (quantized_obj, e8m0_scale, x.shape, x_2d.shape)
#
#     def dequantize(self, compressed: Any, original_shape: Tuple[int, ...]) -> torch.Tensor:
#         quantized_obj, e8m0_scale, orig_shape, padded_shape = compressed
#         result = mxfp4_dequantize(quantized_obj, e8m0_scale, torch.float32, block_size=32)
#         k = orig_shape[-1]
#         result = result[..., :k]
#         return result.reshape(orig_shape)
#
#     def compression_ratio(self, head_dim: int) -> float:
#         return 16.0 / self.bits_per_elem

# ──────────────────────────────────────────────────────────────────────────────
# NVFP4 method (COMMENTED OUT — uncomment on Blackwell)
# ──────────────────────────────────────────────────────────────────────────────

# class NVFP4Method(QuantMethod):
#     """NVFP4: block_size=16, E4M3 block scale + global scale."""
#
#     def __init__(self):
#         self.name = "nvfp4_b16"
#         # 4 bits per element + 8 bits per 16-element block for scale + amortized global scale
#         self.bits_per_elem = 4.0 + 8.0 / 16.0  # = 4.5 (ignoring tiny global scale)
#
#     def quantize(self, x: torch.Tensor) -> Any:
#         x_2d = x.reshape(-1, x.shape[-1]).contiguous()
#         if x_2d.dtype not in (torch.float16, torch.bfloat16):
#             x_2d = x_2d.to(torch.bfloat16)
#         k = x_2d.shape[-1]
#         # Pad to multiple of 16
#         if k % 16 != 0:
#             pad = 16 - (k % 16)
#             x_2d = F.pad(x_2d, (0, pad))
#         packed_fp4, block_scale, global_scale, m, n = nvfp4_quantize(x_2d)
#         return (packed_fp4, block_scale, global_scale, m, n, x.shape)
#
#     def dequantize(self, compressed: Any, original_shape: Tuple[int, ...]) -> torch.Tensor:
#         packed_fp4, block_scale, global_scale, m, n, orig_shape = compressed
#         result = nvfp4_dequantize(packed_fp4, block_scale, global_scale, m, n)
#         k = orig_shape[-1]
#         result = result[..., :k]
#         return result.reshape(orig_shape)
#
#     def compression_ratio(self, head_dim: int) -> float:
#         return 16.0 / self.bits_per_elem

# ──────────────────────────────────────────────────────────────────────────────
# Timing utility
# ──────────────────────────────────────────────────────────────────────────────


def timed_call(fn: Callable, warmup: int = 5, trials: int = 20) -> Tuple[Any, float]:
    """Run fn, return (result, median_time_us)."""
    # Warmup
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # to microseconds

    times.sort()
    median_us = times[len(times) // 2]
    return result, median_us


# ──────────────────────────────────────────────────────────────────────────────
# Accuracy metrics
# ──────────────────────────────────────────────────────────────────────────────


def compute_metrics(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> Dict[str, float]:
    """Compute accuracy metrics between original and reconstructed tensors."""
    orig_f = original.float().flatten()
    recon_f = reconstructed.float().flatten()

    diff = orig_f - recon_f
    mse = (diff**2).mean().item()
    orig_energy = (orig_f**2).mean().item() + 1e-10
    rel_mse = mse / orig_energy
    max_abs_err = diff.abs().max().item()

    # Cosine similarity
    cos_sim = F.cosine_similarity(orig_f.unsqueeze(0), recon_f.unsqueeze(0)).item()

    return {
        "mse": mse,
        "rel_mse": rel_mse,
        "max_abs_err": max_abs_err,
        "cosine_sim": cos_sim,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────────────────


def run_single_benchmark(
    method: QuantMethod,
    x: torch.Tensor,
    scenario: str,
    distribution: str,
    warmup: int = 5,
    trials: int = 20,
) -> QuantResult:
    """Run a single quantization benchmark."""

    head_dim = x.shape[-1]

    # Time quantization
    compressed, quant_us = timed_call(lambda: method.quantize(x), warmup, trials)

    # Time dequantization
    _, dequant_us = timed_call(
        lambda: method.dequantize(compressed, x.shape), warmup, trials
    )

    # Compute accuracy (single pass, no timing)
    reconstructed = method.dequantize(compressed, x.shape)
    metrics = compute_metrics(x.float(), reconstructed.float())

    return QuantResult(
        method=method.name,
        bits_per_elem=method.bits_per_elem,
        compression_ratio=method.compression_ratio(head_dim),
        mse=metrics["mse"],
        rel_mse=metrics["rel_mse"],
        max_abs_err=metrics["max_abs_err"],
        cosine_sim=metrics["cosine_sim"],
        quant_time_us=quant_us,
        dequant_time_us=dequant_us,
        total_time_us=quant_us + dequant_us,
        shape=tuple(x.shape),
        distribution=distribution,
        scenario=scenario,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test scenarios
# ──────────────────────────────────────────────────────────────────────────────

# Typical KV cache tensor shapes: (num_tokens * num_kv_heads, head_dim)
# We benchmark with flattened (tokens*heads, head_dim) which is the actual
# shape fed to quantization kernels.

MODEL_KV_SCENARIOS = {
    # ── DeepSeek-V3 ──
    # 61 layers, 128 KV heads, head_dim=128
    # (MLA uses compressed latent but standard attention path has head_dim=128)
    "DeepSeek-V3_short_1k": {
        "num_tokens": 1024,
        "num_kv_heads": 128,
        "head_dim": 128,
        "desc": "DeepSeek-V3 (128 KV heads, d=128), 1K ctx",
    },
    "DeepSeek-V3_medium_8k": {
        "num_tokens": 8192,
        "num_kv_heads": 128,
        "head_dim": 128,
        "desc": "DeepSeek-V3 (128 KV heads, d=128), 8K ctx",
    },
    "DeepSeek-V3_long_32k": {
        "num_tokens": 32768,
        "num_kv_heads": 128,
        "head_dim": 128,
        "desc": "DeepSeek-V3 (128 KV heads, d=128), 32K ctx",
    },
    # ── GPT-OSS 120B ──
    # 96 attention heads, 8 KV heads (GQA), head_dim=128, 80 layers
    "GPT-OSS-120B_short_1k": {
        "num_tokens": 1024,
        "num_kv_heads": 8,
        "head_dim": 128,
        "desc": "GPT-OSS 120B (8 KV heads GQA, d=128), 1K ctx",
    },
    "GPT-OSS-120B_medium_8k": {
        "num_tokens": 8192,
        "num_kv_heads": 8,
        "head_dim": 128,
        "desc": "GPT-OSS 120B (8 KV heads GQA, d=128), 8K ctx",
    },
    "GPT-OSS-120B_long_32k": {
        "num_tokens": 32768,
        "num_kv_heads": 8,
        "head_dim": 128,
        "desc": "GPT-OSS 120B (8 KV heads GQA, d=128), 32K ctx",
    },
    # ── Qwen3-235B-A22B (MoE) ──
    # 64 attention heads, 4 KV heads (GQA), head_dim=128, 94 layers
    "Qwen3-235B_short_1k": {
        "num_tokens": 1024,
        "num_kv_heads": 4,
        "head_dim": 128,
        "desc": "Qwen3-235B (4 KV heads GQA, d=128), 1K ctx",
    },
    "Qwen3-235B_medium_8k": {
        "num_tokens": 8192,
        "num_kv_heads": 4,
        "head_dim": 128,
        "desc": "Qwen3-235B (4 KV heads GQA, d=128), 8K ctx",
    },
    "Qwen3-235B_long_32k": {
        "num_tokens": 32768,
        "num_kv_heads": 4,
        "head_dim": 128,
        "desc": "Qwen3-235B (4 KV heads GQA, d=128), 32K ctx",
    },
    # ── Qwen3-32B (Dense) ──
    # 64 attention heads, 8 KV heads (GQA), head_dim=128, 64 layers
    "Qwen3-32B_short_1k": {
        "num_tokens": 1024,
        "num_kv_heads": 8,
        "head_dim": 128,
        "desc": "Qwen3-32B (8 KV heads GQA, d=128), 1K ctx",
    },
    "Qwen3-32B_medium_8k": {
        "num_tokens": 8192,
        "num_kv_heads": 8,
        "head_dim": 128,
        "desc": "Qwen3-32B (8 KV heads GQA, d=128), 8K ctx",
    },
    "Qwen3-32B_long_32k": {
        "num_tokens": 32768,
        "num_kv_heads": 8,
        "head_dim": 128,
        "desc": "Qwen3-32B (8 KV heads GQA, d=128), 32K ctx",
    },
}

SYNTHETIC_SCENARIOS = {
    # Small tensors (decode-like, single token per head)
    "synthetic_small": {
        "num_tokens": 1,
        "num_kv_heads": 32,
        "head_dim": 128,
        "desc": "Small: 1 token × 32 heads × 128d (decode step)",
    },
    # Medium tensors
    "synthetic_medium": {
        "num_tokens": 512,
        "num_kv_heads": 32,
        "head_dim": 128,
        "desc": "Medium: 512 tokens × 32 heads × 128d",
    },
    # Large tensors
    "synthetic_large": {
        "num_tokens": 4096,
        "num_kv_heads": 64,
        "head_dim": 128,
        "desc": "Large: 4096 tokens × 64 heads × 128d",
    },
    # Very large (stress test)
    "synthetic_xlarge": {
        "num_tokens": 16384,
        "num_kv_heads": 128,
        "head_dim": 128,
        "desc": "XLarge: 16384 tokens × 128 heads × 128d",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Pretty-print results
# ──────────────────────────────────────────────────────────────────────────────


def print_results_table(results: List[QuantResult], title: str = ""):
    """Print a nicely formatted results table."""
    if title:
        print(f"\n{'═' * 140}")
        print(f"  {title}")
        print(f"{'═' * 140}")

    # Group by scenario
    scenarios = {}
    for r in results:
        key = (r.scenario, r.distribution)
        if key not in scenarios:
            scenarios[key] = []
        scenarios[key].append(r)

    for (scenario, dist), group in scenarios.items():
        shape_str = "×".join(str(s) for s in group[0].shape)
        print(f"\n┌─ {scenario} | dist={dist} | shape=({shape_str})")
        print(
            f"│ {'Method':<28s} │ {'Bits':>5s} │ {'Comp':>5s} │ "
            f"{'RelMSE':>10s} │ {'MSE':>10s} │ {'MaxErr':>8s} │ {'CosSim':>8s} │ "
            f"{'Q(μs)':>9s} │ {'DQ(μs)':>9s} │ {'Total':>9s} │"
        )
        print(f"│{'─' * 28}─┼─{'─' * 5}─┼─{'─' * 5}─┼─"
              f"{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 8}─┼─{'─' * 8}─┼─"
              f"{'─' * 9}─┼─{'─' * 9}─┼─{'─' * 9}─┤")

        # Sort by compression ratio (lower = more compression)
        group.sort(key=lambda r: -r.compression_ratio)

        for r in group:
            print(
                f"│ {r.method:<28s} │ {r.bits_per_elem:>5.2f} │ {r.compression_ratio:>4.1f}x │ "
                f"{r.rel_mse:>10.6f} │ {r.mse:>10.6f} │ {r.max_abs_err:>8.4f} │ {r.cosine_sim:>8.6f} │ "
                f"{r.quant_time_us:>9.1f} │ {r.dequant_time_us:>9.1f} │ {r.total_time_us:>9.1f} │"
            )
    print(f"└{'─' * 139}┘")


def print_summary_table(results: List[QuantResult]):
    """Print a summary comparison across all scenarios."""
    print(f"\n{'═' * 120}")
    print(f"  SUMMARY: Average metrics across all scenarios")
    print(f"{'═' * 120}")

    # Aggregate by method
    method_stats: Dict[str, List[QuantResult]] = {}
    for r in results:
        if r.method not in method_stats:
            method_stats[r.method] = []
        method_stats[r.method].append(r)

    print(
        f"{'Method':<28s} │ {'Bits':>5s} │ {'Comp':>5s} │ "
        f"{'Avg RelMSE':>12s} │ {'Avg CosSim':>10s} │ "
        f"{'Avg Q(μs)':>10s} │ {'Avg DQ(μs)':>10s} │ {'Avg Total':>10s} │ {'#Tests':>6s}"
    )
    print(f"{'─' * 28}─┼─{'─' * 5}─┼─{'─' * 5}─┼─"
          f"{'─' * 12}─┼─{'─' * 10}─┼─"
          f"{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 6}")

    # Sort methods by average compression ratio
    sorted_methods = sorted(
        method_stats.items(),
        key=lambda kv: -sum(r.compression_ratio for r in kv[1]) / len(kv[1]),
    )

    for method_name, rs in sorted_methods:
        n = len(rs)
        avg_bits = sum(r.bits_per_elem for r in rs) / n
        avg_comp = sum(r.compression_ratio for r in rs) / n
        avg_rmse = sum(r.rel_mse for r in rs) / n
        avg_cos = sum(r.cosine_sim for r in rs) / n
        avg_q = sum(r.quant_time_us for r in rs) / n
        avg_dq = sum(r.dequant_time_us for r in rs) / n
        avg_tot = sum(r.total_time_us for r in rs) / n
        print(
            f"{method_name:<28s} │ {avg_bits:>5.2f} │ {avg_comp:>4.1f}x │ "
            f"{avg_rmse:>12.6f} │ {avg_cos:>10.6f} │ "
            f"{avg_q:>10.1f} │ {avg_dq:>10.1f} │ {avg_tot:>10.1f} │ {n:>6d}"
        )


def save_csv(results: List[QuantResult], path: str):
    """Save results to CSV."""
    fieldnames = [
        "scenario",
        "distribution",
        "shape",
        "method",
        "bits_per_elem",
        "compression_ratio",
        "mse",
        "rel_mse",
        "max_abs_err",
        "cosine_sim",
        "quant_time_us",
        "dequant_time_us",
        "total_time_us",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "scenario": r.scenario,
                    "distribution": r.distribution,
                    "shape": str(r.shape),
                    "method": r.method,
                    "bits_per_elem": f"{r.bits_per_elem:.2f}",
                    "compression_ratio": f"{r.compression_ratio:.2f}",
                    "mse": f"{r.mse:.8f}",
                    "rel_mse": f"{r.rel_mse:.8f}",
                    "max_abs_err": f"{r.max_abs_err:.6f}",
                    "cosine_sim": f"{r.cosine_sim:.8f}",
                    "quant_time_us": f"{r.quant_time_us:.1f}",
                    "dequant_time_us": f"{r.dequant_time_us:.1f}",
                    "total_time_us": f"{r.total_time_us:.1f}",
                }
            )
    print(f"\n[INFO] Results saved to {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="KV Cache Quantization Benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--trials", type=int, default=20, help="Timed trial iterations")
    parser.add_argument("--csv", type=str, default=None, help="Path to save CSV results")
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Skip synthetic tensor benchmarks",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model-specific KV cache benchmarks",
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=["dense", "sparse_80", "kv_normal", "kv_lognormal"],
        help="Data distributions to test",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer scenarios and distributions",
    )
    args = parser.parse_args()

    # ── Print system info ──
    print(f"{'═' * 80}")
    print(f"  KV Cache Quantization Benchmark")
    print(f"{'═' * 80}")
    print(f"  Device:      {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch:     {torch.__version__}")
    print(f"  CUDA:        {torch.version.cuda}")
    cc = torch.cuda.get_device_capability()
    print(f"  Compute:     SM{cc[0]}{cc[1]}")
    print(f"  FP8:         {'✓' if FP8_AVAILABLE else '✗'}")
    print(f"  MXFP4:       {'✓' if MXFP4_AVAILABLE else '✗ (commented out, needs Blackwell)'}")
    print(f"  NVFP4:       {'✓' if NVFP4_AVAILABLE else '✗ (commented out, needs Blackwell)'}")
    print(f"  TurboQuant:  ✓ (fast JIT Hadamard: {'✓' if _FAST_HADAMARD_AVAILABLE else '✗ SLOW'})")
    print(f"  Warmup:      {args.warmup}")
    print(f"  Trials:      {args.trials}")
    print(f"{'═' * 80}")

    if args.quick:
        args.distributions = ["dense", "sparse_80"]

    # ── Build method list ──
    methods: List[QuantMethod] = []

    if FP8_AVAILABLE:
        methods.append(FP8BlockwiseMethod(group_size=128))

    # MXFP4 — uncomment on Blackwell
    # if MXFP4_AVAILABLE:
    #     methods.append(MXFP4Method())

    # NVFP4 — uncomment on Blackwell
    # if NVFP4_AVAILABLE:
    #     methods.append(NVFP4Method())

    # TurboQuant at different bit-widths
    for bits in [4, 3.5, 3, 2.5, 2, 1]:
        methods.append(TurboQuantMethod(bits, mode="mse"))
    # Also test prod mode at 4-bit (uses QJL residual)
    methods.append(TurboQuantMethod(4, mode="prod"))

    if not methods:
        print("[ERROR] No quantization methods available!")
        sys.exit(1)

    print(f"\n  Methods ({len(methods)}):")
    for m in methods:
        print(f"    - {m.name}  ({m.bits_per_elem:.2f} bits/elem)")

    all_results: List[QuantResult] = []

    # ──────────────────────────────────────────────────────────────────
    # Part 1: Synthetic tensor benchmarks
    # ──────────────────────────────────────────────────────────────────
    if not args.skip_synthetic:
        scenarios = SYNTHETIC_SCENARIOS
        if args.quick:
            scenarios = {
                k: v
                for k, v in scenarios.items()
                if k in ("synthetic_small", "synthetic_large")
            }

        for scenario_name, cfg in scenarios.items():
            num_tokens = cfg["num_tokens"]
            num_kv_heads = cfg["num_kv_heads"]
            head_dim = cfg["head_dim"]
            flat_rows = num_tokens * num_kv_heads
            shape = (flat_rows, head_dim)

            for dist in args.distributions:
                print(f"\n[RUN] {scenario_name} | dist={dist} | "
                      f"shape=({flat_rows}, {head_dim}) = "
                      f"{flat_rows * head_dim * 2 / 1024:.1f} KB")

                x = generate_tensor(shape, distribution=dist, dtype=torch.bfloat16)

                for method in methods:
                    try:
                        r = run_single_benchmark(
                            method, x, scenario_name, dist,
                            warmup=args.warmup, trials=args.trials,
                        )
                        all_results.append(r)
                        print(
                            f"  {method.name:<28s}  relMSE={r.rel_mse:.6f}  "
                            f"cos={r.cosine_sim:.6f}  "
                            f"Q={r.quant_time_us:.0f}μs  DQ={r.dequant_time_us:.0f}μs"
                        )
                    except Exception as e:
                        print(f"  {method.name:<28s}  ERROR: {e}")

                del x
                torch.cuda.empty_cache()

        print_results_table(
            [r for r in all_results if r.scenario.startswith("synthetic")],
            "SYNTHETIC TENSOR BENCHMARKS",
        )

    # ──────────────────────────────────────────────────────────────────
    # Part 2: Model-specific KV cache benchmarks
    # ──────────────────────────────────────────────────────────────────
    if not args.skip_models:
        scenarios = MODEL_KV_SCENARIOS
        if args.quick:
            # Only test one model at medium context
            scenarios = {
                k: v
                for k, v in scenarios.items()
                if "medium" in k and ("DeepSeek" in k or "Qwen3-32B" in k)
            }

        # For model scenarios, use the most realistic distributions
        model_distributions = ["kv_normal", "kv_lognormal"]
        if "dense" in args.distributions:
            model_distributions.append("dense")

        for scenario_name, cfg in scenarios.items():
            num_tokens = cfg["num_tokens"]
            num_kv_heads = cfg["num_kv_heads"]
            head_dim = cfg["head_dim"]
            flat_rows = num_tokens * num_kv_heads
            shape = (flat_rows, head_dim)
            size_mb = flat_rows * head_dim * 2 / (1024 * 1024)

            for dist in model_distributions:
                print(
                    f"\n[RUN] {scenario_name} | {cfg['desc']} | dist={dist} | "
                    f"shape=({flat_rows}, {head_dim}) = {size_mb:.1f} MB"
                )

                x = generate_tensor(shape, distribution=dist, dtype=torch.bfloat16)

                for method in methods:
                    try:
                        r = run_single_benchmark(
                            method, x, scenario_name, dist,
                            warmup=args.warmup, trials=args.trials,
                        )
                        all_results.append(r)
                        print(
                            f"  {method.name:<28s}  relMSE={r.rel_mse:.6f}  "
                            f"cos={r.cosine_sim:.6f}  "
                            f"Q={r.quant_time_us:.0f}μs  DQ={r.dequant_time_us:.0f}μs"
                        )
                    except Exception as e:
                        print(f"  {method.name:<28s}  ERROR: {e}")

                del x
                torch.cuda.empty_cache()

        print_results_table(
            [r for r in all_results if not r.scenario.startswith("synthetic")],
            "MODEL-SPECIFIC KV CACHE BENCHMARKS",
        )

    # ──────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────
    if all_results:
        print_summary_table(all_results)

    # ──────────────────────────────────────────────────────────────────
    # Theoretical compression ratio table
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  THEORETICAL COMPRESSION RATIOS (head_dim=128, BF16 baseline)")
    print(f"{'═' * 70}")
    print(f"  {'Method':<30s} │ {'Bits/elem':>10s} │ {'Compression':>12s}")
    print(f"  {'─' * 30}─┼─{'─' * 10}─┼─{'─' * 12}")

    if FP8_AVAILABLE:
        m = FP8BlockwiseMethod(128)
        print(f"  {'FP8 blockwise (g=128)':<30s} │ {m.bits_per_elem:>10.2f} │ {m.compression_ratio(128):>11.2f}x")

    # MXFP4 theoretical
    mxfp4_bpe = 4.0 + 8.0 / 32.0
    print(f"  {'MXFP4 (b=32) [disabled]':<30s} │ {mxfp4_bpe:>10.2f} │ {16.0 / mxfp4_bpe:>11.2f}x")

    # NVFP4 theoretical
    nvfp4_bpe = 4.0 + 8.0 / 16.0
    print(f"  {'NVFP4 (b=16) [disabled]':<30s} │ {nvfp4_bpe:>10.2f} │ {16.0 / nvfp4_bpe:>11.2f}x")

    for bits in [4, 3.5, 3, 2.5, 2, 1]:
        cr = compute_compression_ratio(128, bits)
        is_mixed, bh, bl = parse_bits(bits)
        label = f"TurboQuant {bits}b" + (" (mixed)" if is_mixed else "")
        print(f"  {label:<30s} │ {float(bits):>10.2f} │ {cr:>11.2f}x")

    cr_prod = compute_compression_ratio(128, 4, mode="prod")
    print(f"  {'TurboQuant 4b (prod/QJL)':<30s} │ {'4.00':>10s} │ {cr_prod:>11.2f}x")

    # ──────────────────────────────────────────────────────────────────
    # Save CSV
    # ──────────────────────────────────────────────────────────────────
    if args.csv and all_results:
        save_csv(all_results, args.csv)

    print(f"\n{'═' * 80}")
    print(f"  Benchmark complete. {len(all_results)} tests run.")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()

