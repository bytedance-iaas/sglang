from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers import asym_gemm_wrapper
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import (
    dispose_tensor,
    is_cuda,
    is_hip,
    is_npu,
)

_is_hip = is_hip()
_is_npu = is_npu()
_is_cuda = is_cuda()

if not (_is_npu or _is_hip) and _is_cuda:
    from sgl_kernel import silu_and_mul


# NVFP4 micro-scale group size along the K dimension.
_NVFP4_GROUP_SIZE = 16
# E2M1 dynamic range (max |x|) for computing E4M3 scale: sf = amax / 6.0.
_E2M1_MAX = 6.0

# Fast CUDA path: flashinfer.fp4_quantize with global_scale=1.0 and
# is_sf_swizzled_layout=False is numerically equivalent to the single-level
# NVFP4 scheme AsymGEMM consumes (per-group E4M3 scales in row-major
# (..., K/16) layout, packed uint8 codes of shape (..., K/2)). Falls back to
# a pure-PyTorch implementation if flashinfer is unavailable.
try:
    from flashinfer import fp4_quantize as _flashinfer_fp4_quantize
except Exception:  # pragma: no cover - fallback path
    _flashinfer_fp4_quantize = None


# Device-local cache for the global_scale=1.0 tensor used by flashinfer's
# fp4_quantize. Allocating inside the forward would trigger a host->device
# copy and is not allowed during CUDA graph capture.
_FP4_GLOBAL_SCALE_ONE_CACHE: dict = {}


def _get_fp4_global_scale_one(device: torch.device) -> torch.Tensor:
    key = str(device)
    cached = _FP4_GLOBAL_SCALE_ONE_CACHE.get(key)
    if cached is None:
        cached = torch.ones(1, device=device, dtype=torch.float32)
        _FP4_GLOBAL_SCALE_ONE_CACHE[key] = cached
    return cached


# E2M1 magnitude bucket thresholds (midpoints between representable values).
# Only used by the PyTorch fallback. Cached per device so they aren't
# re-allocated on every forward; the host-to-device copy inside
# torch.tensor([...], device=...) is not permitted during CUDA graph capture.
_E2M1_THRESHOLDS_CACHE: dict = {}


def _get_e2m1_thresholds(device: torch.device) -> torch.Tensor:
    key = (str(device), torch.float32)
    cached = _E2M1_THRESHOLDS_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
            device=device,
            dtype=torch.float32,
        )
        _E2M1_THRESHOLDS_CACHE[key] = cached
    return cached


@dataclass
class AsymGemmFp4RunnerInput(RunnerInput):
    hidden_states: torch.Tensor  # packed FP4 (uint8), shape (M, K//2) or (G, M, K//2)
    hidden_states_scale: torch.Tensor  # E4M3 scales, shape (M, K/16) or (G, M, K/16)
    use_masked_gemm: bool
    masked_m: Optional[torch.Tensor] = None
    expected_m: Optional[int] = None
    m_indices: Optional[torch.Tensor] = None
    offsets: Optional[torch.Tensor] = None
    experts: Optional[torch.Tensor] = None
    list_size: Optional[torch.Tensor] = None

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ASYM_GEMM


@dataclass
class AsymGemmFp4RunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ASYM_GEMM


@dataclass
class AsymGemmFp4MoeQuantInfo(MoeQuantInfo):
    """Quantization info for NVFP4 MoE execution via AsymGEMM.

    Weights are packed FP4 (uint8, two E2M1 values per byte). Scales are
    per-group E4M3 bytes with group_size=16 along K. The AsymGEMM kernel
    internally re-packs the scales into its TMA-aligned uint32 layout.
    """

    w13_weight: torch.Tensor  # (num_experts, 2*N, K//2) uint8 (packed FP4)
    w2_weight: torch.Tensor   # (num_experts, K,   N//2) uint8 (packed FP4)
    w13_scale: torch.Tensor   # (num_experts, 2*N, K/16) float8_e4m3fn
    w2_scale: torch.Tensor    # (num_experts, K,   N/16) float8_e4m3fn


def _quantize_bf16_to_nvfp4_e4m3(
    x: torch.Tensor, group_size: int = _NVFP4_GROUP_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row NVFP4 quantization producing packed FP4 + E4M3 scales.

    Matches the layout produced by ``_quantize_a_nvfp4_e4m3`` in AsymGEMM's
    ``tests/test_nvfp4.py``: returns ``(packed_u8[M, K//2], scale_e4m3[M, K/16])``.
    The AsymGEMM kernel handles the TMA-aligned uint32 repack internally.

    Inputs are reshaped to 2D (M, K) before quantization; higher-rank inputs
    (e.g. ``(num_groups, M, K)``) are flattened over the leading dims and
    reshaped back before returning.
    """
    assert x.is_cuda
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert x.shape[-1] % group_size == 0

    leading = x.shape[:-1]
    k = x.shape[-1]
    x2d = x.reshape(-1, k)
    sf_k = k // group_size

    if _flashinfer_fp4_quantize is not None and group_size == _NVFP4_GROUP_SIZE:
        # Single-level NVFP4 (global_scale = 1.0) with un-swizzled per-group
        # E4M3 scales: byte-for-byte the layout AsymGEMM's grouped FP4 kernel
        # consumes, and >80x faster than the pure-PyTorch encoder on
        # decode-sized tensors.
        if x.dtype != torch.bfloat16:
            x2d = x2d.to(torch.bfloat16)
        gs = _get_fp4_global_scale_one(x.device)
        packed, sf = _flashinfer_fp4_quantize(
            x2d,
            gs,
            sf_vec_size=group_size,
            is_sf_swizzled_layout=False,
        )
        # flashinfer returns the scale buffer as raw uint8; reinterpret as
        # float8_e4m3fn (zero-copy view) so AsymGEMM sees the expected dtype.
        if sf.dtype != torch.float8_e4m3fn:
            sf = sf.view(torch.float8_e4m3fn)
        packed = packed.reshape(*leading, k // 2)
        sf = sf.reshape(*leading, sf_k)
        return packed, sf

    return _quantize_bf16_to_nvfp4_e4m3_pytorch(x, group_size=group_size)


def _quantize_bf16_to_nvfp4_e4m3_pytorch(
    x: torch.Tensor, group_size: int = _NVFP4_GROUP_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch fallback encoder used when flashinfer is unavailable."""
    leading = x.shape[:-1]
    k = x.shape[-1]
    x2d = x.reshape(-1, k)
    m = x2d.shape[0]
    sf_k = k // group_size

    x_groups = x2d.to(torch.float32).view(m, sf_k, group_size)
    amax = x_groups.abs().amax(dim=-1).clamp_min_(1e-4)
    # NVFP4 canonical scale = amax / E2M1_MAX, stored as E4M3.
    sf_e4m3 = (amax / _E2M1_MAX).to(torch.float8_e4m3fn)
    sf_decoded = sf_e4m3.to(torch.float32).clamp_min_(1e-12)

    # Quantize values to E2M1. Use the nearest representable magnitude with
    # tie-to-even (consistent with the CPP/numpy reference encoding).
    x_scaled = x_groups / sf_decoded.unsqueeze(-1)
    sign_bits = (x_scaled < 0).to(torch.uint8) << 3
    ax = x_scaled.abs().clamp_max_(_E2M1_MAX)
    # Magnitude thresholds (midpoints between E2M1 values).
    thresholds = _get_e2m1_thresholds(x.device)
    mag_idx = torch.bucketize(ax, thresholds).to(torch.uint8)
    # Zero is encoded as +0 (no sign bit) to mirror the reference encoder.
    sign_bits = torch.where(mag_idx == 0, torch.zeros_like(sign_bits), sign_bits)
    codes = (sign_bits | mag_idx).view(m, k)

    # Pack two 4-bit codes into one byte: low nibble = even index, high = odd.
    codes = codes.view(m, k // 2, 2)
    packed = (codes[..., 0] & 0x0F) | ((codes[..., 1] & 0x0F) << 4)
    packed = packed.to(torch.uint8).contiguous()

    packed = packed.view(*leading, k // 2)
    sf_e4m3 = sf_e4m3.view(*leading, sf_k)
    return packed, sf_e4m3


@triton.jit
def _masked_silu_and_mul_kernel(
    gateup_ptr,   # (G*m, N) bf16
    out_ptr,      # (G*m, N//2) bf16
    masked_m_ptr, # (G,) int32
    m,            # int — rows per group
    N,            # int — gateup width
    HALF_N,       # int — N // 2
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    g = pid // m
    row = pid % m
    active = tl.load(masked_m_ptr + g)
    if row >= active:
        return

    row_off = pid.to(tl.int64) * N
    out_off = pid.to(tl.int64) * HALF_N
    for start in tl.range(0, HALF_N, BLOCK_K):
        k = start + tl.arange(0, BLOCK_K)
        mask = k < HALF_N
        gate = tl.load(gateup_ptr + row_off + k, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(gateup_ptr + row_off + HALF_N + k, mask=mask, other=0.0).to(tl.float32)
        val = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16)
        tl.store(out_ptr + out_off + k, val, mask=mask)


@triton.jit
def _masked_fp4_quant_kernel(
    input_ptr,    # (G*m, K) bf16
    packed_ptr,   # (G*m, K//2) uint8
    scale_ptr,    # (G*m, K//GROUP_SIZE) float8_e4m3fn
    masked_m_ptr, # (G,) int32
    m,            # int — rows per group
    K,            # int — last dim of input
    GROUP_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_GRP: tl.constexpr,     # == BLOCK_K // GROUP_SIZE
    HALF_BK: tl.constexpr,     # == BLOCK_K // 2
):
    pid = tl.program_id(0)
    g = pid // m
    row = pid % m
    active = tl.load(masked_m_ptr + g)
    if row >= active:
        return

    row_off = pid.to(tl.int64) * K
    pack_off = pid.to(tl.int64) * (K // 2)
    sf_off = pid.to(tl.int64) * (K // GROUP_SIZE)

    E2M1_MAX: tl.constexpr = 6.0

    for start in tl.range(0, K, BLOCK_K):
        k = start + tl.arange(0, BLOCK_K)
        mask = k < K
        vals = tl.load(input_ptr + row_off + k, mask=mask, other=0.0).to(tl.float32)

        vals_g = tl.reshape(vals, [NUM_GRP, GROUP_SIZE])
        amax = tl.max(tl.abs(vals_g), axis=1)
        amax = tl.maximum(amax, 1e-4)
        sf_f32 = amax / E2M1_MAX
        sf_e4m3 = sf_f32.to(tl.float8e4nv)
        sf_dec = tl.maximum(sf_e4m3.to(tl.float32), 1e-12)

        scaled = vals_g / sf_dec[:, None]
        ax = tl.minimum(tl.abs(scaled), E2M1_MAX)

        codes = tl.zeros([NUM_GRP, GROUP_SIZE], dtype=tl.uint8)
        codes = tl.where(ax >= 0.25, codes + 1, codes)
        codes = tl.where(ax >= 0.75, codes + 1, codes)
        codes = tl.where(ax >= 1.25, codes + 1, codes)
        codes = tl.where(ax >= 1.75, codes + 1, codes)
        codes = tl.where(ax >= 2.5, codes + 1, codes)
        codes = tl.where(ax >= 3.5, codes + 1, codes)
        codes = tl.where(ax >= 5.0, codes + 1, codes)

        neg = scaled < 0
        sign = tl.where(neg, tl.full([NUM_GRP, GROUP_SIZE], 8, dtype=tl.uint8),
                        tl.zeros([NUM_GRP, GROUP_SIZE], dtype=tl.uint8))
        sign = tl.where(codes == 0, tl.zeros([NUM_GRP, GROUP_SIZE], dtype=tl.uint8), sign)
        codes = codes | sign
        codes_flat = tl.reshape(codes, [BLOCK_K])

        codes_pairs = tl.reshape(codes_flat, [HALF_BK, 2])
        weights = tl.where(tl.arange(0, 2) == 0, 1, 16).to(tl.int32)
        packed = tl.sum(codes_pairs.to(tl.int32) * weights[None, :], axis=1).to(tl.uint8)

        pk = start // 2 + tl.arange(0, HALF_BK)
        pk_mask = pk < (K // 2)
        tl.store(packed_ptr + pack_off + pk, packed, mask=pk_mask)

        sk = start // GROUP_SIZE + tl.arange(0, NUM_GRP)
        sk_mask = sk < (K // GROUP_SIZE)
        tl.store(scale_ptr + sf_off + sk, sf_e4m3, mask=sk_mask)


@triton.jit
def _fused_masked_silu_mul_fp4_quant_kernel(
    gateup_ptr,   # (G*m, N) bf16 — gate in [0, N/2), up in [N/2, N)
    packed_ptr,   # (G*m, N/4) uint8
    scale_ptr,    # (G*m, N/2/GROUP_SIZE) float8_e4m3fn
    masked_m_ptr, # (G,) int32
    m,            # rows per group
    N,            # gateup width (2 * output_dim)
    HALF_N,       # N // 2 (output dim)
    GROUP_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_GRP: tl.constexpr,     # == BLOCK_K // GROUP_SIZE
    HALF_BK: tl.constexpr,     # == BLOCK_K // 2
):
    pid = tl.program_id(0)
    g = pid // m
    row = pid % m
    active = tl.load(masked_m_ptr + g)
    if row >= active:
        return

    row_off = pid.to(tl.int64) * N
    pack_off = pid.to(tl.int64) * (HALF_N // 2)
    sf_off = pid.to(tl.int64) * (HALF_N // GROUP_SIZE)

    E2M1_MAX: tl.constexpr = 6.0

    for start in tl.range(0, HALF_N, BLOCK_K):
        k = start + tl.arange(0, BLOCK_K)
        mask = k < HALF_N

        gate = tl.load(gateup_ptr + row_off + k, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(gateup_ptr + row_off + HALF_N + k, mask=mask, other=0.0).to(tl.float32)
        vals = gate * tl.sigmoid(gate) * up

        vals_g = tl.reshape(vals, [NUM_GRP, GROUP_SIZE])
        amax = tl.max(tl.abs(vals_g), axis=1)
        amax = tl.maximum(amax, 1e-4)
        sf_f32 = amax / E2M1_MAX
        sf_e4m3 = sf_f32.to(tl.float8e4nv)
        sf_dec = tl.maximum(sf_e4m3.to(tl.float32), 1e-12)

        scaled = vals_g / sf_dec[:, None]
        ax = tl.minimum(tl.abs(scaled), E2M1_MAX)

        codes = tl.zeros([NUM_GRP, GROUP_SIZE], dtype=tl.uint8)
        codes = tl.where(ax >= 0.25, codes + 1, codes)
        codes = tl.where(ax >= 0.75, codes + 1, codes)
        codes = tl.where(ax >= 1.25, codes + 1, codes)
        codes = tl.where(ax >= 1.75, codes + 1, codes)
        codes = tl.where(ax >= 2.5, codes + 1, codes)
        codes = tl.where(ax >= 3.5, codes + 1, codes)
        codes = tl.where(ax >= 5.0, codes + 1, codes)

        neg = scaled < 0
        sign = tl.where(neg, tl.full([NUM_GRP, GROUP_SIZE], 8, dtype=tl.uint8),
                        tl.zeros([NUM_GRP, GROUP_SIZE], dtype=tl.uint8))
        sign = tl.where(codes == 0, tl.zeros([NUM_GRP, GROUP_SIZE], dtype=tl.uint8), sign)
        codes = codes | sign
        codes_flat = tl.reshape(codes, [BLOCK_K])

        codes_pairs = tl.reshape(codes_flat, [HALF_BK, 2])
        weights = tl.where(tl.arange(0, 2) == 0, 1, 16).to(tl.int32)
        packed = tl.sum(codes_pairs.to(tl.int32) * weights[None, :], axis=1).to(tl.uint8)

        pk = start // 2 + tl.arange(0, HALF_BK)
        pk_mask = pk < (HALF_N // 2)
        tl.store(packed_ptr + pack_off + pk, packed, mask=pk_mask)

        sk = start // GROUP_SIZE + tl.arange(0, NUM_GRP)
        sk_mask = sk < (HALF_N // GROUP_SIZE)
        tl.store(scale_ptr + sf_off + sk, sf_e4m3, mask=sk_mask)


def _silu_mul_and_fp4_quant_masked(
    gateup_output: torch.Tensor,
    masked_m: torch.Tensor,
    group_size: int = _NVFP4_GROUP_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused masked SiLU-and-Mul + NVFP4 quantization in a single kernel.

    Reads the gateup BF16 output once, applies SiLU*Up, quantizes to FP4,
    and writes packed uint8 + E4M3 scales — no intermediate BF16 buffer.
    Skips inactive expert groups (masked_m[g] == 0).
    """
    num_groups, m, n = gateup_output.shape
    half_n = n // 2
    k_packed = half_n // 2
    sf_k = half_n // group_size
    device = gateup_output.device

    packed_out = torch.empty(
        (num_groups, m, k_packed),
        device=device,
        dtype=torch.uint8,
    )
    scale_out = torch.empty(
        (num_groups, m, sf_k),
        device=device,
        dtype=torch.float8_e4m3fn,
    )

    block_k = min(half_n, 1024)
    assert block_k % group_size == 0
    assert block_k % 2 == 0

    _fused_masked_silu_mul_fp4_quant_kernel[(num_groups * m,)](
        gateup_output.view(-1, n),
        packed_out.view(-1, k_packed),
        scale_out.view(-1, sf_k),
        masked_m,
        m,
        n,
        half_n,
        GROUP_SIZE=group_size,
        BLOCK_K=block_k,
        NUM_GRP=block_k // group_size,
        HALF_BK=block_k // 2,
    )

    return packed_out, scale_out


class AsymGemmFp4RunnerCore(MoeRunnerCore):
    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        assert self.config.activation == "silu"
        assert self.config.is_gated

    def run(
        self,
        runner_input: AsymGemmFp4RunnerInput,
        quant_info: AsymGemmFp4MoeQuantInfo,
        running_state: dict,
    ) -> AsymGemmFp4RunnerOutput:
        if not runner_input.use_masked_gemm:
            hidden_states = self._run_contiguous_gemm(
                runner_input, quant_info, running_state
            )
        else:
            hidden_states = self._run_masked_gemm(
                runner_input, quant_info, running_state
            )
        return AsymGemmFp4RunnerOutput(hidden_states=hidden_states)

    def _run_contiguous_gemm(
        self,
        runner_input: AsymGemmFp4RunnerInput,
        quant_info: AsymGemmFp4MoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        all_tokens = running_state["all_tokens"]
        hidden_states_device = running_state["hidden_states_device"]
        hidden_states_shape = running_state["hidden_states_shape"]

        # N is the packed-major output dim of the gateup weights.
        # w13_weight is (E, 2*N, K//2); the gate-up output has 2*N bf16 columns
        # which silu_and_mul reduces to N.
        w13_n = quant_info.w13_weight.size(1)
        K = hidden_states_shape[1]

        gateup_output = torch.empty(
            (all_tokens, w13_n),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        asym_gemm_wrapper.grouped_gemm_nt_fp4fp4bf16_contig(
            (hidden_states, hidden_states_scale),
            (quant_info.w13_weight, quant_info.w13_scale),
            gateup_output,
            runner_input.offsets,
            runner_input.experts,
            runner_input.list_size,
        )

        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        # SiLU-and-mul: (M, 2*N) bf16 -> (M, N) bf16
        down_in_bf16 = torch.empty(
            (all_tokens, w13_n // 2),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        silu_and_mul(gateup_output.view(-1, w13_n), down_in_bf16)
        del gateup_output

        # Re-quantize activations to NVFP4 for the down-projection GEMM.
        down_in_fp4, down_in_scale = _quantize_bf16_to_nvfp4_e4m3(down_in_bf16)
        del down_in_bf16

        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        asym_gemm_wrapper.grouped_gemm_nt_fp4fp4bf16_contig(
            (down_in_fp4, down_in_scale),
            (quant_info.w2_weight, quant_info.w2_scale),
            down_output,
            runner_input.offsets,
            runner_input.experts,
            runner_input.list_size,
        )

        return down_output

    def _run_masked_gemm(
        self,
        runner_input: AsymGemmFp4RunnerInput,
        quant_info: AsymGemmFp4MoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        masked_m = runner_input.masked_m
        expected_m = runner_input.expected_m

        w13_weight = quant_info.w13_weight
        w2_weight = quant_info.w2_weight
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale

        hidden_states_device = running_state["hidden_states_device"]

        num_groups, m, k_packed = hidden_states.shape
        n = w13_weight.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )

        asym_gemm_wrapper.grouped_gemm_nt_fp4fp4bf16_masked(
            (hidden_states, hidden_states_scale),
            (w13_weight, w13_scale),
            gateup_output,
            masked_m,
            expected_m,
        )
        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        # Fused masked SiLU-and-Mul + NVFP4 quantization: only processes rows
        # where masked_m[g] > 0, skipping inactive expert groups entirely.
        down_in_fp4, down_in_scale = _silu_mul_and_fp4_quant_masked(
            gateup_output, masked_m,
        )
        del gateup_output

        n2 = w2_weight.shape[1]
        down_output = torch.empty(
            (num_groups, m, n2), device=hidden_states_device, dtype=torch.bfloat16
        )
        asym_gemm_wrapper.grouped_gemm_nt_fp4fp4bf16_masked(
            (down_in_fp4, down_in_scale),
            (w2_weight, w2_scale),
            down_output,
            masked_m,
            expected_m,
        )

        return down_output

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ASYM_GEMM
