from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

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

logger = logging.getLogger(__name__)

# Set SGLANG_FP4_GEMM_DEBUG=1 to force-enable kernel I/O logging for this module.
import os as _os
if _os.environ.get("SGLANG_FP4_GEMM_DEBUG", "0") == "1":
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("[%(name)s][%(levelname)s] %(message)s"))
        logger.addHandler(_h)
del _os

# NVFP4 micro-scale group size along the K dimension.
_NVFP4_GROUP_SIZE = 16
# E2M1 dynamic range (max |x|) for computing E4M3 scale: sf = amax / 6.0.
_E2M1_MAX = 6.0
# float8_e4m3fn max representable value; values above this overflow to NaN (0x7F).
_E4M3FN_MAX = 448.0


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

    w13_weight_scale_2 / w2_weight_scale_2 are per-expert global float32
    post-GEMM correction factors (ModelOpt two-level quantization).  They
    are applied row-wise after each grouped GEMM to restore the correct
    magnitude.  None means no correction (legacy checkpoints).
    """

    w13_weight: torch.Tensor  # (num_experts, 2*N, K//2) uint8 (packed FP4)
    w2_weight: torch.Tensor   # (num_experts, K,   N//2) uint8 (packed FP4)
    w13_scale: torch.Tensor   # (num_experts, 2*N, K/16) float8_e4m3fn
    w2_scale: torch.Tensor    # (num_experts, K,   N/16) float8_e4m3fn
    w13_weight_scale_2: Optional[torch.Tensor] = None  # (num_local_experts,) float32
    w2_weight_scale_2: Optional[torch.Tensor] = None   # (num_local_experts,) float32


def _quantize_bf16_to_nvfp4_e4m3(
    x: torch.Tensor, group_size: int = _NVFP4_GROUP_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row NVFP4 quantization producing packed FP4 + E4M3 scales.

    Matches the layout produced by ``_quantize_a_nvfp4_e4m3`` in AsymGEMM's
    ``tests/test_nvfp4.py``: returns ``(packed_u8[M, K//2], scale_e4m3[M, K/16])``.
    The AsymGEMM kernel handles the TMA-aligned uint32 repack internally.

    Inputs are reshaped to 2D (M, K) before quantization; 3D inputs
    (num_groups, M, K) are flattened over the leading dims and reshaped back.
    """
    assert x.is_cuda
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert x.shape[-1] % group_size == 0

    leading = x.shape[:-1]
    k = x.shape[-1]
    x2d = x.reshape(-1, k)
    m = x2d.shape[0]
    sf_k = k // group_size

    x_groups = x2d.to(torch.float32).view(m, sf_k, group_size)
    amax = x_groups.abs().amax(dim=-1).clamp_min_(1e-4)
    # NVFP4 canonical scale = amax / E2M1_MAX, stored as E4M3.
    sf_e4m3 = (amax / _E2M1_MAX).clamp(max=_E4M3FN_MAX).to(torch.float8_e4m3fn)
    sf_decoded = sf_e4m3.to(torch.float32).clamp_min_(1e-12)

    # Quantize values to E2M1. Use the nearest representable magnitude with
    # tie-to-even (consistent with the CPP/numpy reference encoding).
    x_scaled = x_groups / sf_decoded.unsqueeze(-1)
    sign_bits = (x_scaled < 0).to(torch.uint8) << 3
    ax = x_scaled.abs().clamp_max_(_E2M1_MAX)
    # Magnitude thresholds (midpoints between E2M1 values).
    thresholds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=x.device,
        dtype=torch.float32,
    )
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


def _tensor_stats(t: torch.Tensor) -> str:
    """Return a compact stats string for a tensor (shape, dtype, nan, min, max)."""
    if t.numel() == 0:
        return f"shape={tuple(t.shape)} dtype={t.dtype} (empty)"
    f32 = t.to(torch.float32)
    has_nan = f32.isnan().any().item()
    return (
        f"shape={tuple(t.shape)} dtype={t.dtype} "
        f"nan={has_nan} min={f32.min().item():.4g} max={f32.max().item():.4g}"
    )


def _scale_stats(s: torch.Tensor) -> str:
    """Return stats for an E4M3 scale tensor, including NaN-byte count."""
    nan_count = s.view(torch.uint8).eq(0x7F).sum().item()
    f32 = s.to(torch.float32)
    return (
        f"shape={tuple(s.shape)} dtype={s.dtype} "
        f"nan_bytes={nan_count} min={f32.min().item():.4g} max={f32.max().item():.4g}"
    )


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
        m_indices = runner_input.m_indices
        
        # N is the packed-major output dim of the gateup weights.
        # w13_weight is (E, 2*N, K//2); the gate-up output has 2*N bf16 columns
        # which silu_and_mul reduces to N.
        w13_n = quant_info.w13_weight.size(1)
        K = hidden_states_shape[1]

        # Zero-init so padding rows (outside the kernel's offset range) are 0
        # rather than uninitialized garbage.  Uninitialized padding can produce
        # NaN scales after silu_and_mul+quantize, which the block-scale TMA
        # loads may then expose into real-token output tiles.
        gateup_output = torch.zeros(
            (all_tokens, w13_n),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[contig][gateup] act      : %s", _tensor_stats(hidden_states)
            )
            logger.debug(
                "[contig][gateup] act_scale: %s", _scale_stats(hidden_states_scale)
            )
            logger.debug(
                "[contig][gateup] w13      : device=%s shape=%s",
                quant_info.w13_weight.device, tuple(quant_info.w13_weight.shape),
            )
            logger.debug(
                "[contig][gateup] w13_scale: %s", _scale_stats(quant_info.w13_scale)
            )
            logger.debug(
                "[contig][gateup] offsets=%s experts=%s list_size=%s",
                runner_input.offsets.tolist(),
                runner_input.experts.tolist(),
                runner_input.list_size.tolist(),
            )
        asym_gemm_wrapper.grouped_gemm_nt_fp4fp4bf16_contig(
            (hidden_states, hidden_states_scale),
            (quant_info.w13_weight, quant_info.w13_scale),
            gateup_output,
            runner_input.offsets,
            runner_input.experts,
            runner_input.list_size,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[contig][gateup] output   : %s", _tensor_stats(gateup_output)
            )

        # Apply per-expert global post-GEMM scale (ModelOpt two-level quant).
        if quant_info.w13_weight_scale_2 is not None:
            total_real = int(runner_input.list_size.sum().item())
            per_expert_scale = quant_info.w13_weight_scale_2[runner_input.experts]
            per_token_scale = per_expert_scale.repeat_interleave(runner_input.list_size)
            gateup_output[:total_real] *= per_token_scale.to(gateup_output.dtype).unsqueeze(1)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[contig][gateup] w13_scale_2 applied: per_expert_scale=%s",
                    per_expert_scale.tolist(),
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

        down_output = torch.zeros(
            (all_tokens, K),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[contig][down]   act      : %s", _tensor_stats(down_in_fp4)
            )
            logger.debug(
                "[contig][down]   act_scale: %s", _scale_stats(down_in_scale)
            )
            logger.debug(
                "[contig][down]   w2       : device=%s shape=%s",
                quant_info.w2_weight.device, tuple(quant_info.w2_weight.shape),
            )
            logger.debug(
                "[contig][down]   w2_scale : %s", _scale_stats(quant_info.w2_scale)
            )
        asym_gemm_wrapper.grouped_gemm_nt_fp4fp4bf16_contig(
            (down_in_fp4, down_in_scale),
            (quant_info.w2_weight, quant_info.w2_scale),
            down_output,
            runner_input.offsets,
            runner_input.experts,
            runner_input.list_size,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[contig][down]   output   : %s", _tensor_stats(down_output)
            )

        # Apply per-expert global post-GEMM scale for the down projection.
        if quant_info.w2_weight_scale_2 is not None:
            total_real = int(runner_input.list_size.sum().item())
            per_expert_scale = quant_info.w2_weight_scale_2[runner_input.experts]
            per_token_scale = per_expert_scale.repeat_interleave(runner_input.list_size)
            down_output[:total_real] *= per_token_scale.to(down_output.dtype).unsqueeze(1)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[contig][down]   w2_scale_2 applied: per_expert_scale=%s",
                    per_expert_scale.tolist(),
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
        # Zero-init so uncomputed rows (outside offsets range) are 0, not garbage.
        # silu(0)=0 → zero FP4 codes → no NaN propagation through the down GEMM.
        gateup_output = torch.zeros(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[masked][gateup] act      : %s", _tensor_stats(hidden_states)
            )
            logger.debug(
                "[masked][gateup] act_scale: %s", _scale_stats(hidden_states_scale)
            )
            logger.debug(
                "[masked][gateup] w13      : device=%s shape=%s",
                w13_weight.device, tuple(w13_weight.shape),
            )
            logger.debug(
                "[masked][gateup] w13_scale: %s", _scale_stats(w13_scale)
            )
            logger.debug(
                "[masked][gateup] masked_m=%s expected_m=%d offsets=%s experts=%s list_size=%s",
                masked_m.tolist(), expected_m,
                runner_input.offsets.tolist(),
                runner_input.experts.tolist(),
                runner_input.list_size.tolist(),
            )
        asym_gemm_wrapper.grouped_gemm_nt_fp4fp4bf16_masked(
            (hidden_states, hidden_states_scale),
            (w13_weight, w13_scale),
            gateup_output,
            masked_m,
            expected_m,
            runner_input.offsets,
            runner_input.experts,
            runner_input.list_size,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[masked][gateup] output   : %s", _tensor_stats(gateup_output)
            )

        # Apply per-expert global post-GEMM scale (ModelOpt two-level quant).
        # gateup_output shape: (num_groups, m, n) — group g corresponds to
        # experts[g].  Scale is broadcast over the m and n dims.
        if quant_info.w13_weight_scale_2 is not None:
            scales = quant_info.w13_weight_scale_2[runner_input.experts]  # (G,)
            gateup_output *= scales.to(gateup_output.dtype)[:, None, None]

        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        # SiLU-and-mul over the whole padded buffer; padding rows are dropped
        # by the masked downstream GEMM, so masked_m isn't needed here and
        # avoiding ``masked_m[i].item()`` keeps CUDA-graph capture working.
        down_in_bf16 = torch.empty(
            (
                gateup_output.shape[0] * gateup_output.shape[1],
                gateup_output.shape[2] // 2,
            ),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        silu_and_mul(
            gateup_output.view(-1, gateup_output.shape[2]),
            down_in_bf16,
        )
        down_in_bf16 = down_in_bf16.view(
            gateup_output.shape[0], gateup_output.shape[1], gateup_output.shape[2] // 2
        )
        del gateup_output

        down_in_fp4, down_in_scale = _quantize_bf16_to_nvfp4_e4m3(down_in_bf16)
        del down_in_bf16

        n2 = w2_weight.shape[1]
        down_output = torch.empty(
            (num_groups, m, n2), device=hidden_states_device, dtype=torch.bfloat16
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[masked][down]   act      : %s", _tensor_stats(down_in_fp4)
            )
            logger.debug(
                "[masked][down]   act_scale: %s", _scale_stats(down_in_scale)
            )
            logger.debug(
                "[masked][down]   w2       : device=%s shape=%s",
                w2_weight.device, tuple(w2_weight.shape),
            )
            logger.debug(
                "[masked][down]   w2_scale : %s", _scale_stats(w2_scale)
            )
        asym_gemm_wrapper.grouped_gemm_nt_fp4fp4bf16_masked(
            (down_in_fp4, down_in_scale),
            (w2_weight, w2_scale),
            down_output,
            masked_m,
            expected_m,
            runner_input.offsets,
            runner_input.experts,
            runner_input.list_size,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[masked][down]   output   : %s", _tensor_stats(down_output)
            )

        # Apply per-expert global post-GEMM scale for the down projection.
        if quant_info.w2_weight_scale_2 is not None:
            scales = quant_info.w2_weight_scale_2[runner_input.experts]  # (G,)
            down_output *= scales.to(down_output.dtype)[:, None, None]

        return down_output

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ASYM_GEMM
