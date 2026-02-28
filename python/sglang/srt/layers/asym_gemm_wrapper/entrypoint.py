import logging
from contextlib import contextmanager
from typing import Any, Optional, Tuple

import torch

from sglang.srt.layers.asym_gemm_wrapper import compile_utils
from sglang.srt.layers.asym_gemm_wrapper.configurer import (  # noqa: F401
    ASYMGEMM_BLACKWELL,
    ASYMGEMM_SCALE_UE8M0,
    ENABLE_JIT_ASYMGEMM,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

if ENABLE_JIT_ASYMGEMM:
    import asym_gemm
    from asym_gemm.utils.layout import get_mn_major_tma_aligned_tensor  # noqa: F401

_SANITY_CHECK = get_bool_env_var("SGLANG_ASYMGEMM_SANITY_CHECK")


def grouped_gemm_nt_f8f8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
):
    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    kernel_type = compile_utils.AsymGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.asym_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        with configure_asym_gemm_num_sms(
            overlap_args.num_sms if overlap_args is not None else None
        ):
            return asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(
                lhs,
                rhs,
                out,
                masked_m,
                expected_m,
                **(
                    dict(
                        enable_overlap=True,
                        max_block_n=max_block_n,
                        signal=overlap_args.signal,
                    )
                    if overlap_args is not None
                    else {}
                ),
            )


def grouped_gemm_nt_f8f8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
):
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = compile_utils.AsymGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.asym_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(lhs, rhs, out, m_indices)


def grouped_gemm_nt_bf16bf16bf16_masked(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
):
    num_groups, _, k = lhs.shape
    _, n, _ = rhs.shape
    kernel_type = compile_utils.AsymGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED

    with compile_utils.asym_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        asym_gemm.m_grouped_bf16_asym_gemm_nt_masked(
            lhs,
            rhs,
            out,
            masked_m,
            expected_m,
        )


def _m_indices_to_offsets_experts(m_indices: torch.Tensor):
    """Convert m_indices (per-token expert IDs) to offsets/experts/list_size format.

    This function matches the C++ build_offsets_experts_from_indices() logic in
    AsymGEMM (sm100_fp8_asym_gemm_1d1d.hpp:83-113).

    Args:
        m_indices: 1D tensor of shape [M] where each element is the expert ID for that token.
                   -1 is treated as a sentinel value (invalid expert).

    Returns:
        offsets: 1D tensor of shape [list_size] containing start index for each expert segment.
        experts: 1D tensor of shape [list_size] containing expert ID for each segment.
        list_size: Number of expert segments (including sentinel).
    """
    # Move to CPU for processing
    m_indices_cpu = m_indices.cpu()
    mi = m_indices_cpu.tolist()

    # Build offsets/experts on CPU
    m = len(mi)
    offsets_list = []
    experts_list = []

    if m > 0:
        # Emit first token if valid (not sentinel -1)
        if mi[0] != -1:
            current_expert = mi[0]
            start_idx = 0
            for i in range(1, m):
                if mi[i] != current_expert:
                    # Emit segment for current_expert
                    offsets_list.append(start_idx)
                    experts_list.append(current_expert)
                    start_idx = i
                    current_expert = mi[i]
            # Emit last segment
            offsets_list.append(start_idx)
            experts_list.append(current_expert)

    # Append sentinel
    offsets_list.append(m)
    experts_list.append(-1)

    list_size = len(offsets_list)

    offsets = torch.tensor(offsets_list, device=m_indices.device, dtype=torch.int32)
    experts = torch.tensor(experts_list, device=m_indices.device, dtype=torch.int32)

    return offsets, experts, list_size


def grouped_gemm_nt_bf16bf16bf16_contig(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
    m_indices: torch.Tensor,
):
    m, k = lhs.shape
    num_groups, n, _ = rhs.shape
    kernel_type = compile_utils.AsymGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG

    with compile_utils.asym_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        # Convert m_indices to offsets/experts/list_size format for AsymGEMM API
        offsets, experts, list_size = _m_indices_to_offsets_experts(m_indices)
        asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
            lhs, rhs, out, offsets, experts, list_size
        )


def update_asym_gemm_config(gpu_id: int, server_args: ServerArgs):
    compile_utils.update_asym_gemm_config(gpu_id, server_args)


@contextmanager
def configure_asym_gemm_num_sms(num_sms):
    if num_sms is None:
        yield
    else:
        original_num_sms = asym_gemm.get_num_sms()
        asym_gemm.set_num_sms(num_sms)
        try:
            yield
        finally:
            asym_gemm.set_num_sms(original_num_sms)


def _sanity_check_input(x_fp8: Tuple[torch.Tensor, torch.Tensor]):
    if not _SANITY_CHECK:
        return

    x, x_scale = x_fp8

    if x_scale.dtype == torch.int:
        return

    from sglang.srt.layers.quantization.fp8_utils import ceil_to_ue8m0

    x_scale_ceil = ceil_to_ue8m0(x_scale)
    assert torch.all(x_scale == x_scale_ceil), f"{x_scale=} {x_scale_ceil=}"
