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

_debug_dump_done = False

# def grouped_gemm_nt_f8f8bf16_masked(
#     lhs: Tuple[torch.Tensor, torch.Tensor],
#     rhs: Tuple[torch.Tensor, torch.Tensor],
#     out: torch.Tensor,
#     masked_m: torch.Tensor,
#     expected_m: int,
#     overlap_args: Optional[Any] = None,
#     max_block_n: int = 256,
# ):
#     num_groups, _, k = lhs[0].shape
#     _, n, _ = rhs[0].shape
#     kernel_type = compile_utils.AsymGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED

#     _sanity_check_input(lhs)
#     _sanity_check_input(rhs)

#     offsets, experts, list_size = build_offsets_experts_from_masked_m(
#         masked_m, num_groups
#     )

#     with compile_utils.asym_gemm_execution_hook(
#         expected_m, n, k, num_groups, kernel_type
#     ):
#         with configure_asym_gemm_num_sms(
#             overlap_args.num_sms if overlap_args is not None else None
#         ):
#             active_experts = experts[:list_size - 1].cpu().tolist()

#             debug_this_call = (17 in active_experts) and (106 in active_experts)

#             if debug_this_call:
#                 _PRINTED = True
#                 idx = active_experts.index(106)
#                 print("\nPY DEBUG >>> first call with expert 106")
#                 print("segment idx =", idx)
#                 print("offset range =", offsets[idx].item(), offsets[idx + 1].item())
#                 print("BEFORE out[106,0,:8] =", out[106, 0, :8])

#                 print("lhs[0][106,0,:8] =", lhs[0][106, 0, :8])
#                 print("rhs[0][106,0,:8] =", rhs[0][106, 0, :8])

#                 print("lhs[1][106,0,:] =", lhs[1][106, 0, :])
#                 print("rhs[1][106,0,:] =", rhs[1][106, 0, :])

#                 print("lhs nan/inf =", torch.isnan(lhs[0][106,0].float()).any().item(),
#                     torch.isinf(lhs[0][106,0].float()).any().item())
#                 print("rhs nan/inf =", torch.isnan(rhs[0][106].float()).any().item(),
#                     torch.isinf(rhs[0][106].float()).any().item())
            
#             ret = asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(
#                 lhs,
#                 rhs,
#                 out,
#                 offsets,
#                 experts,
#                 list_size, 
#                 expected_m,
#                 None,
#                 "nk",
#                 False,
#             )

#             if debug_this_call:
#                 torch.cuda.synchronize()
#                 print("AFTER  out[106,0,:8] =", out[106, 0, :8])

#             return ret

        

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

    offsets, experts, list_size = build_offsets_experts_from_masked_m(
        masked_m, num_groups
    )

    # ---------------- debug trigger ----------------
    active_experts = experts[:list_size - 1].cpu().tolist()
    debug_this_call = (17 in active_experts) and (106 in active_experts)

    global _debug_dump_done

    if debug_this_call and not _debug_dump_done:
        _debug_dump_done = True

        print("\n===== DEBUG DUMP TRIGGERED =====")
        print("active_experts:", active_experts)
        print("list_size:", list_size)
        print("expected_m:", expected_m)
        print("masked_m[17]:", masked_m[17].item())
        print("masked_m[106]:", masked_m[106].item())

        torch.save(
            {
                "lhs0": lhs[0].cpu(),
                "lhs1": lhs[1].cpu(),
                "rhs0": rhs[0].cpu(),
                "rhs1": rhs[1].cpu(),
                "masked_m": masked_m.cpu(),
                "offsets": offsets.cpu(),
                "experts": experts.cpu(),
                "list_size": list_size,
                "expected_m": expected_m,
            },
            "asym_gemm_bug_dump.pt",
        )

        print("Dump saved to asym_gemm_bug_dump.pt")
        print("================================\n")
    # ------------------------------------------------

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
                offsets,
                experts,
                list_size, 
                expected_m,
                disable_ue8m0_cast=False,
                # None,
                # "nk",
                # False,
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

# def build_offsets_experts_from_masked_m(
#     masked_m: torch.Tensor,
#     num_groups: int,
#     block_m: int = 128,
# ):
#     offsets = []
#     experts = []
#     curr_offset = 0

#     for g in range(num_groups):
#         v = masked_m[g].item()
#         if v > 0:
#             offsets.append(curr_offset)
#             experts.append(g)
#             curr_offset += ((v + block_m - 1) // block_m) * block_m

#     offsets.append(curr_offset)
#     experts.append(-1)

#     return (
#         torch.tensor(offsets, dtype=torch.int32, device="cuda"),
#         torch.tensor(experts, dtype=torch.int32, device="cuda"),
#         len(offsets),
#     )
# def build_offsets_experts_from_masked_m(masked_m: torch.Tensor, num_groups: int, block_m: int = 128):
#     offsets = []
#     experts = []
#     curr_offset = 0
#     for g in range(num_groups):
#         v = masked_m[g].item()
#         if v > 0:
#             offsets.append(curr_offset)
#             experts.append(g)
#             curr_offset += ((v + block_m - 1) // block_m) * block_m
#     offsets.append(curr_offset)
#     experts.append(-1)
#     return (torch.tensor(offsets, dtype=torch.int32, device='cuda'), 
#             torch.tensor(experts, dtype=torch.int32, device='cuda'), 
#             len(offsets))

def build_offsets_experts_from_masked_m(masked_m: torch.Tensor, num_groups: int, block_m: int = 128):
    """
    Build offsets and experts for sparse m-grouped masked GEMM.

    Generates LOCAL offset pairs for each active expert group. The kernel will add
    (scheduler.current_group_idx * shape_m) to compute the global offset.

    Only groups with masked_m[g] > 0 are included in the output mapping.
    Each active group generates a pair of offsets (start, end) relative to that group.

    Args:
        masked_m: (num_groups,) tensor of actual token counts per group
        num_groups: number of expert groups
        block_m: block alignment for padding (default 128)

    Returns:
        offsets: flat tensor with LOCAL pairs [start_0, end_0, start_1, end_1, ...]
        experts: expert IDs for each active group + terminator (-1)
        list_size: number of experts in output (including terminator)

    Example:
        masked_m = torch.tensor([0, 12, 0, 129]), num_groups = 4
        offsets = [0, 128, 0, 256]  # LOCAL offsets (padded to 128)
        experts = [1, 3, -1]        # 2 active experts + terminator
        list_size = 3

    Note:
        Kernel computes global M index as:
        m_idx = (scheduler.current_group_idx * shape_m) + local_m_idx
    """
    offsets = []
    experts = []

    for g in range(num_groups):
        v = masked_m[g].item()
        if v > 0:  # Only process active groups
            # LOCAL offsets (relative to this group)
            start = 0
            # Pad actual tokens to block_m alignment
            end = ((v + block_m - 1) // block_m) * block_m
            offsets.append(start)
            offsets.append(end)
            experts.append(g)

    # Add terminator expert
    experts.append(-1)

    return (torch.tensor(offsets, dtype=torch.int32, device=masked_m.device),
            torch.tensor(experts, dtype=torch.int32, device=masked_m.device),
            len(experts))


def grouped_gemm_nt_bf16bf16bf16_masked(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    num_groups: int,
):
    num_groups, m_max, k = lhs.shape
    _, n, _ = rhs.shape
    kernel_type = compile_utils.AsymGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED

    # Compact offsets for the asym_gemm kernel
    offsets, experts, list_size = build_offsets_experts_from_masked_m(
        masked_m, num_groups
    )

    with compile_utils.asym_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        asym_gemm.m_grouped_bf16_asym_gemm_nt_masked(
            lhs,
            rhs,
            out,
            offsets,
            experts,
            list_size,
            expected_m,
            compiled_dims="nk",
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
