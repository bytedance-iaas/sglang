from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from sgl_kernel.scalar_type import ScalarType
    from tvm_ffi.module import Module

@cache_once
def _jit_deepep_moe_wna16_marlin_direct_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "deepep_moe_wna16_marlin_direct",
        *args,
        cuda_files=["gemm/marlin_moe/moe_wna16_marlin.cuh"],
        cuda_wrappers=[
            (
                "deepep_moe_wna16_marlin_direct_gemm",
                f"deepep_moe_wna16_marlin_direct_gemm<{args}>",
            )
        ],
    )


def _or_empty(
    t: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return t if t is not None else torch.empty(0, device=device, dtype=dtype)


@debug_kernel_api
def deepep_moe_wna16_marlin_direct_gemm(
    a: torch.Tensor,
    c_or_none: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_bias_or_none: Optional[torch.Tensor],
    b_scales: torch.Tensor,
    global_scale_or_none: Optional[torch.Tensor],
    b_zeros_or_none: Optional[torch.Tensor],
    g_idx_or_none: Optional[torch.Tensor],
    active_expert_ids_or_none: Optional[torch.Tensor],
    active_expert_count_or_none: Optional[torch.Tensor],
    launch_num_experts: int,
    workspace: torch.Tensor,
    c_tmp_or_none: Optional[torch.Tensor],
    masked_m: torch.Tensor,
    expected_m: Optional[int],
    moe_block_size: int,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_w13_stage: bool,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    device = a.device
    num_experts = a.shape[0]

    if c_or_none is not None:
        c = c_or_none
    else:
        c = torch.empty((num_experts, size_m, size_n), dtype=a.dtype, device=device)

    if size_m == 0 or num_experts == 0:
        return c

    has_bias = b_bias_or_none is not None and b_bias_or_none.numel() > 0
    has_zp = b_zeros_or_none is not None and b_zeros_or_none.numel() > 0
    num_groups = b_scales.size(1)
    group_size = size_k // num_groups if num_groups > 1 else -1
    c_tmp = _or_empty(c_tmp_or_none, device, torch.float32)
    g_idx_t = _or_empty(g_idx_or_none, device, torch.int32)
    active_expert_ids_t = _or_empty(active_expert_ids_or_none, device, torch.int32)
    active_expert_count_t = _or_empty(active_expert_count_or_none, device, torch.int32)
    b_zeros_t = _or_empty(b_zeros_or_none, device, a.dtype)
    b_bias_t = _or_empty(b_bias_or_none, device, a.dtype)
    global_scale_t = _or_empty(global_scale_or_none, device, a.dtype)

    module = _jit_deepep_moe_wna16_marlin_direct_module(a.dtype)
    module.deepep_moe_wna16_marlin_direct_gemm(
        a,
        c,
        b_q_weight,
        b_bias_t,
        b_scales,
        global_scale_t,
        b_zeros_t,
        g_idx_t,
        active_expert_ids_t,
        active_expert_count_t,
        int(launch_num_experts),
        workspace,
        masked_m,
        -1 if expected_m is None else int(expected_m),
        c_tmp,
        moe_block_size,
        b_q_type.id,
        num_experts,
        size_m,
        size_n,
        size_k,
        is_w13_stage,
        has_bias,
        is_k_full,
        has_zp,
        num_groups,
        group_size,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )
    return c
