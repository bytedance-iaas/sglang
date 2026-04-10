from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from sgl_kernel.scalar_type import ScalarType
    from tvm_ffi.module import Module


_MAX_THREAD_N = 256


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


def _get_direct_workspace_size(
    num_experts: int,
    device: torch.device,
) -> int:
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return max(sms * 4 * max(num_experts, 1), 128)


def _get_direct_c_tmp_size(
    num_experts: int,
    size_m: int,
    size_n: int,
    moe_block_size: int,
    device: torch.device,
) -> int:
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    max_per_expert = min(
        size_n * size_m,
        sms * 4 * moe_block_size * _MAX_THREAD_N,
    )
    if moe_block_size == 8:
        max_per_expert *= 2
    return max(max_per_expert * max(num_experts, 1), 1)


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
    masked_m: torch.Tensor,
    expected_m: Optional[int],
    moe_block_size: int,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
    runtime_cache: Optional[dict] = None,
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
    runtime_cache = {} if runtime_cache is None else runtime_cache
    runtime_key = (
        device.index,
        str(a.dtype),
        num_experts,
        size_m,
        -1 if expected_m is None else int(expected_m),
        size_n,
        size_k,
        moe_block_size,
        bool(use_atomic_add),
        bool(use_fp32_reduce),
    )
    runtime = runtime_cache.get(runtime_key)
    workspace_size = _get_direct_workspace_size(num_experts, device)
    logical_m = size_m if expected_m is None or expected_m <= 0 else min(int(expected_m), size_m)
    c_tmp_size = (
        _get_direct_c_tmp_size(num_experts, logical_m, size_n, moe_block_size, device)
        if use_fp32_reduce and not use_atomic_add
        else 0
    )
    if (
        runtime is None
        or runtime["workspace"].numel() < workspace_size
        or runtime["c_tmp"].numel() < c_tmp_size
    ):
        runtime = {
            "workspace": torch.zeros(
                workspace_size,
                dtype=torch.int,
                device=device,
                requires_grad=False,
            ),
            "c_tmp": torch.empty(
                max(c_tmp_size, 0),
                dtype=torch.float32,
                device=device,
            ),
        }
        runtime_cache[runtime_key] = runtime

    workspace = runtime["workspace"]
    c_tmp = runtime["c_tmp"]

    g_idx_t = _or_empty(g_idx_or_none, device, torch.int32)
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
