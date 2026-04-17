from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.kernel_api_logging import debug_kernel_api

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@dataclass(frozen=True)
class MaskedSiluAndMulKernelConfig:
    block_m: int
    block_n: int
    vec_elems: int

    @property
    def threads_n(self) -> int:
        assert self.block_n % self.vec_elems == 0
        return self.block_n // self.vec_elems


DEFAULT_MASKED_SILU_AND_MUL_KERNEL_CONFIG = MaskedSiluAndMulKernelConfig(
    block_m=8,
    block_n=128,
    vec_elems=8,
)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1, "BLOCK_N": 128}, num_warps=4),
            triton.Config({"BLOCK_M": 2, "BLOCK_N": 128}, num_warps=4),
            triton.Config({"BLOCK_M": 4, "BLOCK_N": 128}, num_warps=4),
            triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=4),
            triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=8),
            triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=8),
        ],
        key=["size_n", "logical_m", "expert_num"],
    )
    @triton.jit
    def _masked_silu_and_mul_triton_kernel(
        input_ptr,
        output_ptr,
        masked_m_ptr,
        stride_i0,
        stride_i1,
        stride_o0,
        stride_o1,
        size_n,
        logical_m,
        expert_num,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_e = tl.program_id(2)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

        token_num = tl.load(masked_m_ptr + pid_e)
        mask_n = offs_n < size_n
        mask_m = (offs_m < logical_m) & (offs_m < token_num)
        mask_e = pid_e < expert_num
        mask = mask_e & mask_m[:, None] & mask_n[None, :]

        input_base = input_ptr + pid_e * stride_i0
        output_base = output_ptr + pid_e * stride_o0

        gate_ptrs = input_base + offs_m[:, None] * stride_i1 + offs_n[None, :]
        up_ptrs = gate_ptrs + size_n
        out_ptrs = output_base + offs_m[:, None] * stride_o1 + offs_n[None, :]

        gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)
        out = gate * tl.sigmoid(gate) * up
        tl.store(out_ptrs, out, mask=mask)


def _validate_kernel_config(config: MaskedSiluAndMulKernelConfig) -> None:
    assert config.block_m > 0
    assert config.block_n > 0
    assert config.vec_elems > 0
    assert config.block_n % config.vec_elems == 0
    assert config.block_m * config.threads_n <= 1024


def _default_cuda_kernel_config(
    expert_num: int,
    logical_m: int,
    size_n: int,
) -> MaskedSiluAndMulKernelConfig:
    del expert_num, logical_m, size_n
    return DEFAULT_MASKED_SILU_AND_MUL_KERNEL_CONFIG


def _should_use_triton(expert_num: int, logical_m: int, size_n: int) -> bool:
    if triton is None:
        return False
    return size_n >= 4096 and expert_num * logical_m >= 1536


def _triton_masked_silu_and_mul(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    masked_m: torch.Tensor,
    logical_m: int,
) -> torch.Tensor:
    assert triton is not None

    grid = lambda meta: (
        triton.cdiv(output_tensor.shape[2], meta["BLOCK_N"]),
        triton.cdiv(logical_m, meta["BLOCK_M"]),
        output_tensor.shape[0],
    )
    _masked_silu_and_mul_triton_kernel[grid](
        input_tensor,
        output_tensor,
        masked_m,
        input_tensor.stride(0),
        input_tensor.stride(1),
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor.shape[2],
        logical_m,
        output_tensor.shape[0],
    )
    return output_tensor


@cache_once
def _jit_masked_silu_and_mul_module(
    dtype: torch.dtype,
    block_m: int,
    block_n: int,
    vec_elems: int,
) -> Module:
    threads_n = block_n // vec_elems
    args = make_cpp_args(dtype, block_m, threads_n, vec_elems)
    return load_jit(
        "masked_silu_and_mul",
        *args,
        cuda_files=["elementwise/masked_silu_and_mul.cuh"],
        cuda_wrappers=[
            (
                "masked_silu_and_mul",
                f"masked_silu_and_mul<{args}>",
            )
        ],
    )

@debug_kernel_api
def masked_silu_and_mul(
    input: torch.Tensor,
    output: torch.Tensor,
    masked_m: torch.Tensor,
    token_upper_hint: Optional[int] = None,
    use_fp32_accum: bool = True,
    kernel_config: Optional[MaskedSiluAndMulKernelConfig] = None,
    backend: Literal["auto", "cuda", "triton"] = "auto",
) -> torch.Tensor:
    assert len(input.shape) == 3
    assert len(output.shape) == 3
    assert input.shape[0] == masked_m.shape[0]
    assert input.shape[0] == output.shape[0]
    assert input.shape[1] == output.shape[1]
    assert input.shape[2] == output.shape[2] * 2
    assert input.stride(2) == 1
    assert output.stride(2) == 1

    expert_num = input.shape[0]
    logical_m = (
        input.shape[1]
        if token_upper_hint is None or token_upper_hint <= 0
        else min(int(token_upper_hint), input.shape[1])
    )
    size_n = output.shape[2]
    if expert_num == 0 or size_n == 0:
        return output

    assert backend in ("auto", "cuda", "triton")

    selected_backend = backend
    if selected_backend == "auto":
        if kernel_config is not None:
            selected_backend = "cuda"
        else:
            selected_backend = (
                "triton" if _should_use_triton(expert_num, logical_m, size_n) else "cuda"
            )

    if selected_backend == "triton":
        assert triton is not None, "Triton is not available"
        assert use_fp32_accum, "Triton path currently expects fp32 accumulation"
        return _triton_masked_silu_and_mul(input, output, masked_m, logical_m)

    config = kernel_config or _default_cuda_kernel_config(expert_num, logical_m, size_n)
    _validate_kernel_config(config)
    module = _jit_masked_silu_and_mul_module(
        input.dtype,
        config.block_m,
        config.block_n,
        config.vec_elems,
    )
    module.masked_silu_and_mul(
        input,
        output,
        masked_m,
        int(expert_num),
        int(logical_m),
        int(size_n),
        bool(use_fp32_accum),
    )
    return output
