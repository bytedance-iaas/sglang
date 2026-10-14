"""Opt-in BF16 O-A projection for DSV4.1 DP-attention verify on H20."""

import torch
import triton
import triton.language as tl


@triton.jit
def _wo_a_bf16_8groups(A, B, C, M: tl.constexpr):
    # More output tiles than the default 64-CTA cuBLAS kernel, without split-K.
    m = (tl.program_id(0) // 16) * 64 + tl.arange(0, 64)
    n = (tl.program_id(0) % 16) * 64 + tl.arange(0, 64)
    g = tl.program_id(1)
    k = tl.arange(0, 64)
    acc = tl.full((64, 64), 0, tl.float32)
    for _ in range(64):
        a = tl.load(
            A + m[:, None] * 32768 + g * 4096 + k[None, :], m[:, None] < M, other=0
        )
        b = tl.load(B + g * 1024 * 4096 + n[None, :] * 4096 + k[:, None])
        acc = tl.dot(a, b, acc)
        k += 64
    # Token-major output is ready for wo_b; einsum otherwise needs a reshape copy.
    tl.store(C + m[:, None] * 8192 + g * 1024 + n[None, :], acc, m[:, None] < M)


def wo_a_bf16_sm90(o: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """[M, 8, 4096] x [8, 1024, 4096] -> contiguous [M, 8, 1024].

    The caller gates architecture and target-verify mode. This kernel uses BF16
    operands and FP32 accumulation; no additional activation quantization.
    """
    assert o.ndim == 3 and 96 <= o.shape[0] <= 192
    assert o.shape[1:] == (8, 4096) and weight.shape == (8, 1024, 4096)
    assert o.dtype == weight.dtype == torch.bfloat16
    assert o.is_contiguous() and weight.is_contiguous()
    result = torch.empty((o.shape[0], 8, 1024), device=o.device, dtype=o.dtype)
    if o.shape[0] < 160:
        # Smaller verify batches favor cuBLAS. Write token-major output directly
        # to avoid the einsum output's reshape copy before wo_b.
        with torch.no_grad():
            torch.bmm(
                o.transpose(0, 1), weight.transpose(1, 2), out=result.transpose(0, 1)
            )
        return result
    _wo_a_bf16_8groups[(triton.cdiv(o.shape[0], 64) * 16, 8)](
        o, weight, result, o.shape[0], num_warps=4, num_stages=4
    )
    return result
