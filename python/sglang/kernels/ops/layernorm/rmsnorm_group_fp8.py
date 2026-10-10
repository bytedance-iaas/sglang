"""RMSNorm with BF16 and block-FP8 outputs for SM90 Q-LoRA projections."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_group_fp8_kernel(
    X,
    Weight,
    Y,
    Q,
    Scale,
    SX: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    K: tl.constexpr,
    EPS: tl.constexpr,
    GROUP: tl.constexpr,
    UE8M0: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * SX + col, col < K, 0).to(tl.float32)
    w = tl.load(Weight + col, col < K, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / K + EPS)
    y = (x * inv * w).to(tl.bfloat16)
    tl.store(Y + row * K + col, y, col < K)
    # Quantize the rounded BF16 row, also used by the indexer, not the FP32 norm.
    values = tl.reshape(y.to(tl.float32), (BLOCK // GROUP, GROUP))
    amax = tl.maximum(tl.max(tl.abs(values), 1), 1.0e-10)
    scale = amax * (1.0 / 448.0)
    if UE8M0:
        # SGLang's row-major UE8M0 path stores powers of two in FP32.
        bits = scale.to(tl.int32, bitcast=True)
        exp = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = (exp << 23).to(tl.float32, bitcast=True)
        multiplier = ((254 - exp) << 23).to(tl.float32, bitcast=True)
    else:
        # Match per_token_group_quant's multiplier (448 / amax).
        multiplier = 448.0 / amax
    quant = tl.minimum(tl.maximum(values * multiplier[:, None], -448.0), 448.0)
    quant = tl.reshape(quant, (BLOCK,)).to(tl.float8e4nv)
    tl.store(Q + row * K + col, quant, col < K)
    groups = tl.arange(0, BLOCK // GROUP)
    tl.store(Scale + row * SS0 + groups * SS1, scale, groups < K // GROUP)


def rmsnorm_group_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    group_size: int,
    column_major_scales: bool = False,
    scale_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return BF16 norm, FP8 values and FP32 group scales without padding tokens.

    DeepGEMM uses group-128 column-major scales with a TMA-aligned group stride.
    Humming / Triton UE8M0 use group-32 row-major FP32 powers of two. This is not
    the packed or swizzled MXFP8 scale format.
    """
    assert x.ndim == 2 and x.stride(1) == 1
    m, k = x.shape
    assert group_size in (32, 128) and k % group_size == 0 and k > 0
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_cuda and weight.device == x.device
    assert weight.shape == (k,) and weight.is_contiguous()
    assert not (scale_ue8m0 and column_major_scales)
    y = torch.empty((m, k), device=x.device, dtype=torch.bfloat16)
    q = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
    if column_major_scales:
        s = torch.empty(
            (k // group_size, triton.cdiv(m, 4) * 4),
            device=x.device,
            dtype=torch.float32,
        ).T[:m]
    else:
        s = torch.empty((m, k // group_size), device=x.device, dtype=torch.float32)
    if m:
        _rmsnorm_group_fp8_kernel[(m,)](
            x,
            weight,
            y,
            q,
            s,
            SX=x.stride(0),
            SS0=s.stride(0),
            SS1=s.stride(1),
            K=k,
            EPS=eps,
            GROUP=group_size,
            UE8M0=scale_ue8m0,
            BLOCK=triton.next_power_of_2(k),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return y, q, s
