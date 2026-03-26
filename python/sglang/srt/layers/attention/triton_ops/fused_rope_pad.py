import torch
import triton
import triton.language as tl

# =============================================================================
# 方案：All-in-SRAM Fused RoPE & Quantization
# 适用场景：小图 (Tokens < 1280)，一个 Block 处理一个 Head
# =============================================================================

@triton.jit
def fused_rope_quant_kernel(
    q_ptr, cos_ptr, sin_ptr, out_q_ptr, scale_ptr,
    M, H, N, padded_N,
    stride_qm, stride_qh, stride_qn,
    stride_cm, stride_cn,
    stride_sm, stride_sn,
    stride_om, stride_oh, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # 每个 Block 处理一个独立的 Head
    pid_h = tl.program_id(0)
    
    # 准备索引
    rm = tl.arange(0, BLOCK_SIZE_M)
    rn = tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = rm < M
    mask_n = rn < N
    
    # 1. 第一次加载：计算该 Head 的全局 Max (abs)
    # q 形状: (M, H, N)
    q_offsets = rm[:, None] * stride_qm + pid_h * stride_qh + rn[None, :] * stride_qn
    q = tl.load(q_ptr + q_offsets, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # 在寄存器中求 Max
    abs_q = tl.abs(q.to(tl.float32))
    max_val = tl.max(abs_q)
    
    # 计算并存储该 Head 的 Scale
    scale = max_val / 448.0
    tl.store(scale_ptr + pid_h, scale)
    
    inv_scale = 1.0 / (scale + 1e-12)
    
    # 2. 计算 RoPE
    half_N = N // 2
    
    # 加载 Cos/Sin
    cos_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    sin_offsets = rm[:, None] * stride_sm + rn[None, :] * stride_sn
    c = tl.load(cos_ptr + cos_offsets, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    s = tl.load(sin_ptr + sin_offsets, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # 计算 rotate_half (在寄存器中操作)
    # 为了简化，我们直接重新构造旋转后的 q
    # Triton 中可以使用索引变换，但最稳妥的是直接根据 half_N 逻辑
    # 我们重新加载一次旋转位置的数据（或者在寄存器里做 shuffle，但 2D tensor shuffle 较复杂）
    # 考虑到数据已经在 L1/Shared Memory 中，二次 load 很快
    rn_rotated = tl.where(rn < half_N, rn + half_N, rn - half_N)
    q_rotated_offsets = rm[:, None] * stride_qm + pid_h * stride_qh + rn_rotated[None, :] * stride_qn
    q_rotated = tl.load(q_ptr + q_rotated_offsets, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    sign = tl.where(rn < half_N, -1.0, 1.0)
    q_embed = q * c + (q_rotated * sign) * s
    
    # 3. 量化并写回 FP8 (带 Padding)
    # 此时 q_embed 还在寄存器中
    q_fp8 = (q_embed * inv_scale).to(tl.float8e4nv)
    
    # 写回已 Pad 的 FP8 显存
    out_offsets = rm[:, None] * stride_om + pid_h * stride_oh + rn[None, :] * stride_on
    
    # 处理 Padding 区域写 0
    mask_padded_n = rn < padded_N
    q_fp8_padded = tl.where(rn < N, q_fp8, 0.0)
    
    tl.store(out_q_ptr + out_offsets, q_fp8_padded, mask=mask_m[:, None] & mask_padded_n[None, :])

def fused_rope_quant(q, cos, sin, out_q_fp8, scale):
    """
    q: (M, H, N) - BF16
    cos, sin: (M, N) - BF16
    out_q_fp8: (M, H, padded_N) - FP8
    scale: (H,) - Float32 (Per-head scale)
    """
    M, H, N = q.shape
    padded_N = out_q_fp8.shape[2]
    
    # 约束：BLOCK_SIZE_M 必须覆盖所有 M
    # 如果 M > 1280，此算子不可用，需回退
    BLOCK_SIZE_M = triton.next_power_of_2(M)
    BLOCK_SIZE_N = triton.next_power_of_2(padded_N)
    
    grid = (H,)
    
    fused_rope_quant_kernel[grid](
        q, cos, sin, out_q_fp8, scale,
        M, H, N, padded_N,
        q.stride(0), q.stride(1), q.stride(2),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out_q_fp8.stride(0), out_q_fp8.stride(1), out_q_fp8.stride(2),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=16, # 极致并发延迟隐藏
    )

# =============================================================================
# 辅助算子：原本的 Padding 和 Copy
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def copy_pad_kernel(
    in_ptr, out_ptr,
    M, H, N, padded_N,
    stride_im, stride_ih, stride_in,
    stride_om, stride_oh, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = rm < M
    mask_n = rn < N
    
    in_offsets = rm[:, None] * stride_im + pid_h * stride_ih + rn[None, :] * stride_in
    x = tl.load(in_ptr + in_offsets, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    out_offsets = rm[:, None] * stride_om + pid_h * stride_oh + rn[None, :] * stride_on
    
    mask_padded_n = rn < padded_N
    x_padded = tl.where(rn < N, x, 0.0)
    
    tl.store(out_ptr + out_offsets, x_padded, mask=mask_m[:, None] & mask_padded_n[None, :])

def copy_pad(v, out_v):
    M, H, N = v.shape
    padded_N = out_v.shape[2]
    BLOCK_SIZE_N = triton.next_power_of_2(padded_N)
    
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']), H)
    
    copy_pad_kernel[grid](
        v, out_v,
        M, H, N, padded_N,
        v.stride(0), v.stride(1), v.stride(2),
        out_v.stride(0), out_v.stride(1), out_v.stride(2),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128}, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_rope_pad_kernel(
    q_ptr, cos_ptr, sin_ptr, out_ptr,
    M, H, N, padded_N,
    stride_qm, stride_qh, stride_qn,
    stride_cm, stride_cn,
    stride_sm, stride_sn,
    stride_om, stride_oh, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = rm < M
    mask_n = rn < N
    
    half_N = N // 2
    
    cos_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    sin_offsets = rm[:, None] * stride_sm + rn[None, :] * stride_sn
    c = tl.load(cos_ptr + cos_offsets, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    s = tl.load(sin_ptr + sin_offsets, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    q_offsets = rm[:, None] * stride_qm + pid_h * stride_qh + rn[None, :] * stride_qn
    q = tl.load(q_ptr + q_offsets, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    rn_rotated = tl.where(rn < half_N, rn + half_N, rn - half_N)
    q_rotated_offsets = rm[:, None] * stride_qm + pid_h * stride_qh + rn_rotated[None, :] * stride_qn
    q_rotated = tl.load(q_ptr + q_rotated_offsets, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    sign = tl.where(rn < half_N, -1.0, 1.0)
    q_embed = q * c + (q_rotated * sign) * s
    
    out_offsets = rm[:, None] * stride_om + pid_h * stride_oh + rn[None, :] * stride_on
    mask_padded_n = rn < padded_N
    q_embed_padded = tl.where(rn < N, q_embed, 0.0)
    
    tl.store(out_ptr + out_offsets, q_embed_padded.to(out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_padded_n[None, :])

def fused_rope_pad(q, cos, sin, out_q):
    M, H, N = q.shape
    padded_N = out_q.shape[2]
    BLOCK_SIZE_N = triton.next_power_of_2(padded_N)
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']), H)
    fused_rope_pad_kernel[grid](
        q, cos, sin, out_q,
        M, H, N, padded_N,
        q.stride(0), q.stride(1), q.stride(2),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out_q.stride(0), out_q.stride(1), out_q.stride(2),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
