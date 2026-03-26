import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256}, num_warps=8),
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
    # Process tokens in M dimension
    pid_m = tl.program_id(0)
    # Process heads in H dimension
    pid_h = tl.program_id(1)
    
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = rm < M
    mask_n = rn < N
    
    half_N = N // 2
    
    # Load cos and sin
    # cos/sin shape: (M, N)
    cos_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    sin_offsets = rm[:, None] * stride_sm + rn[None, :] * stride_sn
    
    c = tl.load(cos_ptr + cos_offsets, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    s = tl.load(sin_ptr + sin_offsets, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Load q
    # q shape: (M, H, N)
    q_offsets = rm[:, None] * stride_qm + pid_h * stride_qh + rn[None, :] * stride_qn
    q = tl.load(q_ptr + q_offsets, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Calculate rotate_half(q)
    # For i < half_N: rotate_half(q)[i] = -q[i + half_N]
    # For i >= half_N: rotate_half(q)[i] = q[i - half_N]
    
    rn_rotated = tl.where(rn < half_N, rn + half_N, rn - half_N)
    q_rotated_offsets = rm[:, None] * stride_qm + pid_h * stride_qh + rn_rotated[None, :] * stride_qn
    q_rotated = tl.load(q_ptr + q_rotated_offsets, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    sign = tl.where(rn < half_N, -1.0, 1.0)
    q_embed = q * c + (q_rotated * sign) * s
    
    # Store to out_ptr (padded)
    # out shape: (M, H, padded_N)
    out_offsets = rm[:, None] * stride_om + pid_h * stride_oh + rn[None, :] * stride_on
    
    # We must write 0 to the padded area. We can use a mask for padded_N
    mask_padded_n = rn < padded_N
    
    # Pad q_embed with 0 for rn >= N
    q_embed_padded = tl.where(rn < N, q_embed, 0.0)
    
    tl.store(out_ptr + out_offsets, q_embed_padded.to(out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_padded_n[None, :])

def fused_rope_pad(q, cos, sin, out_q):
    """
    q: (M, H, N) - BF16
    cos: (M, N) - BF16
    sin: (M, N) - BF16
    out_q: (M, H, padded_N) - BF16
    """
    M, H, N = q.shape
    padded_N = out_q.shape[2]
    
    # N is usually small (e.g. 72), so we use a power of 2 >= padded_N to ensure we can write zeros to the padded part
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

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256}, num_warps=8),
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