import torch
import triton
import triton.language as tl

@triton.jit
def fused_qkv_quant_pad_kernel(
    q_ptr, k_ptr, v_ptr,
    qp_ptr, kp_ptr, vp_ptr,
    sq_ptr, sk_ptr, sv_ptr,
    M, N, aligned_N,
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = rm[:, None] < M
    mask_n_aligned = rn[None, :] < aligned_N
    mask_n_ori = rn[None, :] < N
    
    # Load scales (scalars)
    sq = tl.load(sq_ptr)
    sk = tl.load(sk_ptr)
    sv = tl.load(sv_ptr)
    
    iq = 1.0 / (sq + 1e-12)
    ik = 1.0 / (sk + 1e-12)
    iv = 1.0 / (sv + 1e-12)
    
    off_i = rm[:, None] * stride_im + rn[None, :] * stride_in
    off_o = rm[:, None] * stride_om + rn[None, :] * stride_on
    
    # Process Q
    xq = tl.load(q_ptr + off_i, mask=mask_m & mask_n_ori, other=0.0)
    tl.store(qp_ptr + off_o, (xq * iq).to(tl.float8e4nv), mask=mask_m & mask_n_aligned)
    
    # Process K
    xk = tl.load(k_ptr + off_i, mask=mask_m & mask_n_ori, other=0.0)
    tl.store(kp_ptr + off_o, (xk * ik).to(tl.float8e4nv), mask=mask_m & mask_n_aligned)
    
    # Process V
    xv = tl.load(v_ptr + off_i, mask=mask_m & mask_n_ori, other=0.0)
    tl.store(vp_ptr + off_o, (xv * iv).to(tl.float8e4nv), mask=mask_m & mask_n_aligned)

def fused_qkv_per_tensor_quant_pad(q, k, v, qp, kp, vp, sq, sk, sv):
    """
    q, k, v: (tokens, heads, head_dim) - BF16
    qp, kp, vp: (tokens, heads, aligned_head_dim) - FP8
    sq, sk, sv: (1,) - Float32
    """
    # 1. Calculate scales in Python (very fast)
    sq.copy_(torch.max(torch.abs(q)) / 448.0)
    sk.copy_(torch.max(torch.abs(k)) / 448.0)
    sv.copy_(torch.max(torch.abs(v)) / 448.0)
    
    M = q.shape[0] * q.shape[1]
    N = q.shape[2]
    aligned_N = qp.shape[2]
    
    # 2. Launch Fused QKV Triton kernel
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 128 if aligned_N <= 128 else 64
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(aligned_N, BLOCK_SIZE_N))
    
    fused_qkv_quant_pad_kernel[grid](
        q, k, v,
        qp, kp, vp,
        sq, sk, sv,
        M, N, aligned_N,
        q.stride(1), q.stride(2),
        qp.stride(1), qp.stride(2),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=8,
    )
