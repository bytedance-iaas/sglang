import os
import math
import torch

# ----------------------------
# FP8 quant helpers (simple, explicit)
# ----------------------------

FP8_MAX_E4M3FN = 448.0  # max finite value for e4m3fn (commonly used for scaling)

def _ceil_div(a, b):
    return (a + b - 1) // b

@torch.no_grad()
def fp8_quant_per_token_blockwise(x_bf16: torch.Tensor, block_k: int = 128):
    """
    x_bf16: [M, K] bf16
    returns:
      x_q: [M, K] float8_e4m3fn
      x_s: [M, ceil(K/block_k)] float32  (scale per token per K-block)
    dequant approx: x_deq[:, kblock] = x_q[:, kblock].float() * x_s[:, block_id]
    """
    assert x_bf16.dim() == 2
    assert x_bf16.dtype == torch.bfloat16
    M, K = x_bf16.shape
    nb = _ceil_div(K, block_k)

    # pad K to block multiple for easy reshape
    K_pad = nb * block_k
    if K_pad != K:
        pad = torch.zeros((M, K_pad - K), device=x_bf16.device, dtype=x_bf16.dtype)
        x_pad = torch.cat([x_bf16, pad], dim=1)
    else:
        x_pad = x_bf16

    x_blocks = x_pad.view(M, nb, block_k).float()  # compute scale in fp32
    amax = x_blocks.abs().amax(dim=2)              # [M, nb]
    scale = (amax / FP8_MAX_E4M3FN).clamp(min=1e-8)  # avoid div by 0
    x_scaled = (x_blocks / scale.unsqueeze(-1)).clamp(-FP8_MAX_E4M3FN, FP8_MAX_E4M3FN)
    x_q = x_scaled.to(torch.float8_e4m3fn).view(M, K_pad)[:, :K].contiguous()
    x_s = scale.contiguous()  # [M, nb], fp32

    return x_q, x_s


@torch.no_grad()
def fp8_quant_weight_blockwise(w_bf16: torch.Tensor, block_n: int = 128, block_k: int = 128):
    """
    w_bf16: [E, N, K] bf16
    returns:
      w_q: [E, N, K] float8_e4m3fn
      w_s: [E, ceil(N/block_n), ceil(K/block_k)] float32  (scale per (N,K) block)
    dequant approx: w_deq[e, nblock, kblock] = w_q.float() * w_s[e, bn, bk]
    """
    assert w_bf16.dim() == 3
    assert w_bf16.dtype == torch.bfloat16
    E, N, K = w_bf16.shape
    nb_n = _ceil_div(N, block_n)
    nb_k = _ceil_div(K, block_k)
    N_pad = nb_n * block_n
    K_pad = nb_k * block_k

    w = w_bf16
    if N_pad != N:
        pad = torch.zeros((E, N_pad - N, K), device=w.device, dtype=w.dtype)
        w = torch.cat([w, pad], dim=1)
    if K_pad != K:
        pad = torch.zeros((E, N_pad, K_pad - K), device=w.device, dtype=w.dtype)
        w = torch.cat([w, pad], dim=2)

    w_blocks = w.view(E, nb_n, block_n, nb_k, block_k).float()  # [E, bn, bnsize, bk, bksize]
    amax = w_blocks.abs().amax(dim=(2, 4))                      # [E, nb_n, nb_k]
    scale = (amax / FP8_MAX_E4M3FN).clamp(min=1e-8)
    w_scaled = (w_blocks / scale.unsqueeze(2).unsqueeze(4)).clamp(-FP8_MAX_E4M3FN, FP8_MAX_E4M3FN)
    w_q = w_scaled.to(torch.float8_e4m3fn).view(E, N_pad, K_pad)[:, :N, :K].contiguous()
    w_s = scale.contiguous()  # [E, nb_n, nb_k]

    return w_q, w_s


# ----------------------------
# Build a contiguous-by-expert test case
# ----------------------------

def make_contig_by_expert_m_indices(M: int, E: int, device="cuda"):
    """
    Produce m_indices of length M with contiguous segments per expert:
      [0..0, 1..1, ..., E-1..E-1] with random counts summing to M.
    """
    # random positive counts then normalize to sum M
    counts = torch.rand(E, device=device)
    counts = (counts / counts.sum() * M).to(torch.int64)
    # fix rounding
    diff = M - int(counts.sum().item())
    if diff != 0:
        counts[0] += diff
    counts = torch.clamp(counts, min=0)

    m_indices = torch.cat([torch.full((int(counts[e].item()),), e, device=device, dtype=torch.int32)
                           for e in range(E)], dim=0)
    # If something weird caused mismatch, pad/trim
    if m_indices.numel() < M:
        pad = torch.full((M - m_indices.numel(),), 0, device=device, dtype=torch.int32)
        m_indices = torch.cat([m_indices, pad], dim=0)
    elif m_indices.numel() > M:
        m_indices = m_indices[:M]
    return m_indices


# ----------------------------
# Main compare
# ----------------------------

def main():
    assert torch.cuda.is_available(), "CUDA required"
    dev = torch.device("cuda")

    cc = torch.cuda.get_device_capability()
    if cc < (9, 0):
        print(f"WARNING: device capability {cc} might not support float8 well; test may fail.")

    # Import DeepGEMM
    import deep_gemm

    # Import your extension (rename to your actual module)
    # Example: import my_moe_ext as my_ext
    import AsymCompute  # <-- CHANGE THIS to your module name

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Shapes (pick something realistic)
    M = 4096         # number of packed rows (tokens * topk after your contig packing)
    K = 4096         # in_features
    N = 4096         # out_features (e.g., gate+up)
    E = 8            # num experts/groups

    # Make contiguous-by-expert assignment for rows
    m_indices = make_contig_by_expert_m_indices(M, E, device=dev)

    # Generate BF16 inputs then quantize
    A_bf16 = torch.randn((M, K), device=dev, dtype=torch.bfloat16)
    B_bf16 = torch.randn((E, N, K), device=dev, dtype=torch.bfloat16)

    # Quantize to FP8 + scales (simple blockwise scheme)
    A_q, A_s = fp8_quant_per_token_blockwise(A_bf16, block_k=128)          # (A_q, A_s)
    B_q, B_s = fp8_quant_weight_blockwise(B_bf16, block_n=128, block_k=128)  # (B_q, B_s)

    # Output buffers
    C_deep = torch.empty((M, N), device=dev, dtype=torch.bfloat16)
    C_mine = torch.empty((M, N), device=dev, dtype=torch.bfloat16)

    import ipdb
    ipdb.set_trace()
    # Warmup
    for _ in range(3):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous((A_q, A_s), (B_q, B_s), C_deep, m_indices=m_indices)
        AsymCompute.grouped_gemm_nt_f8f8bf16_contig(A_q, A_s, B_q, C_mine, m_indices)
    # torch.cuda.synchronize()

    # Run once for comparison
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous((A_q, A_s), (B_q, B_s), C_deep, m_indices=m_indices)
    AsymCompute.grouped_gemm_nt_f8f8bf16_contig((A_q, A_s), (B_q, B_s), C_mine, m_indices)
    torch.cuda.synchronize()

    # Compare
    diff = (C_mine.float() - C_deep.float())
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()

    denom = C_deep.float().abs().clamp_min(1e-6)
    max_rel = (diff.abs() / denom).max().item()
    mean_rel = (diff.abs() / denom).mean().item()

    print("=== Compare my kernel vs deep_gemm (contig) ===")
    print(f"M,N,K,E = {M},{N},{K},{E}")
    print(f"max_abs  : {max_abs:.6g}")
    print(f"mean_abs : {mean_abs:.6g}")
    print(f"max_rel  : {max_rel:.6g}")
    print(f"mean_rel : {mean_rel:.6g}")

    # Optional: sanity check vs a BF16 “dequant + per-row expert matmul” baseline (slow!)
    # Uncomment only for small sizes.
    # baseline = torch.empty((M, N), device=dev, dtype=torch.float32)
    # A_deq = A_q.float()
    # # apply per-token block scales
    # # expand scales [M, nb] -> [M, K]
    # nb = A_s.shape[1]
    # A_s_exp = A_s.repeat_interleave(128, dim=1)[:, :K]
    # A_deq = A_deq * A_s_exp
    # # dequant weights blockwise similarly is more code; omitted here.

if __name__ == "__main__":
    main()