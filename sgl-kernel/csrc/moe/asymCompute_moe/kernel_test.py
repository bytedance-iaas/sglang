import os
import time
import torch
import AsymCompute

# -----------------------------
# 1) Reference: MoE compute in PyTorch
# -----------------------------
@torch.no_grad()
def moe_reference_sum_combine(
    A: torch.Tensor,
    B: torch.Tensor,
    expert_ids: torch.Tensor,
    offsets: torch.Tensor,
    use_fp32_accum: bool = True,
):
    """
    Reference for grouped-by-expert MoE compute (sum-combine).

    A: [M, K]
    B: [E, K, N]   (expects per-expert matrix is KxN, so y = A @ B[e])
    expert_ids: [G] expert id for each group (order of groups)
    offsets:   [G] end offsets (prefix sums), last == L
    index_list:[L] token indices (0..M-1), ordered by groups:
               routes for expert_ids[0] are index_list[0:offsets[0]],
               routes for expert_ids[1] are index_list[offsets[0]:offsets[1]], etc.

    Returns:
      out: [M, N] where out[t] = sum over all routed entries that point to token t.
    """
    assert A.dim() == 2, A.shape
    assert B.dim() == 3, B.shape
    M, K = A.shape
    E, Kb, N = B.shape
    assert Kb == K, (Kb, K)

    expert_ids = expert_ids.to(device=A.device, dtype=torch.int64).contiguous()
    offsets    = offsets.to(device=A.device, dtype=torch.int64).contiguous()

    G = expert_ids.numel()
    assert offsets.numel() == G

    # compute dtype
    A_compute = A.float() if use_fp32_accum and A.dtype != torch.float32 else A
    B_compute = B.float() if use_fp32_accum and B.dtype != torch.float32 else B
    out_compute = torch.zeros(
        (M, N),
        device=A.device,
        dtype=torch.float32 if use_fp32_accum else A.dtype,
    )

    start = 0
    for g in range(G):
        end = int(offsets[g].item())
        if end <= start:
            start = end
            continue

        e = int(expert_ids[g].item())
        assert 0 <= e < E, (e, E)

        # gather
        a_e = A_compute[start: end] # [Te, K]
    
        # matmul: [Te,K] @ [K,N] -> [Te,N]
        out_compute[start: end] = a_e @ B_compute[e]

        start = end

    return out_compute.to(dtype=A.dtype) if use_fp32_accum else out_compute

# -----------------------------
# 3) Utility: generate routing lists (topk-style)
# -----------------------------
def make_routing(M, E, topk, device):
    """
    Returns expert_list, index_list of length L=M*topk.
    index_list repeats tokens topk times.
    expert_list assigns each repeated route to an expert.
    """
    # tokens: 0..M-1 repeated topk times -> [L]
    index_list = torch.arange(M, device=device, dtype=torch.int32).repeat_interleave(topk)

    # random expert per route (simulates top-k choices without enforcing uniqueness)
    expert_list = torch.randint(0, E, (M * topk,), device=device, dtype=torch.int32)
    import ipdb
    ipdb.set_trace()
    return expert_list, index_list

def make_grouped_routing_from_counts(M_tokens: int,
                                    expert_ids: torch.Tensor,
                                    counts: torch.Tensor,
                                    device="cuda"):
    """
    Build a routing schedule in "grouped" form:
      offsets: [G] end offsets (prefix sums), total = L
      expert_ids: [G] expert id for each group
    And also return a per-route expert_list/index_list consistent with that schedule.

    M_tokens here is just for choosing token indices (0..M_tokens-1).
    L = counts.sum() is the number of routed entries ("token copies").

    Returns:
      expert_ids (int32)     : [G]
      offsets (int32)        : [G]  (end offsets)
      expert_list (int32)    : [L]  (expert id per routed entry, grouped)
      index_list (int32)     : [L]  (token index per routed entry)
    """
    assert expert_ids.numel() == counts.numel()
    G = expert_ids.numel()
    L = int(counts.sum().item())

    # offsets = inclusive prefix sum (end positions)
    offsets = torch.cumsum(counts.to(torch.int32), dim=0).to(device)

    # expand expert ids into per-route list, grouped
    expert_list = torch.repeat_interleave(expert_ids.to(torch.int32), counts.to(torch.int32)).to(device)

    # choose token indices for each routed entry; simplest: 0..L-1 mapped mod M_tokens
    # (you can replace this with your topk replication scheme if you want)
    index_list = (torch.arange(L, device=device, dtype=torch.int32) % M_tokens)
    return expert_ids.to(device).to(torch.int32), offsets, expert_list, index_list

# -----------------------------
# 4) Main test
# -----------------------------
def main():
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "Need CUDA for kernel timing"

    device = "cuda"

    # Shapes (adjust to match your kernel’s intended use)
    M = 500         # tokens
    K = 7168         # hidden
    N = 4096         # w13 output (2*moe_intermediate_size) OR 7168 for w2 output
    E = 64           # number of experts (e.g., 256 in DeepSeek, but use smaller for test)
    dtype = torch.float16
    expert_ids = torch.tensor([0, 3, 2, 4, 6, 5, 9, 21], device=device, dtype=torch.int32)
    counts = torch.tensor([20, 30, 50, 200, 300, 400, 450, 500], device=device, dtype=torch.int32)

    # A: [M, K], B: [E, K, N]
    A = torch.randn((M, K), device=device, dtype=dtype)
    B = torch.randn((E, N, K), device=device, dtype=dtype)
    out_custom = torch.zeros((M, N), device=device, dtype=dtype)

    B_transpose = B.transpose(1, 2)
    # Reference
    # torch.cuda.synchronize()
    # out_ref = moe_reference_sum_combine(A, B_transpose, expert_ids, counts, use_fp32_accum=True)

    torch.cuda.synchronize()
    # import ipdb
    # ipdb.set_trace()
    # Custom
    AsymCompute.fuseKernel(A, B, out_custom, expert_ids, counts)
    torch.cuda.synchronize()

    # Compare
    max_abs = (out_custom - out_ref).abs().max().item()
    mean_abs = (out_custom - out_ref).abs().mean().item()
    # print(f"[REF ] {t_ref:.3f} ms (one run)")
    # print(f"[CUST] {t_custom:.3f} ms/iter (avg over 50)")
    print(f"max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}")
    # torch.testing.assert_close(out_custom, out_ref, rtol=1e-2, atol=1e-2)

    # print(f"Reference done: {t_ref:.3f} ms (one run). Now wire call_custom() and uncomment timing/compare.")

if __name__ == "__main__":
    main()
