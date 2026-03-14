import copy
import numpy as np
import random
import torch

import asym_gemm
import deep_gemm
from asym_gemm.testing import (
    bench_kineto,
    calc_diff, count_bytes
)
from generators import (
    get_arch_major,
    enumerate_normal, enumerate_m_grouped_contiguous, enumerate_m_grouped_masked, enumerate_k_grouped_contiguous,
    generate_normal, generate_m_grouped_contiguous, generate_m_grouped_masked, generate_k_grouped_contiguous
)


def log_top_left_2x2_ab(a: torch.Tensor, b: torch.Tensor, log_path: str, tag: str) -> None:
    def _select_matrix(x: torch.Tensor) -> torch.Tensor:
        y = x
        while y.dim() > 2:
            y = y[0]
        if y.dim() != 2:
            raise ValueError(f"Expected 2D tensor after slicing leading dims, got shape={tuple(y.shape)}")
        return y

    def _format_2x2(x2d: torch.Tensor) -> str:
        rows = min(2, x2d.shape[0])
        cols = min(2, x2d.shape[1])
        x_cpu = x2d[:rows, :cols].detach().float().cpu()
        return np.array2string(x_cpu.numpy(), precision=6, suppress_small=False)

    a2d = _select_matrix(a)
    b2d = _select_matrix(b)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{tag}]\n")
        f.write(f"A shape={tuple(a.shape)} top-left 2x2:\n{_format_2x2(a2d)}\n")
        f.write(f"B shape={tuple(b.shape)} top-left 2x2:\n{_format_2x2(b2d)}\n\n")

def debug_dump_m_grouped_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indices: torch.Tensor,
    d_asym: torch.Tensor,
    ref_d: torch.Tensor,
    rows_to_print: int = 4,
    a_cols: int = 128,
    b_rows: int = 2,
    b_cols: int = 128,
    out_cols: int = 8,
) -> None:
    def fmt(x: torch.Tensor) -> str:
        return np.array2string(x.detach().float().cpu().numpy(), precision=6, suppress_small=False)

    valid_rows = torch.nonzero(m_indices != -1, as_tuple=False).flatten()
    if valid_rows.numel() == 0:
        print("No valid rows in m_indices.")
        return

    picked_rows = valid_rows[:rows_to_print].tolist()
    out_cols = min(out_cols, d_asym.shape[1], ref_d.shape[1])
    a_cols = min(a_cols, a.shape[1])
    b_rows = min(b_rows, b.shape[1])
    b_cols = min(b_cols, b.shape[2])

    print("\n===== BF16 Asym Debug Dump =====")
    print(f"a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}, d_asym.shape={tuple(d_asym.shape)}, ref_d.shape={tuple(ref_d.shape)}")
    print(f"Inspect rows={picked_rows}, A cols=0:{a_cols}, B rows=0:{b_rows}, B cols=0:{b_cols}, output cols=0:{out_cols}")

    for row_idx in picked_rows:
        expert = int(m_indices[row_idx].item())
        print(f"\n--- token row={row_idx}, expert={expert} ---")
        print(f"A[row, :{a_cols}] =\n{fmt(a[row_idx, :a_cols])}")
        print(f"B[expert, :{b_rows}, :{b_cols}] =\n{fmt(b[expert, :b_rows, :b_cols])}")

        manual_row = a[row_idx].float() @ b[expert].float().t()
        print(f"manual(A[row] @ B[expert].T)[:{out_cols}] =\n{fmt(manual_row[:out_cols])}")
        print(f"d_asym[row, :{out_cols}] =\n{fmt(d_asym[row_idx, :out_cols])}")
        print(f"ref_d[row, :{out_cols}] =\n{fmt(ref_d[row_idx, :out_cols])}")
        print(f"abs(d_asym - manual)[:{out_cols}] =\n{fmt((d_asym[row_idx, :out_cols].float() - manual_row[:out_cols]).abs())}")
        print(f"abs(ref_d - manual)[:{out_cols}] =\n{fmt((ref_d[row_idx, :out_cols].float() - manual_row[:out_cols]).abs())}")

@torch.no_grad()
def split_k_reference_for_m_grouped_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indices: torch.Tensor,
    k_split: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = a.shape[0], b.shape[1]
    device = a.device
    ref_lo = torch.zeros((m, n), device=device, dtype=torch.float)
    ref_hi = torch.zeros((m, n), device=device, dtype=torch.float)

    valid_rows = torch.nonzero(m_indices != -1, as_tuple=False).flatten()
    if valid_rows.numel() == 0:
        return ref_lo, ref_hi

    for row_idx in valid_rows.tolist():
        expert = int(m_indices[row_idx].item())
        a_row = a[row_idx].float()
        b_mat = b[expert].float()
        k_mid = min(k_split, a_row.shape[0], b_mat.shape[1])
        ref_lo[row_idx] = a_row[:k_mid] @ b_mat[:, :k_mid].t()
        ref_hi[row_idx] = a_row[k_mid:] @ b_mat[:, k_mid:].t()

    return ref_lo, ref_hi

def estimate_time(fn, num_warmups: int = 0, num_tests: int = 1,
                  flush_l2: bool = True):
    # import ipdb
    # ipdb.set_trace()
    # Simple CUDA-event based timing to compare with bench_kineto
    torch.cuda.synchronize()
    if flush_l2:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')
        cache.zero_()

    for _ in range(num_warmups):
        fn()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_tests):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / num_tests / 1e3

def pin_b_to_cuda(b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b_pinned = b.detach().to(device="cpu", pin_memory=True)
    b_cuda = b_pinned.to(device="cuda", non_blocking=True)
    return b_cuda, b_pinned

@torch.no_grad()
def calc_abs_diff_stats(
    x: torch.Tensor,
    y: torch.Tensor,
    valid_rows_mask: torch.Tensor | None = None,
) -> tuple[float, float]:
    x_f = x.float()
    y_f = y.float()
    abs_diff = (x_f - y_f).abs()

    if valid_rows_mask is not None:
        row_mask = valid_rows_mask.to(device=abs_diff.device, dtype=torch.bool)
        abs_diff = abs_diff[row_mask]

    if abs_diff.numel() == 0:
        return 0.0, 0.0

    flat = abs_diff.reshape(-1)
    mean_diff = flat.mean().item()
    # torch.quantile can fail on very large tensors; use order statistic instead.
    n = flat.numel()
    k = int(np.ceil(0.99 * n))
    k = min(max(k, 1), n)
    p99_diff = flat.kthvalue(k).values.item()
    return mean_diff, p99_diff

@torch.no_grad()
def extract_offsets_and_experts_start(m_indices: torch.Tensor, drop_invalid: bool = True):
    """
    Convert a 1D expert-id array m_indices (length M) into:
      offsets: 1D int64 tensor of segment start positions (0-based)
      experts: 1D tensor of expert ids for each segment

    Example:
      m_indices = [0,0,0,2,2,2,2,1,1]
      offsets   = [0,3,7]
      experts   = [0,2,1]

    If drop_invalid=True, segments with expert == -1 are removed.
    A sentinel pair (M, -1) is appended to the output when M > 0.
    """
    assert m_indices.dim() == 1, f"expected 1D m_indices, got {m_indices.shape}"
    M = m_indices.numel()
    device = m_indices.device

    if M == 0:
        offsets = torch.empty((0,), device=device, dtype=torch.long)
        experts = torch.empty((0,), device=device, dtype=m_indices.dtype)
        return offsets, experts

    # Find boundaries where expert id changes.
    # change[i] = True means m_indices[i+1] != m_indices[i]
    change = (m_indices[1:] != m_indices[:-1])

    # Segment start indices: always include 0, plus (change positions + 1)
    starts = torch.nonzero(change, as_tuple=False).flatten().to(torch.long) + 1
    offsets = torch.cat([torch.zeros((1,), device=device, dtype=torch.long), starts], dim=0)

    # Expert id for each segment is the label at the segment start
    experts = m_indices[offsets]

    if drop_invalid:
        keep = (experts != -1)
        offsets = offsets[keep]
        experts = experts[keep]

    # Append sentinel (M, -1) so downstream can read segment ends.
    offsets = torch.cat(
        [offsets, torch.tensor([M], device=device, dtype=torch.long)],
        dim=0,
    )
    experts = torch.cat(
        [experts, torch.tensor([-1], device=device, dtype=m_indices.dtype)],
        dim=0,
    )

    return offsets, experts


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing m-grouped contiguous GEMM:')
    compiled_dims = "mnk"
    log_path = "test_bf16_ab_top_left.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("test_bf16.py A/B top-left 2x2 snapshots\n\n")

    for _, num_groups, expected_m_per_group, n, k, major_a, major_b in enumerate_m_grouped_contiguous(torch.bfloat16):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        
        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b, use_bf16=True)
        log_top_left_2x2_ab(
            a,
            b,
            log_path,
            f"num_groups={num_groups} m={m} n={n} k={k} layout={major_opt}",
        )
        b_pinned = b.detach().to("cpu", non_blocking=False).pin_memory()
        d_deep_gpu = torch.empty_like(d)
        d_asym = torch.empty_like(d)

        b_gpu = b_pinned.to(device="cuda", non_blocking=True)
        torch.cuda.synchronize()

        # noinspection PyShadowingNames
        def test_func_deep():
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b_gpu, d_deep_gpu, m_indices)

        # noinspection PyShadowingNames
        def test_func_asym():
            # deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b_gpu, d_asym, m_indices)
            asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
                a, b_pinned, d_asym, offsets_i32, experts_i32, num_groups + 1, compiled_dims
            )

        offsets, experts = extract_offsets_and_experts_start(m_indices)

        experts_i32 = experts.to(dtype=torch.int32, device="cuda").contiguous()
        offsets_i32 = offsets.to(dtype=torch.int32, device="cuda").contiguous()
        print(f"m: {m}")
        print(f"n: {n}")
        print(f"k: {k}")
        print(f"experts_i32: {experts_i32}")
        print(f"offsets_i32: {offsets_i32}")

        t_deep_gpu = estimate_time(test_func_deep)
        t_asym = estimate_time(test_func_asym)

        d_deep_gpu = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d_deep_gpu), d_deep_gpu)
        d_asym = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d_asym), d_asym)
        print(f"d_deep_gpu sample (2x8):\n{d_deep_gpu[:2, :8]}")
        print(f"d_asym sample (2x8):\n{d_asym[:2, :8]}")
        print(f"ref_d sample (2x8):\n{ref_d[:2, :8]}")
        # debug_dump_m_grouped_bf16(a, b_gpu, m_indices, d_asym, ref_d)
        ref_k0_64, ref_k64_end = split_k_reference_for_m_grouped_bf16(a, b_gpu, m_indices, k_split=64)
        print(f"ref_k0_64 sample (2x8):\n{ref_k0_64[:2, :8].to(dtype=torch.bfloat16)}")
        print(f"ref_k64_end sample (2x8):\n{ref_k64_end[:2, :8].to(dtype=torch.bfloat16)}")
        deep_gpu_diff = calc_diff(d_deep_gpu, ref_d)
        asym_diff = calc_diff(d_asym, ref_d)
        valid_rows = (m_indices != -1)
        deep_mean_diff, deep_p99_diff = calc_abs_diff_stats(d_deep_gpu, ref_d, valid_rows)
        mean_diff, p99_diff = calc_abs_diff_stats(d_asym, ref_d, valid_rows)
        deep_vs_k0_64 = calc_diff(d_deep_gpu, ref_k0_64.to(dtype=d_deep_gpu.dtype))
        deep_vs_k64_end = calc_diff(d_deep_gpu, ref_k64_end.to(dtype=d_deep_gpu.dtype))
        asym_vs_k0_64 = calc_diff(d_asym, ref_k0_64.to(dtype=d_asym.dtype))
        asym_vs_k64_end = calc_diff(d_asym, ref_k64_end.to(dtype=d_asym.dtype))

        active_m = int((m_indices != -1).sum().item())
        flops_total = 2 * m * n * k
        flops_active = 2 * active_m * n * k

        print(f'   deep_gpu  diff={deep_gpu_diff:.5e} | mean_abs_diff={deep_mean_diff:.5e} | p99_abs_diff={deep_p99_diff:.5e}')
        print(f'   asym_gemm diff={asym_diff:.5e} | mean_abs_diff={mean_diff:.5e} | p99_abs_diff={p99_diff:.5e}')
        print(f'   deep_gpu vs ref_k0_64={deep_vs_k0_64:.5e} | vs ref_k64_end={deep_vs_k64_end:.5e}')
        print(f'   asym_gemm vs ref_k0_64={asym_vs_k0_64:.5e} | vs ref_k64_end={asym_vs_k64_end:.5e}')
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:5}, k={k:5}, layout={major_opt}): '
              f'active_m={active_m:5} | '
              f'deep_gpu={t_deep_gpu * 1e6:6.0f} us | '
              f'deep_gpu_tflops(m)={flops_total / t_deep_gpu / 1e12:4.0f} | '
              f'deep_gpu_tflops(active_m)={flops_active / t_deep_gpu / 1e12:4.0f} | '
              f'asym_gemm={t_asym * 1e6:6.0f} us | '
              f'asym_tflops(m)={flops_total / t_asym / 1e12:4.0f} | '
              f'asym_tflops(active_m)={flops_active / t_asym / 1e12:4.0f} | '
              f'{count_bytes(a, b, d_asym) / 1e9 / t_asym:4.0f} GB/s')
    print()

def test_block_k_debug():
    """Minimal test to isolate block_k=128 issue"""
    
    # Use K=128 so block_k=128 means exactly 1 k-iteration (no TMA_REDUCE_ADD)
    # If this FAILS → problem is in UMMA descriptor / multi-atom K
    # If this PASSES → problem is in the k-iteration loop / epilogue accumulation
    print("=== block_k=128 debug test ===")
    for K in [128, 256, 512]:
        num_groups = 1
        m_per_group = 128  # = block_m, single m-block
        n = 128            # = block_n, single n-block
        
        m = m_per_group
        a = torch.randn((m, K), device='cuda', dtype=torch.bfloat16)
        b = torch.randn((num_groups, n, K), device='cuda', dtype=torch.bfloat16)
        m_indices = torch.zeros(m, device='cuda', dtype=torch.int32)
        
        ref_d = (a.float() @ b[0].float().t()).to(torch.bfloat16)
        
        # --- Test deep_gemm ---
        d_deep = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d_deep, m_indices)
        
        diff_deep = (d_deep.float() - ref_d.float()).abs()
        max_diff_deep = diff_deep.max().item()
        mean_diff_deep = diff_deep.mean().item()
        mismatch_deep = (diff_deep > 1e-2).sum().item()
        
        # --- Test asym_gemm ---
        d_asym = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        b_pinned = b.detach().to("cpu", non_blocking=False).pin_memory()
        # Build offsets/experts for single group: offsets=[0, m], experts=[0, -1]
        offsets_i32 = torch.tensor([0, m], device='cuda', dtype=torch.int32)
        experts_i32 = torch.tensor([0, -1], device='cuda', dtype=torch.int32)
        compiled_dims = "mnk"
        asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
            a, b_pinned, d_asym, offsets_i32, experts_i32, num_groups + 1, compiled_dims
        )
        
        diff_asym = (d_asym.float() - ref_d.float()).abs()
        max_diff_asym = diff_asym.max().item()
        mean_diff_asym = diff_asym.mean().item()
        mismatch_asym = (diff_asym > 1e-2).sum().item()
        
        # --- deep vs asym ---
        diff_deep_asym = (d_deep.float() - d_asym.float()).abs()
        max_diff_da = diff_deep_asym.max().item()
        
        print(f"K={K} (k_iters={K}//128={'1 (no reduce)' if K==128 else str(K//128)}):")
        print(f"  deep_gemm:  max_diff={max_diff_deep:.6f}, mean_diff={mean_diff_deep:.6f}, "
              f"mismatches(>1e-2)={mismatch_deep}/{d_deep.numel()}")
        print(f"  asym_gemm:  max_diff={max_diff_asym:.6f}, mean_diff={mean_diff_asym:.6f}, "
              f"mismatches(>1e-2)={mismatch_asym}/{d_asym.numel()}")
        print(f"  deep vs asym: max_diff={max_diff_da:.6f}")
        
        # Show sample values
        print(f"  sample [0,0]: deep={d_deep[0,0].item():.6f}, asym={d_asym[0,0].item():.6f}, ref={ref_d[0,0].item():.6f}")
        
        if mismatch_asym > 0:
            rows, cols = torch.where(diff_asym > 1e-2)
            for i in range(min(3, len(rows))):
                r, c = rows[i].item(), cols[i].item()
                print(f"  mismatch [{r},{c}]: asym={d_asym[r,c].item():.6f}, "
                      f"deep={d_deep[r,c].item():.6f}, ref={ref_d[r,c].item():.6f}")
        
        # --- Python-side partial k-block verification ---
        # Shows what each block_k=128 chunk contributes to the final result.
        # If the kernel's single-iteration result matches partial[0] but the
        # multi-iteration sum diverges, the bug is in TMA_REDUCE_ADD accumulation.
        # If even partial[0] diverges, the bug is in UMMA descriptor / data load.
        block_k = 128
        num_k_iters = K // block_k
        print(f"  --- Partial k-block analysis (block_k={block_k}, {num_k_iters} iterations) ---")
        accumulated_bf16 = torch.zeros((m, n), device='cuda', dtype=torch.float32)
        for ki in range(num_k_iters):
            k_start = ki * block_k
            k_end = k_start + block_k
            # Compute partial product for this k-chunk (FP32 matmul, then cast to BF16)
            partial_fp32 = a[:, k_start:k_end].float() @ b[0, :, k_start:k_end].float().t()
            partial_bf16 = partial_fp32.to(torch.bfloat16)
            # Simulate TMA_REDUCE_ADD: accumulate in BF16 precision
            accumulated_bf16 += partial_bf16.float()
            print(f"    k_iter={ki} [k={k_start}:{k_end}]: "
                  f"partial[0,0]={partial_bf16[0,0].item():.6f}, "
                  f"accumulated[0,0]={accumulated_bf16[0,0].to(torch.bfloat16).item():.6f}")
        
        acc_result = accumulated_bf16.to(torch.bfloat16)
        diff_acc_vs_asym = (acc_result.float() - d_asym.float()).abs().max().item()
        diff_acc_vs_deep = (acc_result.float() - d_deep.float()).abs().max().item()
        diff_acc_vs_ref = (acc_result.float() - ref_d.float()).abs().max().item()
        print(f"    simulated_reduce vs asym: max_diff={diff_acc_vs_asym:.6f}")
        print(f"    simulated_reduce vs deep: max_diff={diff_acc_vs_deep:.6f}")
        print(f"    simulated_reduce vs ref:  max_diff={diff_acc_vs_ref:.6f}")
        print()

def test_cublaslt_gemm() -> None:
    print('Testing cuBLASLt GEMM:')
    for kernel_type, m, n, k, major_a, major_b, accumulate, out_dtype in enumerate_normal(dtype=torch.bfloat16):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'

        a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_bf16=True)
        b, b_pinned = pin_b_to_cuda(b)
        asym_gemm.cublaslt_gemm_nt(a, b, d, c=c)
        diff = calc_diff(d, ref_d)
        assert diff < 6e-7, f'{diff=}, ({m=}, {n=}, {k=}, {major_opt=}, {accumulate=}, {out_dtype=})'

        t = bench_kineto(lambda: asym_gemm.cublaslt_gemm_nt(a, b, d, c=c), 'nvjet', suppress_kineto_output=True,)
        # print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, layout={major_opt}, {out_opt}, {acc_opt}): '
        #       f'{t * 1e6:5.0f} us | '
        #       f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
        #       f'{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {asym_gemm.__path__}\n')

    if get_arch_major() >= 9:
        # test_gemm()
        test_m_grouped_gemm_contiguous()
        # test_block_k_debug()
        # test_m_grouped_gemm_masked()
        # test_k_grouped_gemm_contiguous()

    # test_cublaslt_gemm()