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

def estimate_time(fn, num_warmups: int = 5, num_tests: int = 10,
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

    for _, num_groups, expected_m_per_group, n, k, major_a, major_b in enumerate_m_grouped_contiguous(torch.bfloat16):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        
        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b, use_bf16=True)
        b_pinned = b.detach().to("cpu", non_blocking=False).pin_memory()
        d_deep_gpu = torch.empty_like(d)
        d_asym = torch.empty_like(d)

        b_gpu = b_pinned.to(device="cuda", non_blocking=True)
        torch.cuda.synchronize()

        # noinspection PyShadowingNames
        def test_func_deep():
            asym_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b_gpu, d_deep_gpu, m_indices)

        # noinspection PyShadowingNames
        def test_func_asym():
            # deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b_gpu, d_asym, m_indices)
            asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
                a, b_pinned, d_asym, offsets_i32, experts_i32, num_groups + 1, compiled_dims
            )

        offsets, experts = extract_offsets_and_experts_start(m_indices)

        experts_i32 = experts.to(dtype=torch.int32, device="cuda").contiguous()
        offsets_i32 = offsets.to(dtype=torch.int32, device="cuda").contiguous()
        print(f"experts_i32: {experts_i32}")
        print(f"offsets_i32: {offsets_i32}")

        t_deep_gpu = estimate_time(test_func_deep)
        t_asym = estimate_time(test_func_asym)

        d_asym = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d_asym), d_asym)
        print(f"d_asym sample (2x8):\n{d_asym[:2, :8]}")
        print(f"ref_d sample (2x8):\n{ref_d[:2, :8]}")
        asym_diff = calc_diff(d_asym, ref_d)
        valid_rows = (m_indices != -1)
        mean_diff, p99_diff = calc_abs_diff_stats(d_asym, ref_d, valid_rows)

        active_m = int((m_indices != -1).sum().item())
        flops_total = 2 * m * n * k
        flops_active = 2 * active_m * n * k

        print(f'   asym_gemm diff={asym_diff:.5e} | mean_abs_diff={mean_diff:.5e} | p99_abs_diff={p99_diff:.5e}')
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:5}, k={k:5}, layout={major_opt}): '
              f'active_m={active_m:5} | '
              f'deep_gpu={t_deep_gpu * 1e6:6.0f} us | '
              f'deep_gpu_tflops(m)={flops_total / t_deep_gpu / 1e12:4.0f} | '
              f'deep_gpu_tflops(active_m)={flops_active / t_deep_gpu / 1e12:4.0f} | '
              f'asym_gemm={t_asym * 1e6:6.0f} us | '
              f'asym_tflops(m)={flops_total / t_asym / 1e12:4.0f} | '
              f'asym_tflops(active_m)={flops_active / t_asym / 1e12:4.0f} | '
              f'{count_bytes(a, b, d_asym) / 1e9 / t_asym:4.0f} GB/s')

        # break
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
        # test_m_grouped_gemm_masked()
        # test_k_grouped_gemm_contiguous()

    # test_cublaslt_gemm()
