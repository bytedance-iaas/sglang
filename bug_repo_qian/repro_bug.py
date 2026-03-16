import deep_gemm
import torch
from typing import Any, Optional, Tuple
import asym_gemm
from pathlib import Path
import os

def build_offsets_experts_from_masked_m(masked_m: torch.Tensor, num_groups: int, max_m: int, block_m: int = 128):
    """
    Build offsets and experts for sparse m-grouped masked GEMM with fixed per-group allocation.

    Each group gets fixed allocation of max_m space, regardless of actual token count.
    Only groups with masked_m[g] > 0 are included in the output mapping.
    Each active group generates a pair of offsets (start, end).

    Args:
        masked_m: (num_groups,) tensor of actual token counts per group
        num_groups: number of expert groups
        max_m: maximum allocated space per group
        block_m: block alignment for padding (default 128)

    Returns:
        offsets: flat tensor with pairs [start_0, end_0, start_1, end_1, ...]
        experts: expert IDs for each active group + terminator (-1)
        list_size: number of experts in output (excluding terminator)

    Example:
        masked_m = [0, 12, 0, 129], num_groups = 4, max_m = 4096
        offsets = [4096, 4224, 12288, 12544]  # 4 offsets = 2 pairs
        experts = [1, 3, -1]  # 2 active experts + terminator
    """
    offsets = []
    experts = []

    for g in range(num_groups):
        v = masked_m[g].item()
        if v > 0:  # Only process active groups
            start = g * max_m
            # Pad actual tokens to block_m alignment
            end = start + ((v + block_m - 1) // block_m) * block_m
            offsets.append(start)
            offsets.append(end)
            experts.append(g)

    # Add terminator expert
    experts.append(-1)

    return (torch.tensor(offsets, dtype=torch.int32, device=masked_m.device),
            torch.tensor(experts, dtype=torch.int32, device=masked_m.device),
            len(experts))

def grouped_gemm_nt_f8f8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
):
    

    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    # kernel_type = compile_utils.AsymGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED

    # deep_gemm.m_grouped_fp8_gemm_nt_masked(lhs, rhs, out, masked_m, 1, disable_ue8m0_cast=True)

    offsets, experts, list_size = build_offsets_experts_from_masked_m(
        masked_m, num_groups, lhs[0].size()[1]
    )

    asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(lhs, rhs, out, offsets, experts, list_size, 128, disable_ue8m0_cast=True)

    torch.cuda.synchronize()
    import ipdb; ipdb.set_trace()

    return asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(
        lhs,
        rhs,
        out,
        offsets,
        experts,
        list_size, 
        expected_m,
        None,
        "nk",
        False,
    )

    # return deep_gemm.fp8_m_grouped_gemm_nt_masked(
    # lhs,
    # rhs,
    # out,
    # masked_m,
    # expected_m,
    # **(
    #     dict(
    #         enable_overlap=True,
    #         max_block_n=max_block_n,
    #         signal=overlap_args.signal,
    #     )
    #     if overlap_args is not None
    #     else {}
    # ),
# )

def main():
    script_dir = Path(__file__).resolve().parent
    dump_path = script_dir / "asym_gemm_bug_dump.pt"
    print("Loading dump from:", dump_path)
    # dump = torch.load(dump_path, weights_only=False)
    dump = torch.load(dump_path, map_location="cpu", weights_only=False)
    print("loaded!")


    lhs = (
        dump["lhs0"].cuda(),
        dump["lhs1"].cuda(),
    )

    rhs = (
        dump["rhs0"].cuda(),     # 实际测试中在CPUgi t
        dump["rhs1"].cuda(),
    )

    # rhs = (
    #     dump["rhs0"].contiguous().pin_memory(),
    #     dump["rhs1"].contiguous().pin_memory(),
    # )

    # import ipdb; ipdb.set_trace()

    masked_m = dump["masked_m"].cuda()
    expected_m = dump["expected_m"]

    num_groups, m_max, k = lhs[0].shape
    _, n, _ = rhs[0].shape

    print("lhs[0][106][0]:", lhs[0][106][0])
    print("masked_m[106]:", masked_m[106].item())

    out = torch.zeros(
        (num_groups, m_max, n),
        dtype=torch.bfloat16,
        device="cuda",
    )

    grouped_gemm_nt_f8f8bf16_masked(
        lhs,
        rhs,
        out,
        masked_m,
        expected_m,
    )

    torch.cuda.synchronize()
    print("kernel finished")

    print("out[106][0] =", out[106][0])
    print("isnan =", torch.isnan(out[106][0]))
    print("total NaN:", torch.isnan(out.float()).sum().item())


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    main()