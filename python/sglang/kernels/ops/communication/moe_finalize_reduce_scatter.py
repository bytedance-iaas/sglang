from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi import Module

    from sglang.kernels.ops.communication.all_reduce import Communicator


@cache_once
def _jit_module(
    world_size: int,
    top_k: int,
    weight_dtype: torch.dtype,
) -> Module:
    args = make_cpp_args(
        world_size,
        top_k,
        is_arch_support_pdl(),
        weight_dtype,
    )
    return load_jit(
        "moe_finalize_reduce_scatter",
        *args,
        cuda_files=["distributed/moe_finalize_reduce_scatter.cuh"],
        cuda_wrappers=[
            ("run", f"MoeFinalizeReduceScatter<{args}>::run"),
        ],
    )


def moe_finalize_reduce_scatter(
    comm: Communicator,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
) -> torch.Tensor:
    """Finalize rank-local MoE rows directly into an NVLink reduce-scatter."""
    num_tokens, top_k = expert_weights.shape
    world_size = comm.world_size
    rank = comm.rank
    tokens_avg, tokens_rem = divmod(num_tokens, world_size)
    local_tokens = tokens_avg + int(rank < tokens_rem)
    hidden_dim = (
        shared_output.shape[1] if shared_output is not None else gemm2_out.shape[1]
    )
    out = torch.empty(
        local_tokens,
        hidden_dim,
        dtype=torch.bfloat16,
        device=gemm2_out.device,
    )
    _jit_module(world_size, top_k, expert_weights.dtype).run(
        comm,
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        shared_output,
    )
    return out
