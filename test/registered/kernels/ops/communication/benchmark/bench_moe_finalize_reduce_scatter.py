"""Microbenchmark for deferred MoE finalize fused with NVLink reduce-scatter."""

from __future__ import annotations

import atexit
import os
from functools import cache

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import multigpu_bench_main
from sglang.kernels.ops.communication import nvlink_comm
from sglang.kernels.ops.communication.moe_finalize_reduce_scatter import (
    moe_finalize_reduce_scatter,
)
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.kernels.ops.moe.moe_finalize_fuse_shared import (
    moe_finalize_fuse_shared,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="requires 4 GPUs, self-skips in CI",
)

WORLD_SIZE = 4
HIDDEN_SIZE = 5120
TOP_K = 8
TOKEN_COUNTS = [1024, 2048, 4096, 8192]
PROVIDERS = ["separate_finalize_rs", "fused_finalize_rs"]
MAX_PUSH_BYTES = max(TOKEN_COUNTS) // WORLD_SIZE * HIDDEN_SIZE * 2


@cache
def init_cpu_group() -> dist.ProcessGroup:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    atexit.register(dist.destroy_process_group)
    torch.cuda.set_stream(torch.cuda.Stream())
    return coord.cpu_group


@cache
def init_comm():
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    comm = CustomAllReduceV2(
        init_cpu_group(),
        device,
        max_size=MAX_PUSH_BYTES,
        max_pull_size=0,
        max_push_size=MAX_PUSH_BYTES,
        max_pull_blocks=0,
    )
    if comm.disabled:
        raise RuntimeError("CustomAllReduceV2 is unavailable")
    register_comm_cleanup(comm)
    return comm


@cache
def make_case(num_tokens: int):
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    generator = torch.Generator(device=device).manual_seed(100 + rank)
    num_permuted_rows = num_tokens * TOP_K
    gemm2 = torch.randn(
        num_permuted_rows,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    indices = torch.randperm(
        num_permuted_rows,
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    weights = torch.rand(
        num_tokens,
        TOP_K,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    weights /= weights.sum(dim=1, keepdim=True)
    shared = torch.randn(
        num_tokens,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    output = torch.empty(
        num_tokens // WORLD_SIZE,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=device,
    )
    return gemm2, indices, weights, shared, output


def separate_finalize_rs(num_tokens: int) -> torch.Tensor:
    gemm2, indices, weights, shared, output = make_case(num_tokens)
    routed = moe_finalize_fuse_shared(
        gemm2,
        indices,
        weights,
        shared_output=None,
        top_k=TOP_K,
        enable_pdl=True,
    )
    nvlink_comm.reduce_scatter_push(
        init_comm().obj,
        routed,
        output,
        pre_reduce=shared,
    )
    return output


def fused_finalize_rs(num_tokens: int) -> torch.Tensor:
    gemm2, indices, weights, shared, _ = make_case(num_tokens)
    return moe_finalize_reduce_scatter(
        init_comm().obj,
        gemm2,
        indices,
        weights,
        shared,
    )


@marker.parametrize("num_tokens", TOKEN_COUNTS)
@marker.benchmark("provider", PROVIDERS, unit="us")
def benchmark(num_tokens: int, provider: str):
    if dist.get_world_size(init_cpu_group()) != WORLD_SIZE:
        marker.skip(f"benchmark requires world_size={WORLD_SIZE}")
    init_comm()
    fn = (
        separate_finalize_rs
        if provider == "separate_finalize_rs"
        else fused_finalize_rs
    )
    return marker.do_bench(
        fn,
        input_args=(num_tokens,),
        graph_clone_args=None,
        graph_context_fn=init_comm().capture,
        sync_multigpu_fn=lambda: dist.barrier(init_cpu_group()),
        memory_args=None,
        memory_output=None,
    )


if __name__ == "__main__":
    multigpu_bench_main(
        name=__name__,
        file=__file__,
        num_gpus=(WORLD_SIZE,),
        main_fn=benchmark.run,
        timeout=600,
    )
