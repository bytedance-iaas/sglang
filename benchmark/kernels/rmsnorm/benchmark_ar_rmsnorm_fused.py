"""
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345

torchrun --nproc_per_node $tp_num \
--nnodes $WORLD_SIZE \
--node_rank $RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT benchmark/kernels/rmsnorm/benchmark_ar_rmsnorm_fused.py
"""

import sys
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn

from sglang.srt.distributed import (
    init_distributed_environment,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.parallel_state import (
    graph_capture,
    initialize_model_parallel,
)
from sglang.srt.layers.layernorm import RMSNorm


def fused_ar_rmsnorm(hidden_states: torch.Tensor, residual: torch.Tensor, rmsnorm_op):
    return rmsnorm_op(hidden_states, residual, force_fused=True)


def non_fused_ar_rmsnorm(
    hidden_states: torch.Tensor, residual: torch.Tensor, rmsnorm_op
):
    hidden_states = tensor_model_parallel_all_reduce(hidden_states)
    return rmsnorm_op(hidden_states, residual)


def _bench_graph_time(
    func,
    hidden_states,
    residual,
    rmsnorm_op,
    warmup_loop=2,
    graph_loop=10,
    test_loop=10,
):
    hidden_clone = hidden_states.clone()
    residual_clone = residual.clone()

    with graph_capture() as graph_capture_context:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=graph_capture_context.stream):
            for _ in range(graph_loop):
                a = func(hidden_clone, residual_clone, rmsnorm_op)

    graph.replay()

    for _ in range(warmup_loop):
        graph.replay()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: List[float] = []
    for _ in range(test_loop):
        torch.cuda.synchronize()
        dist.barrier()
        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    func_cost_us = sum(latencies) / len(latencies) / graph_loop * 1000
    graph.reset()
    return func_cost_us


def _bench_eager_time(
    func, hidden_states, residual, rmsnorm_op, warmup_loop=2, test_loop=10
):
    eager_input = hidden_states.clone()
    eager_res = residual.clone()

    for _ in range(warmup_loop):
        func(eager_input, eager_res, rmsnorm_op)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(test_loop):
        func(eager_input, eager_res, rmsnorm_op)
    end_event.record()
    torch.cuda.synchronize()
    func_cost_us = start_event.elapsed_time(end_event) / test_loop * 1000

    return func_cost_us


if __name__ == "__main__":

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    world, world_size = dist.group.WORLD, dist.get_world_size()
    rank = dist.get_rank()

    print("w,ws,r", world, world_size, rank)
    torch.cuda.set_device(rank % 8)
    device = torch.cuda.current_device()
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank % 8,
    )

    initialize_model_parallel(tensor_model_parallel_size=world_size)
    dist.barrier()
    dtype = torch.bfloat16

    hidden_size = int(sys.argv[1])
    rmsnorm_op = RMSNorm(hidden_size)
    rmsnorm_op.weight = nn.Parameter(
        torch.empty(hidden_size, dtype=torch.float32, device=device)
    )

    for seq_len in range(1, 128, 4):

        inp_randn = torch.empty(seq_len, hidden_size, dtype=dtype, device=device)
        res_random = torch.empty(seq_len, hidden_size, dtype=dtype, device=device)

        fused_eager_time = _bench_eager_time(
            fused_ar_rmsnorm, inp_randn, res_random, rmsnorm_op
        )

        non_fused_eager_time = _bench_eager_time(
            non_fused_ar_rmsnorm, inp_randn, res_random, rmsnorm_op
        )

        fused_graph_time = _bench_graph_time(
            fused_ar_rmsnorm, inp_randn, res_random, rmsnorm_op
        )

        non_fused_graph_time = _bench_graph_time(
            non_fused_ar_rmsnorm, inp_randn, res_random, rmsnorm_op
        )
        if rank == 0:
            print("\n")
            print(
                "batch : {} \n | fused_eager_time {:.4f} us | non_fused_eager_time {:.4f} us |\n | fused_graph_time {:.4f} us | non_fused_graph_time {:.4f} us |".format(
                    seq_len,
                    fused_eager_time,
                    non_fused_eager_time,
                    fused_graph_time,
                    non_fused_graph_time,
                )
            )
