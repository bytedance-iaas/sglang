"""Correctness test for NVLink ReduceScatter pre-reduce fusion."""

import os

import pytest
import torch

from sglang.kernels.ops.communication import nvlink_comm
from sglang.srt.distributed import parallel_state as ps
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-h100")


@pytest.fixture(scope="module")
def group():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    ps.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    ps.initialize_model_parallel(tensor_model_parallel_size=world_size)
    yield ps.get_tp_group()
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


def test_reduce_scatter_pre_reduce(group):
    if group.ca_comm is None or group.ca_comm.disabled:
        pytest.skip("CustomAllReduceV2 is unavailable")

    rows, hidden = group.world_size * 32, 5120
    generator = torch.Generator(device="cuda").manual_seed(10 + group.rank_in_group)
    routed = torch.randn(
        rows, hidden, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    shared = torch.randn(
        rows, hidden, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    expected = torch.empty(
        rows // group.world_size,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
    )
    group._reduce_scatter_tensor(expected, routed + shared)

    actual = torch.empty_like(expected)
    nvlink_comm.reduce_scatter_push(
        group.ca_comm.obj,
        routed,
        actual,
        pre_reduce=shared,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=4e-2)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
