"""Four-process Gloo coverage for sender-carried PP tensor ownership.

Adapted from upstream SGLang PR #30095. Replicated and TP-sharded tensors
share one dictionary; only the sharded key bypasses slice/all-gather, and the
receiver deliberately supplies no ownership argument.
"""

import os
from datetime import timedelta

import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, find_available_port

register_cpu_ci(est_time=30, suite="base-a-test-cpu")
WORLD = 4


def _group(ranks, rank, name):
    from sglang.srt.distributed.parallel_state import GroupCoordinator

    return GroupCoordinator(
        group_ranks=ranks,
        local_rank=rank,
        torch_distributed_backend="gloo",
        use_pynccl=False,
        use_pymscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        use_message_queue_broadcaster=False,
        group_name=name,
    )


def _worker(rank, port):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group(
        "gloo",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=60),
    )
    try:
        tp = _group([[0, 1], [2, 3]], rank, "ownership_tp")
        pp = _group([[0, 2], [1, 3]], rank, "ownership_pp")
        sharded = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 100 * (
            tp.rank_in_group + 1
        )
        replicated = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        if pp.rank_in_group == 0:
            pp.send_tensor_dict(
                {"sharded": sharded, "replicated": replicated, "tag": "proxy"},
                all_gather_group=tp,
                all_gather_exclude={"sharded"},
            )
        else:
            got = pp.recv_tensor_dict(all_gather_group=tp)
            torch.testing.assert_close(got["sharded"], sharded, rtol=0, atol=0)
            torch.testing.assert_close(got["replicated"], replicated, rtol=0, atol=0)
            assert got["tag"] == "proxy"
        pp.barrier()
    finally:
        torch.distributed.destroy_process_group()


class TestPPTensorOwnership(CustomTestCase):
    def test_sender_metadata_controls_mixed_dict(self):
        mp.spawn(_worker, args=(find_available_port(24000),), nprocs=WORLD, join=True)


if __name__ == "__main__":
    import unittest

    unittest.main()
