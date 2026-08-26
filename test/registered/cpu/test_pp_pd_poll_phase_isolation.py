import multiprocessing
import os
import traceback
import unittest
from multiprocessing import Process

import torch.distributed as dist

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    find_available_port,
    maybe_stub_sgl_kernel,
)

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.utils import (  # noqa: E402
    poll_and_all_reduce_attn_cp_tp_group,
)

register_cpu_ci(est_time=15, suite="base-b-test-cpu")


class _Poller:
    def __init__(self, status):
        self.status = status

    def poll(self):
        return self.status


def _run_phase_isolation(rank, world_size, master_port, output_writer):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # Mirror Scheduler initialization: every logical poll phase gets its
        # own TP and CP process group, even when both groups have the same ranks.
        bootstrap_tp = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        bootstrap_cp = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        transfer_tp = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        transfer_cp = dist.new_group(ranks=list(range(world_size)), backend="gloo")

        if rank == 0:
            bootstrap_keys = ["bootstrap-common", "bootstrap-local"]
            bootstrap_states = [KVPoll.WaitingForInput, KVPoll.Bootstrapping]
            transfer_keys = ["transfer-common"]
            transfer_states = [KVPoll.Success]
        else:
            bootstrap_keys = ["bootstrap-common"]
            bootstrap_states = [KVPoll.WaitingForInput]
            transfer_keys = ["transfer-common", "transfer-local"]
            transfer_states = [KVPoll.Success, KVPoll.Transferring]

        bootstrap_polls = poll_and_all_reduce_attn_cp_tp_group(
            [_Poller(state) for state in bootstrap_states],
            bootstrap_cp,
            bootstrap_tp,
            ordered_keys=bootstrap_keys,
        )
        transfer_polls = poll_and_all_reduce_attn_cp_tp_group(
            [_Poller(state) for state in transfer_states],
            transfer_cp,
            transfer_tp,
            ordered_keys=transfer_keys,
        )

        output_writer.send(
            (
                rank,
                [int(state) for state in bootstrap_polls],
                [int(state) for state in transfer_polls],
            )
        )
    except BaseException:
        traceback.print_exc()
        output_writer.send((rank, None, None))
    finally:
        output_writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


class TestPPPollPhaseIsolation(CustomTestCase):
    def test_bootstrap_and_transfer_collectives_cannot_cross_match(self):
        world_size = 2
        master_port = find_available_port(29671)
        output_reader, output_writer = multiprocessing.Pipe(duplex=False)
        processes = [
            Process(
                target=_run_phase_isolation,
                args=(rank, world_size, master_port, output_writer),
            )
            for rank in range(world_size)
        ]

        for process in processes:
            process.start()
        output_writer.close()

        results = [output_reader.recv() for _ in range(world_size)]
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join()
                self.fail("Gloo phase-isolation subprocess hung")
            self.assertEqual(process.exitcode, 0)

        by_rank = {rank: (bootstrap, transfer) for rank, bootstrap, transfer in results}
        self.assertEqual(
            by_rank[0],
            (
                [int(KVPoll.WaitingForInput), int(KVPoll.Bootstrapping)],
                [int(KVPoll.Success)],
            ),
        )
        self.assertEqual(
            by_rank[1],
            (
                [int(KVPoll.WaitingForInput)],
                [int(KVPoll.Success), int(KVPoll.Bootstrapping)],
            ),
        )


if __name__ == "__main__":
    unittest.main()
