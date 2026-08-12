import multiprocessing as mp
import socket
import unittest

import torch
import torch.distributed as dist

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_TP_SIZE = 2
_PP_SIZE = 4
_WORLD_SIZE = _TP_SIZE * _PP_SIZE


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_pp_rank(rank: int, port: int, result_queue) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=_WORLD_SIZE,
    )
    try:
        tp_groups = [
            dist.new_group([pp_rank * _TP_SIZE + tp_rank for tp_rank in range(_TP_SIZE)])
            for pp_rank in range(_PP_SIZE)
        ]
        pp_groups = [
            dist.new_group([pp_rank * _TP_SIZE + tp_rank for pp_rank in range(_PP_SIZE)])
            for tp_rank in range(_TP_SIZE)
        ]

        pp_rank, tp_rank = divmod(rank, _TP_SIZE)
        # Mirror #677's topology: only stage 0 decides the vector completion
        # count in its TP group, then each TP lane carries the same decision
        # through its PP group.  The third value models DSpark's accumulated
        # context contribution across four stages.
        payload = torch.tensor(
            [5 - tp_rank, 3 + tp_rank, pp_rank + 1], dtype=torch.int32
        )
        if pp_rank == 0:
            dist.all_reduce(payload[:2], op=dist.ReduceOp.MIN, group=tp_groups[0])
        else:
            prev_rank = (pp_rank - 1) * _TP_SIZE + tp_rank
            received = torch.empty_like(payload)
            dist.recv(received, src=prev_rank, group=pp_groups[tp_rank])
            payload[:2].copy_(received[:2])
            payload[2] += received[2]

        if pp_rank + 1 < _PP_SIZE:
            next_rank = (pp_rank + 1) * _TP_SIZE + tp_rank
            dist.send(payload, dst=next_rank, group=pp_groups[tp_rank])
        else:
            result_queue.put((tp_rank, payload.tolist()))
    finally:
        dist.destroy_process_group()


class TestDSparkPP4Gloo(unittest.TestCase):
    def test_tp2_pp4_completion_and_context_flow(self):
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        port = _free_port()
        processes = [
            context.Process(target=_run_pp_rank, args=(rank, port, result_queue))
            for rank in range(_WORLD_SIZE)
        ]
        for process in processes:
            process.start()

        try:
            results = dict(result_queue.get(timeout=60) for _ in range(_TP_SIZE))
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)

        self.assertEqual(results, {0: [4, 3, 10], 1: [4, 3, 10]})


if __name__ == "__main__":
    unittest.main()
