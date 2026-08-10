import multiprocessing as mp
import socket
import unittest
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components.dspark_draft import (
    make_next_draft_input,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    validate_dspark_decode_input,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_WORLD_SIZE = 8
_ACTIVE_RANKS_BY_ROUND = (
    frozenset({0}),
    frozenset({0, 1, 2, 3}),
    frozenset(range(8)),
    frozenset({0}),
    frozenset({7}),
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_dp_rank(rank: int, port: int, result_queue) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=_WORLD_SIZE,
    )
    try:
        for round_id, active_ranks in enumerate(_ACTIVE_RANKS_BY_ROUND):
            active = rank in active_ranks
            local_bs = 1 if active else 0
            device = torch.device("cpu")
            batch = SimpleNamespace(
                forward_mode=ForwardMode.DECODE if active else ForwardMode.IDLE,
                seq_lens=torch.full(
                    (local_bs,), 32 + round_id, dtype=torch.int64, device=device
                ),
                req_pool_indices=torch.arange(
                    local_bs, dtype=torch.int64, device=device
                ),
                out_cache_loc=torch.arange(
                    local_bs, dtype=torch.int64, device=device
                ),
                global_num_tokens=[
                    1 if dp_rank in active_ranks else 0
                    for dp_rank in range(_WORLD_SIZE)
                ],
                global_num_tokens_for_logprob=[0] * _WORLD_SIZE,
            )
            draft_input = (
                make_next_draft_input(
                    bonus_tokens=torch.tensor([rank], dtype=torch.int64),
                    new_seq_lens=batch.seq_lens,
                )
                if active
                else DFlashDraftInputV2.create_idle_input(device)
            )
            validate_dspark_decode_input(
                batch=batch,
                draft_input=draft_input,
                dp_rank=rank,
                tp_rank=rank,
                enable_dp_attention=True,
            )

            local_state = torch.tensor(
                [
                    local_bs,
                    draft_input.bonus_tokens.numel(),
                    draft_input.new_seq_lens.numel(),
                ],
                dtype=torch.int64,
            )
            gathered = [torch.empty_like(local_state) for _ in range(_WORLD_SIZE)]
            dist.all_gather(gathered, local_state)
            if rank == 0:
                result_queue.put(
                    (round_id, [tensor.tolist() for tensor in gathered])
                )
    finally:
        dist.destroy_process_group()


class TestDSparkDP8ActiveIdleGloo(unittest.TestCase):
    def test_active_idle_contract_across_rounds(self):
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        port = _free_port()
        processes = [
            context.Process(target=_run_dp_rank, args=(rank, port, result_queue))
            for rank in range(_WORLD_SIZE)
        ]
        for process in processes:
            process.start()

        try:
            results = dict(
                result_queue.get(timeout=90)
                for _ in range(len(_ACTIVE_RANKS_BY_ROUND))
            )
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)

        self.assertEqual(set(results), set(range(len(_ACTIVE_RANKS_BY_ROUND))))
        for round_id, active_ranks in enumerate(_ACTIVE_RANKS_BY_ROUND):
            expected = [
                [1, 1, 1] if rank in active_ranks else [0, 0, 0]
                for rank in range(_WORLD_SIZE)
            ]
            self.assertEqual(results[round_id], expected)


if __name__ == "__main__":
    unittest.main()
