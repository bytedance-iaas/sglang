"""CPU contract tests for the MegaMoE single-node topology guard."""

import datetime
import multiprocessing
import os
import socket
import unittest
from unittest.mock import Mock, patch

import torch.distributed as dist

from sglang.srt.distributed import bootstrap
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _mock_moe_ep_group(topology, rank_in_group=0):
    group = Mock()
    group.rank_in_group = rank_in_group
    group.world_size = len(topology)
    group.ranks = [entry["rank"] for entry in topology]
    group.all_gather_object.return_value = topology
    return group


class _GlooMoEEPGroup:
    def __init__(self, rank, world_size):
        self.rank_in_group = rank
        self.world_size = world_size
        self.ranks = list(range(world_size))

    def all_gather_object(self, obj):
        topology = [None] * self.world_size
        dist.all_gather_object(topology, obj)
        return topology


def _run_gloo_guard(rank, world_size, port, node_ranks, result_queue):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    try:
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=15),
        )
        group = _GlooMoEEPGroup(rank, world_size)
        with (
            patch.object(bootstrap, "get_moe_ep_group", return_value=group),
            patch.object(bootstrap.socket, "gethostname", return_value=f"pod-{rank}"),
        ):
            try:
                bootstrap._validate_megamoe_single_node_topology(
                    node_rank=node_ranks[rank]
                )
            except RuntimeError as exc:
                result_queue.put((rank, "error", str(exc)))
            else:
                result_queue.put((rank, "ok", ""))
    except Exception as exc:
        result_queue.put((rank, "worker-error", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _free_tcp_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestMegaMoETopology(unittest.TestCase):
    def _run_two_rank_guard(self, node_ranks):
        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        port = _free_tcp_port()
        processes = [
            context.Process(
                target=_run_gloo_guard,
                args=(rank, 2, port, node_ranks, result_queue),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
        hung = [process for process in processes if process.is_alive()]
        for process in hung:
            process.terminate()
            process.join(timeout=5)
        self.assertFalse(hung, "MegaMoE topology guard Gloo collective hung")
        self.assertTrue(
            all(process.exitcode == 0 for process in processes),
            [process.exitcode for process in processes],
        )
        return sorted(result_queue.get(timeout=2) for _ in range(2))

    def test_accepts_single_node_ep_group(self):
        topology = [
            {
                "rank": rank,
                "rank_in_group": rank,
                "hostname": "pod-a",
                "node_rank": 0,
            }
            for rank in range(8)
        ]
        group = _mock_moe_ep_group(topology)

        with (
            patch.object(bootstrap, "get_moe_ep_group", return_value=group),
            patch.object(bootstrap.dist, "get_rank", return_value=0),
            patch.object(bootstrap.socket, "gethostname", return_value="pod-a"),
        ):
            bootstrap._validate_megamoe_single_node_topology(node_rank=0)

        group.all_gather_object.assert_called_once_with(
            {
                "rank": 0,
                "rank_in_group": 0,
                "hostname": "pod-a",
                "node_rank": 0,
            }
        )

    def test_rejects_cross_node_ep_group(self):
        topology = [
            {
                "rank": 0,
                "rank_in_group": 0,
                "hostname": "pod-a",
                "node_rank": 0,
            },
            {
                "rank": 1,
                "rank_in_group": 1,
                "hostname": "pod-b",
                "node_rank": 1,
            },
        ]
        group = _mock_moe_ep_group(topology)

        with (
            patch.object(bootstrap, "get_moe_ep_group", return_value=group),
            patch.object(bootstrap.dist, "get_rank", return_value=0),
            patch.object(bootstrap.socket, "gethostname", return_value="pod-a"),
            self.assertRaisesRegex(RuntimeError, "single-node.*node_rank.*1"),
        ):
            bootstrap._validate_megamoe_single_node_topology(node_rank=0)

    def test_two_rank_gloo_accepts_single_node_group(self):
        results = self._run_two_rank_guard((0, 0))
        self.assertEqual([status for _, status, _ in results], ["ok", "ok"])

    def test_two_rank_gloo_rejects_cross_node_group_without_hanging(self):
        results = self._run_two_rank_guard((0, 1))
        self.assertEqual([status for _, status, _ in results], ["error", "error"])
        self.assertTrue(
            all("single-node" in message for _, _, message in results), results
        )

    def test_parallel_group_init_only_checks_megamoe_topology(self):
        common_args = dict(
            backend="nccl",
            dist_init_method="tcp://127.0.0.1:12345",
            model_config=Mock(),
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_dp_size=1,
            attn_cp_size=1,
            moe_ep_size=1,
            moe_dp_size=1,
            dcp_size=1,
        )

        with (
            patch.object(bootstrap, "init_distributed_environment") as init_environment,
            patch.object(bootstrap, "initialize_model_parallel"),
            patch.object(bootstrap, "initialize_dp_attention"),
            patch.object(bootstrap, "is_npu", return_value=False),
            patch.object(bootstrap, "_validate_megamoe_single_node_topology") as guard,
        ):
            non_megamoe_args = Mock(
                node_rank=0,
                is_ep_joiner=False,
                is_ep_scale_joiner=False,
                ep_join_rank_offset=0,
                dist_timeout=120,
                moe_a2a_backend="deepep",
                speculative_moe_a2a_backend=None,
                max_ep_size=None,
                enable_pdmux=False,
                enable_symm_mem=False,
            )
            bootstrap._init_parallel_groups(server_args=non_megamoe_args, **common_args)
            guard.assert_not_called()

            megamoe_args = Mock(
                node_rank=0,
                is_ep_joiner=False,
                is_ep_scale_joiner=False,
                ep_join_rank_offset=0,
                dist_timeout=120,
                moe_a2a_backend="megamoe",
                speculative_moe_a2a_backend="megamoe",
                max_ep_size=None,
                enable_pdmux=False,
                enable_symm_mem=False,
            )
            bootstrap._init_parallel_groups(server_args=megamoe_args, **common_args)
            guard.assert_called_once_with(node_rank=0)

            guard.reset_mock()
            megamoe_joiner_args = Mock(
                node_rank=1,
                is_ep_joiner=True,
                is_ep_scale_joiner=False,
                ep_join_rank_offset=0,
                dist_timeout=120,
                moe_a2a_backend="megamoe",
                speculative_moe_a2a_backend="megamoe",
                max_ep_size=None,
                enable_pdmux=False,
                enable_symm_mem=False,
            )
            with self.assertRaisesRegex(RuntimeError, "elastic EP joiners"):
                bootstrap._init_parallel_groups(
                    server_args=megamoe_joiner_args, **common_args
                )
            guard.assert_not_called()
            self.assertEqual(init_environment.call_count, 2)


if __name__ == "__main__":
    unittest.main()
