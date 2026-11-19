# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""All persistent pools must register before workers or an atomic attach rollback."""

import threading
import unittest
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.storage import StorageBackendFactory
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 16


class RecordingBackend:
    """A backend that records the registration calls it receives."""

    def __init__(self, order: List[str], fail_on_complete: bool = False):
        self.order = order
        self.fail_on_complete = fail_on_complete
        self.closed = False

    def register_mem_pool_host(self, mem_pool_host: Any) -> None:
        self.order.append("v1")

    def register_mem_host_pool_v2(self, host_pool: Any, host_pool_name: Any) -> None:
        self.order.append(f"v2:{host_pool_name}")

    def on_host_pools_registered(self) -> None:
        self.order.append("complete")
        if self.fail_on_complete:
            raise RuntimeError("registration failed in the backend")

    def close(self) -> None:
        self.closed = True


def _fake_pool(name: str) -> SimpleNamespace:
    """The HostPoolGroup surface, without CUDA."""
    return SimpleNamespace(
        name=name,
        layout="page_first",
        page_size=PAGE,
        device="cpu",
        size=128,
        logical_size=128,
        can_use_write_back_jit=True,
    )


def _entry(name: PoolName, pool: Any, is_anchor: bool = False) -> PoolEntry:
    return PoolEntry(
        name=name,
        host_pool=pool,
        device_pool=SimpleNamespace(),
        layer_mapper=lambda layer: layer,
        is_primary_index_anchor=is_anchor,
    )


def _group() -> HostPoolGroup:
    return HostPoolGroup(
        [
            _entry(PoolName.KV, _fake_pool("kv"), is_anchor=True),
            _entry(PoolName.MAMBA, _fake_pool("mamba")),
        ]
    )


def _controller(cls: type, storage_host_pool: Any, mem_pool_host: Any) -> Any:
    """Only the attributes ``attach_storage_backend`` reads."""
    controller = object.__new__(cls)
    controller.enable_storage = False
    controller.storage_stop_event = threading.Event()
    controller.prefetch_thread = controller.prefetch_io_aux_thread = None
    controller._storage_worker_failures = {}
    controller.prefetch_sync_thread = controller.backup_thread = None
    controller.storage_host_pool = storage_host_pool
    controller.mem_pool_host = mem_pool_host
    controller.page_size = PAGE
    controller.host_memory_mode = "cache"
    controller.tp_group = None
    controller.attn_cp_group = None
    controller.attn_tp_group = None
    controller.pp_group = None
    controller.prefetch_hits_sync_groups = []
    controller.prefetch_completion_sync_groups = []
    if cls is HybridCacheController:
        controller.pp_prefetch_command_thread = None
        controller.pp_prefetch_command_group = None
        controller.extra_host_mem_release_queues = {}
    return controller


def _attach(
    controller: Any,
    backend: RecordingBackend,
    order: List[str],
    host_pools: Optional[List[PoolEntry]] = None,
) -> None:
    """Run ``attach_storage_backend`` with the heavy pieces stubbed out."""
    storage_config = SimpleNamespace(
        is_mla_model=False, extra_config={}, unified_suffixes=None
    )
    # The base controller normally has only an anchor; the hybrid caller
    # supplies the physical entries as part of the same guarded transaction.
    kwargs = {"host_pools": host_pools} if host_pools is not None else {}
    with (
        mock.patch(
            "sglang.srt.managers.cache_controller.get_memory",
            return_value=SimpleNamespace(hicache_storage_key_scheme="rank-suffix"),
        ),
        mock.patch(
            "sglang.srt.managers.cache_controller.get_parallel",
            return_value=SimpleNamespace(attn_dcp_size=1),
        ),
        mock.patch.object(
            StorageBackendFactory, "create_backend", return_value=backend
        ),
        mock.patch.object(
            controller, "_generate_storage_config", return_value=storage_config
        ),
        mock.patch.object(
            controller,
            "_start_storage_threads",
            side_effect=lambda: order.append("threads"),
        ),
    ):
        controller.attach_storage_backend(storage_backend="dynamic", **kwargs)


class TestRegistrationBurstOrder(CustomTestCase):
    def test_hybrid_hook_fires_after_its_v2_burst(self):
        # All physical pools and the completion hook succeed before workers
        # can consume any registered addresses.
        order: List[str] = []
        group = _group()
        controller = _controller(
            HybridCacheController, group.anchor_entry.host_pool, group
        )

        _attach(controller, RecordingBackend(order), order, host_pools=group.entries)

        self.assertEqual(order, ["v1", "v2:kv", "v2:mamba", "complete", "threads"])

    def test_pp_stage_plans_can_differ_but_same_stage_peers_must_agree(self):
        """Different PP layer shards must attach without accepting a TP mismatch."""
        tp, cp, pp = "tp", "cp", "pp"
        rank_sets = {tp: [0, 1], cp: [0, 2], pp: [0, 4]}
        for same_stage_mismatch in (False, True):
            with self.subTest(same_stage_mismatch=same_stage_mismatch):
                order = []
                group = _group()
                controller = _controller(
                    HybridCacheController, group.anchor_entry.host_pool, group
                )
                controller.pp_group = pp
                votes = []

                def all_reduce(tensor, op, groups):
                    votes.append((tensor.numel(), tuple(groups)))
                    if tensor.numel() > 1 and op == torch.distributed.ReduceOp.MAX:
                        # Other stages have different physical owner layers.
                        if pp in groups or same_stage_mismatch:
                            tensor[-1] += 1

                with (
                    mock.patch.object(
                        controller, "_create_sync_groups", return_value=[tp, cp, pp]
                    ),
                    mock.patch.object(
                        controller, "_all_reduce", side_effect=all_reduce
                    ),
                    mock.patch.object(
                        controller,
                        "_storage_registration_identity",
                        return_value="ab" * 32,
                    ),
                    mock.patch(
                        "torch.distributed.get_process_group_ranks",
                        side_effect=rank_sets.__getitem__,
                    ),
                ):
                    if same_stage_mismatch:
                        with self.assertRaisesRegex(RuntimeError, "namespace differs"):
                            _attach(
                                controller,
                                RecordingBackend(order),
                                order,
                                group.entries,
                            )
                        self.assertFalse(controller.enable_storage)
                    else:
                        _attach(
                            controller, RecordingBackend(order), order, group.entries
                        )
                        self.assertTrue(controller.enable_storage)
                        self.assertEqual(order[-1], "threads")
                self.assertTrue(
                    all(pp in groups for size, groups in votes if size == 1)
                )

    def test_load_consumes_external_rows_only_after_side_pools_succeed(self):
        """A failed side-pool allocation leaves pre-reserved FULL rows owned."""
        group = _group()
        controller = _controller(
            HybridCacheController, group.anchor_entry.host_pool, group
        )
        allocator = SimpleNamespace(alloc=mock.Mock(), free=mock.Mock())
        controller.mem_pool_device_allocator = allocator
        controller.load_queue = []
        host = torch.arange(PAGE)
        reserved = torch.arange(PAGE, 2 * PAGE)

        result = controller.load(host, device_indices=reserved)
        self.assertIs(result, reserved)
        self.assertIs(controller.load_queue[0].device_indices, reserved)
        allocator.alloc.assert_not_called()
        controller.load_queue.clear()

        # Drive the real side-pool resolver through a capacity miss.
        group.entry_map[PoolName.MAMBA].device_pool.alloc = lambda size: None
        group.entry_map[PoolName.MAMBA].device_pool.free = mock.Mock()
        sidecar = PoolTransfer(name=PoolName.MAMBA, host_indices=host)
        result = controller.load(host, extra_pools=[sidecar], device_indices=reserved)
        self.assertIsNone(result)
        self.assertEqual(controller.load_queue, [])
        self.assertIsNone(sidecar.device_indices)
        allocator.alloc.assert_not_called()
        allocator.free.assert_not_called()
        torch.testing.assert_close(reserved, torch.arange(PAGE, 2 * PAGE))

    def test_hybrid_hook_failure_rolls_back_before_workers(self):
        # Hybrid registration shares base attach rollback: its final hook
        # cannot leave partially registered storage enabled or start workers.
        order: List[str] = []
        group = _group()
        controller = _controller(
            HybridCacheController, group.anchor_entry.host_pool, group
        )
        backend = RecordingBackend(order, fail_on_complete=True)

        with self.assertRaisesRegex(RuntimeError, "registration failed in the backend"):
            _attach(controller, backend, order, host_pools=group.entries)

        self.assertEqual(order, ["v1", "v2:kv", "v2:mamba", "complete"])
        self.assertTrue(backend.closed)
        self.assertFalse(controller.enable_storage)
        self.assertIsNone(controller.storage_backend)
        self.assertIsNone(controller.storage_backend_type)


if __name__ == "__main__":
    unittest.main()
