"""Focused tests for completion-only layered HiCache PP synchronization."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hybrid_cache.pp_completion_coordinator import (
    CompletionKind,
    CompletionTargets,
    PPHiCacheCompletionCoordinator,
)
from sglang.srt.mem_cache.hybrid_cache.pp_layered_completion import (
    PPHiCacheLayeredCompletion,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _coordinator(rank: int = 0, world_size: int = 2):
    return PPHiCacheCompletionCoordinator(
        process_group=object(),
        interval_ms=1,
        stall_timeout_s=30,
        rank=rank,
        world_size=world_size,
        gather_fn=lambda _outputs, _local: None,
    )


class _FakeCoordinator:
    def __init__(self, targets: CompletionTargets):
        self._targets = targets
        self.published = []

    def targets(self):
        return self._targets

    def publish_local(self, kind, **state):
        self.published.append((kind, state))

    def report_scheduler_fatal(self, message):
        raise RuntimeError(message)


class TestPPHiCacheCompletionCoordinator(unittest.TestCase):
    def test_ready_prepare_and_commit_are_separate_frontiers(self):
        first = _coordinator(rank=0)
        second = _coordinator(rank=1)

        for coordinator in (first, second):
            coordinator.publish_local(
                CompletionKind.WRITE,
                observed=3,
                ready=3,
                prepared=0,
                committed=0,
                prepared_digest=0,
            )
        first._process_rows(
            [first._snapshot_tensor().tolist(), second._snapshot_tensor().tolist()]
        )
        self.assertEqual(first.targets().write_prepare, 3)
        self.assertEqual(first.targets().write_commit, 0)

        for coordinator in (first, second):
            coordinator.publish_local(
                CompletionKind.WRITE,
                observed=3,
                ready=3,
                prepared=3,
                committed=0,
                prepared_digest=1234,
            )
        first._process_rows(
            [first._snapshot_tensor().tolist(), second._snapshot_tensor().tolist()]
        )
        self.assertEqual(first.targets().write_commit, 3)

    def test_prepared_fingerprint_mismatch_is_fatal(self):
        first = _coordinator(rank=0)
        second = _coordinator(rank=1)
        first.publish_local(
            CompletionKind.LOAD,
            observed=2,
            ready=2,
            prepared=2,
            committed=0,
            prepared_digest=11,
        )
        second.publish_local(
            CompletionKind.LOAD,
            observed=2,
            ready=2,
            prepared=2,
            committed=0,
            prepared_digest=22,
        )

        first._process_rows(
            [first._snapshot_tensor().tolist(), second._snapshot_tensor().tolist()]
        )

        with self.assertRaisesRegex(RuntimeError, "fingerprint diverged"):
            first.targets()

    def test_epoch_reset_requires_no_outstanding_frontier(self):
        coordinator = _coordinator()
        coordinator.publish_local(
            CompletionKind.WRITE,
            observed=1,
            ready=1,
            prepared=0,
            committed=0,
            prepared_digest=0,
        )
        with self.assertRaisesRegex(RuntimeError, "outstanding write"):
            coordinator.reset_epoch()

        coordinator.publish_local(
            CompletionKind.WRITE,
            observed=1,
            ready=1,
            prepared=1,
            committed=1,
            prepared_digest=7,
        )
        coordinator.reset_epoch()
        self.assertEqual(coordinator.targets().write_prepare, 0)

    def test_manager_write_commit_has_no_collective_or_event_wait(self):
        manager = object.__new__(PPHiCacheLayeredCompletion)
        finish_event = mock.Mock()
        finish_event.query.return_value = True
        node = SimpleNamespace(
            key=RadixKey([1, 2, 3, 4]),
            hash_value=["logical-page-hash"],
        )
        lock_params = object()
        cache = SimpleNamespace(
            cache_controller=SimpleNamespace(
                ack_write_queue=[(None, finish_event, [17])],
                ack_load_queue=[],
            ),
            ongoing_write_through={17: (node, lock_params)},
            ongoing_load_back={},
            enable_storage=False,
            pp_rank=2,
            dec_lock_ref=mock.Mock(),
        )
        manager.cache = cache
        manager.coordinator = _FakeCoordinator(
            CompletionTargets(write_prepare=1, write_commit=1)
        )
        manager.observed = {kind: 0 for kind in CompletionKind}
        manager.ready = {kind: 0 for kind in CompletionKind}
        manager.prepared = {kind: 0 for kind in CompletionKind}
        manager.committed = {kind: 0 for kind in CompletionKind}
        manager.prepared_digest = {kind: 0 for kind in CompletionKind}
        manager.committed_digest = {kind: 0 for kind in CompletionKind}

        with mock.patch.object(
            torch.distributed, "all_reduce"
        ) as all_reduce, mock.patch.object(
            torch.distributed, "recv"
        ) as recv, mock.patch.object(
            torch.distributed, "isend"
        ) as isend:
            manager.check_write()

        all_reduce.assert_not_called()
        recv.assert_not_called()
        isend.assert_not_called()
        finish_event.synchronize.assert_not_called()
        cache.dec_lock_ref.assert_called_once_with(node, lock_params)
        self.assertEqual(cache.cache_controller.ack_write_queue, [])
        self.assertEqual(cache.ongoing_write_through, {})


if __name__ == "__main__":
    unittest.main()
