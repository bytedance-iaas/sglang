"""Focused tests for completion-only layered HiCache PP synchronization."""

import unittest

from sglang.srt.mem_cache.hybrid_cache.pp_completion_coordinator import (
    CompletionKind,
    PPHiCacheCompletionCoordinator,
)
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


if __name__ == "__main__":
    unittest.main()
