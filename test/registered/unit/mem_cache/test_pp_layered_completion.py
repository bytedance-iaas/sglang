"""Focused tests for minimal layered HiCache PP synchronization."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache.hybrid_cache.pp_layered_completion import (
    PPHiCacheLayeredCompletion,
    _BackgroundFrontier,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeFrontier:
    def __init__(self, prepare=(0, 0), commit=(0, 0)):
        self.prepare = prepare
        self.commit = commit
        self.published = []

    def publish(self, ready, prepared, committed):
        self.published.append(
            (tuple(ready), tuple(prepared), tuple(committed))
        )

    def targets(self):
        return self.prepare, self.commit


class TestBackgroundFrontier(unittest.TestCase):
    def test_targets_use_minimum_ready_and_prepared(self):
        frontier = object.__new__(_BackgroundFrontier)
        frontier._lock = mock.MagicMock()
        frontier._prepare = [0, 0]
        frontier._commit = [0, 0]

        frontier._apply_rows(
            [
                [4, 3, 0, 0, 0, 0],
                [3, 5, 0, 0, 0, 0],
                [5, 4, 0, 0, 0, 0],
            ]
        )

        self.assertEqual(frontier._prepare, [3, 3])
        self.assertEqual(frontier._commit, [0, 0])

        frontier._apply_rows(
            [
                [4, 3, 3, 3, 0, 0],
                [3, 5, 3, 3, 0, 0],
                [5, 4, 3, 3, 0, 0],
            ]
        )
        self.assertEqual(frontier._commit, [3, 3])

    def test_next_prepare_waits_until_all_ranks_commit(self):
        frontier = object.__new__(_BackgroundFrontier)
        frontier._lock = mock.MagicMock()
        frontier._prepare = [3, 0]
        frontier._commit = [3, 0]

        frontier._apply_rows(
            [
                [5, 0, 3, 0, 3, 0],
                [5, 0, 3, 0, 2, 0],
            ]
        )

        self.assertEqual(frontier._prepare, [3, 0])


class TestPPHiCacheLayeredCompletion(unittest.TestCase):
    def test_write_commit_does_not_wait_or_run_collective(self):
        manager = object.__new__(PPHiCacheLayeredCompletion)
        finish_event = mock.Mock()
        finish_event.query.return_value = True
        node = object()
        lock_params = object()
        cache = SimpleNamespace(
            cache_controller=SimpleNamespace(
                ack_write_queue=[(None, finish_event, [17])],
                ack_load_queue=[],
            ),
            ongoing_write_through={17: (node, lock_params)},
            ongoing_load_back={},
            dec_lock_ref=mock.Mock(),
            dec_host_lock_ref=mock.Mock(),
        )
        manager.cache = cache
        manager.ready = [0, 0]
        manager.prepared = [0, 0]
        manager.committed = [0, 0]
        manager.frontier = _FakeFrontier(prepare=(1, 0), commit=(1, 0))

        manager.check_write()

        finish_event.synchronize.assert_not_called()
        cache.dec_lock_ref.assert_called_once_with(node, lock_params)
        self.assertEqual(cache.cache_controller.ack_write_queue, [])
        self.assertEqual(cache.ongoing_write_through, {})


if __name__ == "__main__":
    unittest.main()
