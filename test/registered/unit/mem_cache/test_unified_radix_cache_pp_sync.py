"""Focused tests for UnifiedRadixCache HiCache PP completion sync."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.distributed.communication_tags import P2PTag
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeWork:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class _Holder:
    pass


class TestUnifiedRadixCachePPSync(unittest.TestCase):
    def test_completion_reduce_covers_attention_cp_and_tp_groups(self):
        holder = _Holder()
        holder.attn_cp_group = object()
        holder.attn_tp_group = object()
        holder.tp_group = object()
        holder.tp_world_size = 4
        data = torch.tensor(1, dtype=torch.int)

        with mock.patch.object(
            torch.distributed, "get_world_size", return_value=2
        ), mock.patch.object(torch.distributed, "all_reduce") as all_reduce:
            UnifiedRadixCache._all_reduce_attn_groups(
                holder, data, torch.distributed.ReduceOp.MIN
            )

        self.assertEqual(all_reduce.call_count, 2)
        self.assertEqual(
            all_reduce.call_args_list[0].kwargs["group"], holder.attn_cp_group
        )
        self.assertEqual(
            all_reduce.call_args_list[1].kwargs["group"], holder.attn_tp_group
        )

    def test_drain_waits_all_and_clears(self):
        holder = _Holder()
        works = [_FakeWork(), _FakeWork()]
        holder.work_list = list(works)

        UnifiedRadixCache._drain_async_work(holder)

        self.assertTrue(all(work.waited for work in works))
        self.assertEqual(holder.work_list, [])

    def test_pp_sync_uses_dedicated_tag_and_tracks_send(self):
        holder = _Holder()
        holder.pp_rank = 1
        holder.pp_size = 3
        holder.pp_group = object()
        holder.work_list = []
        data = torch.tensor(0, dtype=torch.int)
        send_work = _FakeWork()

        with mock.patch.object(torch.distributed, "recv") as recv, mock.patch.object(
            torch.distributed, "isend", return_value=send_work
        ) as isend:
            UnifiedRadixCache._pp_sync(holder, data)

        recv.assert_called_once_with(
            data,
            group_src=0,
            group=holder.pp_group,
            tag=P2PTag.HIRADIX_PP_SYNC,
        )
        isend.assert_called_once()
        self.assertEqual(isend.call_args.kwargs["group_dst"], 2)
        self.assertEqual(isend.call_args.kwargs["tag"], P2PTag.HIRADIX_PP_SYNC)
        self.assertEqual(holder.work_list, [send_work])

    def test_loading_check_participates_with_empty_local_queue(self):
        holder = _Holder()
        holder.cache_controller = SimpleNamespace(ack_load_queue=[])
        holder.ongoing_load_back = {}
        holder.pp_rank = 1
        holder._all_reduce = mock.Mock()

        UnifiedRadixCache.loading_check(holder)

        holder._all_reduce.assert_called_once()

    def test_loading_check_releases_device_and_host_locks_after_completion(self):
        holder = _Holder()
        finish_event = mock.Mock()
        finish_event.query.return_value = True
        holder.cache_controller = SimpleNamespace(
            ack_load_queue=[(None, finish_event, [7])]
        )
        node = object()
        device_lock = object()
        host_lock = object()
        holder.ongoing_load_back = {7: (node, device_lock, host_lock)}
        holder.pp_rank = 0
        holder._all_reduce = mock.Mock()
        holder.dec_lock_ref = mock.Mock()
        holder.dec_host_lock_ref = mock.Mock()

        UnifiedRadixCache.loading_check(holder)

        finish_event.synchronize.assert_called_once()
        holder.dec_lock_ref.assert_called_once_with(node, device_lock)
        holder.dec_host_lock_ref.assert_called_once_with(node, host_lock)
        self.assertEqual(holder.cache_controller.ack_load_queue, [])

    def test_nonzero_pp_stage_consumes_propagated_write_count(self):
        holder = _Holder()
        finish_event = mock.Mock()
        holder.cache_controller = SimpleNamespace(
            ack_write_queue=[(None, finish_event, [11])]
        )
        node = object()
        lock_params = object()
        holder.ongoing_write_through = {11: (node, lock_params)}
        holder.pp_rank = 1
        holder.enable_storage = False
        holder.dec_lock_ref = mock.Mock()

        def propagate_one(count, _op):
            count.fill_(1)

        holder._all_reduce = mock.Mock(side_effect=propagate_one)

        UnifiedRadixCache.writing_check(holder)

        finish_event.query.assert_not_called()
        finish_event.synchronize.assert_called_once()
        holder.dec_lock_ref.assert_called_once_with(node, lock_params)
        self.assertEqual(holder.cache_controller.ack_write_queue, [])


if __name__ == "__main__":
    unittest.main()
