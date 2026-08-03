"""Focused tests for UnifiedRadixCache HiCache PP completion sync."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.distributed.communication_tags import P2PTag
from sglang.srt.environ import envs
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeWork:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class _Holder:
    _uses_batched_hicache_pp_sync = (
        UnifiedRadixCache._uses_batched_hicache_pp_sync
    )
    _count_ready_hicache_acks = staticmethod(
        UnifiedRadixCache._count_ready_hicache_acks
    )
    _sync_hicache_completion_counts = (
        UnifiedRadixCache._sync_hicache_completion_counts
    )
    _validated_hicache_acks = UnifiedRadixCache._validated_hicache_acks


class TestUnifiedRadixCachePPSync(unittest.TestCase):
    @staticmethod
    def _make_batched_holder(
        *,
        write_acks=None,
        load_acks=None,
        ongoing_writes=None,
        ongoing_loads=None,
    ):
        holder = _Holder()
        holder._hicache_pp_sync_mode = "batched"
        holder._hicache_pp_sync_counts = torch.zeros(2, dtype=torch.int32)
        holder.cache_controller = SimpleNamespace(
            ack_write_queue=list(write_acks or []),
            ack_load_queue=list(load_acks or []),
        )
        holder.ongoing_write_through = dict(ongoing_writes or {})
        holder.ongoing_load_back = dict(ongoing_loads or {})
        holder.pp_rank = 0
        holder.enable_storage = False
        holder.enable_storage_metrics = False
        holder.storage_metrics_collector = None
        holder._all_reduce = mock.Mock()
        holder.dec_lock_ref = mock.Mock()
        holder.dec_host_lock_ref = mock.Mock()
        return holder

    @staticmethod
    def _make_batched_init_args(**overrides):
        values = dict(
            hicache_storage_backend=None,
            hicache_write_policy="write_through",
            enable_eic_cache=False,
            enable_dp_attention=False,
            dp_size=1,
            tp_size=2,
            pp_size=4,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_batched_mode_initializes_without_a_new_process_group(self):
        holder = _Holder()
        holder.pp_size = 4
        kvcache = object.__new__(DeepSeekV4TokenToKVPool)
        params = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=mock.Mock(return_value=kvcache)
            )
        )
        with envs.SGLANG_HICACHE_PP_SYNC_MODE.override(
            "batched"
        ), mock.patch.object(torch.distributed, "new_group") as new_group:
            UnifiedRadixCache._init_hicache_pp_sync_mode(
                holder, self._make_batched_init_args(), params
            )

        new_group.assert_not_called()
        self.assertEqual(holder._hicache_pp_sync_mode, "batched")
        self.assertFalse(hasattr(holder, "_hicache_pp_sync_group"))
        torch.testing.assert_close(
            holder._hicache_pp_sync_counts,
            torch.zeros(2, dtype=torch.int32),
        )

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

    def test_batched_sync_queries_pp0_and_propagates_one_vector(self):
        write_event = mock.Mock()
        write_event.query.return_value = True
        load_event = mock.Mock()
        load_event.query.return_value = True
        holder = self._make_batched_holder(
            write_acks=[(None, write_event, [11])],
            load_acks=[(None, load_event, [12])],
        )

        counts = UnifiedRadixCache._sync_hicache_completion_counts(holder)

        self.assertEqual(counts, (1, 1))
        write_event.query.assert_called_once()
        load_event.query.assert_called_once()
        holder._all_reduce.assert_called_once()
        synced_counts, reduce_op = holder._all_reduce.call_args.args
        torch.testing.assert_close(
            synced_counts, torch.tensor([1, 1], dtype=torch.int32)
        )
        self.assertEqual(reduce_op, torch.distributed.ReduceOp.MIN)

    def test_batched_sync_nonzero_pp_stage_uses_propagated_counts(self):
        write_event = mock.Mock()
        load_event = mock.Mock()
        holder = self._make_batched_holder(
            write_acks=[(None, write_event, [11])],
            load_acks=[(None, load_event, [12])],
        )
        holder.pp_rank = 1

        def propagate_counts(counts, _op):
            counts.copy_(torch.tensor([1, 1], dtype=torch.int32))

        holder._all_reduce.side_effect = propagate_counts

        counts = UnifiedRadixCache._sync_hicache_completion_counts(holder)

        self.assertEqual(counts, (1, 1))
        write_event.query.assert_not_called()
        load_event.query.assert_not_called()
        holder._all_reduce.assert_called_once()

    def test_batched_empty_queues_still_participate_in_one_sync(self):
        holder = self._make_batched_holder()

        counts = UnifiedRadixCache._sync_hicache_completion_counts(holder)

        self.assertEqual(counts, (0, 0))
        holder._all_reduce.assert_called_once()

    def test_batched_check_consumes_both_after_local_event_completion(self):
        write_event = mock.Mock()
        write_event.query.return_value = True
        load_event = mock.Mock()
        load_event.query.return_value = True
        write_node, write_lock = object(), object()
        load_node, device_lock, host_lock = object(), object(), object()
        holder = self._make_batched_holder(
            write_acks=[(None, write_event, [11])],
            load_acks=[(None, load_event, [12])],
            ongoing_writes={11: (write_node, write_lock)},
            ongoing_loads={12: (load_node, device_lock, host_lock)},
        )
        holder._drain_async_work = mock.Mock()
        call_order = []
        write_event.synchronize.side_effect = lambda: call_order.append("write_event")
        load_event.synchronize.side_effect = lambda: call_order.append("load_event")
        holder.dec_lock_ref.side_effect = lambda *_: call_order.append("device_unlock")
        holder.dec_host_lock_ref.side_effect = lambda *_: call_order.append(
            "host_unlock"
        )

        UnifiedRadixCache.check_hicache_events(holder)

        holder._drain_async_work.assert_called_once()
        holder._all_reduce.assert_called_once()
        write_event.synchronize.assert_called_once()
        load_event.synchronize.assert_called_once()
        self.assertEqual(
            call_order,
            [
                "write_event",
                "load_event",
                "device_unlock",
                "device_unlock",
                "host_unlock",
            ],
        )
        holder.dec_lock_ref.assert_has_calls(
            [
                mock.call(write_node, write_lock),
                mock.call(load_node, device_lock),
            ]
        )
        holder.dec_host_lock_ref.assert_called_once_with(load_node, host_lock)
        self.assertEqual(holder.cache_controller.ack_write_queue, [])
        self.assertEqual(holder.cache_controller.ack_load_queue, [])

    def test_batched_nonzero_pp_stage_waits_for_its_local_event(self):
        finish_event = mock.Mock()
        finish_event.query.return_value = False
        node, device_lock, host_lock = object(), object(), object()
        holder = self._make_batched_holder(
            load_acks=[(None, finish_event, [13])],
            ongoing_loads={13: (node, device_lock, host_lock)},
        )
        holder.pp_rank = 1
        holder._drain_async_work = mock.Mock()

        def propagate_load_count(counts, _op):
            counts.copy_(torch.tensor([0, 1], dtype=torch.int32))

        holder._all_reduce.side_effect = propagate_load_count

        UnifiedRadixCache.check_hicache_events(holder)

        finish_event.query.assert_not_called()
        finish_event.synchronize.assert_called_once()
        holder.dec_lock_ref.assert_called_once_with(node, device_lock)
        holder.dec_host_lock_ref.assert_called_once_with(node, host_lock)
        self.assertEqual(holder.cache_controller.ack_load_queue, [])

    def test_batched_scheduler_flush_does_not_repeat_collective(self):
        write_event = mock.Mock()
        write_event.query.return_value = True
        write_node, write_lock = object(), object()
        holder = self._make_batched_holder(
            write_acks=[(None, write_event, [23])],
            ongoing_writes={23: (write_node, write_lock)},
        )

        UnifiedRadixCache.flush_write_through_acks(holder)

        holder._all_reduce.assert_not_called()
        write_event.query.assert_not_called()
        self.assertEqual(len(holder.cache_controller.ack_write_queue), 1)
        self.assertIn(23, holder.ongoing_write_through)

    def test_batched_consume_fails_before_mutation_on_missing_ack_id(self):
        finish_event = mock.Mock()
        finish_event.query.return_value = True
        holder = self._make_batched_holder(
            write_acks=[(None, finish_event, [31])],
        )

        with self.assertRaisesRegex(RuntimeError, "write ACK IDs diverged"):
            UnifiedRadixCache._validated_hicache_acks(
                holder,
                holder.cache_controller.ack_write_queue,
                holder.ongoing_write_through,
                1,
                kind="write",
            )

        self.assertEqual(len(holder.cache_controller.ack_write_queue), 1)
        finish_event.synchronize.assert_not_called()
        finish_event.query.assert_not_called()

    def test_batched_consume_fails_before_mutation_on_short_queue(self):
        holder = self._make_batched_holder(ongoing_writes={41: (object(), object())})

        with self.assertRaisesRegex(RuntimeError, "write ACK queue diverged"):
            UnifiedRadixCache._validated_hicache_acks(
                holder,
                holder.cache_controller.ack_write_queue,
                holder.ongoing_write_through,
                1,
                kind="write",
            )

        self.assertEqual(holder.cache_controller.ack_write_queue, [])
        self.assertIn(41, holder.ongoing_write_through)


if __name__ == "__main__":
    unittest.main()
