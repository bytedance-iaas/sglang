import unittest
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation import decode as decode_module
from sglang.srt.disaggregation import decode_kvcache_offload_manager as offload_module
from sglang.srt.disaggregation.decode import (
    DSV4HiSparseSafePrefixProbe,
    DecodePreallocQueue,
    DecodeTransferQueue,
)
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.managers import schedule_batch as schedule_batch_module
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDecodeSWAAccounting(unittest.TestCase):
    def _queue(self, *, page_size: int = 256, sliding_window_size: int = 4096):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.scheduler = SimpleNamespace(sliding_window_size=sliding_window_size)
        queue._uses_swa_tail_prealloc = lambda: True
        queue._is_dsv4_hisparse_swa_tail_prealloc = lambda: True
        queue._logical_kv_page_size = lambda: page_size
        queue._relax_decode_output_reserve = False
        queue.num_reserved_decode_tokens = 128
        return queue

    def _req(self, *, input_len: int, output_len: int, max_new_tokens: int):
        return SimpleNamespace(
            origin_input_ids=[0] * input_len,
            output_ids=[1] * output_len,
            sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
        )

    def test_long_request_swa_reserve_is_window_bounded(self):
        queue = self._queue()
        req = self._req(input_len=32000, output_len=1, max_new_tokens=1500)

        fill_len, swa_len = queue._prealloc_kv_lens(req)
        old_linear_required = swa_len + queue._output_reserve_tokens_for_admission(req)
        bounded_required = queue._swa_required_tokens_for_admission(
            req, fill_len=fill_len
        )

        self.assertEqual(fill_len, 32000)
        self.assertEqual(swa_len, 4096)
        self.assertEqual(old_linear_required, 5596)
        self.assertEqual(bounded_required, 4096 + 255)
        self.assertLess(bounded_required, old_linear_required)

    def test_non_dsv4_swa_reserve_keeps_linear_output_reserve(self):
        queue = self._queue()
        queue._is_dsv4_hisparse_swa_tail_prealloc = lambda: False
        req = self._req(input_len=32000, output_len=1, max_new_tokens=1500)

        fill_len, swa_len = queue._prealloc_kv_lens(req)
        required = queue._swa_required_tokens_for_admission(req, fill_len=fill_len)
        retractable = queue._swa_retractable_len(
            self._req(input_len=32000, output_len=1500, max_new_tokens=1500)
        )

        self.assertEqual(swa_len, 4096)
        self.assertEqual(required, 4096 + 1500)
        self.assertEqual(retractable, 4096 + 1500)

    def test_retractable_swa_len_uses_current_window_tail(self):
        queue = self._queue()
        req = self._req(input_len=32000, output_len=1500, max_new_tokens=1500)

        old_linear_retractable = queue._swa_tail_len(32000) + 1500
        bounded_retractable = queue._swa_retractable_len(req)

        self.assertEqual(old_linear_retractable, 5596)
        self.assertEqual(bounded_retractable, queue._swa_tail_len(33500))
        self.assertLess(bounded_retractable, old_linear_retractable)

    def test_active_swa_reserve_uses_future_tail_growth(self):
        queue = self._queue()
        queue.num_reserved_decode_tokens = 512
        req = self._req(input_len=32000, output_len=1, max_new_tokens=1500)
        queue.scheduler.running_batch = SimpleNamespace(reqs=[req])
        queue.scheduler.waiting_queue = []
        queue.scheduler.last_batch = None
        queue.transfer_queue = SimpleNamespace(queue=[])
        queue._active_req_count = lambda: 1
        queue._active_reserved_tokens = lambda n_active=None: 1500
        queue._last_prebuilt_reqs_needing_reserve = lambda: []
        queue._hisparse_eagle_draft_logical_reserve_tokens = lambda n_active=None: 0
        queue.retracted_queue = []
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            size_swa=10000,
            swa_available_size=lambda: 10000,
        )

        current_tail = queue._swa_tail_len(32000)
        future_tail = queue._swa_tail_len(33500)
        allocatable = queue._swa_tail_allocatable_token_budget(
            retractable_swa_tokens=current_tail,
            count_retracted=False,
            n_active=1,
        )

        # The final tail only grows by 220, but the maximum live tail across the
        # reserved interval reaches one extra page minus one token.
        self.assertEqual(future_tail - current_tail, 220)
        self.assertEqual(allocatable, 10000 - 255)

    def test_active_swa_reserve_does_not_double_count_running_footprint(self):
        queue = self._queue()
        req = self._req(input_len=32000, output_len=1, max_new_tokens=1500)
        queue.scheduler.running_batch = SimpleNamespace(reqs=[req])
        queue.scheduler.waiting_queue = []
        queue.scheduler.last_batch = None
        queue.transfer_queue = SimpleNamespace(queue=[])
        queue._active_req_count = lambda: 1
        queue._active_reserved_tokens = lambda n_active=None: 1500
        queue._last_prebuilt_reqs_needing_reserve = lambda: []
        queue._hisparse_eagle_draft_logical_reserve_tokens = lambda n_active=None: 0
        queue.retracted_queue = []
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            size_swa=10000,
            swa_available_size=lambda: 10000,
        )

        allocatable = queue._swa_tail_allocatable_token_budget(
            count_retracted=False,
            n_active=1,
        )

        self.assertEqual(allocatable, 10000 - 255)

    def test_retracted_full_budget_uses_zero_after_safe_prefix_hit(self):
        queue = self._queue()
        req = self._req(input_len=32000, output_len=1, max_new_tokens=0)
        queue.retracted_queue = [req]
        queue._enable_decode_radix_prefix_reuse = lambda: True
        queue._probe_dsv4_safe_prefix_for_req = lambda *args, **kwargs: (
            DSV4HiSparseSafePrefixProbe(
                matched_prefix_len=32000,
                safe_prefix_len=32000,
                c4_host_ready_prefix_len=8000,
                required_full_tokens_after_prefix=0,
                cold_prefix_or_routing_key_miss=False,
            )
        )

        self.assertEqual(queue._retracted_full_required_tokens(), 0)

    def test_safe_prefix_probe_requires_reuse_env(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.scheduler = SimpleNamespace(
            server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=True),
            hisparse_coordinator=SimpleNamespace(host_radix_cache=None),
        )
        queue._is_dsv4_hisparse_swa_tail_prealloc = lambda: True

        with envs.SGLANG_HISPARSE_DECODE_SAFE_PREFIX_REUSE.override(False):
            self.assertFalse(queue._can_probe_dsv4_safe_prefix())

        with envs.SGLANG_HISPARSE_DECODE_SAFE_PREFIX_REUSE.override(True):
            self.assertTrue(queue._can_probe_dsv4_safe_prefix())

    def test_transfer_metadata_buffer_release_is_idempotent(self):
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        freed = []
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            free=lambda idx: freed.append(idx)
        )
        decode_req = SimpleNamespace(metadata_buffer_index=-1)

        queue._release_metadata_buffer(decode_req)
        self.assertEqual(freed, [])

        decode_req.metadata_buffer_index = 3
        queue._release_metadata_buffer(decode_req)
        queue._release_metadata_buffer(decode_req)

        self.assertEqual(freed, [3])
        self.assertEqual(decode_req.metadata_buffer_index, -1)

    def test_prealloc_abort_releases_allocated_hisparse_req(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        calls = []

        class Receiver:
            def abort(self):
                calls.append("receiver_abort")

            def clear(self):
                calls.append("receiver_clear")

        req = SimpleNamespace(
            rid="rid-1",
            return_logprob=False,
            req_pool_idx=7,
            finished_reason=None,
            bootstrap_room=5,
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=Receiver(),
            metadata_buffer_index=11,
        )
        queue.queue = [decode_req]
        queue.pending_reqs = [decode_req]
        queue.transfer_queue = SimpleNamespace(
            enable_staging=True,
            staging_handler=SimpleNamespace(
                is_staging_room=lambda room: room == 5,
                unregister_decode_req=lambda room: calls.append(
                    ("unregister_staging", room)
                ),
            ),
        )
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            free=lambda idx: calls.append(("free_metadata", idx))
        )
        queue.scheduler = SimpleNamespace(
            enable_hisparse=True,
            batch_result_processor=SimpleNamespace(
                flush_online_c128_pending_for_reqs=lambda reqs, require_forward_progress=True: calls.append(
                    ("flush_c128", [r.rid for r in reqs], require_forward_progress)
                )
            ),
            decode_offload_manager=None,
            hisparse_coordinator=SimpleNamespace(
                request_finished=lambda finished_req: calls.append(
                    ("hisparse_finished", finished_req.rid)
                )
            ),
            output_streamer=SimpleNamespace(
                stream_output=lambda reqs, return_logprob: calls.append(
                    ("stream", [r.rid for r in reqs], return_logprob)
                )
            ),
        )
        queue.tree_cache = SimpleNamespace()
        old_release_kv_cache = decode_module.release_kv_cache
        decode_module.release_kv_cache = lambda released_req, tree_cache, is_insert=True: calls.append(
            ("release_kv", released_req.rid, is_insert)
        )
        try:
            queue.abort_requests("rid-1")
        finally:
            decode_module.release_kv_cache = old_release_kv_cache

        self.assertEqual(queue.queue, [])
        self.assertEqual(queue.pending_reqs, [])
        self.assertIn("receiver_abort", calls)
        self.assertIn("receiver_clear", calls)
        self.assertIn(("flush_c128", ["rid-1"], False), calls)
        self.assertIn(("hisparse_finished", "rid-1"), calls)
        self.assertIn(("release_kv", "rid-1", False), calls)
        self.assertIn(("unregister_staging", 5), calls)
        self.assertIn(("free_metadata", 11), calls)
        self.assertEqual(decode_req.metadata_buffer_index, -1)
        self.assertIn(("stream", ["rid-1"], False), calls)

    def test_pop_preallocated_finished_abort_cleans_entry(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        calls = []

        class Receiver:
            def abort(self):
                calls.append("receiver_abort")

            def clear(self):
                calls.append("receiver_clear")

        req = SimpleNamespace(
            rid="rid-1",
            return_logprob=False,
            req_pool_idx=7,
            finished_reason=FINISH_ABORT("abort"),
            bootstrap_room=5,
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=Receiver(),
            metadata_buffer_index=11,
            waiting_for_input=False,
        )
        queue.queue = [decode_req]
        queue.pending_reqs = [decode_req]
        queue._resolve_pending_reqs = lambda: None
        queue._update_handshake_waiters = lambda rids_to_check=None: None
        queue._uses_swa_tail_prealloc = lambda: False
        queue._allocatable_token_budgets = (
            lambda retractable_tokens=0, count_retracted=False, **kwargs: 1024
        )
        queue.transfer_queue = SimpleNamespace(
            queue=[],
            enable_staging=True,
            staging_handler=SimpleNamespace(
                is_staging_room=lambda room: room == 5,
                unregister_decode_req=lambda room: calls.append(
                    ("unregister_staging", room)
                ),
            ),
        )
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            free=lambda idx: calls.append(("free_metadata", idx))
        )
        queue.scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[]),
            enable_priority_scheduling=False,
            enable_hisparse=True,
            batch_result_processor=SimpleNamespace(
                flush_online_c128_pending_for_reqs=lambda reqs, require_forward_progress=True: calls.append(
                    ("flush_c128", [r.rid for r in reqs], require_forward_progress)
                )
            ),
            decode_offload_manager=None,
            hisparse_coordinator=SimpleNamespace(
                is_dsv4_hisparse=False,
                device_buffer_slots_for_request=lambda req=None: 1,
                request_finished=lambda finished_req: calls.append(
                    ("hisparse_finished", finished_req.rid)
                ),
            ),
            output_streamer=SimpleNamespace(
                stream_output=lambda reqs, return_logprob: calls.append(
                    ("stream", [r.rid for r in reqs], return_logprob)
                )
            ),
        )
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            hisparse_attn_allocator=SimpleNamespace(available_size=lambda: 1024)
        )
        queue.tree_cache = SimpleNamespace()
        old_release_kv_cache = decode_module.release_kv_cache
        decode_module.release_kv_cache = lambda released_req, tree_cache, is_insert=True: calls.append(
            ("release_kv", released_req.rid, is_insert)
        )
        try:
            preallocated, failed = queue.pop_preallocated()
        finally:
            decode_module.release_kv_cache = old_release_kv_cache

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertEqual(queue.pending_reqs, [])
        self.assertIn("receiver_abort", calls)
        self.assertIn("receiver_clear", calls)
        self.assertIn(("flush_c128", ["rid-1"], False), calls)
        self.assertIn(("hisparse_finished", "rid-1"), calls)
        self.assertIn(("release_kv", "rid-1", False), calls)
        self.assertIn(("unregister_staging", 5), calls)
        self.assertIn(("free_metadata", 11), calls)
        self.assertEqual(decode_req.metadata_buffer_index, -1)
        self.assertIsNone(decode_req.kv_receiver)
        self.assertIn(("stream", ["rid-1"], False), calls)

    def test_scheduler_waiting_abort_flushes_c128_without_forward_progress(self):
        scheduler = Scheduler.__new__(Scheduler)
        calls = []
        req = SimpleNamespace(
            rid="rid-1",
            req_pool_idx=7,
            finished_reason=None,
            return_logprob=False,
            mamba_pool_idx=None,
        )

        scheduler.waiting_queue = [req]
        scheduler.enable_hicache_storage = False
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.enable_hisparse = True
        scheduler.hisparse_coordinator = SimpleNamespace(
            request_finished=lambda finished_req: calls.append(
                ("hisparse_finished", finished_req.rid)
            )
        )
        scheduler.batch_result_processor = SimpleNamespace(
            flush_online_c128_pending_for_reqs=lambda reqs, require_forward_progress=True: calls.append(
                ("flush_c128", [r.rid for r in reqs], require_forward_progress)
            )
        )
        scheduler.decode_offload_manager = None
        scheduler.tree_cache = SimpleNamespace()
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(
                send_output=lambda output, output_req: calls.append(
                    ("stream_abort", output_req.rid)
                )
            )
        )
        scheduler.grammar_manager = SimpleNamespace(
            abort_requests=lambda recv_req: calls.append("grammar_abort")
        )
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            abort_requests=lambda rid, abort_all: calls.append(
                ("prealloc_abort", rid, abort_all)
            ),
            retracted_queue=[],
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(
            abort_requests=lambda rid, abort_all: calls.append(
                ("transfer_abort", rid, abort_all)
            )
        )
        scheduler.cur_batch = None
        scheduler.running_batch = SimpleNamespace(
            reqs=[],
            is_empty=lambda: True,
        )

        old_release_kv_cache = scheduler_module.release_kv_cache
        scheduler_module.release_kv_cache = lambda released_req, tree_cache, is_insert=True: calls.append(
            ("release_kv", released_req.rid, is_insert)
        )
        try:
            scheduler.abort_request(AbortReq(rid="rid-1"))
        finally:
            scheduler_module.release_kv_cache = old_release_kv_cache

        self.assertEqual(scheduler.waiting_queue, [])
        self.assertIn(("flush_c128", ["rid-1"], False), calls)
        self.assertIn(("hisparse_finished", "rid-1"), calls)
        self.assertIn(("release_kv", "rid-1", False), calls)

    def test_scheduler_retracted_abort_uses_offload_release(self):
        scheduler = Scheduler.__new__(Scheduler)
        calls = []
        req = SimpleNamespace(
            rid="rid-1",
            req_pool_idx=7,
            finished_reason=None,
            return_logprob=False,
            kv_cache_cpu=object(),
        )

        scheduler.waiting_queue = []
        scheduler.enable_hicache_storage = False
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.enable_hisparse = True
        scheduler.hisparse_coordinator = SimpleNamespace(
            release_retracted_decode_req=lambda released_req: calls.append(
                ("release_retracted", released_req.rid)
            )
        )
        scheduler.batch_result_processor = SimpleNamespace(
            flush_online_c128_pending_for_reqs=lambda *args, **kwargs: calls.append(
                "unexpected_flush"
            )
        )
        scheduler.decode_offload_manager = SimpleNamespace(
            release_on_abort=lambda released_req: calls.append(
                ("offload_release_on_abort", released_req.rid)
            ),
            abort_request=lambda released_req: calls.append(
                ("unexpected_offload_abort_only", released_req.rid)
            ),
        )
        scheduler.tree_cache = SimpleNamespace()
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(
                send_output=lambda output, output_req: calls.append(
                    ("stream_abort", output_req.rid)
                )
            )
        )
        scheduler.grammar_manager = SimpleNamespace(
            abort_requests=lambda recv_req: calls.append("grammar_abort")
        )
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            abort_requests=lambda rid, abort_all: calls.append(
                ("prealloc_abort", rid, abort_all)
            ),
            retracted_queue=[req],
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(
            abort_requests=lambda rid, abort_all: calls.append(
                ("transfer_abort", rid, abort_all)
            )
        )
        scheduler.cur_batch = None
        scheduler.running_batch = SimpleNamespace(
            reqs=[],
            is_empty=lambda: True,
        )

        scheduler.abort_request(AbortReq(rid="rid-1"))

        self.assertEqual(scheduler.disagg_decode_prealloc_queue.retracted_queue, [])
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertFalse(hasattr(req, "kv_cache_cpu"))
        self.assertIn(("offload_release_on_abort", "rid-1"), calls)
        self.assertIn(("release_retracted", "rid-1"), calls)
        self.assertIn(("stream_abort", "rid-1"), calls)
        self.assertNotIn(("unexpected_offload_abort_only", "rid-1"), calls)
        self.assertNotIn("unexpected_flush", calls)

    def test_scheduler_not_idle_while_decode_offload_state_exists(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = SimpleNamespace(is_empty=lambda: True)
        scheduler.chunked_req = None
        scheduler.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
        scheduler.last_batch = None
        scheduler.cur_batch = None
        scheduler.enable_overlap = False
        scheduler.result_queue = []
        scheduler.ps = SimpleNamespace(pp_size=1)
        scheduler.running_mbs = []
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[],
            retracted_queue=[],
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False
        scheduler.decode_offload_manager = SimpleNamespace(
            ongoing_offload={},
            ongoing_backup={},
            aborted_req_ids=set(),
            offloaded_state={"rid-1": object()},
        )

        self.assertFalse(scheduler.is_fully_idle())

        scheduler.decode_offload_manager.offloaded_state = {}
        self.assertTrue(scheduler.is_fully_idle())

    def test_deferred_decode_validation_preserves_requested_output(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.server_args = SimpleNamespace(allow_auto_truncate=False)
        scheduler.model_config = SimpleNamespace(context_len=262144)
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            _can_probe_dsv4_safe_prefix=lambda: True
        )
        scheduler.page_size = 256
        scheduler.max_req_len = 56064
        scheduler.max_total_num_tokens = 56064
        req = SimpleNamespace(
            origin_input_ids=[0] * 66896,
            sampling_params=SimpleNamespace(max_new_tokens=1500),
        )

        scheduler.init_req_max_new_tokens(req)

        self.assertEqual(req.sampling_params.max_new_tokens, 1500)

    def _scheduler_for_prealloc_cap(self, *, running: int, waiting: int):
        scheduler = Scheduler.__new__(Scheduler)
        calls = []

        class Stats:
            def record_staging_diagnostics(self, **kwargs):
                calls.append(("diagnostics", kwargs.get("max_new_prealloc_reqs")))

            def record_poll(self, *, interval_due):
                calls.append(("poll", interval_due))

            def record_poll_skip(self):
                calls.append("poll_skip")

        scheduler.server_args = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False,
            disaggregation_decode_polling_interval=1,
            cuda_graph_max_bs=None,
        )
        scheduler.enable_hisparse = True
        scheduler.req_to_token_pool = SimpleNamespace(size=64)
        scheduler.max_running_requests = 64
        scheduler.running_batch = SimpleNamespace(batch_size=lambda: running)
        scheduler.waiting_queue = [object()] * waiting
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(
            queue=[],
            pop_transferred=lambda: calls.append("pop_transfer") or [],
        )
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            retracted_queue=[],
            queue=[SimpleNamespace(req=SimpleNamespace())],
            resume_retracted_reqs=lambda: [],
            pop_preallocated=lambda max_new_reqs=None: calls.append(
                ("pop_prealloc", max_new_reqs)
            )
            or ([], []),
        )
        scheduler._record_disagg_decode_pipeline_sample = lambda: None
        scheduler._get_disagg_decode_pipeline_stats = lambda: Stats()
        return scheduler, calls

    def test_hisparse_prealloc_is_capped_by_pipeline_slots(self):
        scheduler, calls = self._scheduler_for_prealloc_cap(running=60, waiting=0)

        scheduler.process_decode_queue()

        self.assertIn(("diagnostics", 20), calls)
        self.assertIn(("pop_prealloc", 20), calls)

    def test_hisparse_prealloc_skips_when_pipeline_backlog_full(self):
        scheduler, calls = self._scheduler_for_prealloc_cap(running=64, waiting=16)

        scheduler.process_decode_queue()

        self.assertIn(("diagnostics", 0), calls)
        self.assertNotIn(("pop_prealloc", 0), calls)
        self.assertFalse(
            any(call[0] == "pop_prealloc" for call in calls if isinstance(call, tuple))
        )

    def test_hisparse_over_capacity_prefix_hit_passes_decode_gate(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        req = SimpleNamespace(
            rid="rid-prefix",
            origin_input_ids=[0] * 66896,
            output_ids=[1],
            return_logprob=False,
        )
        queue.scheduler = SimpleNamespace(
            enable_hisparse=True,
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(max_token_pool_size=56064)
            ),
            output_streamer=SimpleNamespace(
                stream_output=lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("prefix hit should not be aborted")
                )
            ),
        )
        queue._can_probe_dsv4_safe_prefix = lambda: True
        queue._probe_dsv4_safe_prefix_for_req = lambda req, fill_len: (
            DSV4HiSparseSafePrefixProbe(
                matched_prefix_len=65536,
                safe_prefix_len=65536,
                c4_host_ready_prefix_len=16384,
                required_full_tokens_after_prefix=2048,
                cold_prefix_or_routing_key_miss=False,
            )
        )
        queue._uses_swa_tail_prealloc = lambda: False

        self.assertFalse(queue._check_if_req_exceed_kv_capacity(req))

    def test_hisparse_over_capacity_prefix_miss_fails_decode_gate(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        calls = []
        req = SimpleNamespace(
            rid="rid-miss",
            origin_input_ids=[0] * 66896,
            output_ids=[1],
            return_logprob=False,
            finished_reason=None,
            time_stats=SimpleNamespace(
                trace_ctx=SimpleNamespace(abort=lambda *args, **kwargs: None)
            ),
        )
        queue.scheduler = SimpleNamespace(
            enable_hisparse=True,
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(max_token_pool_size=56064)
            ),
            output_streamer=SimpleNamespace(
                stream_output=lambda reqs, return_logprob: calls.append(
                    ("stream", [r.rid for r in reqs], return_logprob)
                )
            ),
        )
        queue._can_probe_dsv4_safe_prefix = lambda: True
        queue._probe_dsv4_safe_prefix_for_req = lambda req, fill_len: (
            DSV4HiSparseSafePrefixProbe(
                matched_prefix_len=0,
                safe_prefix_len=0,
                c4_host_ready_prefix_len=0,
                required_full_tokens_after_prefix=66896,
                cold_prefix_or_routing_key_miss=True,
            )
        )

        self.assertTrue(queue._check_if_req_exceed_kv_capacity(req))
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertIn(("stream", ["rid-miss"], False), calls)

    def test_prefill_info_retry_exhaustion_aborts_decode_request(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.kv_manager = SimpleNamespace(
            try_ensure_parallel_info=lambda bootstrap_addr: False
        )
        queue._ensure_retry_count = {}
        queue._ensure_last_attempt_time = {}
        queue._ensure_retry_interval = 0
        queue._max_ensure_retries = 1

        class Receiver:
            def __init__(self):
                self.aborted = False

            def abort(self):
                self.aborted = True

        receiver = Receiver()
        req = SimpleNamespace(rid="rid-1", return_logprob=False, finished_reason=None)
        decode_req = SimpleNamespace(req=req, kv_receiver=receiver)

        ready, remaining = queue._ensure_prefill_info(
            {"127.0.0.1:10000": [decode_req]}
        )

        self.assertEqual(ready, {})
        self.assertEqual(remaining, [])
        self.assertTrue(receiver.aborted)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertIn(
            "Could not fetch prefill parallel info", req.finished_reason.message
        )
        self.assertEqual(queue._ensure_retry_count, {})
        self.assertEqual(queue._ensure_last_attempt_time, {})

    def test_retracted_resume_load_failure_releases_allocated_state(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        calls = []

        class Req(SimpleNamespace):
            def load_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
                calls.append("load_kv_cache")
                raise RuntimeError("load failed")

        req = Req(
            rid="rid-1",
            origin_input_ids=[0] * 8,
            output_ids=[1],
            req_pool_idx=None,
            is_retracted=True,
            finished_reason=None,
        )
        queue.retracted_queue = [req]
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: 1)
        queue.token_to_kv_pool_allocator = SimpleNamespace()
        queue.tree_cache = SimpleNamespace()
        queue._uses_swa_tail_prealloc = lambda: False
        queue._enable_decode_radix_prefix_reuse = lambda: False
        queue._allocatable_token_budgets = lambda **kwargs: 1024
        queue._required_alloc_tokens = lambda fill_len, prefix_len: 0
        queue._output_reserve_tokens_for_admission = lambda req: 0
        queue._future_full_tokens_for_admission = lambda *args, **kwargs: 0

        def pre_alloc(prealloc_req, prefix_indices, prefix_len):
            calls.append("pre_alloc")
            prealloc_req.req_pool_idx = 7

        queue._pre_alloc = pre_alloc
        queue.scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[]),
            enable_hisparse=True,
            batch_result_processor=SimpleNamespace(
                flush_online_c128_pending_for_reqs=lambda reqs, require_forward_progress=True: calls.append(
                    ("flush_c128", [r.rid for r in reqs], require_forward_progress)
                )
            ),
            decode_offload_manager=None,
            hisparse_coordinator=SimpleNamespace(
                request_finished=lambda finished_req: calls.append(
                    ("hisparse_finished", finished_req.rid)
                )
            ),
            output_streamer=SimpleNamespace(
                stream_output=lambda reqs, return_logprob: calls.append(
                    ("stream", [r.rid for r in reqs], return_logprob)
                )
            ),
        )
        req.return_logprob = False
        req.kv_cache_cpu = object()

        old_release_kv_cache = decode_module.release_kv_cache
        decode_module.release_kv_cache = lambda released_req, tree_cache, is_insert=True: calls.append(
            ("release_kv", released_req.rid, is_insert)
        )
        try:
            resumed = queue.resume_retracted_reqs()
        finally:
            decode_module.release_kv_cache = old_release_kv_cache

        self.assertEqual(resumed, [])
        self.assertIn("pre_alloc", calls)
        self.assertIn("load_kv_cache", calls)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertFalse(req.is_retracted)
        self.assertIn(("flush_c128", ["rid-1"], False), calls)
        self.assertIn(("hisparse_finished", "rid-1"), calls)
        self.assertIn(("release_kv", "rid-1", False), calls)
        self.assertIn(("stream", ["rid-1"], False), calls)
        self.assertFalse(hasattr(req, "kv_cache_cpu"))
        self.assertEqual(queue.retracted_queue, [])

    def test_retracted_resume_prealloc_failure_aborts_request(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        calls = []

        req = SimpleNamespace(
            rid="rid-1",
            origin_input_ids=[0] * 8,
            output_ids=[1],
            req_pool_idx=None,
            is_retracted=True,
            finished_reason=None,
            return_logprob=False,
            kv_cache_cpu=object(),
        )
        queue.retracted_queue = [req]
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: 1)
        queue.token_to_kv_pool_allocator = SimpleNamespace()
        queue.tree_cache = SimpleNamespace()
        queue._uses_swa_tail_prealloc = lambda: False
        queue._enable_decode_radix_prefix_reuse = lambda: False
        queue._allocatable_token_budgets = lambda **kwargs: 1024
        queue._required_alloc_tokens = lambda fill_len, prefix_len: 0
        queue._output_reserve_tokens_for_admission = lambda req: 0
        queue._future_full_tokens_for_admission = lambda *args, **kwargs: 0
        queue._pre_alloc = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("prealloc failed")
        )
        queue.scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[]),
            enable_hisparse=True,
            batch_result_processor=SimpleNamespace(
                flush_online_c128_pending_for_reqs=lambda *args, **kwargs: calls.append(
                    "unexpected_flush"
                )
            ),
            decode_offload_manager=None,
            hisparse_coordinator=SimpleNamespace(
                release_retracted_decode_req=lambda released_req: calls.append(
                    ("release_retracted", released_req.rid)
                )
            ),
            output_streamer=SimpleNamespace(
                stream_output=lambda reqs, return_logprob: calls.append(
                    ("stream", [r.rid for r in reqs], return_logprob)
                )
            ),
        )

        resumed = queue.resume_retracted_reqs()

        self.assertEqual(resumed, [])
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertFalse(req.is_retracted)
        self.assertFalse(hasattr(req, "kv_cache_cpu"))
        self.assertIn(("release_retracted", "rid-1"), calls)
        self.assertIn(("stream", ["rid-1"], False), calls)
        self.assertNotIn("unexpected_flush", calls)
        self.assertEqual(queue.retracted_queue, [])

    def test_transfer_cleanup_skips_already_released_req_slot(self):
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        calls = []
        req = SimpleNamespace(rid="rid-1", req_pool_idx=None, bootstrap_room=3)
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=None,
            metadata_buffer_index=-1,
        )
        queue.enable_staging = False
        queue.scheduler = SimpleNamespace(
            enable_hisparse=True,
            batch_result_processor=SimpleNamespace(
                flush_online_c128_pending_for_reqs=lambda *args, **kwargs: calls.append(
                    "unexpected_flush"
                )
            ),
            decode_offload_manager=None,
            hisparse_coordinator=SimpleNamespace(
                request_finished=lambda *args, **kwargs: calls.append(
                    "unexpected_hisparse_finished"
                )
            ),
        )
        queue.tree_cache = SimpleNamespace()
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            free=lambda idx: calls.append(("free_metadata", idx))
        )
        old_release_kv_cache = decode_module.release_kv_cache
        decode_module.release_kv_cache = lambda *args, **kwargs: calls.append(
            "unexpected_release_kv"
        )
        try:
            queue._cleanup_failed_decode_req(decode_req)
        finally:
            decode_module.release_kv_cache = old_release_kv_cache

        self.assertEqual(calls, [])

    def test_transfer_failed_cleanup_releases_staging_once(self):
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        calls = []

        req = SimpleNamespace(
            rid="rid-1",
            req_pool_idx=None,
            bootstrap_room=3,
            return_logprob=False,
            finished_reason=None,
        )
        receiver = SimpleNamespace(
            failure_exception=lambda: RuntimeError("transfer failed"),
            abort=lambda: calls.append("receiver_abort"),
            clear=lambda: calls.append("receiver_clear"),
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=7,
        )
        queue.queue = [decode_req]
        queue.enable_staging = True
        queue._poll_with_staging = lambda: [decode_module.KVPoll.Failed]
        queue.staging_handler = SimpleNamespace(
            is_staging_room=lambda room: room == 3,
            unregister_decode_req=lambda room: calls.append(
                ("unregister_staging", room)
            ),
        )
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            free=lambda idx: calls.append(("free_metadata", idx))
        )
        queue.scheduler = SimpleNamespace(
            enable_hisparse=False,
            decode_offload_manager=None,
            output_streamer=SimpleNamespace(
                stream_output=lambda reqs, return_logprob: calls.append(
                    ("stream", [r.rid for r in reqs], return_logprob)
                )
            ),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
        )
        queue.tree_cache = SimpleNamespace()

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertEqual(decode_req.metadata_buffer_index, -1)
        self.assertEqual(calls.count(("free_metadata", 7)), 1)
        self.assertEqual(calls.count(("unregister_staging", 3)), 1)
        self.assertIn("receiver_abort", calls)
        self.assertIn("receiver_clear", calls)
        self.assertIn(("stream", ["rid-1"], False), calls)

    def test_retract_decode_offload_failure_aborts_request(self):
        batch = ScheduleBatch.__new__(ScheduleBatch)
        calls = []

        class Req(SimpleNamespace):
            def finished(self):
                return False

            def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
                calls.append(("offload", self.rid))
                raise RuntimeError("offload failed")

            def reset_for_retract(self):
                calls.append(("reset_for_retract", self.rid))

        keep_req = Req(
            rid="keep",
            origin_input_ids=[0] * 4,
            output_ids=[1],
            sampling_params=SimpleNamespace(max_new_tokens=8),
            to_finish=None,
        )
        fail_req = Req(
            rid="fail",
            origin_input_ids=[0] * 4,
            output_ids=[1],
            sampling_params=SimpleNamespace(max_new_tokens=8),
            to_finish=None,
        )
        batch.reqs = [keep_req, fail_req]
        batch.check_decode_mem = lambda selected_indices=None: True

        def filter_batch(keep_indices):
            batch.reqs = [batch.reqs[i] for i in keep_indices]

        batch.filter_batch = filter_batch
        batch.req_to_token_pool = SimpleNamespace()
        batch.token_to_kv_pool_allocator = SimpleNamespace()
        batch.tree_cache = SimpleNamespace()
        batch.hisparse_coordinator = SimpleNamespace(
            is_dsv4_hisparse=True,
            retract_decode_req=lambda req: calls.append(("retract_hisparse", req.rid)),
            release_retracted_decode_req=lambda req: calls.append(
                ("release_retracted_hisparse", req.rid)
            ),
        )
        old_release_kv_cache = schedule_batch_module.release_kv_cache
        schedule_batch_module.release_kv_cache = lambda req, tree_cache, is_insert=True: calls.append(
            ("release_kv", req.rid, is_insert)
        )
        try:
            retracted, _, to_abort = batch.retract_decode(
                SimpleNamespace(
                    speculative_algorithm=None,
                    disaggregation_mode="decode",
                )
            )
        finally:
            schedule_batch_module.release_kv_cache = old_release_kv_cache

        self.assertEqual(retracted, [])
        self.assertEqual(to_abort, [fail_req])
        self.assertIsInstance(fail_req.to_finish, FINISH_ABORT)
        self.assertEqual(batch.reqs, [keep_req])
        self.assertIn(("offload", "fail"), calls)
        self.assertIn(("retract_hisparse", "fail"), calls)
        self.assertIn(("release_retracted_hisparse", "fail"), calls)
        self.assertIn(("release_kv", "fail", False), calls)
        self.assertNotIn(("reset_for_retract", "fail"), calls)

    def test_retract_all_skips_offload_failure_requeue(self):
        batch = ScheduleBatch.__new__(ScheduleBatch)
        calls = []

        class Req(SimpleNamespace):
            def finished(self):
                return False

            def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
                calls.append(("offload", self.rid))
                if self.rid == "fail":
                    raise RuntimeError("offload failed")

            def reset_for_retract(self):
                calls.append(("reset_for_retract", self.rid))

        ok_req = Req(rid="ok", req_pool_idx=1, to_finish=None)
        fail_req = Req(rid="fail", req_pool_idx=2, to_finish=None)
        batch.reqs = [ok_req, fail_req]
        batch.req_to_token_pool = SimpleNamespace()
        batch.token_to_kv_pool_allocator = SimpleNamespace()
        batch.tree_cache = SimpleNamespace()
        batch.hisparse_coordinator = SimpleNamespace(
            is_dsv4_hisparse=True,
            retract_decode_req=lambda req: calls.append(("retract_hisparse", req.rid)),
            release_retracted_decode_req=lambda req: calls.append(
                ("release_retracted_hisparse", req.rid)
            ),
        )

        def filter_batch(*, keep_indices=None, **_):
            batch.reqs = [batch.reqs[i] for i in keep_indices]

        batch.filter_batch = filter_batch
        old_release_kv_cache = schedule_batch_module.release_kv_cache
        old_evict = schedule_batch_module.evict_from_tree_cache
        schedule_batch_module.release_kv_cache = lambda req, tree_cache, is_insert=True: calls.append(
            ("release_kv", req.rid, is_insert)
        )
        schedule_batch_module.evict_from_tree_cache = lambda *args, **kwargs: calls.append(
            ("evict", kwargs)
        )
        try:
            retracted, to_abort = batch.retract_all(
                SimpleNamespace(
                    speculative_algorithm=None,
                    disaggregation_mode="decode",
                )
            )
        finally:
            schedule_batch_module.release_kv_cache = old_release_kv_cache
            schedule_batch_module.evict_from_tree_cache = old_evict

        self.assertEqual(retracted, [ok_req])
        self.assertEqual(to_abort, [fail_req])
        self.assertEqual(batch.reqs, [])
        self.assertIsInstance(fail_req.to_finish, FINISH_ABORT)
        self.assertIn(("reset_for_retract", "ok"), calls)
        self.assertNotIn(("reset_for_retract", "fail"), calls)
        self.assertIn(("release_retracted_hisparse", "fail"), calls)

    def test_release_req_aborted_request_does_not_create_retracted_state(self):
        batch = ScheduleBatch.__new__(ScheduleBatch)
        calls = []

        class Req(SimpleNamespace):
            def finished(self):
                return False

            def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
                calls.append(("unexpected_offload", self.rid))

            def reset_for_retract(self):
                calls.append(("unexpected_reset", self.rid))

        req = Req(rid="abort", req_pool_idx=3, to_finish=FINISH_ABORT("abort"))
        batch.reqs = [req]
        batch.req_to_token_pool = SimpleNamespace()
        batch.token_to_kv_pool_allocator = SimpleNamespace()
        batch.tree_cache = SimpleNamespace()
        batch.hisparse_coordinator = SimpleNamespace(
            is_dsv4_hisparse=True,
            request_finished=lambda finished_req: calls.append(
                (
                    "hisparse_finished",
                    finished_req.rid,
                    isinstance(finished_req.finished_reason, FINISH_ABORT),
                )
            ),
            retract_decode_req=lambda retracted_req: calls.append(
                ("unexpected_retract", retracted_req.rid)
            ),
            retract_req=lambda retracted_req: calls.append(
                ("unexpected_retract", retracted_req.rid)
            ),
        )
        old_release_kv_cache = schedule_batch_module.release_kv_cache
        old_evict = schedule_batch_module.evict_from_tree_cache
        schedule_batch_module.release_kv_cache = lambda req, tree_cache, is_insert=True: calls.append(
            ("release_kv", req.rid, is_insert)
        )
        schedule_batch_module.evict_from_tree_cache = lambda *args, **kwargs: calls.append(
            ("evict", kwargs)
        )
        try:
            batch.release_req(
                0,
                0,
                SimpleNamespace(
                    speculative_algorithm=None,
                    disaggregation_mode="decode",
                ),
            )
        finally:
            schedule_batch_module.release_kv_cache = old_release_kv_cache
            schedule_batch_module.evict_from_tree_cache = old_evict

        self.assertIn(("hisparse_finished", "abort", True), calls)
        self.assertIn(("release_kv", "abort", False), calls)
        self.assertNotIn(("unexpected_offload", "abort"), calls)
        self.assertNotIn(("unexpected_retract", "abort"), calls)
        self.assertNotIn(("unexpected_reset", "abort"), calls)

    def test_retract_offload_success_clears_c128_state_after_cpu_copy(self):
        calls = []
        req = Req.__new__(Req)
        req.rid = "rid-1"
        req.req_pool_idx = 2
        req.origin_input_ids = [0, 1, 2, 3]
        req.output_ids = [4, 5]
        req.mamba_pool_idx = None

        class Allocator:
            def get_cpu_copy(self, indices, mamba_indices=None, req_pool_idx=None):
                calls.append(("copy", indices.cpu().tolist(), req_pool_idx))
                return {"saved": True}

            def get_kvcache(self):
                return SimpleNamespace(
                    clear_c128_req_state=lambda req_pool_idx: calls.append(
                        ("clear_c128_state", req_pool_idx)
                    )
                )

        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(32, dtype=torch.int64).view(4, 8)
        )

        req.offload_kv_cache(req_to_token_pool, Allocator())

        self.assertEqual(req.kv_cache_cpu, {"saved": True})
        self.assertEqual(
            calls,
            [
                ("copy", [16, 17, 18, 19, 20], 2),
                ("clear_c128_state", 2),
            ],
        )

    def test_retract_offload_failure_keeps_c128_state(self):
        calls = []
        req = Req.__new__(Req)
        req.rid = "rid-1"
        req.req_pool_idx = 2
        req.origin_input_ids = [0, 1, 2, 3]
        req.output_ids = [4, 5]
        req.mamba_pool_idx = None

        class Allocator:
            def get_cpu_copy(self, indices, mamba_indices=None, req_pool_idx=None):
                calls.append(("copy", req_pool_idx))
                raise RuntimeError("copy failed")

            def get_kvcache(self):
                calls.append("unexpected_get_kvcache")
                return SimpleNamespace(
                    clear_c128_req_state=lambda req_pool_idx: calls.append(
                        ("unexpected_clear_c128_state", req_pool_idx)
                    )
                )

        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(32, dtype=torch.int64).view(4, 8)
        )

        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            req.offload_kv_cache(req_to_token_pool, Allocator())

        self.assertEqual(calls, [("copy", 2)])
        self.assertFalse(hasattr(req, "kv_cache_cpu"))

    def test_hisparse_request_finished_abort_clears_c128_state(self):
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        calls = []
        req = SimpleNamespace(
            rid="abort",
            req_pool_idx=3,
            finished_reason=FINISH_ABORT("abort"),
            kv_allocated_len=8,
        )

        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            get_kvcache=lambda: SimpleNamespace(
                clear_c128_req_state=lambda req_pool_idx: calls.append(
                    ("clear_c128_state", req_pool_idx)
                )
            )
        )
        coordinator.decode_producer_stream = None
        coordinator.wait_for_pending_backup = lambda: calls.append("wait_backup")
        coordinator._free_device_buffer_slots = lambda finished_req: calls.append(
            ("free_device_buffer", finished_req.rid)
        )
        coordinator.is_dsv4_hisparse = True
        coordinator._dsv4_c4_compressed_locs_for_prefix = (
            lambda req_pool_idx, allocated_len: torch.empty(0, dtype=torch.int64)
        )
        coordinator._release_host_slots = lambda finished_req: calls.append(
            ("release_host", finished_req.rid)
        )
        coordinator._clear_request_runtime_state = lambda req_pool_idx: calls.append(
            ("clear_runtime", req_pool_idx)
        )
        coordinator._clear_c4_prefix_req_state = lambda req_pool_idx: calls.append(
            ("clear_c4_prefix", req_pool_idx)
        )
        coordinator._debug_assert_request_state_cleared = lambda req_pool_idx: calls.append(
            ("assert_clear", req_pool_idx)
        )

        coordinator.request_finished(req)

        self.assertEqual(calls[0], "wait_backup")
        self.assertEqual(calls[1], ("clear_c128_state", 3))
        self.assertIn(("free_device_buffer", "abort"), calls)
        self.assertIn(("release_host", "abort"), calls)
        self.assertIn(("clear_runtime", 3), calls)

    def test_hisparse_partial_prealloc_abort_clears_c128_state(self):
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        calls = []
        req = SimpleNamespace(rid="partial", req_pool_idx=5)

        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            get_kvcache=lambda: SimpleNamespace(
                clear_c128_req_state=lambda req_pool_idx: calls.append(
                    ("clear_c128_state", req_pool_idx)
                )
            )
        )
        coordinator.decode_producer_stream = None
        coordinator.wait_for_pending_backup = lambda: calls.append("wait_backup")
        coordinator._free_device_buffer_slots = lambda aborted_req: calls.append(
            ("free_device_buffer", aborted_req.rid)
        )
        coordinator.host_radix_cache = None
        coordinator.is_dsv4_hisparse = True
        coordinator._release_aborted_dsv4_c4_host_slots = lambda aborted_req: calls.append(
            ("release_dsv4_c4_host", aborted_req.rid)
        )
        coordinator._clear_request_runtime_state = lambda req_pool_idx: calls.append(
            ("clear_runtime", req_pool_idx)
        )
        coordinator._clear_c4_prefix_req_state = lambda req_pool_idx: calls.append(
            ("clear_c4_prefix", req_pool_idx)
        )
        coordinator._debug_assert_request_state_cleared = lambda req_pool_idx: calls.append(
            ("assert_clear", req_pool_idx)
        )

        coordinator.abort_partial_prealloc_req(req)

        self.assertEqual(calls[0], "wait_backup")
        self.assertEqual(calls[1], ("clear_c128_state", 5))
        self.assertIn(("free_device_buffer", "partial"), calls)
        self.assertIn(("release_dsv4_c4_host", "partial"), calls)
        self.assertIn(("clear_runtime", 5), calls)

    def test_decode_offload_finish_releases_hisparse_state(self):
        manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
        calls = []

        class Req(SimpleNamespace):
            def pop_committed_kv_cache(self):
                calls.append("pop_committed")
                return 3

            def pop_overallocated_kv_cache(self):
                calls.append("pop_overallocated")
                return 3, 3

        req = Req(
            rid="rid-1",
            req_pool_idx=2,
            last_node="node",
            swa_uuid_for_lock=17,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
        )
        manager.hisparse_coordinator = SimpleNamespace(
            request_finished=lambda finished_req: calls.append(
                ("hisparse_finished", finished_req.rid)
            )
        )
        manager.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(16, dtype=torch.int64).view(4, 4),
            free=lambda freed_req: calls.append(("free_req", freed_req.rid)),
        )
        manager.token_to_kv_pool_allocator = SimpleNamespace(
            free=lambda indices: calls.append(("free_kv", indices.cpu().tolist()))
        )
        manager.tree_cache = SimpleNamespace(
            dec_lock_ref=lambda node, params=None: calls.append(
                (
                    "dec_lock_ref",
                    node,
                    getattr(params, "swa_uuid_for_lock", None),
                )
            )
        )
        manager.page_size = 1
        manager.offloaded_state = {"rid-1": object()}

        manager._release_finished_req(req, start_offset=1)

        self.assertEqual(calls[0], ("hisparse_finished", "rid-1"))
        self.assertIn("pop_committed", calls)
        self.assertIn(("free_kv", [9, 10]), calls)
        self.assertIn(("dec_lock_ref", "node", 17), calls)
        self.assertIn(("free_req", "rid-1"), calls)
        self.assertNotIn("rid-1", manager.offloaded_state)

    def test_decode_offload_finish_waits_for_all_write_acks(self):
        manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
        calls = []

        class Event:
            def synchronize(self):
                calls.append("sync")

        class Req(SimpleNamespace):
            def finished(self):
                return True

        req = Req(rid="rid-1", req_pool_idx=0)
        manager.cache_controller = SimpleNamespace(
            ack_write_queue=[(None, Event(), [1])],
        )
        manager.ongoing_offload = {
            1: (
                req,
                torch.tensor([1]),
                torch.arange(10, 20, dtype=torch.int64),
                [11],
                1.0,
                10,
                20,
            ),
            2: (
                req,
                torch.tensor([2]),
                torch.arange(20, 30, dtype=torch.int64),
                [22],
                2.0,
                20,
                30,
            ),
        }
        manager.ongoing_backup = {}
        manager.aborted_req_ids = set()
        manager.offloaded_state = {
            "rid-1": SimpleNamespace(prefill_len=0, inc_len=30, last_hash=None)
        }
        manager.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(64, dtype=torch.int64).view(1, 64)
        )
        manager.token_to_kv_pool_allocator = SimpleNamespace(
            free=lambda indices: calls.append(("free", indices.cpu().tolist()))
        )
        manager._trigger_backup = lambda *args: calls.append(
            ("backup", args[1].cpu().tolist(), args[2])
        ) or "hash"
        manager._release_finished_req = lambda released_req, start_offset: calls.append(
            ("release_finished", released_req.rid, start_offset)
        )

        manager._check_offload_progress(1)

        self.assertIn(("free", list(range(10, 20))), calls)
        self.assertIn(("backup", [1], [11]), calls)
        self.assertNotIn(("release_finished", "rid-1", 30), calls)

        manager.cache_controller.ack_write_queue = [(None, Event(), [2])]
        manager._check_offload_progress(1)

        self.assertIn(("free", list(range(20, 30))), calls)
        self.assertIn(("backup", [2], [22]), calls)
        self.assertIn(("release_finished", "rid-1", 30), calls)

    def test_decode_offload_abort_waits_for_write_ack_before_host_free(self):
        manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
        calls = []
        req = SimpleNamespace(rid="rid-1", req_pool_idx=None)
        host_indices = torch.tensor([3, 4], dtype=torch.int64)
        manager.offloaded_state = {"rid-1": object()}
        manager.aborted_req_ids = set()
        manager.ongoing_offload = {
            1: (
                req,
                host_indices,
                torch.tensor([7, 8], dtype=torch.int64),
                [1, 2],
                0.0,
                0,
                2,
            )
        }
        manager.ongoing_backup = {}
        manager.decode_host_mem_pool = SimpleNamespace(
            free=lambda indices: calls.append(("free_host", indices.cpu().tolist()))
        )
        manager.token_to_kv_pool_allocator = SimpleNamespace(
            free=lambda indices: calls.append(("free_device", indices.cpu().tolist()))
        )
        manager.cache_controller = SimpleNamespace(
            ack_write_queue=[
                (
                    None,
                    SimpleNamespace(synchronize=lambda: calls.append("sync")),
                    [1],
                )
            ]
        )

        manager.abort_request(req)

        self.assertEqual(calls, [])
        self.assertIn("rid-1", manager.offloaded_state)
        self.assertIn("rid-1", manager.aborted_req_ids)

        manager._check_offload_progress(1)

        self.assertIn("sync", calls)
        self.assertIn(("free_device", [7, 8]), calls)
        self.assertIn(("free_host", [3, 4]), calls)
        self.assertEqual(manager.ongoing_offload, {})
        self.assertNotIn("rid-1", manager.aborted_req_ids)
        self.assertNotIn("rid-1", manager.offloaded_state)

    def test_decode_offload_host_write_failure_does_not_create_offload_state(self):
        manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
        calls = []
        req = SimpleNamespace(
            rid="rid-1",
            req_pool_idx=0,
            origin_input_ids=[1, 2, 3, 4],
            output_ids=[5, 6],
        )
        manager.cache_controller = SimpleNamespace(write=lambda **kwargs: None)
        manager.decode_host_mem_pool = object()
        manager.page_size = 2
        manager.offload_stride = 1
        manager.request_counter = 0
        manager.offloaded_state = {}
        manager.ongoing_offload = {}
        manager.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(16, dtype=torch.int64).view(1, 16)
        )
        manager.token_to_kv_pool_allocator = SimpleNamespace(
            free=lambda indices: calls.append(("free", indices.cpu().tolist()))
        )
        manager._compute_prefix_hash = lambda tokens, prior_hash="": ["h"]

        self.assertFalse(manager.offload_kv_cache(req))

        self.assertEqual(calls, [])
        self.assertEqual(manager.ongoing_offload, {})
        self.assertNotIn("rid-1", manager.offloaded_state)

    def test_decode_offload_without_aligned_increment_does_not_create_state(self):
        manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
        req = SimpleNamespace(
            rid="rid-1",
            req_pool_idx=0,
            origin_input_ids=[1, 2, 3, 4],
            output_ids=[5],
        )
        manager.page_size = 2
        manager.offload_stride = 2
        manager.offloaded_state = {}
        manager.ongoing_offload = {}
        manager.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(16, dtype=torch.int64).view(1, 16)
        )
        manager._compute_prefix_hash = lambda tokens, prior_hash="": ["h"]

        self.assertFalse(manager.offload_kv_cache(req))

        self.assertEqual(manager.ongoing_offload, {})
        self.assertNotIn("rid-1", manager.offloaded_state)

    def test_decode_offload_abort_without_offload_state_uses_normal_release(self):
        manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
        calls = []
        req = SimpleNamespace(
            rid="rid-1",
            req_pool_idx=0,
        )
        manager.offloaded_state = {}
        manager.ongoing_offload = {}
        manager.ongoing_backup = {}
        manager.aborted_req_ids = set()
        manager.hisparse_coordinator = SimpleNamespace(
            request_finished=lambda finished_req: calls.append(
                ("hisparse_finished", finished_req.rid)
            )
        )
        manager.tree_cache = SimpleNamespace()
        manager.token_to_kv_pool_allocator = SimpleNamespace(
            free=lambda indices: calls.append(("unexpected_free", indices))
        )
        old_release_kv_cache = offload_module.release_kv_cache
        offload_module.release_kv_cache = lambda released_req, tree_cache, is_insert=True: calls.append(
            ("release_kv", released_req.rid, is_insert)
        )
        try:
            manager.release_on_abort(req)
        finally:
            offload_module.release_kv_cache = old_release_kv_cache

        self.assertIn(("hisparse_finished", "rid-1"), calls)
        self.assertIn(("release_kv", "rid-1", False), calls)
        self.assertFalse(
            any(call[0] == "unexpected_free" for call in calls if isinstance(call, tuple))
        )

    def test_decode_offload_inflight_abort_keeps_state_until_release(self):
        manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
        calls = []

        class Event:
            def synchronize(self):
                calls.append("sync")

        class Req(SimpleNamespace):
            def pop_committed_kv_cache(self):
                return 8

            def pop_overallocated_kv_cache(self):
                return 8, 8

        req = Req(
            rid="rid-1",
            req_pool_idx=0,
            origin_input_ids=[0, 1, 2, 3],
            output_ids=[4, 5],
            last_node="node",
            swa_uuid_for_lock=3,
        )
        manager.cache_controller = SimpleNamespace(
            ack_write_queue=[(None, Event(), [1])]
        )
        manager.offloaded_state = {
            "rid-1": SimpleNamespace(prefill_len=4, inc_len=2, last_hash=None)
        }
        manager.aborted_req_ids = set()
        manager.ongoing_offload = {
            1: (
                req,
                torch.tensor([10, 11], dtype=torch.int64),
                torch.tensor([4, 5], dtype=torch.int64),
                [4, 5],
                0.0,
                4,
                6,
            )
        }
        manager.ongoing_backup = {}
        manager.decode_host_mem_pool = SimpleNamespace(
            free=lambda indices: calls.append(("free_host", indices.cpu().tolist()))
        )
        manager.token_to_kv_pool_allocator = SimpleNamespace(
            free=lambda indices: calls.append(("free_device", indices.cpu().tolist()))
        )
        manager.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(16, dtype=torch.int64).view(1, 16),
            free=lambda freed_req: calls.append(("free_req", freed_req.rid)),
        )
        manager.hisparse_coordinator = SimpleNamespace(
            request_finished=lambda finished_req: calls.append(
                ("hisparse_finished", finished_req.rid)
            )
        )
        manager.tree_cache = SimpleNamespace(
            dec_lock_ref=lambda node, params=None: calls.append(
                ("dec_lock_ref", node, getattr(params, "swa_uuid_for_lock", None))
            )
        )
        manager.page_size = 1

        manager.abort_request(req)
        manager._check_offload_progress(1)

        self.assertIn(("free_device", [4, 5]), calls)
        self.assertIn(("free_host", [10, 11]), calls)
        self.assertIn("rid-1", manager.offloaded_state)

        manager.release_on_abort(req)

        self.assertIn(("free_device", [6, 7]), calls)
        self.assertNotIn(("free_device", [0, 1, 2, 3]), calls)
        self.assertIn(("dec_lock_ref", "node", 3), calls)
        self.assertIn(("free_req", "rid-1"), calls)
        self.assertNotIn("rid-1", manager.offloaded_state)

    def test_decode_offload_release_on_abort_before_write_ack(self):
        manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
        calls = []

        class Event:
            def synchronize(self):
                calls.append("sync")

        class Req(SimpleNamespace):
            def pop_committed_kv_cache(self):
                return 8

            def pop_overallocated_kv_cache(self):
                return 8, 8

        req = Req(
            rid="rid-1",
            req_pool_idx=0,
            origin_input_ids=[0, 1, 2, 3],
            output_ids=[4, 5],
            last_node="node",
            swa_uuid_for_lock=3,
        )
        host_indices = torch.tensor([10, 11], dtype=torch.int64)
        device_indices = torch.tensor([4, 5], dtype=torch.int64)
        manager.cache_controller = SimpleNamespace(
            ack_write_queue=[(None, Event(), [1])]
        )
        manager.offloaded_state = {
            "rid-1": SimpleNamespace(prefill_len=4, inc_len=2, last_hash=None)
        }
        manager.aborted_req_ids = set()
        manager.ongoing_offload = {
            1: (
                req,
                host_indices,
                device_indices,
                [4, 5],
                0.0,
                4,
                6,
            )
        }
        manager.ongoing_backup = {}
        manager.decode_host_mem_pool = SimpleNamespace(
            free=lambda indices: calls.append(("free_host", indices.cpu().tolist()))
        )
        manager.token_to_kv_pool_allocator = SimpleNamespace(
            free=lambda indices: calls.append(("free_device", indices.cpu().tolist()))
        )
        manager.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(16, dtype=torch.int64).view(1, 16),
            free=lambda freed_req: (
                calls.append(("free_req", freed_req.rid)),
                setattr(freed_req, "req_pool_idx", None),
            ),
        )
        manager.hisparse_coordinator = SimpleNamespace(
            request_finished=lambda finished_req: calls.append(
                ("hisparse_finished", finished_req.rid)
            )
        )
        manager.tree_cache = SimpleNamespace(
            dec_lock_ref=lambda node, params=None: calls.append(
                ("dec_lock_ref", node, getattr(params, "swa_uuid_for_lock", None))
            )
        )
        manager.page_size = 1

        manager.release_on_abort(req)

        self.assertIn(("hisparse_finished", "rid-1"), calls)
        self.assertIn(("free_device", [6, 7]), calls)
        self.assertIn(("free_req", "rid-1"), calls)
        self.assertIn("rid-1", manager.aborted_req_ids)
        self.assertNotIn("rid-1", manager.offloaded_state)
        self.assertNotIn(("free_device", [4, 5]), calls)
        self.assertNotIn(("free_host", [10, 11]), calls)

        manager._check_offload_progress(1)

        self.assertIn("sync", calls)
        self.assertIn(("free_device", [4, 5]), calls)
        self.assertIn(("free_host", [10, 11]), calls)
        self.assertEqual(manager.ongoing_offload, {})
        self.assertNotIn("rid-1", manager.aborted_req_ids)

    def test_decode_offload_abort_waits_for_backup_ack_before_host_free(self):
        manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
        calls = []

        class Req(SimpleNamespace):
            def pop_committed_kv_cache(self):
                return 8

            def pop_overallocated_kv_cache(self):
                return 8, 8

        class BackupQueue:
            def get(self):
                return SimpleNamespace(id=2)

        req = Req(
            rid="rid-1",
            req_pool_idx=0,
            origin_input_ids=[0, 1, 2, 3],
            output_ids=[4, 5],
            last_node="node",
            swa_uuid_for_lock=3,
        )
        host_indices = torch.tensor([10, 11], dtype=torch.int64)
        manager.cache_controller = SimpleNamespace(ack_backup_queue=BackupQueue())
        manager.offloaded_state = {
            "rid-1": SimpleNamespace(prefill_len=4, inc_len=2, last_hash=None)
        }
        manager.aborted_req_ids = set()
        manager.ongoing_offload = {}
        manager.ongoing_backup = {2: (req, host_indices, 0.0)}
        manager.decode_host_mem_pool = SimpleNamespace(
            free=lambda indices: calls.append(("free_host", indices.cpu().tolist()))
        )
        manager.token_to_kv_pool_allocator = SimpleNamespace(
            free=lambda indices: calls.append(("free_device", indices.cpu().tolist()))
        )
        manager.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(16, dtype=torch.int64).view(1, 16),
            free=lambda freed_req: (
                calls.append(("free_req", freed_req.rid)),
                setattr(freed_req, "req_pool_idx", None),
            ),
        )
        manager.hisparse_coordinator = SimpleNamespace(
            request_finished=lambda finished_req: calls.append(
                ("hisparse_finished", finished_req.rid)
            )
        )
        manager.tree_cache = SimpleNamespace(
            dec_lock_ref=lambda node, params=None: calls.append(
                ("dec_lock_ref", node, getattr(params, "swa_uuid_for_lock", None))
            )
        )
        manager.page_size = 1

        manager.release_on_abort(req)

        self.assertIn(("hisparse_finished", "rid-1"), calls)
        self.assertIn(("free_device", [6, 7]), calls)
        self.assertIn(("free_req", "rid-1"), calls)
        self.assertNotIn(("free_host", [10, 11]), calls)
        self.assertNotIn("rid-1", manager.offloaded_state)
        self.assertIn("rid-1", manager.aborted_req_ids)

        manager._check_backup_progress(1)

        self.assertIn(("free_host", [10, 11]), calls)
        self.assertEqual(manager.ongoing_backup, {})
        self.assertNotIn("rid-1", manager.aborted_req_ids)

    def test_decode_offload_backup_ack_clears_released_aborted_state(self):
        manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
        calls = []

        class BackupQueue:
            def get(self):
                return SimpleNamespace(id=2)

        req = SimpleNamespace(rid="rid-1", req_pool_idx=None)
        host_indices = torch.tensor([10, 11], dtype=torch.int64)
        manager.cache_controller = SimpleNamespace(ack_backup_queue=BackupQueue())
        manager.offloaded_state = {"rid-1": object()}
        manager.aborted_req_ids = {"rid-1"}
        manager.ongoing_offload = {}
        manager.ongoing_backup = {2: (req, host_indices, 0.0)}
        manager.decode_host_mem_pool = SimpleNamespace(
            free=lambda indices: calls.append(("free_host", indices.cpu().tolist()))
        )

        manager._check_backup_progress(1)

        self.assertIn(("free_host", [10, 11]), calls)
        self.assertEqual(manager.ongoing_backup, {})
        self.assertNotIn("rid-1", manager.aborted_req_ids)
        self.assertNotIn("rid-1", manager.offloaded_state)

    def test_decode_offload_backup_ack_keeps_live_aborted_state_until_release(self):
        manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
        calls = []

        class BackupQueue:
            def get(self):
                return SimpleNamespace(id=2)

        req = SimpleNamespace(rid="rid-1", req_pool_idx=0)
        host_indices = torch.tensor([10, 11], dtype=torch.int64)
        manager.cache_controller = SimpleNamespace(ack_backup_queue=BackupQueue())
        manager.offloaded_state = {"rid-1": object()}
        manager.aborted_req_ids = {"rid-1"}
        manager.ongoing_offload = {}
        manager.ongoing_backup = {2: (req, host_indices, 0.0)}
        manager.decode_host_mem_pool = SimpleNamespace(
            free=lambda indices: calls.append(("free_host", indices.cpu().tolist()))
        )

        manager._check_backup_progress(1)

        self.assertIn(("free_host", [10, 11]), calls)
        self.assertEqual(manager.ongoing_backup, {})
        self.assertNotIn("rid-1", manager.aborted_req_ids)
        self.assertIn("rid-1", manager.offloaded_state)

    def test_pre_alloc_allocator_exception_rolls_back_req_slot(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        calls = []

        class ReqToTokenPool:
            def alloc(self, reqs):
                reqs[0].req_pool_idx = 9
                calls.append(("alloc_req", reqs[0].rid))
                return [9]

            def free(self, req):
                calls.append(("free_req", req.req_pool_idx))
                req.req_pool_idx = None

            def write(self, indices, values):
                calls.append(("write", indices))

        req = SimpleNamespace(
            rid="rid-1",
            origin_input_ids=[0] * 8,
            output_ids=[1],
            req_pool_idx=None,
            kv_allocated_len=3,
            kv_committed_len=3,
            swa_evicted_seqlen=2,
            fill_ids=[7],
            prefix_indices="old-prefix",
            extend_input_len=4,
        )
        queue.req_to_token_pool = ReqToTokenPool()
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=1,
            available_size=lambda: 1024,
            alloc=lambda delta_len: (_ for _ in ()).throw(
                RuntimeError("alloc failed")
            ),
        )
        queue.scheduler = SimpleNamespace(enable_hisparse=False)
        queue.tree_cache = SimpleNamespace()
        queue._required_alloc_tokens = lambda fill_len, prefix_len: 0
        queue._enable_decode_radix_prefix_reuse = lambda: False

        with self.assertRaisesRegex(RuntimeError, "alloc failed"):
            queue._pre_alloc(req)

        self.assertIn(("alloc_req", "rid-1"), calls)
        self.assertIn(("free_req", 9), calls)
        self.assertIsNone(req.req_pool_idx)
        self.assertEqual(req.kv_allocated_len, 3)
        self.assertEqual(req.kv_committed_len, 3)
        self.assertEqual(req.swa_evicted_seqlen, 2)
        self.assertEqual(req.fill_ids, [7])
        self.assertEqual(req.prefix_indices, "old-prefix")
        self.assertEqual(req.extend_input_len, 4)

    def test_pre_alloc_token_write_exception_frees_allocated_kv(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        calls = []

        class FakeKVLoc:
            def __len__(self):
                return 4

            def numel(self):
                return 4

        class ReqToTokenPool:
            def alloc(self, reqs):
                reqs[0].req_pool_idx = 9
                return [9]

            def free(self, req):
                calls.append(("free_req", req.req_pool_idx))
                req.req_pool_idx = None

            def write(self, indices, values):
                calls.append(("write", indices))
                raise RuntimeError("write failed")

        fake_kv_loc = FakeKVLoc()
        req = SimpleNamespace(
            rid="rid-1",
            origin_input_ids=[0] * 8,
            output_ids=[1],
            req_pool_idx=None,
            kv_allocated_len=3,
            kv_committed_len=3,
            swa_evicted_seqlen=2,
            fill_ids=[7],
            prefix_indices="old-prefix",
            extend_input_len=4,
        )
        queue.req_to_token_pool = ReqToTokenPool()
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=1,
            available_size=lambda: 1024,
            alloc=lambda delta_len: fake_kv_loc,
            free=lambda kv_loc: calls.append(("free_kv", kv_loc)),
        )
        queue.scheduler = SimpleNamespace(enable_hisparse=False)
        queue.tree_cache = SimpleNamespace()
        queue._required_alloc_tokens = lambda fill_len, prefix_len: 0
        queue._enable_decode_radix_prefix_reuse = lambda: False

        with self.assertRaisesRegex(RuntimeError, "write failed"):
            queue._pre_alloc(req)

        self.assertIn(("free_kv", fake_kv_loc), calls)
        self.assertIn(("free_req", 9), calls)
        self.assertIsNone(req.req_pool_idx)
        self.assertEqual(req.kv_allocated_len, 3)
        self.assertEqual(req.kv_committed_len, 3)
        self.assertEqual(req.swa_evicted_seqlen, 2)
        self.assertEqual(req.fill_ids, [7])
        self.assertEqual(req.prefix_indices, "old-prefix")
        self.assertEqual(req.extend_input_len, 4)

    def test_pre_alloc_hisparse_prefix_hit_uses_swa_tail_allocator(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        calls = []
        prefix_len = 4096
        fill_len = 8192

        class Req(SimpleNamespace):
            def set_extend_input_len(self, value):
                self.extend_input_len = value

        class ReqToTokenPool:
            def alloc(self, reqs):
                reqs[0].req_pool_idx = 3
                return [3]

            def free(self, req):
                calls.append(("free_req", req.req_pool_idx))
                req.req_pool_idx = None

            def write(self, indices, values):
                calls.append(("write", indices, len(values)))

        class HostPool:
            def __getitem__(self, key):
                _, host_slice = key
                return torch.arange(host_slice.stop, dtype=torch.int64)

        def alloc_extend_swa_tail(
            *,
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            swa_tail_len,
        ):
            calls.append(
                (
                    "alloc_extend_swa_tail",
                    int(prefix_lens_cpu[0].item()),
                    int(seq_lens_cpu[0].item()),
                    extend_num_tokens,
                    swa_tail_len,
                    int(last_loc[0].item()),
                )
            )
            return torch.arange(prefix_len, fill_len, dtype=torch.int64)

        req = Req(
            rid="rid-prefix",
            origin_input_ids=[0] * fill_len,
            output_ids=[1],
            req_pool_idx=None,
            kv_allocated_len=0,
            kv_committed_len=0,
            swa_evicted_seqlen=0,
            fill_ids=[],
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            extend_input_len=0,
        )
        coordinator = SimpleNamespace(
            is_dsv4_hisparse=True,
            device="cpu",
            reserve_dsv4_c4_host_transfer_slots=lambda req, host_len: calls.append(
                ("reserve_c4_host", req.req_pool_idx, host_len)
            ),
            req_to_host_pool=HostPool(),
        )
        queue.req_to_token_pool = ReqToTokenPool()
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=256,
            device="cpu",
            logical_attn_allocator=SimpleNamespace(
                full_available_size=lambda: 1 << 20,
                available_size=lambda: 1 << 20,
            ),
            c4_tokens_for_full_tokens=lambda full_tokens: full_tokens // 4,
            alloc_extend_swa_tail=alloc_extend_swa_tail,
            alloc_logical_only=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("alloc_logical_only should not be used")
            ),
            free=lambda kv_loc: calls.append(("free_kv", len(kv_loc))),
        )
        queue.token_to_kv_pool = SimpleNamespace()
        queue.scheduler = SimpleNamespace(
            enable_hisparse=True,
            sliding_window_size=4096,
            hisparse_coordinator=coordinator,
        )
        queue.kv_manager = SimpleNamespace(kv_args=SimpleNamespace(state_types=[]))
        queue.tree_cache = SimpleNamespace()
        queue._uses_swa_tail_prealloc = lambda: True
        queue._logical_kv_page_size = lambda: 256
        queue._required_alloc_tokens = (
            lambda fill_len, prefix_len: fill_len - prefix_len
        )
        queue._enable_decode_radix_prefix_reuse = lambda: False

        host_indices = queue._pre_alloc(
            req,
            prefix_indices=torch.arange(prefix_len, dtype=torch.int64),
            prefix_len=prefix_len,
        )

        self.assertIn(
            (
                "alloc_extend_swa_tail",
                prefix_len,
                fill_len,
                fill_len - prefix_len,
                4096,
                prefix_len - 1,
            ),
            calls,
        )
        self.assertIn(("reserve_c4_host", 3, fill_len // 4), calls)
        self.assertEqual(host_indices.numel(), fill_len // 4)
        self.assertEqual(req.swa_evicted_seqlen, prefix_len)
        self.assertEqual(req.kv_allocated_len, fill_len)
        self.assertEqual(req.kv_committed_len, fill_len)
        self.assertEqual(req.extend_input_len, fill_len - prefix_len)

    def test_full_only_radix_swa_release_includes_page_margin(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.page_size = 256
        cache.sliding_window_size = 4096
        cache.hisparse_mode = True
        values = torch.arange(32000, dtype=torch.int64)

        tail = cache._swa_release_tail_values(values)

        self.assertEqual(tail.numel(), 4096 + 255)
        self.assertEqual(int(tail[0].item()), 32000 - 4096 - 255)

    def test_normal_radix_swa_release_keeps_window_tail(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.page_size = 256
        cache.sliding_window_size = 4096
        cache.hisparse_mode = False
        values = torch.arange(32000, dtype=torch.int64)

        tail = cache._swa_release_tail_values(values)

        self.assertEqual(tail.numel(), 4096)
        self.assertEqual(int(tail[0].item()), 32000 - 4096)

    def _dsv4_pool_for_state_remap(self):
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool.device = "cpu"
        pool.swa_page_size = 16
        pool.compression_ratios = [4]
        pool.get_ring_size = lambda compress_ratio: 8
        return pool

    def test_dsv4_compress_state_load_collapses_by_state_loc(self):
        dsv4_pool = self._dsv4_pool_for_state_remap()

        class DummyStatePool:
            def __init__(self):
                self.records = []

            def set_state_by_state_loc(self, state_locs, value):
                self.records.append((state_locs.cpu(), value.kv_score.cpu()))

        state_pool = DummyStatePool()
        old_swa_locs = torch.cat(
            [
                torch.arange(0, 8),
                torch.arange(16, 24),
                torch.arange(32, 40),
                torch.arange(48, 56),
            ]
        )
        new_swa_locs = torch.arange(0, 32)
        saved_state_locs = torch.arange(8, dtype=torch.int64)
        copied_state = torch.arange(16, dtype=torch.float32).view(8, 2)

        dsv4_pool._load_remapped_compress_state_pools(
            [copied_state],
            saved_state_locs,
            old_swa_locs,
            new_swa_locs,
            [state_pool],
            4,
        )

        self.assertEqual(len(state_pool.records), 1)
        target_locs, restored_state = state_pool.records[0]
        self.assertEqual(target_locs.tolist(), [0, 1, 2, 3])
        self.assertEqual(restored_state.shape[0], 4)

    def test_dsv4_compress_state_load_rejects_state_split(self):
        dsv4_pool = self._dsv4_pool_for_state_remap()

        with self.assertRaisesRegex(RuntimeError, "cannot split one saved"):
            dsv4_pool._load_remapped_compress_state_pools(
                [None],
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([0, 1, 2, 3], dtype=torch.int64),
                torch.tensor([0, 4, 8, 12], dtype=torch.int64),
                [None],
                4,
            )


if __name__ == "__main__":
    unittest.main()
