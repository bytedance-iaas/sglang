import unittest
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import torch

from sglang.srt.distributed.pipeline_layout import (
    PipelineControlEnvelope,
    PipelineControlKind,
    PipelinePrefixRegistry,
    PipelineRankSchedule,
    PipelineResourceGate,
    PipelineResourceSnapshot,
    PipelineWavefrontAction,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_pp_mixin import (  # noqa: E402
    PPBatchMetadata,
    SchedulerPPMixin,
)
from sglang.srt.managers.schedule_batch import Req  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Event:
    def __init__(self):
        self.recorded_stream = None

    def record(self, stream):
        self.recorded_stream = stream


class _PollHandle:
    def __init__(self, result=None):
        self.result = result
        self.poll_count = 0

    def poll(self):
        self.poll_count += 1
        return self.result


def _make_scheduler(*, is_first_rank=False, is_last_rank=False):
    scheduler = SchedulerPPMixin()
    scheduler.ps = SimpleNamespace(
        pp_rank=3 if is_last_rank else 1,
        pp_size=4,
        tp_rank=0,
        tp_size=2,
    )
    scheduler.pp_group = SimpleNamespace(
        is_first_rank=is_first_rank,
        is_last_rank=is_last_rank,
    )
    scheduler.world_group = SimpleNamespace(
        rank_in_group=scheduler.ps.pp_rank * scheduler.ps.tp_size,
    )
    scheduler.attn_tp_group = SimpleNamespace(
        broadcast_tensor_dict=MagicMock(
            side_effect=lambda value, src: value or {"next_token_ids": "tokens"}
        )
    )
    scheduler.device = "cpu"
    scheduler.forward_stream_ctx = nullcontext()
    scheduler.forward_stream = MagicMock()
    scheduler.schedule_stream = object()
    scheduler.last_rank_comm_queue = deque()
    scheduler.device_module = SimpleNamespace(
        Event=_Event,
        current_stream=lambda: "forward-stream",
    )
    scheduler._pp_send_dict_to_next_stage = MagicMock(
        side_effect=lambda *_args, **_kwargs: [object()]
    )
    scheduler._pp_commit_comm_work = MagicMock()
    scheduler._pp_recv_vpp_proxy_tensors = MagicMock(
        return_value=PPProxyTensors({"stage": 4})
    )
    scheduler._pp_prepare_tensor_dict = MagicMock(
        return_value={"next_token_ids": "tokens"}
    )
    return scheduler


class TestSchedulerVPP(unittest.TestCase):
    @patch("sglang.srt.managers.scheduler_pp_mixin.get_vpp_pp_reverse_group")
    def test_pp2_activation_transport_separates_peer_directions(
        self, get_reverse_group
    ):
        scheduler = _make_scheduler()
        scheduler.ps.pp_size = 2
        reverse_group = SimpleNamespace(
            recv_tensor_dict_async=MagicMock(return_value=object()),
            send_tensor_dict=MagicMock(return_value=[object()]),
        )
        get_reverse_group.return_value = reverse_group

        scheduler.ps.pp_rank = 0
        scheduler._pp_vpp_pending_recv = None
        scheduler._pp_vpp_start_receiver()
        reverse_group.recv_tensor_dict_async.assert_called_once_with(
            all_gather_group=scheduler.attn_tp_group,
            batch_p2p=True,
            tag=1,
        )

        scheduler.ps.pp_rank = 1
        send_work = SchedulerPPMixin._pp_send_dict_to_next_stage(
            scheduler,
            {"hidden_states": torch.arange(2)},
            msg_type="vpp_proxy",
            batch_p2p=True,
            tag=1,
        )
        self.assertEqual(len(send_work), 1)
        reverse_group.send_tensor_dict.assert_called_once()

    @patch("sglang.srt.managers.scheduler_pp_mixin.get_vpp_pp_reverse_group")
    def test_pp2_activation_transport_keeps_pp0_to_pp1_on_base_group(
        self, get_reverse_group
    ):
        scheduler = _make_scheduler()
        scheduler.ps.pp_size = 2
        scheduler.ps.pp_rank = 0
        scheduler.pp_group.send_tensor_dict = MagicMock(return_value=[object()])

        SchedulerPPMixin._pp_send_dict_to_next_stage(
            scheduler,
            {"hidden_states": torch.arange(2)},
            msg_type="vpp_proxy",
            batch_p2p=True,
            tag=1,
        )

        scheduler.pp_group.send_tensor_dict.assert_called_once()
        get_reverse_group.assert_not_called()

    def test_vpp_event_loop_defaults_to_rank_local_scheduler(self):
        scheduler = SchedulerPPMixin()
        scheduler._event_loop_pp_disagg_prefill_vpp_rank_local = MagicMock(
            return_value="rank-local"
        )

        result = scheduler._event_loop_pp_disagg_prefill_vpp()

        self.assertEqual(result, "rank-local")
        scheduler._event_loop_pp_disagg_prefill_vpp_rank_local.assert_called_once()

    def test_vpp_prefill_uses_dedicated_collective_loop(self):
        scheduler = SchedulerPPMixin()
        scheduler._pp_vpp_enabled = MagicMock(return_value=True)
        scheduler._event_loop_pp_disagg_prefill_vpp = MagicMock()

        scheduler.event_loop_pp_disagg_prefill()

        scheduler._event_loop_pp_disagg_prefill_vpp.assert_called_once_with()

    def test_prewarm_initializes_full_vpp_device_group(self):
        scheduler = SchedulerPPMixin()
        device_group = object()
        scheduler.pp_group = SimpleNamespace(
            device="cuda",
            device_group=device_group,
            device_module=SimpleNamespace(synchronize=MagicMock()),
        )
        warmup_tensor = object()

        with (
            patch.object(scheduler, "_pp_vpp_enabled", return_value=True),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.torch.zeros",
                return_value=warmup_tensor,
            ) as zeros,
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.torch.distributed.all_reduce"
            ) as all_reduce,
        ):
            scheduler._pp_prewarm_vpp_device_group()

        zeros.assert_called_once_with(1, dtype=torch.int32, device="cuda")
        all_reduce.assert_called_once_with(warmup_tensor, group=device_group)
        scheduler.pp_group.device_module.synchronize.assert_called_once_with()

    def test_prewarm_is_noop_without_vpp(self):
        scheduler = SchedulerPPMixin()
        scheduler.pp_group = MagicMock()

        with (
            patch.object(scheduler, "_pp_vpp_enabled", return_value=False),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.torch.distributed.all_reduce"
            ) as all_reduce,
        ):
            scheduler._pp_prewarm_vpp_device_group()

        all_reduce.assert_not_called()

    def test_recv_first_visit_skips_only_on_first_physical_rank(self):
        scheduler = SchedulerPPMixin()
        scheduler.pp_group = SimpleNamespace(is_first_rank=True)
        scheduler.attn_tp_group = object()
        scheduler._pp_recv_typed_dict = MagicMock(return_value={"stage": 4})

        self.assertIsNone(scheduler._pp_recv_vpp_proxy_tensors(first_visit=True))
        scheduler._pp_recv_typed_dict.assert_not_called()

        proxy = scheduler._pp_recv_vpp_proxy_tensors(first_visit=False)

        self.assertEqual(proxy.tensors, {"stage": 4})
        scheduler._pp_recv_typed_dict.assert_called_once_with(
            expected_kind="vpp_proxy",
            all_gather_group=scheduler.attn_tp_group,
            batch_p2p=True,
        )

    def test_vpp_proxy_rejects_wrong_batch_identity(self):
        scheduler = SchedulerPPMixin()
        scheduler.pp_group = SimpleNamespace(is_first_rank=False)
        scheduler.attn_tp_group = object()
        scheduler._pp_recv_typed_dict = MagicMock(
            return_value={"vpp_batch_seq": 3, "vpp_stage_id": 5}
        )

        with self.assertRaisesRegex(RuntimeError, "batch mismatch"):
            scheduler._pp_recv_vpp_proxy_tensors(
                first_visit=False,
                expected_batch_seq=4,
                expected_stage_id=5,
            )

    def test_vpp_receiver_routes_arrival_by_envelope_identity(self):
        scheduler = _make_scheduler()
        scheduler.ps.pp_rank = 1
        scheduler._pp_vpp_pending_recv = None
        scheduler._pp_vpp_ready_proxies = {}
        scheduler._pp_vpp_arrivals = deque()
        scheduler._pp_vpp_slot_batch_seqs = [None, None, None, 7]
        scheduler._pp_vpp_start_receiver = MagicMock()

        scheduler._pp_vpp_accept_received_tensors(
            {
                "__msg_type__": "vpp_proxy",
                "vpp_protocol_version": 1,
                "vpp_batch_seq": 7,
                "vpp_generation": 1,
                "vpp_src_stage_id": 4,
                "vpp_stage_id": 5,
            }
        )
        stage5_action = PipelineWavefrontAction(0, 7, 3, 5, 1)
        self.assertEqual(scheduler._pp_vpp_arrivals.popleft(), (7, 5))
        proxy = scheduler._pp_vpp_take_ready_proxy(stage5_action)
        self.assertEqual(proxy.tensors["vpp_stage_id"], 5)
        scheduler._pp_vpp_start_receiver.assert_called_once_with()

    def test_vpp_receiver_rejects_activation_for_recycled_slot(self):
        scheduler = _make_scheduler()
        scheduler.ps.pp_rank = 1
        scheduler._pp_vpp_ready_proxies = {}
        scheduler._pp_vpp_slot_batch_seqs = [None] * 4

        with self.assertRaisesRegex(RuntimeError, "Stale VPP activation"):
            scheduler._pp_vpp_accept_received_tensors(
                {
                    "__msg_type__": "vpp_proxy",
                    "vpp_protocol_version": 1,
                    "vpp_batch_seq": 7,
                    "vpp_generation": 1,
                    "vpp_src_stage_id": 4,
                    "vpp_stage_id": 5,
                }
            )

    def test_vpp_send_work_polling_preserves_incomplete_payload(self):
        scheduler = _make_scheduler()
        incomplete = MagicMock()
        incomplete.is_completed.return_value = False
        complete = MagicMock()
        complete.is_completed.return_value = True
        pending = deque(
            [
                [SimpleNamespace(work=incomplete, payload=object())],
                [SimpleNamespace(work=complete, payload=object())],
            ]
        )

        scheduler._pp_vpp_reap_send_work(pending)

        self.assertEqual(len(pending), 2)
        incomplete.wait.assert_not_called()
        complete.wait.assert_not_called()

        incomplete.is_completed.return_value = True
        scheduler._pp_vpp_reap_send_work(pending)

        self.assertEqual(len(pending), 0)
        incomplete.wait.assert_called_once_with()
        complete.wait.assert_called_once_with()

    def test_vpp_control_receiver_validates_and_rearms_fifo(self):
        scheduler = _make_scheduler()
        scheduler._pp_vpp_runtime_epoch = 17
        scheduler._pp_vpp_control_layout_digest = "layout"
        scheduler._pp_vpp_pending_control_recv = _PollHandle(
            PipelineControlEnvelope(
                protocol_version=1,
                runtime_epoch=17,
                layout_digest="layout",
                kind=PipelineControlKind.RESOURCE,
                source_rank=2,
                payload={
                    "request_slots": 4,
                    "kv_tokens": 8192,
                    "activation_bytes": 0,
                    "pending_sends": 0,
                },
            ).to_dict()
        )
        scheduler.pp_group.recv_tensor_dict_async = MagicMock(
            return_value=_PollHandle()
        )

        envelope, _ = scheduler._pp_vpp_poll_control_receiver()

        self.assertEqual(envelope.kind, PipelineControlKind.RESOURCE)
        scheduler.pp_group.recv_tensor_dict_async.assert_called_once_with(
            batch_p2p=True,
            tag=2,
        )

    def test_vpp_control_tp0_broadcasts_pp_ring_message_to_tp_lanes(self):
        scheduler = _make_scheduler()
        scheduler._pp_vpp_runtime_epoch = 17
        scheduler._pp_vpp_control_layout_digest = "layout"
        envelope = PipelineControlEnvelope(
            protocol_version=1,
            runtime_epoch=17,
            layout_digest="layout",
            kind=PipelineControlKind.BOOTSTRAP_STATUS,
            source_rank=0,
            hops=1,
            payload={"phase": "collect", "good": [], "bad": []},
        )
        message = (envelope, envelope.to_dict())
        scheduler._pp_vpp_poll_control_receiver = MagicMock(return_value=message)
        scheduler.attn_tp_group.broadcast_object = MagicMock(
            side_effect=lambda value, src: value
        )

        result = scheduler._pp_vpp_poll_control_receiver_tp_broadcast()

        self.assertEqual(result, message)
        scheduler._pp_vpp_poll_control_receiver.assert_called_once_with()
        scheduler.attn_tp_group.broadcast_object.assert_called_once_with(
            envelope.to_dict(),
            src=0,
        )

    def test_vpp_control_tp_follower_only_consumes_tp0_broadcast(self):
        scheduler = _make_scheduler()
        scheduler.ps.tp_rank = 1
        envelope = PipelineControlEnvelope(
            protocol_version=1,
            runtime_epoch=17,
            layout_digest="layout",
            kind=PipelineControlKind.CACHE_UPDATE,
            source_rank=3,
            batch_seq=0,
            generation=0,
            hops=3,
            payload={},
        )
        scheduler._pp_vpp_runtime_epoch = 17
        scheduler._pp_vpp_control_layout_digest = "layout"
        scheduler._pp_vpp_poll_control_receiver = MagicMock()
        scheduler.attn_tp_group.broadcast_object = MagicMock(
            return_value=envelope.to_dict()
        )

        result = scheduler._pp_vpp_poll_control_receiver_tp_broadcast()

        self.assertEqual(result[0], envelope)
        scheduler._pp_vpp_poll_control_receiver.assert_not_called()
        scheduler.attn_tp_group.broadcast_object.assert_called_once_with(None, src=0)

    def test_vpp_control_tp_follower_does_not_use_pp_ring(self):
        scheduler = _make_scheduler()
        scheduler.ps.tp_rank = 1
        scheduler._pp_vpp_pending_control_recv = None
        scheduler._pp_vpp_control_outbox = deque()
        scheduler.pp_group.recv_tensor_dict_async = MagicMock()
        scheduler.pp_group.send_tensor_dict = MagicMock()
        envelope = PipelineControlEnvelope(
            protocol_version=1,
            runtime_epoch=17,
            layout_digest="layout",
            kind=PipelineControlKind.RESOURCE,
            source_rank=1,
        )

        scheduler._pp_vpp_start_control_receiver()
        scheduler._pp_vpp_queue_control(envelope)
        scheduler._pp_vpp_flush_control_outbox(deque(), 4)

        scheduler.pp_group.recv_tensor_dict_async.assert_not_called()
        scheduler.pp_group.send_tensor_dict.assert_not_called()
        self.assertEqual(len(scheduler._pp_vpp_control_outbox), 0)

    def test_vpp_resource_snapshot_tracks_allocator_and_activation_bytes(self):
        scheduler = _make_scheduler()
        scheduler.req_to_token_pool = SimpleNamespace(
            available_size=MagicMock(return_value=3)
        )
        scheduler.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=MagicMock(return_value=5)
        )
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            full_available_size=MagicMock(return_value=9000),
            swa_available_size=MagicMock(return_value=7000),
        )
        scheduler._pp_vpp_ready_proxies = {
            (0, 1): PPProxyTensors(
                {"hidden_states": torch.empty(8, dtype=torch.float32)}
            )
        }

        snapshot = scheduler._pp_vpp_resource_snapshot(2)

        self.assertEqual(snapshot.request_slots, 3)
        self.assertEqual(snapshot.kv_tokens, 7000)
        self.assertEqual(snapshot.activation_bytes, 32)
        self.assertEqual(snapshot.pending_sends, 2)
        self.assertEqual(snapshot.metadata_slots, 5)

    @patch(
        "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
        return_value=SimpleNamespace(pp_virtual_stages=2),
    )
    def test_vpp_prefix_plan_preserves_physical_kv_frontier(self, _get_parallel):
        scheduler = _make_scheduler()
        scheduler._pp_vpp_prefix_registry = PipelinePrefixRegistry()
        req = SimpleNamespace(
            rid="r0",
            session_generation=None,
            extend_range=SimpleNamespace(start=0, end=256),
            kv=SimpleNamespace(kv_committed_len=4, kv_allocated_len=4),
            skip_radix_cache_insert=False,
        )

        scheduler._pp_vpp_register_prefix_batch(SimpleNamespace(reqs=[req]))

        entry = scheduler._pp_vpp_prefix_registry.get("r0", 0)
        self.assertEqual(entry.planned_end, 256)
        self.assertEqual(req.kv.kv_committed_len, req.kv.kv_allocated_len)
        self.assertTrue(req.skip_radix_cache_insert)

        for stage_id in range(8):
            scheduler._pp_vpp_apply_prefix_materialized(
                PipelineControlEnvelope(
                    protocol_version=1,
                    runtime_epoch=17,
                    layout_digest="layout",
                    kind=PipelineControlKind.PREFIX_MATERIALIZED,
                    source_rank=stage_id % 4,
                    payload={
                        "rid": req.rid,
                        "request_generation": 0,
                        "stage_id": stage_id,
                        "end": 256,
                    },
                )
            )

        self.assertEqual(entry.committed_end, 256)
        self.assertEqual(req.kv.kv_committed_len, req.kv.kv_allocated_len)

    def test_vpp_batch_manifest_snapshots_request_ranges(self):
        scheduler = _make_scheduler()
        scheduler.mbs = [None] * 4
        batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(
                    rid="r0",
                    extend_range=SimpleNamespace(start=1024, end=4096),
                )
            ]
        )

        manifest = scheduler._pp_vpp_batch_manifest(4, 0, batch)

        self.assertEqual(manifest["generation"], 1)
        self.assertEqual(manifest["requests"], (("r0", 1024, 4096),))

    def test_vpp_bootstrap_uses_minimum_prefix_boundary_across_pp_ranks(self):
        scheduler = _make_scheduler()
        req = SimpleNamespace(
            rid="r0",
            prefix_indices=torch.arange(7936),
            vpp_prefix_limit=None,
            init_next_round_input=MagicMock(),
        )
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[req])
        scheduler.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=MagicMock(return_value=8)
        )
        scheduler.attn_tp_group.all_gather_object = MagicMock(return_value=[8, 6])
        scheduler.tree_cache = object()
        payload = {
            "good": ["r0"],
            "bad": [],
            "prefix_boundaries": {"r0": 0},
            "metadata_slots": 16,
        }

        scheduler._pp_vpp_merge_bootstrap_status(payload, ["r0"], [])
        req.init_next_round_input.side_effect = lambda _cache: setattr(
            req, "prefix_indices", torch.empty(0, dtype=torch.int64)
        )
        scheduler._pp_vpp_apply_bootstrap_prefix_boundaries(
            payload["prefix_boundaries"],
            {"r0"},
        )

        self.assertEqual(payload["prefix_boundaries"], {"r0": 0})
        self.assertEqual(payload["metadata_slots"], 6)
        self.assertEqual(req.vpp_prefix_limit, 0)
        self.assertEqual(len(req.prefix_indices), 0)
        req.init_next_round_input.assert_called_once_with(scheduler.tree_cache)

    def test_vpp_bootstrap_apply_is_bounded_by_common_metadata_capacity(self):
        scheduler = _make_scheduler()
        payload = {
            "good": [f"r{i}" for i in range(48)],
            "bad": ["bad"],
            "metadata_slots": 2,
        }

        good, bad = scheduler._pp_vpp_bootstrap_apply_status(payload)

        self.assertEqual(good, ["r0", "r1"])
        self.assertEqual(bad, ["bad"])
        payload["metadata_slots"] = 0
        self.assertEqual(
            scheduler._pp_vpp_bootstrap_apply_status(payload),
            [[], ["bad"]],
        )

    def test_vpp_prefix_limit_caps_next_prefix_match(self):
        req = Req.__new__(Req)
        req.return_logprob = False
        req.logprob_start_len = -1
        req.vpp_prefix_limit = 4096

        self.assertEqual(req._compute_max_prefix_len(8184), 4096)

    def test_prepare_wavefront_batch_claims_slot(self):
        scheduler = _make_scheduler()
        scheduler.mbs = [None] * 4
        scheduler.last_mbs = [None] * 4
        scheduler.running_mbs = [SimpleNamespace() for _ in range(4)]
        scheduler.mb_metadata = [None] * 4
        scheduler.process_prefill_chunk = MagicMock()
        batch = SimpleNamespace(chunked_req=None)
        next_running = SimpleNamespace()
        scheduler.get_new_batch_prefill = MagicMock(
            return_value=SimpleNamespace(
                batch_to_run=batch,
                running_batch=next_running,
            )
        )
        scheduler.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=MagicMock(return_value=batch)
        )
        slot_batch_seqs = [None] * 4

        result = scheduler._pp_vpp_prepare_wavefront_batch(5, slot_batch_seqs)

        self.assertIs(result, batch)
        self.assertEqual(slot_batch_seqs, [None, 5, None, None])
        self.assertIs(scheduler.mbs[1], batch)
        self.assertIs(scheduler.running_mbs[1], next_running)
        scheduler.process_prefill_chunk.assert_called_once_with(
            last_batch=None,
            running_batch=ANY,
        )

        with self.assertRaisesRegex(RuntimeError, "still owned"):
            scheduler._pp_vpp_prepare_wavefront_batch(9, slot_batch_seqs)

    def test_prepare_wavefront_batch_snapshots_chunk_end(self):
        scheduler = _make_scheduler()
        scheduler.mbs = [None] * 4
        scheduler.last_mbs = [None] * 4
        scheduler.running_mbs = [SimpleNamespace() for _ in range(4)]
        scheduler.mb_metadata = [None] * 4
        scheduler.process_prefill_chunk = MagicMock()
        req = SimpleNamespace(
            rid="r0",
            extend_range=SimpleNamespace(end=4096),
            origin_input_ids=list(range(8192)),
            vpp_prefix_limit=0,
        )
        batch = SimpleNamespace(chunked_req=req)
        scheduler.get_new_batch_prefill = MagicMock(
            return_value=SimpleNamespace(
                batch_to_run=batch,
                running_batch=SimpleNamespace(),
            )
        )
        scheduler.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=MagicMock(return_value=batch)
        )

        scheduler._pp_vpp_prepare_wavefront_batch(0, [None] * 4)
        req.extend_range.end = 8192

        self.assertEqual(batch.disagg_prefill_chunk_end_by_rid, {"r0": 4096})
        self.assertIsNone(req.vpp_prefix_limit)

    def test_execute_wavefront_action_routes_expected_batch_and_stage(self):
        scheduler = _make_scheduler()
        batch = SimpleNamespace()
        scheduler.mbs = [batch, None, None, None]
        scheduler.last_mbs = [None] * 4
        scheduler.running_mbs = [SimpleNamespace() for _ in range(4)]
        scheduler.mb_metadata = [None] * 4
        scheduler.enable_staging = False
        proxy = PPProxyTensors({"vpp_batch_seq": 4, "vpp_stage_id": 5})
        scheduler._pp_recv_vpp_proxy_tensors = MagicMock(return_value=proxy)
        send_work = [object()]
        scheduler._pp_launch_vpp_stage = MagicMock(
            return_value=(object(), object(), send_work)
        )
        action = PipelineWavefrontAction(
            tick=13,
            batch_seq=4,
            slot_id=0,
            stage_id=5,
            physical_rank=1,
        )

        executed, result_send_work = scheduler._pp_vpp_execute_wavefront_action(
            action,
            [4, None, None, None],
        )

        self.assertTrue(executed)
        self.assertIs(result_send_work, send_work)
        scheduler._pp_recv_vpp_proxy_tensors.assert_called_once_with(
            first_visit=False,
            expected_batch_seq=4,
            expected_stage_id=5,
        )
        scheduler._pp_launch_vpp_stage.assert_called_once_with(
            action,
            batch,
            proxy,
            scheduler.mb_metadata,
            scheduler.last_rank_comm_queue,
        )

    @patch(
        "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
        return_value=SimpleNamespace(pp_virtual_stages=2),
    )
    @patch("sglang.srt.managers.scheduler_pp_mixin.set_time_batch")
    def test_non_final_wavefront_action_runs_one_stage(
        self, _set_time_batch, _get_parallel
    ):
        scheduler = _make_scheduler()
        stage_input = PPProxyTensors({"vpp_stage_id": 1, "vpp_batch_seq": 4})
        stage_output = PPProxyTensors({"vpp_stage_id": 2})
        stage_result = SimpleNamespace(
            pp_hidden_states_proxy_tensors=stage_output,
            can_run_cuda_graph=False,
        )
        scheduler.run_batch = MagicMock(return_value=stage_result)
        batch = SimpleNamespace(reqs=[])
        metadata = [None] * 4
        action = PipelineWavefrontAction(
            tick=9,
            batch_seq=4,
            slot_id=0,
            stage_id=1,
            physical_rank=1,
        )

        result, event, send_work = scheduler._pp_launch_vpp_stage(
            action,
            batch,
            stage_input,
            metadata,
            deque(),
        )

        self.assertIs(result, stage_result)
        self.assertEqual(event.recorded_stream, "forward-stream")
        self.assertEqual(len(send_work), 1)
        self.assertEqual(metadata[0], PPBatchMetadata(can_run_cuda_graph=False))
        scheduler.run_batch.assert_called_once_with(batch, stage_input)
        scheduler._pp_send_dict_to_next_stage.assert_called_once_with(
            {
                "vpp_stage_id": 2,
                "vpp_batch_seq": 4,
                "vpp_generation": 1,
                "vpp_protocol_version": 1,
                "vpp_src_stage_id": 1,
            },
            async_send=True,
            msg_type="vpp_proxy",
            batch_p2p=True,
            tag=1,
        )
        scheduler._pp_commit_comm_work.assert_not_called()

    @patch(
        "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
        return_value=SimpleNamespace(pp_virtual_stages=2),
    )
    @patch("sglang.srt.managers.scheduler_pp_mixin.set_time_batch")
    def test_final_wavefront_action_queues_output(self, _set_time_batch, _get_parallel):
        scheduler = _make_scheduler(is_last_rank=True)
        final_result = SimpleNamespace(
            pp_hidden_states_proxy_tensors=None,
            can_run_cuda_graph=True,
        )
        scheduler.run_batch = MagicMock(return_value=final_result)
        batch = SimpleNamespace(reqs=[])
        metadata = [None] * 4
        output_queue = deque()
        action = PipelineWavefrontAction(
            tick=7,
            batch_seq=0,
            slot_id=0,
            stage_id=7,
            physical_rank=3,
        )
        stage_input = PPProxyTensors({"vpp_stage_id": 7, "vpp_batch_seq": 0})

        result, event, send_work = scheduler._pp_launch_vpp_stage(
            action,
            batch,
            stage_input,
            metadata,
            output_queue,
        )

        self.assertIs(result, final_result)
        self.assertEqual(send_work, [])
        self.assertEqual(metadata[0], PPBatchMetadata(can_run_cuda_graph=True))
        scheduler._pp_send_dict_to_next_stage.assert_not_called()
        scheduler._pp_prepare_tensor_dict.assert_called_once_with(final_result, batch)
        queued_event, queued_output = output_queue.pop()
        self.assertIs(queued_event, event)
        self.assertEqual(
            queued_output.tensors,
            {"next_token_ids": "tokens", "vpp_batch_seq": 0},
        )

    @patch(
        "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
        return_value=SimpleNamespace(pp_virtual_stages=2),
    )
    @patch("sglang.srt.managers.scheduler_pp_mixin.set_time_batch")
    def test_final_wavefront_action_queues_output_on_each_tp_lane(
        self, _set_time_batch, _get_parallel
    ):
        scheduler = _make_scheduler(is_last_rank=True)
        scheduler.ps.tp_rank = 1
        scheduler.run_batch = MagicMock(
            return_value=SimpleNamespace(
                pp_hidden_states_proxy_tensors=None,
                can_run_cuda_graph=True,
            )
        )
        output_queue = deque()

        scheduler._pp_launch_vpp_stage(
            PipelineWavefrontAction(
                tick=7,
                batch_seq=0,
                slot_id=0,
                stage_id=7,
                physical_rank=3,
            ),
            SimpleNamespace(reqs=[]),
            PPProxyTensors({"vpp_stage_id": 7, "vpp_batch_seq": 0}),
            [None] * 4,
            output_queue,
        )

        self.assertEqual(len(output_queue), 1)
        scheduler._pp_prepare_tensor_dict.assert_not_called()
        scheduler.attn_tp_group.broadcast_tensor_dict.assert_called_once_with(
            None,
            src=0,
        )

    @patch(
        "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
        return_value=SimpleNamespace(pp_virtual_stages=3),
    )
    def test_launch_rejects_non_vpp2_layout(self, _get_parallel):
        scheduler = _make_scheduler()
        action = PipelineWavefrontAction(
            tick=0,
            batch_seq=0,
            slot_id=0,
            stage_id=1,
            physical_rank=1,
        )

        with self.assertRaisesRegex(RuntimeError, "supports VPP2 only"):
            scheduler._pp_launch_vpp_stage(
                action,
                SimpleNamespace(reqs=[]),
                PPProxyTensors({"vpp_stage_id": 1, "vpp_batch_seq": 0}),
                [None] * 4,
                deque(),
            )

    def test_tp_leader_waits_until_task_is_ready_on_every_lane(self):
        scheduler = _make_scheduler()
        scheduler.ps.pp_rank = 1
        scheduler.ps.tp_rank = 0
        scheduler.attn_tp_group = SimpleNamespace(
            all_gather_object=MagicMock(return_value=[((0, 1),), ()]),
            broadcast_object=MagicMock(side_effect=lambda value, src: value),
        )
        schedule = PipelineRankSchedule(
            physical_rank=1,
            physical_size=4,
            virtual_stages=2,
            max_inflight=4,
        )
        schedule.admit(0)
        schedule.mark_ready(0, 1)

        action = scheduler._pp_vpp_select_tp_action(
            schedule,
            tick=3,
            is_ready=lambda _batch_seq, _stage_id: True,
        )

        self.assertIsNone(action)
        self.assertIsNone(schedule.running)
        scheduler.attn_tp_group.all_gather_object.assert_called_once()

    def test_rank_local_loop_advances_bootstrap_and_transfer_via_control_ring(self):
        scheduler = _make_scheduler(is_first_rank=True)
        scheduler.ps.pp_rank = 0
        scheduler.world_group.rank_in_group = 0
        scheduler.world_group.broadcast_object = MagicMock(return_value=17)
        scheduler.attn_tp_group.all_gather_object = MagicMock(
            side_effect=lambda value: [value, value]
        )
        scheduler.attn_tp_group.broadcast_object = MagicMock(
            side_effect=lambda value, src: value
        )
        scheduler.model_config = SimpleNamespace(
            hidden_size=16,
            num_hidden_layers=40,
            hf_config=SimpleNamespace(hc_mult=1),
        )
        bootstrap_req = SimpleNamespace(
            rid="bootstrap-rid",
            finished_reason=None,
            prefix_indices=torch.empty(0, dtype=torch.int64),
            vpp_prefix_limit=None,
            init_next_round_input=MagicMock(),
        )
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
            queue=[bootstrap_req]
        )
        scheduler.disagg_prefill_inflight_queue = [SimpleNamespace(rid="transfer-rid")]
        scheduler.waiting_queue = []
        scheduler.req_to_token_pool = SimpleNamespace(
            available_size=MagicMock(return_value=0)
        )
        scheduler.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=MagicMock(return_value=1)
        )
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            available_size=MagicMock(return_value=8192)
        )
        scheduler.chunked_req = None
        scheduler._pending_chunked_abort_req = None
        scheduler.process_pending_chunked_abort = MagicMock()
        scheduler.on_idle = MagicMock()
        scheduler._pp_vpp_layout_digest = MagicMock(return_value="layout")
        scheduler._pp_vpp_start_receiver = MagicMock()
        scheduler._pp_vpp_start_control_receiver = MagicMock()
        scheduler._pp_vpp_poll_receiver_tp_consensus = MagicMock(return_value=False)
        scheduler._pp_vpp_reap_send_work = MagicMock()
        scheduler._pp_vpp_select_tp_action = MagicMock(return_value=None)
        scheduler._pp_vpp_prepare_wavefront_batch = MagicMock(return_value=None)
        scheduler._pp_vpp_resource_snapshot = MagicMock(
            return_value=PipelineResourceSnapshot(1, 8192, 0, 0, 1)
        )
        scheduler.get_rids = MagicMock(
            side_effect=lambda queue, _is_send, *statuses: (
                (["bootstrap-rid"], []) if len(statuses) == 2 else ["transfer-rid"]
            )
        )

        bootstrap_attempts = 0

        def process_bootstrap(status):
            nonlocal bootstrap_attempts
            bootstrap_attempts += 1
            if bootstrap_attempts == 1:
                return [[], []]
            scheduler.disagg_prefill_bootstrap_queue.queue.clear()
            scheduler.waiting_queue.append("bootstrap-rid")
            return status

        scheduler.process_bootstrapped_queue = MagicMock(side_effect=process_bootstrap)

        def process_transfer(rids):
            scheduler.disagg_prefill_inflight_queue.clear()
            return rids

        scheduler.process_disagg_prefill_inflight_queue = MagicMock(
            side_effect=process_transfer
        )
        control_inbox = deque()
        sent_kinds = []

        def poll_control():
            return control_inbox.popleft() if control_inbox else None

        def flush_control(_pending_work, _max_pending):
            while scheduler._pp_vpp_control_outbox:
                wire = scheduler._pp_vpp_control_outbox.popleft()
                envelope = PipelineControlEnvelope.from_dict(wire)
                sent_kinds.append(envelope.kind)
                while envelope.hops < scheduler.ps.pp_size:
                    envelope = envelope.forwarded()
                control_inbox.append((envelope, envelope.to_dict()))

        scheduler._pp_vpp_poll_control_receiver = poll_control
        scheduler._pp_vpp_flush_control_outbox = flush_control
        ingest_count = 0

        def ingest_requests():
            nonlocal ingest_count
            ingest_count += 1
            if ingest_count == 1:
                return ["request"]
            if ingest_count > 2:
                raise StopIteration
            return []

        scheduler.ingest_requests = ingest_requests

        with (
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
                return_value=SimpleNamespace(
                    pp_virtual_stages=2,
                    pp_async_batch_depth=0,
                ),
            ),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.max_prefill_buffer_tokens",
                return_value=128,
            ),
            patch.object(PipelineResourceGate, "can_admit", return_value=True),
            patch.object(PipelineResourceGate, "can_bootstrap", return_value=True),
            self.assertRaises(StopIteration),
        ):
            scheduler._event_loop_pp_disagg_prefill_vpp_rank_local()

        self.assertEqual(scheduler.waiting_queue, ["bootstrap-rid"])
        self.assertEqual(scheduler.process_bootstrapped_queue.call_count, 2)
        scheduler.process_bootstrapped_queue.assert_called_with([["bootstrap-rid"], []])
        scheduler.process_disagg_prefill_inflight_queue.assert_called_once_with(
            ["transfer-rid"]
        )
        scheduler._pp_vpp_prepare_wavefront_batch.assert_not_called()
        self.assertLess(
            sent_kinds.index(PipelineControlKind.REQUEST),
            sent_kinds.index(PipelineControlKind.BOOTSTRAP_STATUS),
        )


if __name__ == "__main__":
    unittest.main()
