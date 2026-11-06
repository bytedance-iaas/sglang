import inspect
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.pp_dspark_draft import (  # noqa: E402
    PPDSparkDraftCoordinator,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # noqa: E402
from sglang.srt.speculative.dflash_info_v2 import (  # noqa: E402
    DFlashDraftInputV2,
    _device_matches,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeEvent:
    def __init__(self):
        self.synchronize_count = 0

    def record(self, stream):
        pass

    def synchronize(self):
        self.synchronize_count += 1


class _FakeStream:
    def wait_stream(self, stream):
        pass


class TestSchedulerPPRelayOrder(unittest.TestCase):
    @staticmethod
    def _make_dflash_input():
        return DFlashDraftInputV2(
            topk_p=torch.empty((1, 1)),
            topk_index=torch.empty((1, 1), dtype=torch.int64),
            bonus_tokens=torch.empty((1,), dtype=torch.int64),
            new_seq_lens=torch.empty((1,), dtype=torch.int64),
            hidden_states=torch.empty((1, 1)),
        )

    def _make_scheduler(self, is_last_rank, pp_rank=0, speculative=False):
        scheduler = SchedulerPPMixin()
        scheduler.pp_group = SimpleNamespace(is_last_rank=is_last_rank)
        scheduler.ps = SimpleNamespace(pp_rank=pp_rank)
        scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: not speculative)
        scheduler.copy_stream_ctx = nullcontext()
        scheduler.pp_comm_stream_ctx = nullcontext()
        scheduler.copy_stream = _FakeStream()
        scheduler.schedule_stream = _FakeStream()
        scheduler._pp_pending_serial_output_exchange = None
        scheduler.device_module = SimpleNamespace(
            Event=_FakeEvent,
            current_stream=lambda: _FakeStream(),
        )
        return scheduler

    def _run_relay(self, is_last_rank, pp_rank=0, speculative=False):
        scheduler = self._make_scheduler(is_last_rank, pp_rank, speculative)
        events = []
        target = SimpleNamespace(
            forward_mode=SimpleNamespace(is_prebuilt=lambda: False)
        )
        scheduler._pp_send_output_to_next_stage = lambda *args: (
            events.append("send") or ["output-work"]
        )
        scheduler._pp_recv_dict_from_prev_stage = lambda: (
            events.append("recv") or {"next_token_ids": object()}
        )
        scheduler._pp_prep_batch_result = lambda *args, **kwargs: events.append("prep")
        scheduler._pp_send_dict_to_next_stage = lambda *args, **kwargs: (
            events.append("relay") or ["relay-work"]
        )
        scheduler._pp_commit_comm_work = lambda work: events.append(("commit", work))
        scheduler._pp_spec_relay = speculative

        with (
            patch(
                "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
                return_value=False,
            ),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin._pp_use_batched_result_relay",
                return_value=False,
            ),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin._pp_use_serial_dspark_generation",
                return_value=False,
            ),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.get_spec",
                return_value=SimpleNamespace(
                    speculative_dspark_pp_replicated_draft=False
                ),
            ),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
                return_value=SimpleNamespace(pp_rank=pp_rank),
            ),
        ):
            _, _, _, send_work = scheduler._pp_send_recv_and_preprocess_output_tensors(
                next_first_rank_mb_id=0,
                next_mb_id=1,
                mbs=[target, target],
                mb_metadata=[None, None],
                last_rank_comm_queue=[],
                pp_outputs=None,
            )

        return events, send_work

    def test_last_rank_leaves_output_send_pending_for_next_slot(self):
        events, send_work = self._run_relay(is_last_rank=True)

        self.assertEqual(events, ["send", "recv", "prep"])
        self.assertEqual(send_work, ["output-work"])

    def test_non_last_rank_leaves_prior_output_send_pending(self):
        events, send_work = self._run_relay(is_last_rank=False)

        self.assertEqual(events, ["send", "recv", "prep"])
        self.assertEqual(send_work, ["output-work"])

    def test_speculative_output_pairs_adjacent_ranks_without_waiting(self):
        for pp_rank in range(4):
            with self.subTest(pp_rank=pp_rank):
                events, send_work = self._run_relay(
                    is_last_rank=pp_rank == 3,
                    pp_rank=pp_rank,
                    speculative=True,
                )
                expected = (
                    ["recv", "prep", "send"]
                    if pp_rank % 2
                    else ["send", "recv", "prep"]
                )
                self.assertEqual(events, expected)
                self.assertEqual(send_work, ["output-work"])

    def test_non_speculative_cuda_odd_rank_keeps_send_first(self):
        with patch("sglang.srt.managers.scheduler_pp_mixin.is_xpu", return_value=False):
            events, send_work = self._run_relay(is_last_rank=True, pp_rank=1)
        self.assertEqual(events, ["send", "recv", "prep"])
        self.assertEqual(send_work, ["output-work"])

    def _assert_disagg_proxy_order(self, event_loop, control_marker):
        source = inspect.getsource(event_loop)
        launch_pos = source.index("result, self.launch_event = self._pp_launch_batch")
        output_pos = source.index(
            "self._pp_commit_send_output_work_and_preprocess_output_tensors",
            launch_pos,
        )
        control_pos = source.index(control_marker, output_pos)
        send_pos = source.index(
            "self._pp_queue_proxy_send",
            control_pos,
        )

        self.assertLess(launch_pos, output_pos)
        self.assertLess(output_pos, control_pos)
        self.assertLess(control_pos, send_pos)

    def test_prefill_proxy_send_stays_after_control_ring_and_async(self):
        self._assert_disagg_proxy_order(
            SchedulerPPMixin.event_loop_pp_disagg_prefill,
            "self._pp_pd_send_consensus_bootstrapped_ids",
        )

    def test_decode_proxy_send_stays_after_control_ring_and_async(self):
        self._assert_disagg_proxy_order(
            SchedulerPPMixin.event_loop_pp_disagg_decode,
            "self._pp_pd_send_consensus_bootstrapped_ids",
        )

    def test_proxy_send_is_queued_without_wait(self):
        scheduler = self._make_scheduler(is_last_rank=False)
        events = []
        scheduler._pp_send_dict_to_next_stage = lambda *args, **kwargs: (
            events.append(("send", kwargs["async_send"], kwargs["msg_type"]))
            or ["proxy-work"]
        )
        scheduler._pp_commit_comm_work = lambda work: events.append(("commit", work))

        scheduler._pp_queue_proxy_send({"hidden_states": object()})

        self.assertEqual(events, [("send", True, "proxy")])
        self.assertEqual(scheduler.send_proxy_work, ["proxy-work"])

    def test_generation_envelope_rejects_stale_sequence(self):
        envelope = SchedulerPPMixin._pp_make_generation_envelope(7)

        self.assertEqual(
            SchedulerPPMixin._pp_validate_generation_envelope(envelope, 7),
            7,
        )
        with self.assertRaisesRegex(RuntimeError, "expected 8, got 7"):
            SchedulerPPMixin._pp_validate_generation_envelope(envelope, 8)

    def test_serial_proxy_envelope_covers_active_and_idle_generations(self):
        scheduler = self._make_scheduler(is_last_rank=True)
        scheduler.pp_group.is_first_rank = False
        scheduler.require_attn_tp_allgather = False
        scheduler.attn_tp_group = None

        scheduler._pp_recv_typed_dict = lambda **_: {
            "__pp_generation__": 3,
            "__pp_active__": False,
        }
        self.assertIsNone(
            scheduler._pp_recv_proxy_tensors(
                expected_generation=3,
                expected_active=False,
            )
        )

        hidden_states = object()
        scheduler._pp_recv_typed_dict = lambda **_: {
            "__pp_generation__": 4,
            "__pp_active__": True,
            "hidden_states": hidden_states,
        }
        proxy = scheduler._pp_recv_proxy_tensors(
            expected_generation=4,
            expected_active=True,
        )
        self.assertIs(proxy.tensors["hidden_states"], hidden_states)

    def test_serial_proxy_envelope_rejects_stale_generation(self):
        scheduler = self._make_scheduler(is_last_rank=True)
        scheduler.pp_group.is_first_rank = False
        scheduler.require_attn_tp_allgather = False
        scheduler.attn_tp_group = None
        scheduler._pp_recv_typed_dict = lambda **_: {
            "__pp_generation__": 4,
            "__pp_active__": True,
        }

        with self.assertRaisesRegex(RuntimeError, "expected generation 5, got 4"):
            scheduler._pp_recv_proxy_tensors(
                expected_generation=5,
                expected_active=True,
            )

    def test_serial_idle_output_always_sends_typed_envelope(self):
        scheduler = self._make_scheduler(is_last_rank=True)
        scheduler._pp_local_output_to_process = None
        scheduler._pp_local_output_ready_event = None
        scheduler._pp_send_dict_to_next_stage = lambda tensor_dict, **kwargs: (
            tensor_dict,
            kwargs,
        )

        with patch(
            "sglang.srt.managers.scheduler_pp_mixin._pp_use_serial_dspark_generation",
            return_value=True,
        ):
            tensor_dict, kwargs = scheduler._pp_send_output_to_next_stage(
                next_first_rank_mb_id=0,
                mbs=[None, None],
                last_rank_comm_queue=[],
                pp_outputs=None,
            )

        self.assertEqual(
            tensor_dict,
            {
                "__pp_generation__": None,
                "__pp_active__": False,
            },
        )
        self.assertEqual(kwargs["msg_type"], "output")
        self.assertTrue(kwargs["async_send"])

    def test_serial_first_rank_never_relays_active_output(self):
        scheduler = self._make_scheduler(is_last_rank=False)
        scheduler._pp_local_output_to_process = None
        scheduler._pp_local_output_ready_event = None
        scheduler._pp_send_dict_to_next_stage = lambda tensor_dict, **_: tensor_dict

        with patch(
            "sglang.srt.managers.scheduler_pp_mixin._pp_use_serial_dspark_generation",
            return_value=True,
        ):
            tensor_dict = scheduler._pp_send_output_to_next_stage(
                next_first_rank_mb_id=0,
                mbs=[None, None],
                last_rank_comm_queue=[],
                pp_outputs=SimpleNamespace(
                    tensors={
                        "__pp_generation__": 5,
                        "__pp_active__": True,
                    }
                ),
            )

        self.assertEqual(
            tensor_dict,
            {
                "__pp_generation__": None,
                "__pp_active__": False,
            },
        )

    def test_serial_output_uses_atomic_bidirectional_exchange(self):
        scheduler = self._make_scheduler(is_last_rank=False)
        scheduler.require_attn_tp_allgather = False
        scheduler.attn_tp_group = None
        scheduler._pp_received_output_generation = None
        scheduler._pp_deferred_output = None
        calls = []

        def exchange(**kwargs):
            calls.append(kwargs)
            return {
                "__msg_type__": "output",
                "__pp_generation__": None,
                "__pp_active__": False,
            }

        scheduler.pp_group.send_recv_tensor_dict = exchange
        pp_outputs, batch_result, d2h_event, send_work = (
            scheduler._pp_exchange_serial_output_tensors([])
        )

        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["send_tensor_dict"]["__pp_active__"])
        self.assertEqual(
            pp_outputs.tensors["__msg_type__"],
            "output",
        )
        self.assertIsNone(batch_result)
        self.assertIsNone(d2h_event)
        self.assertEqual(send_work, [])

    def test_output_overlap_keeps_comm_event_until_explicit_completion(self):
        scheduler = self._make_scheduler(is_last_rank=False)
        scheduler.require_attn_tp_allgather = False
        scheduler.attn_tp_group = None
        scheduler._pp_received_output_generation = None
        scheduler._pp_deferred_output = None
        scheduler._pp_pending_serial_output_exchange = None
        scheduler.pp_group.send_recv_tensor_dict = lambda **_: {
            "__msg_type__": "output",
            "__pp_generation__": None,
            "__pp_active__": False,
        }

        with patch(
            "sglang.srt.managers.scheduler_pp_mixin."
            "_pp_use_serial_dspark_output_overlap",
            return_value=True,
        ):
            pp_outputs, _, _, _ = scheduler._pp_exchange_serial_output_tensors([])

        self.assertIsNone(pp_outputs)
        pending_event = scheduler._pp_pending_serial_output_exchange[2]
        self.assertEqual(pending_event.synchronize_count, 0)

        pp_outputs, _, _, _ = scheduler._pp_complete_serial_output_exchange()

        self.assertEqual(pending_event.synchronize_count, 1)
        self.assertFalse(pp_outputs.tensors["__pp_active__"])
        self.assertIsNone(scheduler._pp_pending_serial_output_exchange)

    def test_serial_output_keeps_current_generation_in_outbox(self):
        scheduler = self._make_scheduler(is_last_rank=True)
        scheduler.require_attn_tp_allgather = False
        scheduler.attn_tp_group = None
        scheduler.pp_comm_stream_ctx = nullcontext()
        scheduler._pp_serial_generation = 9
        scheduler._pp_received_output_generation = None
        scheduler._pp_deferred_output = None
        current = SimpleNamespace(
            tensors={
                "__pp_generation__": 9,
                "__pp_active__": True,
                "next_token_ids": object(),
            }
        )
        queue = [(_FakeEvent(), current)]
        scheduler.pp_group.send_recv_tensor_dict = lambda **_: {
            "__msg_type__": "output",
            "__pp_generation__": None,
            "__pp_active__": False,
        }

        pp_outputs, _, _, _ = scheduler._pp_exchange_serial_output_tensors(queue)

        self.assertEqual(len(queue), 1)
        self.assertFalse(pp_outputs.tensors["__pp_active__"])

    def test_local_output_skips_relay_only_dspark_hooks(self):
        source = inspect.getsource(SchedulerPPMixin._pp_prep_batch_result)

        self.assertIn("if not local_output:", source)
        self.assertIn("if is_decode and not local_output:", source)

    def test_serial_generation_disables_deferred_bubble_draft(self):
        source = inspect.getsource(PPDSparkDraftCoordinator.on_batch_launched)

        self.assertIn(
            "self._is_bubble_policy() and not self._is_serial_generation_protocol()",
            source,
        )

    def test_serial_proposal_relay_sends_all_owner_payloads(self):
        scheduler = SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=True),
            attn_tp_group=None,
        )
        calls = []
        scheduler._pp_send_dict_to_next_stage = lambda payload, **kwargs: (
            calls.append((payload, kwargs)) or ["proposal-work"]
        )
        scheduler._pp_commit_comm_work = lambda work: calls.append(("commit", work))
        coordinator = PPDSparkDraftCoordinator(scheduler)
        pp_outputs = SimpleNamespace(
            tensors={
                "dspark_next_0_identities": [("a", (1, 0), 2)],
                "dspark_next_1_identities": [("b", (1, 0), 2)],
            }
        )
        ready_event = _FakeEvent()

        with patch(
            "sglang.srt.managers.scheduler_components.pp_dspark_draft.get_parallel",
            return_value=SimpleNamespace(pp_size=2),
        ):
            coordinator.relay_serial_proposals(
                object(),
                pp_outputs,
                transport_generation=7,
                ready_event=ready_event,
            )

        payload, kwargs = calls[0]
        self.assertEqual(payload["__pp_generation__"], 7)
        self.assertTrue(payload["__pp_active__"])
        self.assertIn("dspark_next_0_identities", payload)
        self.assertIn("dspark_next_1_identities", payload)
        self.assertEqual(kwargs["msg_type"], "dspark_draft_serial")
        self.assertIs(kwargs["ready_event"], ready_event)
        self.assertEqual(calls[1], ("commit", ["proposal-work"]))

    def test_serial_proposal_relay_installs_all_owners_on_last_rank(self):
        installs = []
        scheduler = SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=False),
            attn_tp_group=None,
            model_worker=SimpleNamespace(
                install_pp_draft=lambda batch, owner, payload: installs.append(
                    (batch, owner, payload)
                )
            ),
        )
        scheduler._pp_recv_typed_dict = lambda **_: {
            "__pp_generation__": 11,
            "__pp_active__": True,
            "dspark_next_0_identities": [("a", (1, 0), 2)],
            "dspark_next_1_identities": [("b", (1, 0), 2)],
        }
        scheduler._pp_commit_comm_work = lambda work: None
        coordinator = PPDSparkDraftCoordinator(scheduler)
        batch = object()

        with patch(
            "sglang.srt.managers.scheduler_components.pp_dspark_draft.get_parallel",
            return_value=SimpleNamespace(pp_size=2),
        ):
            coordinator.relay_serial_proposals(
                batch,
                SimpleNamespace(tensors={}),
                transport_generation=11,
                ready_event=None,
            )

        self.assertEqual([owner for _, owner, _ in installs], [0, 1])
        self.assertTrue(
            all(installed_batch is batch for installed_batch, _, _ in installs)
        )

    def test_serial_proposal_relay_rejects_stale_generation(self):
        scheduler = SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=False),
            attn_tp_group=None,
        )
        scheduler._pp_recv_typed_dict = lambda **_: {
            "__pp_generation__": 12,
            "__pp_active__": False,
        }
        scheduler._pp_commit_comm_work = lambda work: None
        coordinator = PPDSparkDraftCoordinator(scheduler)

        with self.assertRaisesRegex(RuntimeError, "expected generation 13, got 12"):
            coordinator.relay_serial_proposals(
                None,
                SimpleNamespace(tensors={}),
                transport_generation=13,
                ready_event=None,
            )

    def test_merge_invalidates_prepared_lengths_for_unprepared_prebuilt(self):
        running = self._make_dflash_input()
        prebuilt = self._make_dflash_input()
        running.nxt_kv_lens_cpu = torch.tensor([8], dtype=torch.int32)
        running.nxt_kv_lens_sum = 8

        running.merge_batch(prebuilt)

        self.assertIsNone(running.nxt_kv_lens_cpu)
        self.assertIsNone(running.nxt_kv_lens_sum)

    def test_unspecified_cuda_index_matches_local_cuda_device(self):
        self.assertTrue(_device_matches(torch.device("cuda:3"), "cuda"))
        self.assertTrue(_device_matches(torch.device("cuda:3"), "cuda:3"))
        self.assertFalse(_device_matches(torch.device("cuda:3"), "cuda:2"))

    def test_serial_generation_orders_transport_before_launch(self):
        source = inspect.getsource(
            SchedulerPPMixin._event_loop_pp_disagg_decode_serial_generation
        )
        forward_pos = source.index("self.send_req_work =")
        plan_pos = source.index("self.get_next_disagg_decode_batch_to_run")
        reverse_pos = source.index("PP2 reverse consensus")
        proxy_pos = source.index("self._pp_recv_proxy_tensors")
        launch_pos = source.index("result, self.launch_event = self._pp_launch_batch")
        queue_pos = source.index("self._pp_queue_proxy_send", launch_pos)
        proxy_commit_pos = source.index(
            "self._pp_commit_comm_work(self.send_proxy_work)",
            queue_pos,
        )

        self.assertLess(forward_pos, plan_pos)
        self.assertLess(plan_pos, reverse_pos)
        self.assertLess(reverse_pos, proxy_pos)
        self.assertLess(proxy_pos, launch_pos)
        self.assertLess(launch_pos, queue_pos)
        self.assertLess(queue_pos, proxy_commit_pos)
        self.assertEqual(source.count("self._pp_serial_generation_fence()"), 4)

    def test_output_overlap_submits_before_proxy_and_completes_after_launch(self):
        source = inspect.getsource(
            SchedulerPPMixin._event_loop_pp_disagg_decode_serial_generation
        )
        overlap_pos = source.index(
            "_pp_use_serial_dspark_output_overlap()",
        )
        output_submit_pos = source.index(
            "self._pp_commit_send_output_work_and_preprocess_output_tensors",
            overlap_pos,
        )
        proxy_recv_pos = source.index("self._pp_recv_proxy_tensors", output_submit_pos)
        launch_pos = source.index("result, self.launch_event = self._pp_launch_batch")
        proxy_commit_pos = source.index(
            "self._pp_commit_comm_work(self.send_proxy_work)",
            launch_pos,
        )
        output_complete_pos = source.index(
            "self._pp_complete_serial_output_exchange",
            proxy_commit_pos,
        )

        self.assertLess(output_submit_pos, proxy_recv_pos)
        self.assertLess(proxy_recv_pos, launch_pos)
        self.assertLess(launch_pos, proxy_commit_pos)
        self.assertLess(proxy_commit_pos, output_complete_pos)

    def test_proxy_overlap_commits_after_output_processing_before_fence(self):
        source = inspect.getsource(
            SchedulerPPMixin._event_loop_pp_disagg_decode_serial_generation
        )
        queue_pos = source.index("self._pp_queue_proxy_send")
        immediate_gate_pos = source.index(
            "if not _pp_use_serial_dspark_proxy_overlap():",
            queue_pos,
        )
        output_complete_pos = source.index(
            "self._pp_complete_serial_output_exchange",
            immediate_gate_pos,
        )
        process_pos = source.index(
            "self._pp_process_batch_result",
            output_complete_pos,
        )
        deferred_gate_pos = source.index(
            "if _pp_use_serial_dspark_proxy_overlap():",
            process_pos,
        )
        deferred_commit_pos = source.index(
            "self._pp_commit_comm_work(self.send_proxy_work)",
            deferred_gate_pos,
        )
        final_fence_pos = source.index(
            "self._pp_serial_generation_fence()",
            deferred_commit_pos,
        )

        self.assertLess(queue_pos, immediate_gate_pos)
        self.assertLess(immediate_gate_pos, output_complete_pos)
        self.assertLess(output_complete_pos, process_pos)
        self.assertLess(process_pos, deferred_commit_pos)
        self.assertLess(deferred_commit_pos, final_fence_pos)

    def test_prelaunch_result_apply_installs_proposal_before_target_launch(self):
        source = inspect.getsource(
            SchedulerPPMixin._event_loop_pp_disagg_decode_serial_generation
        )
        prelaunch_gate_pos = source.index(
            "_pp_use_serial_dspark_prelaunch_result_apply()"
        )
        output_complete_pos = source.index(
            "self._pp_complete_serial_output_exchange",
            prelaunch_gate_pos,
        )
        preprocess_pos = source.index(
            "self._pp_preprocess_deferred_serial_output",
            output_complete_pos,
        )
        draft_pos = source.index(
            "dspark_draft.drain()",
            preprocess_pos,
        )
        decode_gate_pos = source.index(
            "next_batch_result.accept_lens is not None",
            draft_pos,
        )
        proposal_relay_pos = source.index(
            "dspark_draft.relay_serial_proposals",
            decode_gate_pos,
        )
        launch_pos = source.index(
            "result, self.launch_event = self._pp_launch_batch",
            proposal_relay_pos,
        )
        process_pos = source.index(
            "self._pp_process_batch_result",
            launch_pos,
        )

        self.assertLess(output_complete_pos, preprocess_pos)
        self.assertLess(preprocess_pos, draft_pos)
        self.assertLess(draft_pos, decode_gate_pos)
        self.assertLess(decode_gate_pos, proposal_relay_pos)
        self.assertLess(proposal_relay_pos, launch_pos)
        self.assertLess(launch_pos, process_pos)

    def test_preplan_result_apply_commits_before_batch_planning(self):
        source = inspect.getsource(
            SchedulerPPMixin._event_loop_pp_disagg_decode_serial_generation
        )
        preplan_gate_pos = source.index(
            "if _pp_use_serial_dspark_preplan_result_apply():"
        )
        output_pos = source.index(
            "self._pp_complete_serial_output_exchange",
            preplan_gate_pos,
        )
        proposal_pos = source.index(
            "dspark_draft.relay_serial_proposals",
            output_pos,
        )
        process_pos = source.index(
            "self._pp_process_batch_result",
            proposal_pos,
        )
        plan_pos = source.index(
            "self.get_next_disagg_decode_batch_to_run",
            process_pos,
        )
        launch_pos = source.index(
            "result, self.launch_event = self._pp_launch_batch",
            plan_pos,
        )

        self.assertLess(output_pos, proposal_pos)
        self.assertLess(proposal_pos, process_pos)
        self.assertLess(process_pos, plan_pos)
        self.assertLess(plan_pos, launch_pos)

    def test_committed_snapshot_updates_only_equivalent_running_slots(self):
        scheduler = SchedulerPPMixin()
        req_a = SimpleNamespace(rid="a", bootstrap_room=1, retraction_count=0)
        committed = SimpleNamespace(
            reqs=[req_a],
            spec_info=object(),
            seq_lens=object(),
            seq_lens_cpu=object(),
            seq_lens_sum=9,
            orig_seq_lens=object(),
            input_ids=object(),
        )
        matching = SimpleNamespace(
            reqs=[req_a],
            spec_info=None,
            seq_lens=None,
            seq_lens_cpu=None,
            seq_lens_sum=0,
            orig_seq_lens=None,
            input_ids=None,
        )
        scheduler.running_mbs = [matching]

        identity = SimpleNamespace(rid="a", generation=(1, 0))
        scheduler._pp_sync_committed_snapshot_to_running_slots(committed, (identity,))

        self.assertIs(matching.spec_info, committed.spec_info)
        self.assertIs(matching.seq_lens, committed.seq_lens)
        self.assertEqual(matching.seq_lens_sum, 9)

    def test_committed_snapshot_merges_recomposed_rows_by_generation(self):
        scheduler = SchedulerPPMixin()
        committed_reqs = [
            SimpleNamespace(rid="b"),
            SimpleNamespace(rid="a"),
        ]
        committed = SimpleNamespace(
            reqs=committed_reqs,
            spec_info=DFlashDraftInputV2(
                topk_p=torch.empty((2, 0)),
                topk_index=torch.empty((2, 0), dtype=torch.int64),
                bonus_tokens=torch.tensor([20, 10]),
                new_seq_lens=torch.tensor([220, 110]),
                hidden_states=torch.empty((2, 0)),
            ),
            seq_lens=torch.tensor([220, 110]),
            seq_lens_cpu=torch.tensor([220, 110]),
            seq_lens_sum=330,
            orig_seq_lens=torch.tensor([200, 100], dtype=torch.int32),
            input_ids=None,
        )
        live_reqs = [
            SimpleNamespace(rid="a", bootstrap_room=7, retraction_count=0),
            SimpleNamespace(rid="c", bootstrap_room=8, retraction_count=0),
            SimpleNamespace(rid="b", bootstrap_room=9, retraction_count=1),
        ]
        live_spec = DFlashDraftInputV2(
            topk_p=torch.empty((3, 0)),
            topk_index=torch.empty((3, 0), dtype=torch.int64),
            bonus_tokens=torch.tensor([1, 3, 2]),
            new_seq_lens=torch.tensor([11, 33, 22]),
            hidden_states=torch.empty((3, 0)),
            nxt_kv_lens_cpu=torch.tensor([12, 34, 23], dtype=torch.int32),
            nxt_kv_lens_sum=69,
            future_indices=torch.tensor([4, 5, 6]),
        )
        live = SimpleNamespace(
            reqs=live_reqs,
            spec_info=live_spec,
            seq_lens=torch.tensor([11, 33, 22]),
            seq_lens_cpu=torch.tensor([11, 33, 22]),
            seq_lens_sum=66,
            orig_seq_lens=torch.tensor([10, 30, 20], dtype=torch.int32),
            input_ids=torch.tensor([1, 3, 2]),
        )
        scheduler.running_mbs = [live]
        identities = (
            SimpleNamespace(rid="b", generation=(9, 0)),
            SimpleNamespace(rid="a", generation=(7, 0)),
        )

        scheduler._pp_sync_committed_snapshot_to_running_slots(committed, identities)

        self.assertEqual(live.spec_info.bonus_tokens.tolist(), [10, 3, 2])
        self.assertEqual(live.spec_info.new_seq_lens.tolist(), [110, 33, 22])
        self.assertIsNone(live.spec_info.nxt_kv_lens_cpu)
        self.assertIsNone(live.spec_info.future_indices)
        self.assertEqual(live.seq_lens.tolist(), [110, 33, 22])
        self.assertEqual(live.seq_lens_cpu.tolist(), [110, 33, 22])
        self.assertEqual(live.seq_lens_sum, 165)
        self.assertEqual(live.orig_seq_lens.tolist(), [100, 30, 20])
        self.assertIsNone(live.input_ids)

    def test_committed_snapshot_requires_frozen_identities(self):
        scheduler = SchedulerPPMixin()
        committed = SimpleNamespace(
            reqs=[SimpleNamespace(rid="a")],
        )
        scheduler.running_mbs = []

        with self.assertRaisesRegex(RuntimeError, "require matching frozen identities"):
            scheduler._pp_sync_committed_snapshot_to_running_slots(committed)


if __name__ == "__main__":
    unittest.main()
