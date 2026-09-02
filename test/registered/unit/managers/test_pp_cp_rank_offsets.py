import unittest
from collections import defaultdict, deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.distributed.parallel_state_wrapper import ParallelState  # noqa: E402
from sglang.srt.distributed.bootstrap import _prewarm_nccl  # noqa: E402
from sglang.srt.managers.scheduler_components.request_receiver import (  # noqa: E402
    SchedulerRequestReceiver,
)
from sglang.srt.managers.scheduler_pp_mixin import (  # noqa: E402
    _PP_DISAGG_SCHEDULER_FENCE_PHASES,
    SchedulerPPMixin,
    _pp_attention_dp_control_ranks,
    _pp_fence_scheduler_phase,
    _pp_pack_control_ring_message,
    _pp_unpack_control_ring_message,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_ps(**overrides) -> ParallelState:
    defaults = dict(
        tp_size=8,
        pp_rank=1,
        pp_size=2,
        dp_rank=None,
        attn_tp_size=2,
        attn_cp_size=2,
        attn_dp_rank=1,
        attn_dp_size=2,
        moe_dp_rank=None,
    )
    defaults.update(overrides)
    return ParallelState.trivial(**defaults)


def _fake_group() -> SimpleNamespace:
    return SimpleNamespace(rank=0, ranks=[0], cpu_group=object())


def _make_receiver(ps: ParallelState) -> SchedulerRequestReceiver:
    group = _fake_group()
    return SchedulerRequestReceiver(
        recv_from_tokenizer=None,
        recv_from_rpc=None,
        recv_skipper=None,
        input_blocker=None,
        mm_receiver=None,
        ps=ps,
        tp_group=group,
        tp_cpu_group=group,
        attn_tp_group=group,
        attn_tp_cpu_group=group,
        attn_cp_group=group,
        attn_cp_cpu_group=group,
        world_group=group,
        server_args=SimpleNamespace(
            enable_dp_attention=True,
            enable_dp_attention_local_control_broadcast=False,
        ),
        model_config=SimpleNamespace(is_multimodal=False),
        max_recv_per_poll=-1,
        stream_output=lambda *args, **kwargs: None,
        get_last_batch=lambda: None,
    )


class TestPPCPRankOffsets(unittest.TestCase):
    def test_nccl_prewarm_initializes_proxy_and_output_pp_channels(self):
        tp_handle = object()
        proxy_handle = object()
        output_handle = object()
        proxy_group = SimpleNamespace(
            device_group=proxy_handle, ranks=[0, 8], rank_in_group=0
        )
        output_group = SimpleNamespace(
            device_group=output_handle, ranks=[0, 8], rank_in_group=0
        )
        warmup_tensor = object()
        recv_tensor = object()
        send_work = Mock()
        recv_work = Mock()

        with (
            patch(
                "sglang.srt.distributed.bootstrap.get_tp_group",
                return_value=SimpleNamespace(device_group=tp_handle),
            ),
            patch(
                "sglang.srt.distributed.bootstrap.get_pp_group",
                return_value=proxy_group,
            ),
            patch(
                "sglang.srt.distributed.bootstrap.get_pp_output_group",
                return_value=output_group,
            ),
            patch(
                "sglang.srt.distributed.bootstrap.torch.zeros",
                return_value=warmup_tensor,
            ),
            patch(
                "sglang.srt.distributed.bootstrap.torch.empty_like",
                return_value=recv_tensor,
            ),
            patch(
                "sglang.srt.distributed.bootstrap.torch.cuda.current_device",
                return_value=0,
            ),
            patch(
                "sglang.srt.distributed.bootstrap.dist.all_reduce"
            ) as all_reduce,
            patch(
                "sglang.srt.distributed.bootstrap.dist.isend",
                return_value=send_work,
            ) as isend,
            patch(
                "sglang.srt.distributed.bootstrap.dist.irecv",
                return_value=recv_work,
            ) as irecv,
            patch(
                "sglang.srt.distributed.bootstrap.dist.barrier"
            ) as barrier,
            patch(
                "sglang.srt.distributed.bootstrap.current_platform.synchronize"
            ),
        ):
            _prewarm_nccl(tp_size=8, pp_size=2, moe_ep_size=8)

        self.assertEqual(
            all_reduce.call_args_list,
            [
                call(warmup_tensor, group=tp_handle),
                call(warmup_tensor, group=proxy_handle),
                call(warmup_tensor, group=output_handle),
            ],
        )
        self.assertEqual(
            isend.call_args_list,
            [
                call(warmup_tensor, dst=8, group=proxy_handle),
                call(warmup_tensor, dst=8, group=output_handle),
            ],
        )
        self.assertEqual(
            irecv.call_args_list,
            [
                call(recv_tensor, src=8, group=proxy_handle),
                call(recv_tensor, src=8, group=output_handle),
            ],
        )
        self.assertEqual(send_work.wait.call_count, 2)
        self.assertEqual(recv_work.wait.call_count, 2)
        self.assertEqual(
            barrier.call_args_list,
            [
                call(group=proxy_handle),
                call(group=proxy_handle),
                call(group=output_handle),
                call(group=output_handle),
            ],
        )

    def test_pp_proxy_and_output_use_independent_tensor_channels(self):
        proxy_group = SimpleNamespace(
            send_tensor_dict=Mock(return_value=[]),
            recv_tensor_dict=Mock(return_value={"__msg_type__": "proxy"}),
        )
        output_group = SimpleNamespace(
            send_tensor_dict=Mock(return_value=[]),
            recv_tensor_dict=Mock(return_value={"__msg_type__": "output"}),
        )
        all_gather_group = object()
        scheduler = SimpleNamespace(
            pp_group=proxy_group,
            pp_output_group=output_group,
            _pp_tensor_dict_inbox=defaultdict(deque),
            require_attn_tp_allgather=False,
            attn_tp_group=all_gather_group,
        )
        proxy_tensors = {"hidden_states": object()}
        output_tensors = {"next_token_ids": object()}

        SchedulerPPMixin._pp_send_dict_to_next_stage(
            scheduler, proxy_tensors, async_send=True, msg_type="proxy"
        )
        SchedulerPPMixin._pp_send_dict_to_next_stage(
            scheduler, output_tensors, async_send=True, msg_type="output"
        )
        SchedulerPPMixin._pp_recv_typed_dict(
            scheduler, expected_kind="proxy", all_gather_group=all_gather_group
        )
        SchedulerPPMixin._pp_recv_typed_dict(
            scheduler, expected_kind="output", all_gather_group=all_gather_group
        )

        proxy_group.send_tensor_dict.assert_called_once_with(
            tensor_dict=proxy_tensors, all_gather_group=None, async_send=True
        )
        output_group.send_tensor_dict.assert_called_once_with(
            tensor_dict=output_tensors, all_gather_group=None, async_send=True
        )
        proxy_group.recv_tensor_dict.assert_called_once_with(
            all_gather_group=all_gather_group
        )
        output_group.recv_tensor_dict.assert_called_once_with(
            all_gather_group=all_gather_group
        )

    def test_pp_disagg_output_ring_relays_fresh_payload_before_control_ring(self):
        events = []
        received_tensors = {"next_token_ids": object()}
        send_work = [object()]
        recorded_event = Mock()
        target = SimpleNamespace(
            forward_mode=SimpleNamespace(is_prebuilt=lambda: False)
        )
        scheduler = SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=False),
            copy_stream_ctx=nullcontext(),
            copy_stream=SimpleNamespace(
                wait_stream=lambda stream: events.append(("wait_stream", stream))
            ),
            schedule_stream=object(),
            device_module=SimpleNamespace(
                Event=Mock(return_value=recorded_event),
                current_stream=Mock(return_value=object()),
            ),
            _pp_recv_dict_from_prev_stage=Mock(
                side_effect=lambda: events.append("recv") or received_tensors
            ),
            _pp_prep_batch_result=Mock(
                side_effect=lambda batch, metadata, outputs: events.append("prep")
                or object()
            ),
            _pp_send_dict_to_next_stage=Mock(
                side_effect=lambda tensors, async_send, msg_type: events.append(
                    ("send", tensors, async_send, msg_type)
                )
                or send_work
            ),
            _pp_send_output_to_next_stage=Mock(),
            _pp_commit_comm_work=Mock(
                side_effect=lambda work: events.append(("commit", work))
            ),
        )

        with patch(
            "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
            return_value=False,
        ):
            outputs, _, event, work = (
                SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors(
                    scheduler,
                    next_first_rank_mb_id=0,
                    next_mb_id=0,
                    mbs=[target],
                    mb_metadata=[object()],
                    last_rank_comm_queue=deque(),
                    pp_outputs=None,
                    relay_output_immediately=True,
                )
            )

        self.assertIs(outputs.tensors, received_tensors)
        self.assertIs(event, recorded_event)
        self.assertEqual(work, [])
        self.assertEqual(
            events[-2:],
            [
                ("send", received_tensors, True, "output"),
                ("commit", send_work),
            ],
        )
        scheduler._pp_send_output_to_next_stage.assert_not_called()

    def test_pp_disagg_output_ring_last_stage_starts_relay_chain(self):
        events = []
        send_work = [object()]
        target = SimpleNamespace(
            forward_mode=SimpleNamespace(is_prebuilt=lambda: False)
        )
        recorded_event = Mock()
        scheduler = SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=True),
            copy_stream_ctx=nullcontext(),
            copy_stream=SimpleNamespace(wait_stream=Mock()),
            schedule_stream=object(),
            device_module=SimpleNamespace(
                Event=Mock(return_value=recorded_event),
                current_stream=Mock(return_value=object()),
            ),
            _pp_send_output_to_next_stage=Mock(
                side_effect=lambda *args: events.append("send") or send_work
            ),
            _pp_recv_dict_from_prev_stage=Mock(
                side_effect=lambda: events.append("recv")
                or {"next_token_ids": object()}
            ),
            _pp_prep_batch_result=Mock(return_value=object()),
            _pp_commit_comm_work=Mock(
                side_effect=lambda work: events.append(("commit", work))
            ),
        )

        with patch(
            "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
            return_value=False,
        ):
            _, _, _, work = (
                SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors(
                    scheduler,
                    next_first_rank_mb_id=0,
                    next_mb_id=0,
                    mbs=[target],
                    mb_metadata=[object()],
                    last_rank_comm_queue=deque(),
                    pp_outputs=None,
                    relay_output_immediately=True,
                )
            )

        self.assertIs(work, send_work)
        self.assertEqual(events, ["send", "recv"])
        scheduler._pp_commit_comm_work.assert_not_called()

    def test_pp_linear_payload_is_forwarded_before_following_control_phase(self):
        events = []
        previous_work = [object()]
        next_work = [object()]
        scheduler = SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=False),
            _pp_commit_comm_work=Mock(
                side_effect=lambda work: events.append(("commit", work))
            ),
            _pp_send_pyobj_to_next_stage=Mock(
                side_effect=lambda payload, async_send: events.append(
                    ("send", payload, async_send)
                )
                or next_work
            ),
        )

        result = SchedulerPPMixin._pp_forward_stage_payload(
            scheduler, previous_work, ["request"]
        )

        self.assertIs(result, next_work)
        self.assertEqual(
            events,
            [
                ("commit", previous_work),
                ("send", ["request"], True),
            ],
        )

    def test_pp_last_stage_consumes_linear_payload_without_forwarding(self):
        previous_work = [object()]
        scheduler = SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=True),
            _pp_commit_comm_work=Mock(),
            _pp_send_pyobj_to_next_stage=Mock(),
        )

        result = SchedulerPPMixin._pp_forward_stage_payload(
            scheduler, previous_work, ["request"]
        )

        self.assertEqual(result, [])
        scheduler._pp_commit_comm_work.assert_called_once_with(previous_work)
        scheduler._pp_send_pyobj_to_next_stage.assert_not_called()

    def test_pp_proxy_exchange_is_committed_before_control_ring(self):
        events = []
        proxy_work = [object()]
        tensor_dict = {"hidden_states": object()}
        scheduler = SimpleNamespace(
            send_proxy_work=[],
            _pp_send_dict_to_next_stage=Mock(
                side_effect=lambda tensors, async_send, msg_type: events.append(
                    ("send", tensors, async_send, msg_type)
                )
                or proxy_work
            ),
            _pp_commit_comm_work=Mock(
                side_effect=lambda work: events.append(("commit", work))
            ),
        )

        SchedulerPPMixin._pp_send_and_commit_proxy(scheduler, tensor_dict)

        self.assertIs(scheduler.send_proxy_work, proxy_work)
        self.assertEqual(
            events,
            [
                ("send", tensor_dict, True, "proxy"),
                ("commit", proxy_work),
            ],
        )

    def test_pp_control_ring_forwards_typed_payload_on_dedicated_group(self):
        events = []
        payload = [["ready"], []]
        incoming = _pp_pack_control_ring_message(
            "prefill_bootstrap_consensus", True, payload
        )
        control_group = object()
        local_control_group = object()
        scheduler = SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=False),
            pp_disagg_control_group=control_group,
            pp_disagg_local_control_group=local_control_group,
            _pp_recv_pyobj_from_prev_stage=Mock(
                side_effect=lambda group, local_group: events.append(
                    ("recv", group, local_group)
                )
                or incoming
            ),
            _pp_send_pyobj_to_next_stage=Mock(
                side_effect=lambda message, async_send, group: events.append(
                    ("send", group)
                )
                or [object()]
            ),
            _pp_commit_comm_work=Mock(
                side_effect=lambda work: events.append(("commit", work))
            ),
        )
        process_payload = Mock(
            side_effect=lambda value: events.append(("process", value)) or value
        )

        result = SchedulerPPMixin._pp_run_control_ring_phase(
            scheduler,
            phase="prefill_bootstrap_consensus",
            origin_has_payload=False,
            origin_payload=None,
            process_payload=process_payload,
        )

        self.assertEqual(result, payload)
        self.assertEqual(
            events[0], ("recv", control_group, local_control_group)
        )
        self.assertEqual(events[1], ("process", payload))
        self.assertEqual(events[2], ("send", control_group))
        forwarded = scheduler._pp_send_pyobj_to_next_stage.call_args.args[0]
        self.assertEqual(
            _pp_unpack_control_ring_message(
                forwarded, "prefill_bootstrap_consensus"
            ),
            (True, payload),
        )

    def test_pp_control_ring_last_stage_emits_typed_noop_every_slot(self):
        events = []
        incoming = _pp_pack_control_ring_message(
            "prefill_release_consensus", False, None
        )
        control_group = object()
        local_control_group = object()
        scheduler = SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=True),
            pp_disagg_control_group=control_group,
            pp_disagg_local_control_group=local_control_group,
            _pp_recv_pyobj_from_prev_stage=Mock(
                side_effect=lambda group, local_group: events.append(
                    ("recv", group, local_group)
                )
                or incoming
            ),
            _pp_send_pyobj_to_next_stage=Mock(
                side_effect=lambda message, async_send, group: events.append(
                    ("send", group)
                )
                or [object()]
            ),
            _pp_commit_comm_work=Mock(),
        )
        process_payload = Mock()

        result = SchedulerPPMixin._pp_run_control_ring_phase(
            scheduler,
            phase="prefill_release_consensus",
            origin_has_payload=False,
            origin_payload=["stale-must-not-leak"],
            process_payload=process_payload,
        )

        self.assertIsNone(result)
        self.assertEqual(
            events,
            [
                ("send", control_group),
                ("recv", control_group, local_control_group),
            ],
        )
        process_payload.assert_not_called()
        originated = scheduler._pp_send_pyobj_to_next_stage.call_args.args[0]
        self.assertEqual(
            _pp_unpack_control_ring_message(
                originated, "prefill_release_consensus"
            ),
            (False, None),
        )

    def test_pp_control_ring_rejects_phase_cross_match(self):
        message = _pp_pack_control_ring_message(
            "prefill_bootstrap_consensus", False, None
        )
        with self.assertRaisesRegex(
            RuntimeError, "prefill_release_consensus"
        ):
            _pp_unpack_control_ring_message(
                message, "prefill_release_consensus"
            )

    def test_pp_scheduler_fences_use_full_attention_dp_group_and_distinct_phases(
        self,
    ):
        ps = _make_ps()
        shifted_tp_ranks = list(range(24, 32))
        dp1_ranks = _pp_attention_dp_control_ranks(ps, shifted_tp_ranks)
        self.assertEqual(dp1_ranks, [28, 29, 30, 31])
        dp0_ranks = _pp_attention_dp_control_ranks(
            _make_ps(attn_dp_rank=0), shifted_tp_ranks
        )
        self.assertEqual(dp0_ranks, [24, 25, 26, 27])
        self.assertEqual(set(dp0_ranks).intersection(dp1_ranks), set())

        fence_groups = {phase: object() for phase in _PP_DISAGG_SCHEDULER_FENCE_PHASES}
        with patch(
            "sglang.srt.managers.scheduler_pp_mixin.torch.distributed.barrier"
        ) as barrier:
            for phase in _PP_DISAGG_SCHEDULER_FENCE_PHASES:
                _pp_fence_scheduler_phase(fence_groups[phase])
            _pp_fence_scheduler_phase(None)

        self.assertEqual(
            [call.kwargs["group"] for call in barrier.call_args_list],
            [fence_groups[phase] for phase in _PP_DISAGG_SCHEDULER_FENCE_PHASES],
        )
        self.assertEqual(len(set(fence_groups.values())), 4)

    def test_request_receiver_uses_cp_size_for_pp_recv_rank(self):
        ps = _make_ps()
        calls = []

        def fake_point_to_point_pyobj(data, rank, group, src, dst, **kwargs):
            calls.append((rank, src, dst))
            return ["req"]

        receiver = _make_receiver(ps)
        with patch(
            "sglang.srt.managers.scheduler_components.request_receiver."
            "point_to_point_pyobj",
            side_effect=fake_point_to_point_pyobj,
        ):
            self.assertEqual(receiver._pull_raw_reqs(), ["req"])

        self.assertEqual(calls, [(12, 4, 12)])

    def test_pp_mixin_uses_cp_size_for_pyobj_send_and_recv_rank(self):
        ps = _make_ps()
        scheduler = SchedulerPPMixin()
        scheduler.ps = ps
        scheduler.world_group = _fake_group()
        scheduler.tp_group = SimpleNamespace(ranks=list(range(8, 16)))
        control_group = object()
        local_control_group = object()
        scheduler.pp_disagg_control_group = control_group
        scheduler.pp_disagg_local_control_group = local_control_group
        scheduler.attn_tp_group = _fake_group()
        scheduler.attn_tp_cpu_group = _fake_group()
        scheduler.attn_cp_group = _fake_group()
        scheduler.attn_cp_cpu_group = _fake_group()
        calls = []
        broadcasts = []

        def fake_point_to_point_pyobj(data, rank, group, src, dst, **kwargs):
            calls.append(
                (rank, group, src, dst, kwargs.get("async_send", False))
            )
            return ["work"]

        with (
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.point_to_point_pyobj",
                side_effect=fake_point_to_point_pyobj,
            ),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.broadcast_pyobj",
                side_effect=lambda data, rank, group, **kwargs: broadcasts.append(
                    (rank, group, kwargs.get("src"))
                )
                or data,
            ),
        ):
            self.assertEqual(
                scheduler._pp_send_pyobj_to_next_stage(["data"], async_send=True),
                ["work"],
            )
            self.assertEqual(scheduler._pp_recv_pyobj_from_prev_stage(), ["work"])
            self.assertEqual(
                scheduler._pp_send_pyobj_to_next_stage(
                    ["control"], async_send=True, group=control_group
                ),
                ["work"],
            )
            self.assertEqual(
                scheduler._pp_recv_pyobj_from_prev_stage(group=control_group),
                ["work"],
            )
            with patch(
                "sglang.srt.managers.scheduler_pp_mixin."
                "_pp_attention_dp_control_ranks",
                return_value=[12, 13, 14, 15],
            ), patch(
                "sglang.srt.managers.scheduler_pp_mixin.torch.distributed.get_rank",
                return_value=12,
            ):
                self.assertEqual(
                    scheduler._pp_recv_pyobj_from_prev_stage(
                        group=control_group, local_group=local_control_group
                    ),
                    ["work"],
                )

        self.assertEqual(
            calls,
            [
                (12, scheduler.world_group.cpu_group, 12, 4, True),
                (12, scheduler.world_group.cpu_group, 4, 12, False),
                (12, control_group, 12, 4, True),
                (12, control_group, 4, 12, False),
                (12, control_group, 4, 12, False),
            ],
        )
        self.assertIn((12, local_control_group, 12), broadcasts)


if __name__ == "__main__":
    unittest.main()
