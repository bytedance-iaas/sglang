import inspect
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeEvent:
    def record(self, stream):
        pass


class _FakeStream:
    def wait_stream(self, stream):
        pass


class TestSchedulerPPRelayOrder(unittest.TestCase):
    def _make_scheduler(self, is_last_rank, pp_rank=0, speculative=False):
        scheduler = SchedulerPPMixin()
        scheduler.pp_group = SimpleNamespace(is_last_rank=is_last_rank)
        scheduler.ps = SimpleNamespace(pp_rank=pp_rank)
        scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: not speculative)
        scheduler.copy_stream_ctx = nullcontext()
        scheduler.copy_stream = _FakeStream()
        scheduler.schedule_stream = _FakeStream()
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
        scheduler._pp_prep_batch_result = lambda *args: events.append("prep")
        scheduler._pp_send_dict_to_next_stage = lambda *args, **kwargs: (
            events.append("relay") or ["relay-work"]
        )
        scheduler._pp_commit_comm_work = lambda work: events.append(("commit", work))

        with patch(
            "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
            return_value=False,
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


if __name__ == "__main__":
    unittest.main()
