import unittest
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_pp_mixin import (  # noqa: E402
    PPBatchMetadata,
    SchedulerPPMixin,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Event:
    def __init__(self):
        self.recorded_stream = None

    def record(self, stream):
        self.recorded_stream = stream


def _make_scheduler(*, is_first_rank=False, is_last_rank=False):
    scheduler = SchedulerPPMixin()
    scheduler.pp_group = SimpleNamespace(
        is_first_rank=is_first_rank,
        is_last_rank=is_last_rank,
    )
    scheduler.attn_tp_group = object()
    scheduler.forward_stream_ctx = nullcontext()
    scheduler.forward_stream = MagicMock()
    scheduler.schedule_stream = object()
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
        )

    @patch(
        "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
        return_value=SimpleNamespace(pp_virtual_stages=2),
    )
    @patch("sglang.srt.managers.scheduler_pp_mixin.set_time_batch")
    def test_non_final_rank_runs_two_visits_and_forwards_both(
        self, _set_time_batch, _get_parallel
    ):
        scheduler = _make_scheduler()
        first_input = PPProxyTensors({"stage": 1})
        first_output = PPProxyTensors({"stage": 2})
        second_input = scheduler._pp_recv_vpp_proxy_tensors.return_value
        second_output = PPProxyTensors({"stage": 5})
        first_result = SimpleNamespace(
            pp_hidden_states_proxy_tensors=first_output,
            can_run_cuda_graph=False,
        )
        second_result = SimpleNamespace(
            pp_hidden_states_proxy_tensors=second_output,
            can_run_cuda_graph=False,
        )
        scheduler.run_batch = MagicMock(side_effect=[first_result, second_result])
        batch = SimpleNamespace(reqs=[])
        metadata = [None]

        result, event = scheduler._pp_launch_vpp_batch(
            0, batch, first_input, metadata, deque()
        )

        self.assertIs(result, second_result)
        self.assertEqual(event.recorded_stream, "forward-stream")
        self.assertEqual(metadata, [PPBatchMetadata(can_run_cuda_graph=False)])
        self.assertEqual(
            scheduler.run_batch.call_args_list,
            [call(batch, first_input), call(batch, second_input)],
        )
        self.assertEqual(
            scheduler._pp_send_dict_to_next_stage.call_args_list,
            [
                call(
                    first_output.tensors,
                    async_send=True,
                    msg_type="vpp_proxy",
                ),
                call(
                    second_output.tensors,
                    async_send=True,
                    msg_type="vpp_proxy",
                ),
            ],
        )
        self.assertEqual(scheduler._pp_commit_comm_work.call_count, 2)

    @patch(
        "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
        return_value=SimpleNamespace(pp_virtual_stages=2),
    )
    @patch("sglang.srt.managers.scheduler_pp_mixin.set_time_batch")
    def test_final_rank_queues_second_visit_output(
        self, _set_time_batch, _get_parallel
    ):
        scheduler = _make_scheduler(is_last_rank=True)
        first_output = PPProxyTensors({"stage": 4})
        first_result = SimpleNamespace(
            pp_hidden_states_proxy_tensors=first_output,
            can_run_cuda_graph=False,
        )
        final_result = SimpleNamespace(
            pp_hidden_states_proxy_tensors=None,
            can_run_cuda_graph=True,
        )
        scheduler.run_batch = MagicMock(side_effect=[first_result, final_result])
        batch = SimpleNamespace(reqs=[])
        metadata = [None]
        output_queue = deque()

        result, event = scheduler._pp_launch_vpp_batch(
            0, batch, PPProxyTensors({"stage": 3}), metadata, output_queue
        )

        self.assertIs(result, final_result)
        self.assertEqual(metadata, [PPBatchMetadata(can_run_cuda_graph=True)])
        scheduler._pp_send_dict_to_next_stage.assert_called_once_with(
            first_output.tensors,
            async_send=True,
            msg_type="vpp_proxy",
        )
        scheduler._pp_prepare_tensor_dict.assert_called_once_with(final_result, batch)
        queued_event, queued_output = output_queue.pop()
        self.assertIs(queued_event, event)
        self.assertEqual(
            queued_output.tensors,
            {"next_token_ids": "tokens"},
        )

    @patch(
        "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
        return_value=SimpleNamespace(pp_virtual_stages=3),
    )
    def test_launch_rejects_non_vpp2_layout(self, _get_parallel):
        scheduler = _make_scheduler()

        with self.assertRaisesRegex(RuntimeError, "supports VPP2 only"):
            scheduler._pp_launch_vpp_batch(
                0,
                SimpleNamespace(reqs=[]),
                None,
                [None],
                deque(),
            )


if __name__ == "__main__":
    unittest.main()
