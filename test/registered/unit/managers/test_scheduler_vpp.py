import unittest
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

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
    def test_vpp_prefill_uses_dedicated_collective_loop(self):
        scheduler = SchedulerPPMixin()
        scheduler._pp_vpp_enabled = MagicMock(return_value=True)
        scheduler._event_loop_pp_disagg_prefill_vpp = MagicMock()

        scheduler.event_loop_pp_disagg_prefill()

        scheduler._event_loop_pp_disagg_prefill_vpp.assert_called_once_with()

    def test_vpp_request_broadcast_uses_first_rank_ingress(self):
        scheduler = SchedulerPPMixin()
        requests = [object()]
        scheduler.ingest_requests = MagicMock(return_value=requests)
        scheduler.pp_group = SimpleNamespace(
            is_first_rank=True,
            broadcast_object=MagicMock(return_value=requests),
        )

        result = scheduler._pp_vpp_ingest_requests()

        self.assertIs(result, requests)
        scheduler.ingest_requests.assert_called_once_with()
        scheduler.pp_group.broadcast_object.assert_called_once_with(requests, src=0)

    def test_vpp_request_broadcast_processes_non_first_rank(self):
        scheduler = SchedulerPPMixin()
        requests = [object()]
        scheduler.pp_group = SimpleNamespace(
            is_first_rank=False,
            broadcast_object=MagicMock(return_value=requests),
        )
        scheduler.metrics_reporter = SimpleNamespace(
            record_scheduler_active=MagicMock()
        )
        scheduler.process_input_requests = MagicMock()

        result = scheduler._pp_vpp_ingest_requests()

        self.assertIs(result, requests)
        scheduler.pp_group.broadcast_object.assert_called_once_with(None, src=0)
        scheduler.metrics_reporter.record_scheduler_active.assert_called_once()
        scheduler.process_input_requests.assert_called_once_with(requests)

    def test_vpp_bootstrap_consensus_intersects_good_and_unions_bad(self):
        scheduler = SchedulerPPMixin()
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
        scheduler.get_rids = MagicMock(return_value=(["a", "b"], ["x"]))
        scheduler.pp_group = SimpleNamespace(
            all_gather_object=MagicMock(
                return_value=[
                    [["a", "b"], ["x"]],
                    [["a", "c"], ["y"]],
                    [["a"], []],
                    [["a", "d"], ["b"]],
                ]
            )
        )

        result = scheduler._pp_vpp_collect_bootstrapped_ids()

        self.assertEqual(result, [["a"], ["b", "x", "y"]])

    def test_vpp_transfer_consensus_intersects_all_ranks(self):
        scheduler = SchedulerPPMixin()
        scheduler.disagg_prefill_inflight_queue = []
        scheduler.get_rids = MagicMock(return_value=["a", "b"])
        scheduler.pp_group = SimpleNamespace(
            all_gather_object=MagicMock(
                return_value=[
                    ["a", "b"],
                    ["a", "c"],
                    ["a"],
                    ["a", "d"],
                ]
            )
        )

        result = scheduler._pp_vpp_collect_transferred_ids()

        self.assertEqual(result, ["a"])

    def test_vpp_output_is_broadcast_from_last_physical_rank(self):
        scheduler = SchedulerPPMixin()
        output_tensors = {"next_token_ids": torch.tensor([1])}
        scheduler.pp_group = SimpleNamespace(
            is_last_rank=True,
            world_size=4,
            broadcast_tensor_dict=MagicMock(return_value=output_tensors),
        )
        scheduler.device_module = SimpleNamespace(
            current_stream=MagicMock(),
            Event=MagicMock(return_value=MagicMock()),
        )
        scheduler.copy_stream_ctx = nullcontext()
        scheduler.copy_stream = MagicMock()
        scheduler.schedule_stream = object()
        expected = object()
        scheduler._pp_prep_batch_result = MagicMock(return_value=expected)
        output_event = MagicMock()
        output_proxy = PPProxyTensors(output_tensors)

        result = scheduler._pp_vpp_broadcast_batch_result(
            SimpleNamespace(),
            PPBatchMetadata(can_run_cuda_graph=False),
            deque([(output_event, output_proxy)]),
        )

        self.assertIs(result, expected)
        scheduler.pp_group.broadcast_tensor_dict.assert_called_once_with(
            output_tensors,
            src=3,
        )
        scheduler.device_module.current_stream().wait_event.assert_called_once_with(
            output_event
        )

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
                    batch_p2p=True,
                ),
                call(
                    second_output.tensors,
                    async_send=True,
                    msg_type="vpp_proxy",
                    batch_p2p=True,
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
            batch_p2p=True,
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
