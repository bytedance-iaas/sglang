import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

from sglang.srt.managers.scheduler_pp_mixin import (
    SchedulerPPMixin,
    _pp_snapshot_graph_output_tensors,
)


class TestSchedulerPPCudaGraphOutput(unittest.TestCase):
    def test_proxy_send_waits_for_launch_on_schedule_stream(self):
        calls = []
        launch_event = object()
        payload = {"hidden_states": torch.tensor([1])}
        expected_work = [object()]
        scheduler = SimpleNamespace(
            schedule_stream=SimpleNamespace(
                wait_event=lambda event: calls.append(("wait", event))
            ),
            launch_event=launch_event,
            _pp_send_dict_to_next_stage=lambda tensor_dict, **kwargs: (
                calls.append(("send", tensor_dict, kwargs)) or expected_work
            ),
        )

        actual_work = SchedulerPPMixin._pp_send_proxy_after_launch(scheduler, payload)

        self.assertIs(actual_work, expected_work)
        self.assertEqual(calls[0], ("wait", launch_event))
        self.assertEqual(calls[1][0], "send")
        self.assertIs(calls[1][1], payload)
        self.assertEqual(calls[1][2], {"async_send": True, "msg_type": "proxy"})

    def test_graph_output_is_detached_recursively(self):
        tensor = torch.tensor([1, 2, 3])
        nested = torch.tensor([4, 5])
        source = {"token_ids": tensor, "nested": [nested], "metadata": None}

        snapshot = _pp_snapshot_graph_output_tensors(source, True)
        tensor.add_(10)
        nested.add_(10)

        self.assertEqual(snapshot["token_ids"].tolist(), [1, 2, 3])
        self.assertEqual(snapshot["nested"][0].tolist(), [4, 5])
        self.assertNotEqual(snapshot["token_ids"].data_ptr(), tensor.data_ptr())
        self.assertNotEqual(snapshot["nested"][0].data_ptr(), nested.data_ptr())

    def test_eager_output_keeps_original_objects(self):
        source = {"token_ids": torch.tensor([1])}

        self.assertIs(_pp_snapshot_graph_output_tensors(source, False), source)


if __name__ == "__main__":
    unittest.main()
