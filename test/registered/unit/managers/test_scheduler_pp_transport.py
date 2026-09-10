import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.scheduler_pp_mixin import (
    SchedulerPPMixin,
    should_pp_allgather_tensors,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestShouldPPAllgatherTensors(unittest.TestCase):
    def test_replicated_attention_tp_layout_uses_allgather(self):
        self.assertTrue(
            should_pp_allgather_tensors(
                enable_dsa_prefill_context_parallel=False,
                require_attn_tp_gather_=False,
            )
        )

    def test_token_scattered_a2a_layout_sends_each_lane_intact(self):
        self.assertFalse(
            should_pp_allgather_tensors(
                enable_dsa_prefill_context_parallel=False,
                require_attn_tp_gather_=True,
            )
        )

    def test_prefill_pp_output_callback_observes_serialized_target_token(self):
        callback = MagicMock()
        scheduler = SimpleNamespace(
            draft_worker=SimpleNamespace(record_prefill_pp_output=callback)
        )
        token = torch.tensor([8451], dtype=torch.int64)
        result = GenerationBatchResult(
            logits_output=None,
            next_token_ids=token,
            next_draft_input=SimpleNamespace(topk_p=None, bonus_tokens=token),
        )
        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, return_logprob=False)

        tensors = SchedulerPPMixin._pp_prepare_tensor_dict(scheduler, result, batch)

        self.assertIs(tensors["next_token_ids"], token)
        callback.assert_called_once_with(
            batch=batch, result=result, pp_next_token_ids=token
        )

    def test_dsa_prefill_cp_sends_each_lane_intact(self):
        self.assertFalse(
            should_pp_allgather_tensors(
                enable_dsa_prefill_context_parallel=True,
                require_attn_tp_gather_=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
