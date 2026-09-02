import unittest

from sglang.srt.managers.scheduler_pp_mixin import should_pp_allgather_tensors
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

    def test_dsa_prefill_cp_sends_each_lane_intact(self):
        self.assertFalse(
            should_pp_allgather_tensors(
                enable_dsa_prefill_context_parallel=True,
                require_attn_tp_gather_=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
