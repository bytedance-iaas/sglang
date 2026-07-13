import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.models.minimax_m3 import MiniMaxM3Attention
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMiniMaxM3AttentionConfig(CustomTestCase):
    @patch("sglang.srt.models.minimax_m3.get_parallel")
    def test_attention_output_gate_is_rejected_at_initialization(self, get_parallel):
        get_parallel.return_value = SimpleNamespace(attn_tp_rank=0, attn_tp_size=1)
        config = SimpleNamespace(
            hidden_size=128,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=128,
            rope_theta=10000,
            rope_scaling=None,
            max_position_embeddings=1024,
            rotary_dim=64,
            use_qk_norm=True,
            qk_norm_type="per_head",
            use_gemma_norm=True,
            attention_output_gate=True,
        )

        with self.assertRaisesRegex(NotImplementedError, "attention_output_gate"):
            MiniMaxM3Attention(config)


if __name__ == "__main__":
    unittest.main()
