import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention.dsa.utils import cal_padded_tokens
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSAPaddedTokens(unittest.TestCase):
    def _cal(self, global_num_tokens, *, dp_mode=DpPaddingMode.MAX_LEN):
        forward_batch = SimpleNamespace(
            global_num_tokens_cpu=global_num_tokens,
            dp_padding_mode=dp_mode,
            is_extend_in_batch=False,
        )
        parallel = SimpleNamespace(
            attn_tp_size=2,
            attn_cp_size=1,
            attn_dp_rank=0,
        )
        with (
            patch(
                "sglang.srt.layers.attention.dsa.utils.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
                return_value=1,
            ),
            patch(
                "sglang.srt.layers.cp.utils.enable_cp_v2",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.cp.utils.is_cp_v2_active",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.attention.dsa.utils."
                "can_dsa_prefill_cp_round_robin_split",
                return_value=False,
            ),
        ):
            return cal_padded_tokens(forward_batch)

    def test_single_draft_row_aligns_to_attention_tp(self):
        self.assertEqual(self._cal([1]), 2)

    def test_each_dp_rank_is_aligned_before_max_padding(self):
        self.assertEqual(self._cal([1, 3]), 4)


if __name__ == "__main__":
    unittest.main()
