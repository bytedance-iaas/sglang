import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention.dsa import utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _PerRankPadding:
    def is_max_len(self):
        return False


class TestDSAPaddedTokens(CustomTestCase):
    def test_matches_attention_tp_then_cp_padding(self):
        forward_batch = SimpleNamespace(
            global_num_tokens_cpu=[4],
            is_extend_in_batch=False,
            forward_mode=SimpleNamespace(
                is_context_parallel_extend=lambda: False,
            ),
        )
        parallel = SimpleNamespace(
            attn_cp_size=1,
            attn_tp_size=32,
            attn_dp_rank=0,
        )

        with (
            patch.object(utils, "get_parallel", return_value=parallel),
            patch.object(
                utils.DpPaddingMode,
                "get_dp_padding_mode",
                return_value=_PerRankPadding(),
            ),
            patch(
                "sglang.srt.layers.utils.cp_utils.get_cp_padding_align_size",
                return_value=1,
            ),
        ):
            self.assertEqual(utils.cal_padded_tokens(forward_batch), 32)


if __name__ == "__main__":
    unittest.main()
