import unittest

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.base_runner import (
    resolve_pp_proxy_num_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestResolvePPProxyNumTokens(unittest.TestCase):
    def _resolve(self, **overrides):
        kwargs = dict(
            tensor_name="hidden_states",
            num_tokens=64,
            forward_mode=ForwardMode.TARGET_VERIFY,
            pp_rank=1,
            attn_tp_size=2,
            attn_cp_size=1,
            require_attn_tp_gather_=True,
        )
        kwargs.update(overrides)
        return resolve_pp_proxy_num_tokens(**kwargs)

    def test_first_pp_rank_keeps_full_width(self):
        self.assertEqual(self._resolve(pp_rank=0), 64)

    def test_attention_tp_scattered_proxy_is_rank_local(self):
        self.assertEqual(self._resolve(), 32)

    def test_attention_tp_topk_metadata_keeps_full_width(self):
        self.assertEqual(self._resolve(tensor_name="topk_indices"), 64)

    def test_tp_full_proxy_keeps_full_width(self):
        self.assertEqual(self._resolve(require_attn_tp_gather_=False), 64)

    def test_extend_cp_uses_cp_local_width(self):
        self.assertEqual(
            self._resolve(
                forward_mode=ForwardMode.EXTEND,
                attn_cp_size=4,
            ),
            16,
        )

    def test_extend_cp_topk_metadata_is_also_cp_local(self):
        self.assertEqual(
            self._resolve(
                tensor_name="topk_indices",
                forward_mode=ForwardMode.EXTEND,
                attn_cp_size=4,
            ),
            16,
        )

    def test_unaligned_width_matches_attention_tp_padding(self):
        self.assertEqual(self._resolve(num_tokens=3), 2)


if __name__ == "__main__":
    unittest.main()
