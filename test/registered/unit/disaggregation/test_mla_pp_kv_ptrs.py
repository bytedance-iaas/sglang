import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMLAPPKVPtrs(unittest.TestCase):
    def _manager(self, start_layer: int):
        manager = CommonKVManager.__new__(CommonKVManager)
        manager.kv_args = SimpleNamespace(
            prefill_start_layer=start_layer,
            mla_compression_ratios=None,
        )
        return manager

    def test_matching_pp_layout_without_draft(self):
        src = list(range(17))
        dst = list(range(100, 117))
        got_src, got_dst, num_layers = self._manager(61).get_mla_kv_ptrs_with_pp(
            src, dst
        )
        self.assertEqual(got_src, src)
        self.assertEqual(got_dst, dst)
        self.assertEqual(num_layers, 17)

    def test_matching_pp_layout_ignores_decode_only_draft_cache(self):
        src = list(range(17))
        dst_main = list(range(100, 117))
        dst_draft = [999]
        got_src, got_dst, num_layers = self._manager(61).get_mla_kv_ptrs_with_pp(
            src, dst_main + dst_draft
        )
        self.assertEqual(got_src, src)
        self.assertEqual(got_dst, dst_main)
        self.assertEqual(num_layers, 17)

    def test_prefill_pp_to_decode_pp1_uses_global_layer_slice(self):
        src = list(range(20))
        dst_full_model = list(range(100, 178))
        dst_draft = [999]
        got_src, got_dst, num_layers = self._manager(21).get_mla_kv_ptrs_with_pp(
            src, dst_full_model + dst_draft
        )
        self.assertEqual(got_src, src)
        self.assertEqual(got_dst, dst_full_model[21:41])
        self.assertEqual(num_layers, 20)

    def test_short_local_decode_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shorter"):
            self._manager(61).get_mla_kv_ptrs_with_pp(
                list(range(17)), list(range(16))
            )


if __name__ == "__main__":
    unittest.main()
