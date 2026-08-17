import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMLAPPKVPtrs(unittest.TestCase):
    def _manager(
        self,
        start_layer: int,
        *,
        end_layer: int | None = None,
        compression_ratios: list[int] | None = None,
    ) -> CommonKVManager:
        manager = CommonKVManager.__new__(CommonKVManager)
        manager.kv_args = SimpleNamespace(
            prefill_start_layer=start_layer,
            prefill_end_layer=end_layer,
            mla_compression_ratios=compression_ratios,
        )
        return manager

    def test_matching_pp_layout_without_draft(self):
        src = list(range(38))
        dst = list(range(100, 138))
        got_src, got_dst, num_layers = self._manager(40).get_mla_kv_ptrs_with_pp(
            src, dst
        )
        self.assertEqual(got_src, src)
        self.assertEqual(got_dst, dst)
        self.assertEqual(num_layers, 38)

    def test_matching_pp_layout_ignores_decode_only_draft_cache(self):
        src = list(range(38))
        dst_main = list(range(100, 138))
        got_src, got_dst, num_layers = self._manager(40).get_mla_kv_ptrs_with_pp(
            src, dst_main + [999]
        )
        self.assertEqual(got_src, src)
        self.assertEqual(got_dst, dst_main)
        self.assertEqual(num_layers, 38)

    def test_prefill_pp_to_decode_pp1_uses_global_layer_slice(self):
        src = list(range(38))
        dst_full_model = list(range(100, 178))
        got_src, got_dst, num_layers = self._manager(40).get_mla_kv_ptrs_with_pp(
            src, dst_full_model + [999]
        )
        self.assertEqual(got_src, src)
        self.assertEqual(got_dst, dst_full_model[40:78])
        self.assertEqual(num_layers, 38)

    def test_short_local_decode_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shorter"):
            self._manager(40).get_mla_kv_ptrs_with_pp(list(range(38)), list(range(37)))

    def test_compressed_matching_pp_layout_ignores_decode_only_draft(self):
        ratios = [4, 128, 4, 128, 4, 128, 4, 128]
        src = list(range(6))
        dst_main = list(range(100, 106))
        got_src, got_dst, num_entries = self._manager(
            4, end_layer=8, compression_ratios=ratios
        ).get_mla_kv_ptrs_with_pp(src, dst_main + [999])
        self.assertEqual(got_src, src)
        self.assertEqual(got_dst, dst_main)
        self.assertEqual(num_entries, 6)

    def test_compressed_decode_pp1_ignores_appended_draft_after_bucket_slice(self):
        ratios = [4, 128, 4, 128, 4, 128, 4, 128]
        src = list(range(6))
        dst_full_target = (
            list(range(100, 104)) + list(range(200, 204)) + list(range(300, 304))
        )
        got_src, got_dst, num_entries = self._manager(
            4, end_layer=8, compression_ratios=ratios
        ).get_mla_kv_ptrs_with_pp(src, dst_full_target + [999])
        self.assertEqual(got_src, src)
        self.assertEqual(got_dst, [102, 103, 202, 203, 302, 303])
        self.assertEqual(num_entries, 6)


if __name__ == "__main__":
    unittest.main()
