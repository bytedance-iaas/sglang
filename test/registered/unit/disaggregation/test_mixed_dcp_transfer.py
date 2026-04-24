import unittest
from types import SimpleNamespace

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.common.conn import CommonKVManager, PrefillServerInfo
from sglang.srt.disaggregation.common.utils import (
    get_dcp_compatible_token_positions,
    get_dcp_compatible_transfer_page_size,
)

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


class TestMixedDCPTransfer(CustomTestCase):
    def test_transfer_page_size_expands_for_decode_only_dcp(self):
        self.assertEqual(get_dcp_compatible_transfer_page_size(16, 1, 2), 32)
        self.assertEqual(get_dcp_compatible_transfer_page_size(16, 2, 2), 16)

    def test_token_positions_stride_for_decode_only_dcp(self):
        src_pos, dst_pos, page_stride = get_dcp_compatible_token_positions(
            page_size=4,
            prefill_dcp_size=1,
            decode_dcp_size=2,
            decode_dcp_rank=1,
        )

        np.testing.assert_array_equal(src_pos, np.array([1, 3, 5, 7], dtype=np.int64))
        np.testing.assert_array_equal(dst_pos, np.array([0, 1, 2, 3], dtype=np.int64))
        self.assertEqual(page_stride, 2)

    def test_resolve_rank_mapping_uses_prefill_dcp_zero_for_decode_only_dcp(self):
        manager = CommonKVManager.__new__(CommonKVManager)
        manager.dcp_size = 2
        manager.dcp_rank = 1
        manager.attn_tp_size = 2
        manager.attn_cp_size = 1
        manager.attn_cp_rank = 0
        manager.pp_size = 1
        manager.pp_rank = 0
        manager.is_mla_backend = False
        manager.enable_all_cp_ranks_for_transfer = False
        manager.kv_args = SimpleNamespace(engine_rank=1)

        info = PrefillServerInfo(
            attn_tp_size=2,
            dcp_size=1,
            attn_cp_size=1,
            dp_size=1,
            pp_size=1,
            page_size=16,
            kv_cache_dtype="auto",
            follow_bootstrap_room=True,
        )

        CommonKVManager._resolve_rank_mapping(manager, info)

        self.assertEqual(info.target_dcp_ranks, [0])
        self.assertEqual(info.target_tp_ranks, [1])
        self.assertEqual(info.target_pp_ranks, [0])


if __name__ == "__main__":
    unittest.main()
