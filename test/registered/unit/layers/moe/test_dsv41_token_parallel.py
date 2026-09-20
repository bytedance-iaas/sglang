"""Scope and token ownership checks for DSV4.1's residual SP region."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.dsv41_token_parallel import (
    DSV41TokenParallel,
    can_use_dsv41_token_parallel,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSV41TokenParallel(unittest.TestCase):
    def test_unsupported_paths_retain_existing_layout(self):
        supported = dict(
            enabled=True,
            model_type="deepseek_v41",
            sm90=True,
            cuda=True,
            decode=True,
            capture_hidden=False,
            capture_dspark=False,
            hc_pre_from_prev=True,
            pp_size=1,
            tp_size=8,
            attn_tp_size=8,
            attn_dp_size=1,
            attn_cp_size=1,
            moe_ep_size=8,
            moe_tp_size=1,
            megamoe=True,
            other_sp=False,
            rows=32,
        )
        self.assertTrue(can_use_dsv41_token_parallel(**supported))
        unsupported = dict(
            enabled=False,
            model_type="deepseek_v4",
            sm90=False,
            cuda=False,
            decode=False,
            capture_hidden=True,
            capture_dspark=True,
            hc_pre_from_prev=False,
            pp_size=2,
            tp_size=4,
            attn_tp_size=4,
            attn_dp_size=8,
            attn_cp_size=8,
            moe_ep_size=4,
            moe_tp_size=2,
            megamoe=False,
            other_sp=True,
            rows=7,
        )
        for key, value in unsupported.items():
            with self.subTest(key=key):
                self.assertFalse(
                    can_use_dsv41_token_parallel(**{**supported, key: value})
                )
        self.assertFalse(can_use_dsv41_token_parallel(**{**supported, "rows": 0}))

    def test_ids_and_four_way_residual_share_token_ownership(self):
        for rows in (8, 16, 24, 32, 256):
            ids = torch.arange(rows)
            residual = torch.arange(rows * 4 * 8).view(rows, 4, 8)
            local_ids, local_residuals = [], []
            for rank in range(8):
                layout = DSV41TokenParallel(
                    SimpleNamespace(world_size=8, rank_in_group=rank), rows
                )
                local_ids.append(layout.local(ids))
                local_residuals.append(layout.local(residual))
                self.assertEqual(local_ids[-1].numel(), rows // 8)
            torch.testing.assert_close(torch.cat(local_ids), ids)
            torch.testing.assert_close(torch.cat(local_residuals), residual)


if __name__ == "__main__":
    unittest.main()
