import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_INDEXER = "sglang.srt.layers.attention.dsa.dsa_indexer"


class TestDSACPPrepareFusionGate(unittest.TestCase):
    def setUp(self):
        self.indexer = Indexer.__new__(Indexer)
        self.batch = SimpleNamespace(attn_cp_metadata=SimpleNamespace())

    def test_requires_opt_in(self):
        self.indexer.use_dsa_cp_local_prepare_fusion = False
        with patch(f"{_INDEXER}.is_cp_v2_active", return_value=True):
            self.assertFalse(
                self.indexer._can_use_cp_local_prepare_fusion(
                    self.batch,
                    in_piecewise_or_breakable_cuda_graph=False,
                )
            )

    def test_requires_cp_metadata(self):
        self.indexer.use_dsa_cp_local_prepare_fusion = True
        batch = SimpleNamespace(attn_cp_metadata=None)
        with patch(f"{_INDEXER}.is_cp_v2_active", return_value=True):
            self.assertFalse(
                self.indexer._can_use_cp_local_prepare_fusion(
                    batch,
                    in_piecewise_or_breakable_cuda_graph=False,
                )
            )

    def test_requires_cp_v2(self):
        self.indexer.use_dsa_cp_local_prepare_fusion = True
        with patch(f"{_INDEXER}.is_cp_v2_active", return_value=False):
            self.assertFalse(
                self.indexer._can_use_cp_local_prepare_fusion(
                    self.batch,
                    in_piecewise_or_breakable_cuda_graph=False,
                )
            )

    def test_rejects_graph_surface(self):
        self.indexer.use_dsa_cp_local_prepare_fusion = True
        with patch(f"{_INDEXER}.is_cp_v2_active", return_value=True):
            self.assertFalse(
                self.indexer._can_use_cp_local_prepare_fusion(
                    self.batch,
                    in_piecewise_or_breakable_cuda_graph=True,
                )
            )

    def test_accepts_eager_cp_v2(self):
        self.indexer.use_dsa_cp_local_prepare_fusion = True
        with patch(f"{_INDEXER}.is_cp_v2_active", return_value=True):
            self.assertTrue(
                self.indexer._can_use_cp_local_prepare_fusion(
                    self.batch,
                    in_piecewise_or_breakable_cuda_graph=False,
                )
            )

    def test_prepare_runs_fused_local_math_then_one_bf16_gather(self):
        self.indexer.use_dsa_cp_local_prepare_fusion = True
        self.indexer.alt_stream = None
        self.indexer.head_dim = 4
        self.indexer.n_heads = 2
        self.indexer.softmax_scale = 0.5
        self.indexer.scale_fmt = "ue8m0"
        self.indexer.k_norm = SimpleNamespace(
            weight=torch.ones(4),
            bias=torch.zeros(4),
            variance_epsilon=1e-6,
        )
        self.indexer.rotary_emb = SimpleNamespace(
            cos_sin_cache=torch.arange(64, dtype=torch.float32).view(8, 8)
        )

        x = torch.arange(18, dtype=torch.float32).view(3, 6)
        q_lora = torch.arange(15, dtype=torch.float32).view(3, 5)
        positions = torch.tensor([0, 3, 7], dtype=torch.int32)
        key_raw = torch.arange(12, dtype=torch.float32).view(3, 4)
        weights_raw = torch.arange(6, dtype=torch.float32).view(3, 2)
        q_raw = torch.arange(24, dtype=torch.float32).view(3, 8)
        q_fp8 = torch.full((3, 2, 4), 11, dtype=torch.int8)
        weights = torch.full((3, 2, 1), 13, dtype=torch.float32)
        key_local = torch.full((3, 4), 17, dtype=torch.bfloat16)
        key_full = torch.full((9, 4), 19, dtype=torch.bfloat16)

        self.indexer.wk_weights_proj = Mock(
            return_value=(
                torch.cat([key_raw, weights_raw], dim=-1),
                None,
            )
        )
        self.indexer.wq_b = Mock(return_value=(q_raw, None))
        strategy = SimpleNamespace(
            materialize_full_indexer_k_cache=Mock(return_value=key_full)
        )

        with (
            patch(f"{_INDEXER}.is_cp_v2_active", return_value=True),
            patch(f"{_INDEXER}.get_cp_strategy", return_value=strategy),
            patch(
                f"{_INDEXER}.fused_q_indexer_rope_first_quant",
                return_value=(q_fp8, weights),
            ) as fused_q,
            patch(
                f"{_INDEXER}.fused_k_indexer_norm_rope",
                return_value=key_local,
            ) as fused_k,
        ):
            actual_q, actual_weights, actual_key = self.indexer._fused_cp_local_prepare(
                x, q_lora, positions, self.batch
            )

        self.assertIs(actual_q, q_fp8)
        self.assertIs(actual_weights, weights)
        self.assertIs(actual_key, key_full)
        fused_q.assert_called_once()
        fused_k.assert_called_once()
        self.assertIs(fused_q.call_args.args[4], positions)
        self.assertEqual(fused_q.call_args.args[5], "ue8m0")
        self.assertIs(fused_k.call_args.args[5], positions)
        strategy.materialize_full_indexer_k_cache.assert_called_once_with(
            key_local, self.batch
        )


if __name__ == "__main__":
    unittest.main()
