import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import communicator_dsa_cp as dsa_cp
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDsv41CollectiveFusionGate(CustomTestCase):
    def _eligible(self, *, rank_tokens="default", shared_tp1=False):
        if rank_tokens == "default":
            rank_tokens = [16] * 8
        local_tokens = rank_tokens[0] if rank_tokens is not None else 16
        comm = types.SimpleNamespace(disabled=False, world_size=8)
        mlp = types.SimpleNamespace(
            num_fused_shared_experts=0,
            _shared_expert_tp1=shared_tp1,
            shared_experts_is_fp8=True,
            shared_experts_weight_block_size=[128, 128],
            shared_experts=types.SimpleNamespace(
                gate_up_proj=types.SimpleNamespace(
                    weight=object(),
                    weight_scale_inv=object(),
                )
            ),
        )
        batch = types.SimpleNamespace(
            attn_cp_metadata=types.SimpleNamespace(per_rank_actual_token=rank_tokens),
            extend_seq_lens_cpu=[
                sum(rank_tokens) if rank_tokens is not None else local_tokens * 8
            ],
        )
        hidden = torch.empty(local_tokens, 128, dtype=torch.bfloat16)

        with (
            patch.object(
                dsa_cp.envs.SGLANG_DSV41_CP_AG_SHARED_GEMM,
                "get",
                return_value=True,
            ),
            patch.object(dsa_cp, "dsa_use_prefill_cp", return_value=True),
            patch.object(dsa_cp, "max_prefill_buffer_tokens", return_value=16384),
            patch(
                "sglang.srt.distributed.get_tp_group",
                return_value=types.SimpleNamespace(torch_symm_mem_comm=comm),
            ),
            patch(
                "sglang.srt.layers.attention.dsa.utils.is_dsa_prefill_cp_round_robin_split",
                return_value=True,
            ),
            patch(
                "sglang.srt.model_executor.runner.get_is_capture_mode",
                return_value=False,
            ),
            get_parallel().override(
                attn_dp_size=1,
                attn_tp_size=1,
                attn_cp_size=8,
            ),
        ):
            return dsa_cp.dsa_cp_fused_ag_shared_experts_eligible(mlp, batch, hidden)

    def test_target_pp2_tp8_cp8_surface_is_eligible(self):
        self.assertTrue(self._eligible())

    def test_unequal_cp_shards_fall_back(self):
        self.assertFalse(self._eligible(rank_tokens=[17, 16, 16, 16, 16, 16, 16, 15]))

    def test_legacy_metadata_uses_global_divisibility(self):
        self.assertTrue(self._eligible(rank_tokens=None))

    def test_tp1_shared_expert_falls_back(self):
        self.assertFalse(self._eligible(shared_tp1=True))


if __name__ == "__main__":
    unittest.main()
