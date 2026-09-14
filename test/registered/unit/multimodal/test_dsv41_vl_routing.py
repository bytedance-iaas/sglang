"""Unit tests for srt/multimodal/dsv41/vl_routing"""

import unittest
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.topk import TopKConfig
from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.models.deepseek_v4 import (
    _dsv41_multimodal_enabled,
    _dsv41_weight_skip_group,
    _normalize_dsv41_prompt_input_ids,
    _should_build_dsv41_vision,
    _should_prepare_dsv41_vision,
)
from sglang.srt.multimodal.dsv41.vl_routing import vision_topk
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_EXPERTS = 16
NUM_ROUTED_TOPK = 4
ROUTED_SCALING_FACTOR = 1.5


def _make_moe(
    num_fused_shared_experts: int,
    fused_shared_experts_scaling_factor: Optional[float] = None,
):
    return SimpleNamespace(
        gate=SimpleNamespace(
            e_score_correction_bias=torch.zeros(NUM_EXPERTS),
            e_score_correction_bias_vl=torch.zeros(NUM_EXPERTS),
        ),
        config=SimpleNamespace(image_token_id=0),
        topk=SimpleNamespace(
            topk_config=TopKConfig(
                top_k=NUM_ROUTED_TOPK + num_fused_shared_experts,
                renormalize=True,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=ROUTED_SCALING_FACTOR,
                apply_routed_scaling_factor_on_output=True,
                scoring_func="sqrtsoftplus",
                fused_shared_experts_scaling_factor=fused_shared_experts_scaling_factor,
            )
        ),
    )


class TestDsv41VisionTopK(CustomTestCase):
    @patch("sglang.srt.multimodal.dsv41.vl_routing.is_cuda", return_value=False)
    def test_fused_shared_expert_slot(self, _mock_is_cuda):
        torch.manual_seed(0)
        logits = torch.randn(8, NUM_EXPERTS)

        unfused = vision_topk(_make_moe(0), logits, None)
        fused = vision_topk(_make_moe(1), logits, None)

        self.assertEqual(fused.topk_ids.shape, (8, NUM_ROUTED_TOPK + 1))
        # The shared expert occupies the slot past the routed experts and, with
        # renormalization, contributes with weight 1.0.
        torch.testing.assert_close(
            fused.topk_ids[:, -1],
            torch.full((8,), NUM_EXPERTS, dtype=fused.topk_ids.dtype),
        )
        torch.testing.assert_close(fused.topk_weights[:, -1], torch.ones(8))
        # Routing of the non-shared slots is unchanged by fusion.
        torch.testing.assert_close(fused.topk_ids[:, :-1], unfused.topk_ids)
        torch.testing.assert_close(fused.topk_weights[:, :-1], unfused.topk_weights)

    @patch("sglang.srt.multimodal.dsv41.vl_routing.is_cuda", return_value=False)
    def test_fused_shared_expert_ep_scaling(self, _mock_is_cuda):
        torch.manual_seed(0)
        logits = torch.randn(8, NUM_EXPERTS)
        ep_size = 8

        fused = vision_topk(_make_moe(1), logits, None)
        scaled = vision_topk(_make_moe(1, 1 / ep_size), logits, None)

        # Standard EP replicates the shared expert per rank, so its weight is
        # divided by ep_size while the routed slots are untouched.
        torch.testing.assert_close(scaled.topk_ids, fused.topk_ids)
        torch.testing.assert_close(
            scaled.topk_weights[:, -1], fused.topk_weights[:, -1] / ep_size
        )
        torch.testing.assert_close(
            scaled.topk_weights[:, :-1], fused.topk_weights[:, :-1]
        )

    def test_vpp_builds_vision_only_on_first_physical_stage(self):
        config = SimpleNamespace(
            model_type="deepseek_v41",
            vision_n_layers=12,
            language_only=False,
            language_model_only=False,
        )

        self.assertTrue(
            _should_build_dsv41_vision(
                config,
                SimpleNamespace(is_first_rank=True),
            )
        )
        self.assertFalse(
            _should_build_dsv41_vision(
                config,
                SimpleNamespace(is_first_rank=False),
            )
        )
        self.assertTrue(_dsv41_multimodal_enabled(config))

    def test_text_only_modes_skip_vision_on_first_stage(self):
        for field in ("language_only", "language_model_only"):
            config = SimpleNamespace(
                model_type="deepseek_v41",
                vision_n_layers=12,
                language_only=False,
                language_model_only=False,
            )
            setattr(config, field, True)
            with self.subTest(field=field):
                self.assertFalse(
                    _should_build_dsv41_vision(
                        config,
                        SimpleNamespace(is_first_rank=True),
                    )
                )

    def test_all_vpp_stages_normalize_prompt_image_ids(self):
        config = SimpleNamespace(
            model_type="deepseek_v41",
            vision_n_layers=12,
            image_token_id=99,
            language_only=False,
            language_model_only=False,
        )
        input_ids = torch.tensor([7, MM_PAD_SHIFT_VALUE + 123, 8])

        normalized = _normalize_dsv41_prompt_input_ids(
            config,
            input_ids,
            ForwardMode.EXTEND,
        )

        torch.testing.assert_close(normalized, torch.tensor([7, 99, 8]))
        torch.testing.assert_close(
            input_ids, torch.tensor([7, MM_PAD_SHIFT_VALUE + 123, 8])
        )

    def test_decode_ids_are_not_remapped(self):
        config = SimpleNamespace(
            model_type="deepseek_v41",
            vision_n_layers=12,
            image_token_id=99,
            language_only=False,
            language_model_only=False,
        )
        input_ids = torch.tensor([MM_PAD_SHIFT_VALUE + 123])

        normalized = _normalize_dsv41_prompt_input_ids(
            config,
            input_ids,
            ForwardMode.DECODE,
        )

        self.assertIs(normalized, input_ids)

    def test_vpp_wraparound_does_not_repeat_vision(self):
        vision = object()
        mm_inputs = [object()]

        self.assertTrue(
            _should_prepare_dsv41_vision(
                vision,
                None,
                ForwardMode.EXTEND,
                mm_inputs,
            )
        )
        self.assertFalse(
            _should_prepare_dsv41_vision(
                vision,
                PPProxyTensors({"vpp_stage_id": 4}),
                ForwardMode.EXTEND,
                mm_inputs,
            )
        )

    def test_non_owner_keeps_vl_routing_weights(self):
        self.assertEqual(
            _dsv41_weight_skip_group(
                "vision.blocks.0.attn.qkv_proj.weight",
                owns_vision=False,
                multimodal_enabled=True,
            ),
            "vision",
        )
        self.assertIsNone(
            _dsv41_weight_skip_group(
                "model.layers.25.mlp.gate.e_score_correction_bias_vl",
                owns_vision=False,
                multimodal_enabled=True,
            )
        )

    def test_text_only_rank_skips_vl_routing_weights(self):
        self.assertEqual(
            _dsv41_weight_skip_group(
                "model.layers.25.mlp.gate.e_score_correction_bias_vl",
                owns_vision=False,
                multimodal_enabled=False,
            ),
            "gate.bias_vl",
        )


if __name__ == "__main__":
    unittest.main()
