import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
    FlashInferCutlassMxfp4MoeQuantInfo,
    fused_experts_deepep_to_flashinfer_mxfp4,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
    DeepEPNormalCombineInput,
    DeepEPNormalDispatchOutput,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _quant_info(*, ep_rank: int = 1):
    return FlashInferCutlassMxfp4MoeQuantInfo(
        w13_weight=torch.empty(2, 1),
        w2_weight=torch.empty(2, 1),
        w13_weight_scale=torch.empty(2, 1),
        w2_weight_scale=torch.empty(2, 1),
        w13_humming_residual_scale=torch.ones(2),
        w2_humming_residual_scale=torch.ones(2),
        humming_fc2_act_scale=torch.ones(()),
        moe_ep_size=4,
        moe_ep_rank=ep_rank,
    )


class TestFlashInferMxfp4DeepEPAdapter(CustomTestCase):
    @patch(
        "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass."
        "_fused_experts_flashinfer_mxfp4_cutlass"
    )
    def test_low_latency_maps_masked_rows_to_local_experts(self, run_cutlass):
        hidden_states = torch.randn(2, 3, 4)
        topk_ids = torch.tensor([[2, 5], [3, 6]], dtype=torch.int64)
        topk_weights = torch.rand(2, 2)
        dispatch_output = DeepEPLLDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            masked_m=torch.tensor([2, 1], dtype=torch.int32),
            expected_m=3,
        )
        run_cutlass.side_effect = lambda output, *_: StandardCombineInput(
            output.hidden_states + 1
        )

        result = fused_experts_deepep_to_flashinfer_mxfp4(
            dispatch_output,
            _quant_info(),
            MoeRunnerConfig(),
        )

        self.assertIsInstance(result, DeepEPLLCombineInput)
        self.assertEqual(result.hidden_states.shape, hidden_states.shape)
        self.assertIs(result.topk_ids, topk_ids)
        self.assertIs(result.topk_weights, topk_weights)
        standard_output = run_cutlass.call_args.args[0]
        self.assertEqual(
            standard_output.topk_output.topk_ids.tolist(),
            [[2], [2], [2], [3], [3], [3]],
        )
        self.assertTrue(
            torch.equal(
                standard_output.topk_output.topk_weights,
                torch.ones(6, 1),
            )
        )

    @patch(
        "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass."
        "_fused_experts_flashinfer_mxfp4_cutlass"
    )
    def test_normal_preserves_deepep_routing(self, run_cutlass):
        hidden_states = torch.randn(3, 4)
        topk_ids = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int64)
        topk_weights = torch.rand(3, 2)
        dispatch_output = DeepEPNormalDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            num_recv_tokens_per_expert=[1, 2],
        )
        run_cutlass.side_effect = lambda output, *_: StandardCombineInput(
            output.hidden_states + 1
        )

        result = fused_experts_deepep_to_flashinfer_mxfp4(
            dispatch_output,
            _quant_info(),
            MoeRunnerConfig(),
        )

        self.assertIsInstance(result, DeepEPNormalCombineInput)
        standard_output = run_cutlass.call_args.args[0]
        self.assertIs(standard_output.topk_output.topk_ids, topk_ids)
        self.assertIs(standard_output.topk_output.topk_weights, topk_weights)

    def test_requires_humming_scales_and_bf16_dispatch(self):
        dispatch_output = DeepEPLLDispatchOutput(
            hidden_states=torch.randn(2, 3, 4),
            hidden_states_scale=None,
            topk_ids=torch.zeros(2, 1, dtype=torch.int64),
            topk_weights=torch.ones(2, 1),
            masked_m=torch.ones(2, dtype=torch.int32),
            expected_m=3,
        )
        quant_info = _quant_info()
        quant_info.humming_fc2_act_scale = None
        with self.assertRaisesRegex(ValueError, "Humming MXFP4xFP8"):
            fused_experts_deepep_to_flashinfer_mxfp4(
                dispatch_output,
                quant_info,
                MoeRunnerConfig(),
            )

        quant_info.humming_fc2_act_scale = torch.ones(())
        with self.assertRaisesRegex(ValueError, "BF16 dispatch"):
            fused_experts_deepep_to_flashinfer_mxfp4(
                dispatch_output._replace(hidden_states_scale=torch.ones(2, 3, 1)),
                quant_info,
                MoeRunnerConfig(),
            )


if __name__ == "__main__":
    unittest.main()
