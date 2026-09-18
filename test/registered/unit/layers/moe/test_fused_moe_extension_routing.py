"""Model shortcuts must delegate to the same full-MoE provider as other models."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.mega_moe import _run_mega_routed
from sglang.srt.models.kimi_k3 import KimiK3MoE

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestFusedMoEExtensionRouting(unittest.TestCase):
    def test_common_layer_returns_already_combined_output(self):
        expected = object()
        backend = SimpleNamespace(forward=Mock(return_value=expected))
        layer = SimpleNamespace(fused_moe_backend=backend, _dwdp_bound=False)
        x, topk = object(), object()
        self.assertIs(FusedMoE.forward_impl(layer, x, topk), expected)
        backend.forward.assert_called_once_with(layer, x, topk)

    def test_deepseek_empty_rank_still_calls_provider(self):
        expected = object()
        backend = SimpleNamespace(forward=Mock(return_value=expected))
        experts = SimpleNamespace(fused_moe_backend=backend, top_k=4)
        moe = SimpleNamespace(experts=experts, config=SimpleNamespace(hidden_size=512))
        x = torch.empty((0, 512), dtype=torch.bfloat16)
        self.assertIs(_run_mega_routed(moe, x, None, None, 0), expected)
        layer, actual_x, topk = backend.forward.call_args.args
        self.assertIs(layer, experts)
        self.assertIs(actual_x, x)
        self.assertEqual(topk.topk_ids.shape, (0, 4))
        self.assertEqual(topk.topk_weights.dtype, torch.float32)

    def test_kimi_uses_common_experts_entry(self):
        experts = Mock()
        experts.fused_moe_backend = object()
        moe = SimpleNamespace(experts=experts)
        x, topk = object(), object()
        self.assertIs(
            KimiK3MoE._forward_mega_experts(moe, x, topk), experts.return_value
        )
        experts.assert_called_once_with(x, topk)


if __name__ == "__main__":
    unittest.main()
