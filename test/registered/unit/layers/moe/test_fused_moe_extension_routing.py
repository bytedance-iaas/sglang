"""Model shortcuts must delegate to the same full-MoE provider as other models."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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

    def test_deepseek_v41_preserves_vision_bias_scaling_and_padding(self):
        from sglang.srt.layers.moe import topk as topk_module
        from sglang.srt.multimodal.dsv41 import vl_routing

        logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]]).repeat(3, 1)
        gate = Mock(return_value=logits)
        gate.e_score_correction_bias = torch.tensor([2.0, 1.0, 0.0, 0.0])
        gate.e_score_correction_bias_vl = torch.tensor([0.0, 0.0, 1.0, 2.0])
        config = SimpleNamespace(
            num_fused_shared_experts=0,
            top_k=2,
            renormalize=True,
            routed_scaling_factor=1.5,
            apply_routed_scaling_factor_on_output=True,
            fused_shared_experts_scaling_factor=None,
        )
        backend = SimpleNamespace(forward=Mock(return_value=object()))
        experts = SimpleNamespace(fused_moe_backend=backend, top_k=2)
        moe = SimpleNamespace(
            gate=gate,
            topk=SimpleNamespace(topk_config=config),
            experts=experts,
            config=SimpleNamespace(hidden_size=512, image_token_id=99),
        )
        x = torch.empty((3, 512), dtype=torch.bfloat16)
        ids = torch.tensor([10, 99, 99])
        batch = SimpleNamespace(num_token_non_padded=torch.tensor(2))
        with (
            patch.object(vl_routing, "is_cuda", return_value=False),
            patch.object(topk_module, "_is_cuda", False),
            patch.object(topk_module, "_can_fuse_padded_region", return_value=False),
        ):
            _run_mega_routed(moe, x, batch, ids, 3)

        selected = backend.forward.call_args.args[2]
        self.assertEqual(selected.topk_ids.tolist(), [[0, 1], [3, 2], [-1, -1]])
        torch.testing.assert_close(
            selected.topk_weights[:2].sum(-1), torch.full((2,), 1.5)
        )
        self.assertEqual(selected.topk_weights[2].tolist(), [0.0, 0.0])
        gate.assert_called_once_with(x, forward_batch=batch)

    def test_non_vision_hash_routing_keeps_token_ids(self):
        x = torch.empty((2, 512), dtype=torch.bfloat16)
        ids = torch.tensor([10, 11])
        logits = torch.empty((2, 4))
        gate = Mock(return_value=logits)
        gate.e_score_correction_bias_vl = None
        topk_output = SimpleNamespace(
            topk_ids=torch.tensor([[0, 1], [2, 3]]),
            topk_weights=torch.full((2, 2), 0.5),
        )
        backend = SimpleNamespace(forward=Mock(return_value=object()))
        moe = SimpleNamespace(
            gate=gate,
            topk=Mock(return_value=topk_output),
            experts=SimpleNamespace(fused_moe_backend=backend),
            config=SimpleNamespace(hidden_size=512),
            is_hash=True,
            layer_id=0,
        )
        with patch(
            "sglang.srt.layers.moe.mega_moe.ExpertLocationDispatchInfo.init_new",
            return_value=None,
        ):
            _run_mega_routed(moe, x, None, ids, 2)
        self.assertIs(moe.topk.call_args.kwargs["input_ids"], ids)
        self.assertIs(backend.forward.call_args.args[2], topk_output)

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
