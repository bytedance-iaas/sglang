import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models.deepseek_v4 import (  # noqa: E402
    DeepseekV4ForCausalLM,
    resolve_deepseek_v4_num_fused_shared_experts,
)
from sglang.srt.models.deepseek_v4_dspark import (  # noqa: E402
    DeepseekV4ForCausalLMDSpark,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _RecordingParameter(torch.nn.Parameter):
    def __new__(cls):
        parameter = super().__new__(cls, torch.empty(1), requires_grad=False)
        parameter.loaded = []

        def weight_loader(param, loaded_weight, candidate, *, shard_id, expert_id):
            del loaded_weight
            param.loaded.append((candidate, shard_id, expert_id))

        parameter.weight_loader = weight_loader
        return parameter


class TestDeepseekV4DsparkSharedExpertFusion(CustomTestCase):
    def _resolve(self, *, disable=False, enforce=False, n_shared_experts=1):
        server_args = SimpleNamespace(
            disable_shared_experts_fusion=disable,
            enforce_shared_experts_fusion=enforce,
        )
        config = SimpleNamespace(n_shared_experts=n_shared_experts)
        with patch(
            "sglang.srt.models.deepseek_v4.get_global_server_args",
            return_value=server_args,
        ):
            result = resolve_deepseek_v4_num_fused_shared_experts(
                DeepseekV4ForCausalLMDSpark, config, quant_config=None
            )
        return result, server_args

    def test_default_layout_is_separate_without_mutating_server_args(self):
        result, server_args = self._resolve()

        self.assertEqual(result, 0)
        self.assertFalse(server_args.disable_shared_experts_fusion)

    def test_explicit_fusion_is_resolved_per_model_construction(self):
        result, server_args = self._resolve(enforce=True)

        self.assertEqual(result, 1)
        self.assertFalse(server_args.disable_shared_experts_fusion)

    def test_explicit_disable_takes_precedence(self):
        result, _ = self._resolve(disable=True, enforce=True)

        self.assertEqual(result, 0)

    def test_invalid_fused_shared_expert_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one shared expert"):
            self._resolve(enforce=True, n_shared_experts=2)

    def test_dspark_delegates_the_target_fusion_gate(self):
        config = SimpleNamespace(n_shared_experts=1)
        server_args = SimpleNamespace(enforce_shared_experts_fusion=True)
        with patch(
            "sglang.srt.models.deepseek_v4.get_global_server_args",
            return_value=server_args,
        ):
            target_reason = DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                config, None
            )
            draft_reason = (
                DeepseekV4ForCausalLMDSpark.shared_experts_fusion_disable_reason(
                    config, None
                )
            )

        self.assertEqual(draft_reason, target_reason)

    def test_dspark_shared_weights_load_into_the_fused_expert_slot(self):
        model = DeepseekV4ForCausalLMDSpark.__new__(
            DeepseekV4ForCausalLMDSpark
        )
        torch.nn.Module.__init__(model)
        model.is_lifecycle_only = False
        model.num_stages = 1
        model.num_fused_shared_experts = 1
        model.config = SimpleNamespace(n_routed_experts=256)
        model.confidence_head = None
        model.register_parameter(
            "stages_0_mlp_experts_w13_weight", _RecordingParameter()
        )
        model.register_parameter(
            "stages_0_mlp_experts_w2_weight", _RecordingParameter()
        )
        params = {
            "stages.0.mlp.experts.w13_weight": model.stages_0_mlp_experts_w13_weight,
            "stages.0.mlp.experts.w2_weight": model.stages_0_mlp_experts_w2_weight,
        }
        model.named_parameters = lambda *args, **kwargs: iter(params.items())

        model.load_weights(
            iter(
                (
                    (f"mtp.0.ffn.shared_experts.{projection}.weight", torch.empty(1))
                    for projection in ("gate_proj", "down_proj", "up_proj")
                )
            )
        )

        loaded = (
            model.stages_0_mlp_experts_w13_weight.loaded
            + model.stages_0_mlp_experts_w2_weight.loaded
        )
        self.assertEqual(
            {(shard_id, expert_id) for _, shard_id, expert_id in loaded},
            {("w1", 256), ("w2", 256), ("w3", 256)},
        )


if __name__ == "__main__":
    unittest.main()
