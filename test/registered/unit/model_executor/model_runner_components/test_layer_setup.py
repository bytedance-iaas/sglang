"""Unit tests for model-runner layer discovery."""

import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.model_runner_components.layer_setup import (
    compute_attention_and_moe_layers,
    resolve_layer_indices,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestComputeAttentionAndMoeLayers(unittest.TestCase):
    def test_deepseek_mla_registers_mha_companion(self):
        attn_mqa = SimpleNamespace()
        attn_mha = SimpleNamespace()
        layer_model = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(attn_mqa=attn_mqa, attn_mha=attn_mha)
                )
            ]
        )

        attention_layers, _, _, _, mha_companion_layers = (
            compute_attention_and_moe_layers(layer_model)
        )

        self.assertEqual(attention_layers, [attn_mqa])
        self.assertEqual(mha_companion_layers, [attn_mha])
        self.assertNotIn("_pcg_mha_companion", vars(attn_mqa))


class TestResolveLayerIndices(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(start_layer=0, end_layer=6)
        self.model_config = SimpleNamespace(
            num_nextn_predict_layers=1,
            num_hidden_layers=43,
            num_attention_layers=43,
            hf_config=SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"],
                loop_num=1,
            ),
        )

    def test_pp_mtp_target_allows_dspark(self):
        layer_info = resolve_layer_indices(
            model=self.model,
            model_config=self.model_config,
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
        )

        self.assertEqual(layer_info.start_layer, 0)
        self.assertEqual(layer_info.end_layer, 6)
        self.assertEqual(layer_info.num_effective_layers, 6)

    def test_pp_mtp_target_still_rejects_eagle(self):
        with self.assertRaisesRegex(
            AssertionError, "PP is not compatible with MTP models"
        ):
            resolve_layer_indices(
                model=self.model,
                model_config=self.model_config,
                is_draft_worker=False,
                spec_algorithm=SpeculativeAlgorithm.EAGLE,
            )


if __name__ == "__main__":
    unittest.main()
