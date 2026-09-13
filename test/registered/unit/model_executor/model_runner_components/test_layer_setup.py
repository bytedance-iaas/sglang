"""Unit tests for model-runner layer discovery."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.model_runner_components.layer_setup import (
    compute_attention_and_moe_layers,
    resolve_layer_indices,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


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

    def test_pipeline_placeholders_preserve_global_layer_ids(self):
        local_attention = SimpleNamespace()
        layer_model = SimpleNamespace(
            layers=[SimpleNamespace(), SimpleNamespace()]
            + [SimpleNamespace(self_attn=SimpleNamespace(attn=local_attention))]
        )

        attention_layers, _, _, _, mha_companion_layers = (
            compute_attention_and_moe_layers(layer_model)
        )

        self.assertEqual(attention_layers, [None, None, local_attention])
        self.assertEqual(mha_companion_layers, [None, None, None])

    def test_resolve_layer_indices_preserves_non_contiguous_ownership(self):
        model = SimpleNamespace(
            start_layer=0,
            end_layer=25,
            layer_ids=(0, 1, 2, 3, 4, 20, 21, 22, 23, 24),
        )
        model_config = SimpleNamespace(
            num_hidden_layers=40,
            num_attention_layers=40,
            num_nextn_predict_layers=None,
            hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
        )
        spec_algorithm = SimpleNamespace(is_none=lambda: True)

        with patch(
            "sglang.srt.model_executor.model_runner_components.layer_setup.get_parallel",
            return_value=SimpleNamespace(pp_virtual_stages=2),
        ):
            info = resolve_layer_indices(
                model=model,
                model_config=model_config,
                is_draft_worker=False,
                spec_algorithm=spec_algorithm,
            )

        self.assertEqual(info.start_layer, 0)
        self.assertEqual(info.end_layer, 25)
        self.assertEqual(info.num_effective_layers, 10)
        self.assertEqual(info.layer_ids, model.layer_ids)


if __name__ == "__main__":
    unittest.main()
