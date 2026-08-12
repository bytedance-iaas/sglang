"""Regression tests for cached Hugging Face config isolation."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.configs.model_config import ModelConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestModelConfigCacheIsolation(CustomTestCase):
    def test_dspark_draft_remap_does_not_mutate_cached_target_config(self):
        cached_config = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"],
            model_type="deepseek_v4",
        )

        method_patches = {
            "_validate_quantize_and_serve_config": None,
            "_maybe_pull_model_for_runai": None,
            "_maybe_pull_model_tokenizer_from_remote": None,
            "_get_sliding_window_size": None,
            "_derive_context_length": None,
            "_derive_model_shapes": None,
            "_derive_hybrid_model": None,
            "_verify_quantization": None,
            "_verify_transformers_version": None,
            "_verify_dual_chunk_attention_config": None,
            "_get_hf_eos_token_id": None,
        }

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "sglang.srt.configs.model_config.get_config",
                    return_value=cached_config,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.configs.model_config.get_hf_text_config",
                    side_effect=lambda config: config,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.configs.model_config.get_generation_config",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.configs.model_config.is_deepseek_v4",
                    return_value=False,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.configs.model_config.is_generation_model",
                    return_value=True,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.configs.model_config.is_encoder_decoder_model",
                    return_value=False,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.configs.model_config.is_local_attention_model",
                    return_value=False,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.configs.model_config.is_piecewise_cuda_graph_disabled_model",
                    return_value=False,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.configs.model_config._get_and_verify_dtype",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.speculative.dspark_components.dspark_config.checkpoint_bundles_dspark_draft",
                    return_value=True,
                )
            )
            for method_name, return_value in method_patches.items():
                stack.enter_context(
                    patch.object(ModelConfig, method_name, return_value=return_value)
                )

            target = ModelConfig("cached-dsv4", enable_multimodal=False)
            draft = ModelConfig(
                "cached-dsv4",
                enable_multimodal=False,
                is_draft_model=True,
                speculative_algorithm="DSPARK",
            )

        self.assertIsNot(target.hf_config, cached_config)
        self.assertIsNot(draft.hf_config, cached_config)
        self.assertIsNot(target.hf_config, draft.hf_config)
        self.assertEqual(cached_config.architectures, ["DeepseekV4ForCausalLM"])
        self.assertEqual(target.hf_config.architectures, ["DeepseekV4ForCausalLM"])
        self.assertEqual(
            draft.hf_config.architectures,
            ["DeepseekV4ForCausalLMDSpark"],
        )


if __name__ == "__main__":
    unittest.main()
