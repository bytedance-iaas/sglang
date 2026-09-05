"""Regression tests for NextN embedding ownership under pipeline parallelism."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _FakeParam:
    def __init__(self):
        self.loaded = None

    def weight_loader(self, param, loaded_weight, *args, **kwargs):
        self.loaded = (param, loaded_weight, args, kwargs)


class _NextNLoader(DeepseekV2WeightLoaderMixin):
    def __init__(self):
        self.config = SimpleNamespace(
            num_hidden_layers=78,
            num_nextn_predict_layers=1,
            n_routed_experts=1,
        )
        self.quant_config = None
        # PP EAGLE constructs the one-layer draft under a temporary singleton
        # group even when its target spans multiple pipeline stages.
        self.pp_group = SimpleNamespace(
            world_size=1, is_first_rank=True, is_last_rank=True
        )
        self.num_fused_shared_experts = 0
        self.model = SimpleNamespace()
        self.embed = _FakeParam()

    def named_parameters(self):
        return iter((("model.embed_tokens.weight", self.embed),))

    def post_load_weights(self, is_nextn=False, weight_names=None):
        pass


class TestDeepseekNextNEmbeddingLoading(unittest.TestCase):
    def test_pp_target_loads_checkpoint_embedding_into_singleton_draft(self):
        model = _NextNLoader()
        weight = torch.arange(4)

        with patch(
            "sglang.srt.models.deepseek_common.deepseek_weight_loader.get_parallel",
            return_value=SimpleNamespace(pp_size=2),
        ):
            model.do_load_weights(
                [("model.embed_tokens.weight", weight)], is_nextn=True
            )

        self.assertEqual(model.embed.loaded, (model.embed, weight, (), {}))

    def test_non_pp_target_keeps_shared_embedding_path(self):
        model = _NextNLoader()

        with patch(
            "sglang.srt.models.deepseek_common.deepseek_weight_loader.get_parallel",
            return_value=SimpleNamespace(pp_size=1),
        ):
            model.do_load_weights(
                [("model.embed_tokens.weight", torch.arange(4))], is_nextn=True
            )

        self.assertIsNone(model.embed.loaded)


if __name__ == "__main__":
    unittest.main()
