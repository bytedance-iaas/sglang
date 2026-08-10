import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.model_runner import (  # noqa: E402
    _assert_pp_mtp_compat,
    _compute_model_num_layers,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _model_config(*, num_nextn_predict_layers=1):
    return SimpleNamespace(
        num_nextn_predict_layers=num_nextn_predict_layers,
        num_hidden_layers=43,
        num_attention_layers=43,
        hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
    )


class TestModelRunnerLayerSetup(CustomTestCase):
    def test_multistage_dspark_uses_loaded_model_stage_count(self):
        self.assertEqual(
            _compute_model_num_layers(
                model=SimpleNamespace(num_stages=3),
                model_config=_model_config(),
                is_draft_worker=True,
            ),
            3,
        )

    def test_single_layer_mtp_falls_back_to_checkpoint_count(self):
        self.assertEqual(
            _compute_model_num_layers(
                model=SimpleNamespace(),
                model_config=_model_config(),
                is_draft_worker=True,
            ),
            1,
        )

    def test_target_worker_uses_target_layer_count(self):
        self.assertEqual(
            _compute_model_num_layers(
                model=SimpleNamespace(num_stages=3),
                model_config=_model_config(),
                is_draft_worker=False,
            ),
            43,
        )

    def test_partial_mtp_pp_remains_rejected(self):
        spec_algorithm = SimpleNamespace(is_none=lambda: False)
        _assert_pp_mtp_compat(
            model_has_mtp_layers=True,
            spec_algorithm=spec_algorithm,
            num_effective_layers=3,
            model_num_layers=3,
        )
        with self.assertRaisesRegex(AssertionError, "PP is not compatible"):
            _assert_pp_mtp_compat(
                model_has_mtp_layers=True,
                spec_algorithm=spec_algorithm,
                num_effective_layers=2,
                model_num_layers=3,
            )


if __name__ == "__main__":
    unittest.main()
