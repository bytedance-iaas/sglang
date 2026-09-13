import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.deepseek_v4_hook import validate_deepseek_v41_features
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _config(**overrides):
    values = {
        "pp_virtual_stages": 2,
        "pp_size": 4,
        "tp_size": 2,
        "attn_cp_size": 1,
        "dp_size": 1,
        "dcp_size": 1,
        "enable_prefill_cp": False,
        "speculative_algorithm": None,
        "disaggregation_mode": "prefill",
        "disaggregation_transfer_backend": "mooncake",
        "language_only": False,
        "language_model_only": True,
        "enable_hierarchical_cache": False,
        "pp_async_batch_depth": 0,
        "enable_encoder_swa_bounded_replay": False,
        "enable_decoder_swa_bounded_replay": False,
        "cuda_graph_config": SimpleNamespace(
            prefill=SimpleNamespace(backend=Backend.DISABLED)
        ),
        "enable_mixed_chunk": False,
        "enable_two_batch_overlap": False,
        "enable_hisparse": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _validate(config):
    with (
        patch(
            "sglang.srt.arg_groups.deepseek_v4_hook.resolving_view",
            return_value=config,
        ),
        patch(
            "sglang.srt.arg_groups.deepseek_v4_hook.model_config_of",
            return_value=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="deepseek_v41")
            ),
        ),
        patch(
            "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate."
            "is_unified_kv_triton",
            return_value=False,
        ),
    ):
        validate_deepseek_v41_features(SimpleNamespace())


class TestDeepSeekV41VPPGate(unittest.TestCase):
    def test_pp4_tp2_vpp2_without_cp_is_supported(self):
        _validate(_config())

    def test_vpp2_rejects_prefill_cp(self):
        for config in (
            _config(enable_prefill_cp=True),
            _config(attn_cp_size=2),
        ):
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, "context parallelism"):
                    _validate(config)


if __name__ == "__main__":
    unittest.main()
