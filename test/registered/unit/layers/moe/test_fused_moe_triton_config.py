import json
import sys
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config
from sglang.srt.runtime_context import get_context

BENCHMARK_DIR = Path(__file__).parents[5] / "benchmark" / "kernels" / "fused_moe_triton"
sys.path.insert(0, str(BENCHMARK_DIR))
import common_utils  # noqa: E402


def test_down_moe_reuses_tuned_up_config_when_separate_config_is_absent(
    monkeypatch, tmp_path
):
    config_root = tmp_path / "configs" / "triton_3_6_0"
    config_root.mkdir(parents=True)
    tuned_config = {"128": {"BLOCK_SIZE_M": 64}}
    (config_root / "up.json").write_text(json.dumps(tuned_config))

    monkeypatch.setenv("SGLANG_MOE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(fused_moe_triton_config.triton, "__version__", "3.6.0")
    monkeypatch.setattr(
        fused_moe_triton_config,
        "get_config_file_name",
        lambda *args, down_moe=False, **kwargs: "down.json" if down_moe else "up.json",
    )
    fused_moe_triton_config.get_moe_configs.cache_clear()

    try:
        # get_moe_configs reads get_exec().deterministic.
        with get_context().override_server_args(enable_deterministic_inference=False):
            assert fused_moe_triton_config.get_moe_configs(
                32, 768, None, down_moe=True
            ) == {128: {"BLOCK_SIZE_M": 64}}
    finally:
        fused_moe_triton_config.get_moe_configs.cache_clear()


def test_h20_triton_3_8_loads_paired_glm53_configs(monkeypatch):
    monkeypatch.delenv("SGLANG_MOE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(fused_moe_triton_config.triton, "__version__", "3.8.0")
    monkeypatch.setattr(
        fused_moe_triton_config, "get_device_name", lambda: "NVIDIA H20"
    )
    fused_moe_triton_config.get_moe_configs.cache_clear()

    expected_m = {
        1,
        2,
        4,
        8,
        16,
        24,
        32,
        48,
        64,
        96,
        128,
        256,
        512,
        1024,
        1536,
        2048,
        3072,
        4096,
    }
    try:
        with get_context().override_server_args(enable_deterministic_inference=False):
            up = fused_moe_triton_config.get_moe_configs(72, 2048, "fp8_w8a8", 128, 128)
            down = fused_moe_triton_config.get_moe_configs(
                72, 2048, "fp8_w8a8", 128, 128, down_moe=True
            )

        assert up is not None and down is not None
        assert set(up) == set(down) == expected_m
        assert all(up[m]["BLOCK_SIZE_M"] == down[m]["BLOCK_SIZE_M"] for m in expected_m)
        assert all(down[m]["USE_TMA"] is True for m in expected_m)
        assert all(up[m]["USE_TMA"] is True for m in expected_m if m <= 512)
        assert all(up[m]["USE_TMA"] is False for m in expected_m if m >= 1024)
    finally:
        fused_moe_triton_config.get_moe_configs.cache_clear()


def test_int4_tuner_filename_uses_runtime_down_projection_dimension(monkeypatch):
    monkeypatch.setattr(
        common_utils,
        "get_config_file_name",
        lambda E, N, *_args: f"E={E},N={N}.json",
    )

    filename = common_utils.get_config_filename(
        num_experts=256,
        shard_intermediate_size=512,
        hidden_size=1024,
        topk=8,
        dtype=None,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=True,
        per_channel_quant=False,
        block_shape=[128, 128],
    )

    assert filename == "E=256,N=256.json"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
