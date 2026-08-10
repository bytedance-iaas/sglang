import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner import deep_gemm as deep_gemm_runner
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    _get_compact_all_tokens,
    _should_use_masked_standard_layout,
)


@pytest.mark.parametrize(
    ("num_assignments", "num_experts", "expected"),
    [
        (1, 256, 128),
        (8, 256, 1024),
        (256, 256, 32768),
        (1024, 256, 33536),
    ],
)
def test_compact_all_tokens_is_graph_static_and_tightly_bounded(
    num_assignments, num_experts, expected
):
    assert _get_compact_all_tokens(num_assignments, num_experts) == expected


def _runner_config(num_experts=256, num_local_experts=256):
    return MoeRunnerConfig(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=4096,
        intermediate_size_per_partition=2048,
        top_k=8,
        activation="silu",
        is_gated=True,
    )


def _quant_info():
    return DeepGemmMoeQuantInfo(
        w13_weight=torch.empty((1, 512, 1), dtype=torch.float8_e4m3fn),
        w2_weight=torch.empty((1, 4096, 1), dtype=torch.float8_e4m3fn),
        use_fp8=True,
        block_shape=[128, 128],
    )


def test_standard_dispatch_explicit_layout_override():
    config = _runner_config()
    quant_info = _quant_info()
    hidden_states = torch.empty((8, 4096), device="meta")

    with envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.override("masked"):
        assert _should_use_masked_standard_layout(config, quant_info, hidden_states)
    with envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.override("compact"):
        assert not _should_use_masked_standard_layout(
            config, quant_info, hidden_states
        )


def test_standard_dispatch_auto_layout_uses_memory_budget(monkeypatch):
    config = _runner_config(num_experts=512, num_local_experts=512)
    quant_info = _quant_info()
    monkeypatch.setattr(
        deep_gemm_runner,
        "_masked_standard_layout_memory_budget_bytes",
        int(42.5 * (1 << 30)),
    )

    with envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.override("auto"):
        for num_tokens, expected in ((8192, True), (16384, False)):
            hidden_states = torch.empty((num_tokens, 4096), device="meta")
            assert (
                _should_use_masked_standard_layout(
                    config, quant_info, hidden_states
                )
                is expected
            )


def test_standard_dispatch_rejects_invalid_layout():
    config = _runner_config()
    quant_info = _quant_info()
    hidden_states = torch.empty((8, 4096), device="meta")

    with envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.override("invalid"):
        with pytest.raises(ValueError, match="auto, masked, compact"):
            _should_use_masked_standard_layout(config, quant_info, hidden_states)


def test_masked_layout_budget_fraction_is_validated(monkeypatch):
    monkeypatch.setattr(
        deep_gemm_runner, "_masked_standard_layout_memory_budget_bytes", None
    )
    with envs.SGLANG_DEEPGEMM_MASKED_MEMORY_BUDGET_FRACTION.override(0.5):
        assert deep_gemm_runner.set_masked_standard_layout_memory_budget(100) == 50

    with envs.SGLANG_DEEPGEMM_MASKED_MEMORY_BUDGET_FRACTION.override(0.0):
        with pytest.raises(ValueError, match="must be in"):
            deep_gemm_runner.set_masked_standard_layout_memory_budget(100)


def test_standard_dispatch_auto_capture_without_budget_uses_compact(monkeypatch):
    common = dict(
        config=_runner_config(num_experts=256, num_local_experts=32),
        quant_info=_quant_info(),
        hidden_states=torch.empty((8, 4096), device="meta"),
    )
    monkeypatch.setattr(
        deep_gemm_runner, "_masked_standard_layout_memory_budget_bytes", None
    )
    monkeypatch.setattr(
        "sglang.srt.model_executor.cuda_graph_runner.get_is_capture_mode",
        lambda: True,
    )
    with envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.override("auto"):
        assert not _should_use_masked_standard_layout(**common)


def test_plain_silu_uses_compatible_quantizer_when_jit_is_enabled(monkeypatch):
    calls = []

    monkeypatch.setattr(
        deep_gemm_runner,
        "silu_and_mul_masked_post_quant",
        lambda *args, **kwargs: pytest.fail(
            "plain SiLU must not use the geometry-restricted DSV4 JIT kernel"
        ),
    )
    monkeypatch.setattr(
        "sglang.srt.layers.moe.ep_moe.kernels."
        "silu_and_mul_masked_post_quant_fwd",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    # D=256 and group_size=128 gives G=2, the TP8 DSV4 geometry that cannot
    # satisfy the specialized JIT kernel's G % 4 requirement.
    gateup = torch.empty((2, 4, 512), dtype=torch.bfloat16)
    masked_m = torch.tensor([1, 1], dtype=torch.int32)
    with envs.SGLANG_OPT_USE_JIT_EP_ACTIVATION.override(True):
        output, output_scale = deep_gemm_runner._varlen_deep_gemm_silu_mul_quant(
            gateup,
            masked_m,
            group_size=128,
            topk=8,
            swiglu_limit=None,
            swizzle=False,
        )

    assert len(calls) == 1
    assert output.shape == (2, 4, 256)
    assert output_scale.shape == (2, 4, 2)
