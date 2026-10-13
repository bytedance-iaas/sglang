"""SM90 mHC preserves the original projection and fuses only its reduction."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import sglang.kernels.ops.layernorm.mhc as mhc
from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="Requires SM90",
)


def _inputs(rows, amplitude=1.0):
    torch.manual_seed(11)
    x = torch.randn(rows, 20480, device="cuda", dtype=torch.bfloat16) * amplitude
    weight = torch.randn(24, 20480, device="cuda") * 0.01
    scale = torch.tensor([0.5, 0.25, 0.25], device="cuda")
    base = torch.randn(24, device="cuda")
    return x, weight, scale, base


def _original_coefficients(x, weight, scale, base):
    mixes = mhc.hc_mix_stats(x, weight, 1e-20)
    return tuple(
        value.squeeze(1)
        for value in mhc.hc_split_sinkhorn(mixes[:, None], scale, base, 4, 20, 1e-6)
    )


@pytest.mark.parametrize("rows", [1, 18, 108, 192, 2048])
def test_projection_partials_are_unchanged(monkeypatch, rows):
    x, weight, scale, base = _inputs(rows)
    original_kernel = mhc._hc_mix_stats_partial_kernel
    launches = []

    class CaptureProjection:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                original_kernel[grid](*args, **kwargs)
                launches.append((grid, kwargs, args[2].clone(), args[3].clone()))

            return launch

    monkeypatch.setattr(mhc, "_hc_mix_stats_partial_kernel", CaptureProjection())
    expected = _original_coefficients(x, weight, scale, base)
    actual = mhc.hc_mix_stats_sinkhorn(
        x, weight, scale, base, 4, 20, 1e-20, 1e-6, tree_reduce=True
    )
    assert len(launches) == 2
    assert launches[0][:2] == launches[1][:2]
    for original, optimized in zip(launches[0][2:], launches[1][2:]):
        torch.testing.assert_close(original, optimized, atol=0, rtol=0)
    for got, ref in zip(actual, expected):
        torch.testing.assert_close(got, ref, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("amplitude", [1e-4, 1.0, 100.0])
def test_fused_coefficients_and_batch_consistency(amplitude):
    x, weight, scale, base = _inputs(108, amplitude)
    actual = mhc.hc_mix_stats_sinkhorn(
        x, weight, scale, base, 4, 20, 1e-20, 1e-6, tree_reduce=True
    )
    alone = mhc.hc_mix_stats_sinkhorn(
        x[:1], weight, scale, base, 4, 20, 1e-20, 1e-6, tree_reduce=True
    )
    expected = _original_coefficients(x, weight, scale, base)
    gold = (x.double() @ weight.double().T) * torch.rsqrt(
        x.double().square().mean(-1, keepdim=True) + 1e-20
    )
    gold_coefficients = mhc.hc_split_sinkhorn(
        gold.float()[:, None], scale, base, 4, 20, 1e-6
    )
    for got, ref, single, gold_coefficient in zip(
        actual, expected, alone, gold_coefficients
    ):
        torch.testing.assert_close(got, ref, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(
            got, gold_coefficient.squeeze(1), atol=2e-6, rtol=2e-6
        )
        torch.testing.assert_close(single, got[:1], atol=0, rtol=0)


def test_empty_coefficients():
    x, weight, scale, base = _inputs(0)
    actual = mhc.hc_mix_stats_sinkhorn(
        x, weight, scale, base, 4, 20, 1e-20, 1e-6, tree_reduce=True
    )
    assert [value.shape for value in actual] == [(0, 4), (0, 4), (0, 4, 4)]


@pytest.mark.parametrize(
    "enabled,invariant,model_type,rows,expected_tree",
    [
        (True, False, "deepseek_v41", 108, True),
        (False, False, "deepseek_v41", 108, False),
        (True, True, "deepseek_v41", 108, False),
        (True, False, "deepseek_v4", 108, False),
        (True, False, "deepseek_v41", 2049, False),
    ],
)
def test_model_dispatch(enabled, invariant, model_type, rows, expected_tree):
    import sglang.srt.models.deepseek_v4 as model

    x, weight, scale, base = _inputs(rows)
    layer = SimpleNamespace(
        hc_mult=4,
        hc_sinkhorn_iters=20,
        rms_norm_eps=1e-20,
        hc_eps=1e-6,
        config=SimpleNamespace(model_type=model_type),
    )
    expected = _original_coefficients(x, weight, scale, base)
    with (
        envs.SGLANG_OPT_DSV41_SM90_MHC.override(enabled),
        patch(
            "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
            return_value=invariant,
        ),
        patch.object(model, "_get_mhc_ops", return_value=mhc),
        patch.object(
            mhc, "hc_mix_stats_sinkhorn", wraps=mhc.hc_mix_stats_sinkhorn
        ) as fused,
    ):
        actual = model.DeepseekV4DecoderLayer._hc_mix_stats(
            layer, x.reshape(rows, 4, 5120), weight, scale, base
        )
    assert fused.call_count == int(expected_tree)
    if expected_tree:
        assert fused.call_args.kwargs["tree_reduce"] is True
    for got, ref in zip(actual, expected):
        torch.testing.assert_close(got, ref, atol=2e-6, rtol=2e-6)


def test_sm90_does_not_create_compensated_weights():
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer

    _, weight, _, _ = _inputs(1)
    layer = SimpleNamespace(
        input_layernorm=SimpleNamespace(weight=torch.ones(5120, device="cuda")),
        post_attention_layernorm=SimpleNamespace(
            weight=torch.ones(5120, device="cuda")
        ),
        hc_pre_from_prev_sublayer=True,
        hc_attn_fn=weight,
        hc_ffn_fn=weight.clone(),
        config=SimpleNamespace(model_type="deepseek_v41"),
    )
    with (
        envs.SGLANG_OPT_DSV41_SM90_MHC.override(True),
        patch.object(
            mhc, "split_bf16_hc_weight", side_effect=AssertionError("BF16 split")
        ),
    ):
        DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache(layer)
    assert layer._hc_attn_bf16_parts is None
    assert layer._hc_ffn_bf16_parts is None
    assert layer._hc_attn_tf32_parts is None
    assert layer._hc_ffn_tf32_parts is None


@pytest.mark.parametrize("rows", [1, 18, 108, 192])
def test_post_mix_is_bitwise_exact(monkeypatch, rows):
    monkeypatch.setattr(mhc, "is_dsa_prefill_cp_interleave", lambda: False)
    torch.manual_seed(15)
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(rows, 4, 5120, device="cuda", dtype=torch.bfloat16)
    post = torch.rand(rows, 4, device="cuda")
    comb = torch.rand(rows, 4, 4, device="cuda")
    expected = mhc.mhc_post(x, residual, post, comb)
    actual = mhc_post_split_h(x, residual, post, comb)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
