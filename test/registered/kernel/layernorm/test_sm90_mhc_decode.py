"""Compensated SM90 mHC accuracy, batch consistency and post-mix precision."""

import pytest
import torch

import sglang.kernels.ops.layernorm.mhc as mhc
from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="Requires SM90",
)


@pytest.mark.parametrize("amplitude", [1e-4, 1.0, 100.0])
def test_compensated_coefficients(amplitude):
    torch.manual_seed(11)
    x = torch.randn(108, 20480, device="cuda", dtype=torch.bfloat16) * amplitude
    weight = torch.randn(24, 20480, device="cuda") * 0.01
    scale = torch.tensor([0.5, 0.25, 0.25], device="cuda")
    base = torch.randn(24, device="cuda")
    parts = mhc.split_bf16_hc_weight(weight)
    actual = mhc.hc_mix_stats_sinkhorn_sm90_bf16x3(
        x, parts, scale, base, 20, 1e-20, 1e-6
    )
    alone = mhc.hc_mix_stats_sinkhorn_sm90_bf16x3(
        x[:1], parts, scale, base, 20, 1e-20, 1e-6
    )
    gold = (x.double() @ weight.double().T) * torch.rsqrt(
        x.double().square().mean(-1, keepdim=True) + 1e-20
    )
    expected = mhc.hc_split_sinkhorn(gold.float()[:, None, :], scale, base, 4, 20, 1e-6)
    for got, ref, single in zip(actual, expected, alone):
        torch.testing.assert_close(got, ref.squeeze(1), atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(single, got[:1], atol=0, rtol=0)


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
