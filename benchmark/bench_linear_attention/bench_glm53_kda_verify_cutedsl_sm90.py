"""Complete GLM-5.3 KDA verify+output-norm comparison on SM90.

The fixture is shared with the production dispatch parity test.  Candidate is
the CuTe DSL verify kernel with ReplaySSM ring and fused sigmoid gated RMSNorm;
control is the Triton ReplaySSM verify path followed by the existing norm.
"""

import argparse
import json
import runpy
import statistics
from pathlib import Path
from unittest.mock import patch

import torch
from sglang.kernels.ops.attention.fla.fused_norm_gate import layer_norm_gated_fwd
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend


def _fixture_class():
    path = (
        Path(__file__).parents[2]
        / "test/registered/kernels/ops/attention/test_kda_fused_verify_backend.py"
    )
    return runpy.run_path(str(path))["TestKDAFusedVerifyBackend"]


def _measure(fn, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(samples),
        "p95_us": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


@torch.inference_mode()
def bench(batch_size: int, steps: int, warmup: int, iters: int) -> dict:
    fixture = _fixture_class()()
    layers, initial, slots, batch, rounds = fixture._make_case(
        batch_size=batch_size,
        heads=64,
        v_heads=64,
        lower_bound=-5.0,
        steps=steps,
    )
    candidate, _, _ = fixture._make_backend(
        initial,
        slots,
        steps,
        fused=False,
        ring=True,
        verify_backend=LinearAttnKernelBackend.NV_CUTEDSL,
    )
    control, _, _ = fixture._make_backend(
        initial,
        slots,
        steps,
        fused=False,
        ring=True,
        verify_backend=LinearAttnKernelBackend.TRITON,
    )
    layer = layers[0]
    # Match the production GLM-5.3 KDA contract accepted by the CuTe kernel.
    # The generic parity fixture also covers biased/BF16 conv weights, which
    # intentionally fall back to Triton.
    layer.bias = None
    layer.conv_weights = layer.conv_weights.float()
    mixed, a, b = rounds[0][0]
    gate = torch.randn(
        batch_size * steps,
        64 * 128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    norm_weight = torch.randn(128, device="cuda", dtype=torch.float32)
    layer._k3_onorm_gate = gate
    layer._k3_onorm_args = (norm_weight, 1e-5)
    layer._k3_onorm_consumed = False

    def candidate_fn():
        layer._k3_onorm_consumed = False
        output = candidate.forward_extend(layer, batch, mixed, a, b)
        assert layer._k3_onorm_consumed
        return output

    def control_fn():
        # Keep the control as Triton ReplaySSM verify plus the existing,
        # separate gated RMSNorm. Clear the model handoff while entering the
        # backend so this remains true if another Triton fusion is added later.
        onorm_gate = layer._k3_onorm_gate
        onorm_args = layer._k3_onorm_args
        layer._k3_onorm_gate = None
        layer._k3_onorm_args = None
        layer._k3_onorm_consumed = False
        try:
            output = control.forward_extend(layer, batch, mixed, a, b)
        finally:
            layer._k3_onorm_gate = onorm_gate
            layer._k3_onorm_args = onorm_args
        assert not layer._k3_onorm_consumed
        normalized, _, _, _ = layer_norm_gated_fwd(
            x=output.view(-1, 128),
            g=gate.view(-1, 128),
            weight=norm_weight,
            bias=None,
            activation="sigmoid",
            eps=1e-5,
            out_dtype=torch.bfloat16,
            is_rms_norm=True,
        )
        return normalized.view_as(output)

    # A rejected capability gate silently falls back to Triton. Count one
    # real dispatch outside the timed loop so the result proves CuTe execution
    # without adding mocking overhead to the measurement.
    with patch.object(
        KDAAttnBackend,
        "_run_dspark_cutedsl_mtp",
        wraps=KDAAttnBackend._run_dspark_cutedsl_mtp,
    ) as run_cute:
        actual = candidate_fn()
    dispatch_count = run_cute.call_count
    if dispatch_count != 1:
        raise AssertionError(f"expected one CuTe dispatch, got {dispatch_count}")
    reference = control_fn()
    rel_error = (
        (actual.float() - reference.float()).abs().max()
        / reference.float().abs().max().clamp_min(1e-12)
    ).item()
    if rel_error >= 2e-2:
        raise AssertionError(f"candidate/control max relative error {rel_error}")
    control_timing = _measure(control_fn, warmup, iters)
    candidate_timing = _measure(candidate_fn, warmup, iters)
    return {
        "batch_size": batch_size,
        "draft_token_num": steps,
        "heads": 64,
        "head_dim": 128,
        "cutedsl_dispatch_count": dispatch_count,
        "max_relative_error": rel_error,
        "control": control_timing,
        "candidate": candidate_timing,
        "median_speedup": control_timing["median_us"] / candidate_timing["median_us"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--draft-token-num", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("physical SM90 CUDA is required")
    result = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "cases": [
            bench(
                batch_size,
                args.draft_token_num,
                args.warmup,
                args.iters,
            )
            for batch_size in args.batch_sizes
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
