"""Compare GLM-5.3 KDA target-verify paths on physical SM90.

K0 is the Triton ReplaySSM verify followed by the existing output norm.  K1 is
the CuTe DSL ReplaySSM verify followed by that same output norm.  K2 is the
CuTe DSL verify with the output norm fused into the kernel.  Keeping K1
separate distinguishes PR #44612's kernel from PR #43659's norm fusion.
"""

import argparse
import importlib.metadata
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


def _measure(fn, warmup: int, iters: int) -> dict[str, object]:
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
        "samples_us": samples,
    }


def _max_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float()
    expected = expected.float()
    return (
        (actual - expected).abs().max()
        / expected.abs().max().clamp_min(1e-12)
    ).item()


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
    k0_backend, _, _ = fixture._make_backend(
        initial,
        slots,
        steps,
        fused=False,
        ring=True,
        verify_backend=LinearAttnKernelBackend.TRITON,
    )
    k1_backend, _, _ = fixture._make_backend(
        initial,
        slots,
        steps,
        fused=False,
        ring=True,
        verify_backend=LinearAttnKernelBackend.NV_CUTEDSL,
    )
    k2_backend, _, _ = fixture._make_backend(
        initial,
        slots,
        steps,
        fused=False,
        ring=True,
        verify_backend=LinearAttnKernelBackend.NV_CUTEDSL,
    )

    layer = layers[0]
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

    def separate_norm(output):
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

    def run_separate(backend):
        layer._k3_onorm_gate = None
        layer._k3_onorm_args = None
        layer._k3_onorm_consumed = False
        output = backend.forward_extend(layer, batch, mixed, a, b)
        if layer._k3_onorm_consumed:
            raise AssertionError("separate-norm path unexpectedly consumed handoff")
        return separate_norm(output)

    def k0_fn():
        return run_separate(k0_backend)

    def k1_fn():
        return run_separate(k1_backend)

    def k2_fn():
        layer._k3_onorm_gate = gate
        layer._k3_onorm_args = (norm_weight, 1e-5)
        layer._k3_onorm_consumed = False
        output = k2_backend.forward_extend(layer, batch, mixed, a, b)
        if not layer._k3_onorm_consumed:
            raise AssertionError("fused-norm path did not consume handoff")
        return output

    reference = k0_fn()
    with patch.object(
        KDAAttnBackend,
        "_run_dspark_cutedsl_mtp",
        wraps=KDAAttnBackend._run_dspark_cutedsl_mtp,
    ) as run_cute:
        k1_output = k1_fn()
        k2_output = k2_fn()
    dispatch_count = run_cute.call_count
    if dispatch_count != 2:
        raise AssertionError(f"expected two CuTe dispatches, got {dispatch_count}")

    k1_error = _max_relative_error(k1_output, reference)
    k2_error = _max_relative_error(k2_output, reference)
    for name, error in (("K1", k1_error), ("K2", k2_error)):
        if error >= 2e-2:
            raise AssertionError(f"{name}/K0 max relative error {error}")

    timings = {
        "K0": _measure(k0_fn, warmup, iters),
        "K1": _measure(k1_fn, warmup, iters),
        "K2": _measure(k2_fn, warmup, iters),
    }
    return {
        "batch_size": batch_size,
        "draft_token_num": steps,
        "heads": 64,
        "head_dim": 128,
        "cutedsl_dispatch_count": dispatch_count,
        "max_relative_error": {"K1_vs_K0": k1_error, "K2_vs_K0": k2_error},
        "timings": timings,
        "median_speedup": {
            "K1_vs_K0": timings["K0"]["median_us"]
            / timings["K1"]["median_us"],
            "K2_vs_K0": timings["K0"]["median_us"]
            / timings["K2"]["median_us"],
            "K2_vs_K1": timings["K1"]["median_us"]
            / timings["K2"]["median_us"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--draft-token-num", type=int, choices=range(2, 9), required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("physical SM90 CUDA is required")
    result = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "method": {
            "K0": "Triton ReplaySSM verify plus separate gated RMSNorm",
            "K1": "CuTe ReplaySSM verify plus separate gated RMSNorm",
            "K2": "CuTe ReplaySSM verify with fused gated RMSNorm",
            "warmup": args.warmup,
            "iterations": args.iters,
        },
        "cases": [
            bench(batch_size, args.draft_token_num, args.warmup, args.iters)
            for batch_size in args.batch_sizes
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
