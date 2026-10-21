"""Complete-cost SM90 microbenchmark for GLM-5.3-Flash KDA PTPC.

This compares the exact BF16 fallback with the opt-in FP8 adapter at the two
P3 target shapes. It includes activation quantization and the separate BF16
beta projection for the packed first stage, and gated RMSNorm for o_proj.
"""

import argparse
import json
import statistics

import torch
import torch.nn.functional as F
from sglang.kernels.ops.attention.fla.fused_norm_gate import layer_norm_gated_fwd
from sglang.srt.layers.quantization.unquant import (
    Glm53KdaPackedPtpcLinearMethod,
    Glm53KdaPtpcLinearMethod,
)


class _BenchLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)


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
        samples.append(start.elapsed_time(end))
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[p95_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _accuracy(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float().flatten()
    reference_f = reference.float().flatten()
    error = actual_f - reference_f
    denom = reference_f.square().mean().sqrt().clamp_min(1e-12)
    return {
        "relative_rmse": (error.square().mean().sqrt() / denom).item(),
        "max_abs": error.abs().max().item(),
        "cosine": F.cosine_similarity(actual_f, reference_f, dim=0).item(),
    }


@torch.inference_mode()
def _bench_packed_first_stage(m: int, heads: int, warmup: int, iters: int) -> dict:
    k = 4096
    qkv_size, beta_size, fg_size = 3 * heads * 128, heads, 256
    n = qkv_size + beta_size + fg_size
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    layer = _BenchLinear(
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    )
    bf16_weight_bytes = layer.weight.numel() * layer.weight.element_size()
    method = Glm53KdaPackedPtpcLinearMethod(
        fp8_min_m=m,
        qkv_size=qkv_size,
        beta_size=beta_size,
        fg_size=fg_size,
    )
    before_fp8 = torch.cuda.memory_allocated()
    method.process_weights_after_loading(layer)
    torch.cuda.synchronize()
    fp8_copy_bytes = torch.cuda.memory_allocated() - before_fp8

    def baseline():
        fused = F.linear(x, layer.weight)
        return torch.split(fused, (qkv_size, beta_size, fg_size), dim=-1)

    def candidate():
        return method.apply_ptpc_prefill(layer, x)

    ref = baseline()
    actual = candidate()
    accuracy = {
        name: _accuracy(got, expected)
        for name, got, expected in zip(("qkv", "beta", "fg_a"), actual, ref)
    }
    baseline_ms = _measure(baseline, warmup, iters)
    candidate_ms = _measure(candidate, warmup, iters)
    return {
        "shape": {"m": m, "k": k, "n": n, "fp8_n": qkv_size + fg_size},
        "bf16_weight_bytes": bf16_weight_bytes,
        "fp8_copy_allocated_bytes": fp8_copy_bytes,
        "baseline": baseline_ms,
        "candidate": candidate_ms,
        "median_speedup": baseline_ms["median_ms"] / candidate_ms["median_ms"],
        "accuracy": accuracy,
    }


@torch.inference_mode()
def _bench_o_proj(m: int, heads: int, warmup: int, iters: int) -> dict:
    head_dim = 128
    k = n = heads * head_dim
    core = torch.randn(1, m, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(core)
    norm_weight = torch.ones(head_dim, device="cuda", dtype=torch.bfloat16)
    layer = _BenchLinear(
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    )
    bf16_weight_bytes = layer.weight.numel() * layer.weight.element_size()
    method = Glm53KdaPtpcLinearMethod("o_proj", fp8_min_m=m)
    before_fp8 = torch.cuda.memory_allocated()
    method.process_weights_after_loading(layer)
    torch.cuda.synchronize()
    fp8_copy_bytes = torch.cuda.memory_allocated() - before_fp8

    def norm_input():
        # The public module writes into ``core`` in-place. A benchmark reuses
        # its fixture across iterations, so request a separate BF16 output to
        # preserve the same original input for every sample.
        output, _, _, _ = layer_norm_gated_fwd(
            x=core.view(-1, head_dim),
            g=gate.view(-1, head_dim),
            weight=norm_weight,
            bias=None,
            activation="sigmoid",
            out_dtype=torch.bfloat16,
            is_rms_norm=True,
        )
        return output.view(m, k)

    def baseline():
        return F.linear(norm_input(), layer.weight)

    def candidate():
        return method.apply(layer, norm_input())

    reference = baseline()
    actual = candidate()
    baseline_ms = _measure(baseline, warmup, iters)
    candidate_ms = _measure(candidate, warmup, iters)
    return {
        "shape": {"m": m, "k": k, "n": n, "heads": heads, "head_dim": head_dim},
        "bf16_weight_bytes": bf16_weight_bytes,
        "fp8_copy_allocated_bytes": fp8_copy_bytes,
        "baseline": baseline_ms,
        "candidate": candidate_ms,
        "median_speedup": baseline_ms["median_ms"] / candidate_ms["median_ms"],
        "accuracy": _accuracy(actual, reference),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--heads", type=int, choices=(32, 64), default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (9, 0):
        raise RuntimeError(f"SM90 is required, got capability={capability}")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    result = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": capability,
        "m": args.m,
        "heads": args.heads,
        "warmup": args.warmup,
        "iters": args.iters,
        "packed_first_stage": _bench_packed_first_stage(
            args.m, args.heads, args.warmup, args.iters
        ),
        "o_proj_with_norm": _bench_o_proj(args.m, args.heads, args.warmup, args.iters),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
