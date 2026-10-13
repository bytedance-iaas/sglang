"""Interleaved A/B for the opt-in SM90 indexer, including preparation costs.

Inputs are synthetic fake-FP4 queries and packed FP4 cache pages. Logits timing
includes dense slot construction on the default path and Q packing on CUDA.
Backend timing additionally covers lengths, top-k=512, sorting and slot mapping;
query projection / RoPE and head-weight projection are replaced by prepared
inputs, so these numbers are not end-to-end serving latency.

Optional environment variables: SM90_INDEXER_BENCH_CASES (comma-separated
rows:width:visible_fraction), SM90_INDEXER_BENCH_SEED, SM90_INDEXER_BENCH_ROUNDS,
SM90_INDEXER_BENCH_MS, SM90_INDEXER_BENCH_HEADS (32 or 64, default 64),
SM90_INDEXER_BENCH_RATIO (1 or 2, default 2), and SM90_INDEXER_BENCH_OUTPUT
(a new JSON filename). Set HEADS=32 for the Flash workload; its opt-in path
uses mapped Triton, not the 64-head grouped CUDA kernel.
"""

import json
import math
import os
import random
import statistics
from pathlib import Path

import torch
from test_sm90_fp4_grouped_indexer import TestSm90Fp4GroupedIndexer

from sglang.kernels.jit.benchmark.utils import get_benchmark_range
from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_logits_decode
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def elapsed_ms(graph, repeats):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def interleaved_times(runners, rounds, target_ms, batch=20):
    graphs, repeats = {}, {}
    samples = {name: [] for name in runners}
    try:
        for name, run in runners.items():
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            graphs[name] = graph
            with torch.cuda.graph(graph):
                for _ in range(batch):
                    run()
            elapsed_ms(graph, 5)
            repeats[name] = max(
                1, min(2000, math.ceil(target_ms / (elapsed_ms(graph, 5) / 5)))
            )
        rng = random.Random(20260918)
        names = list(runners)
        for _ in range(rounds):
            rng.shuffle(names)
            for name in names:
                count = repeats[name]
                samples[name].append(
                    elapsed_ms(graphs[name], count) * 1000 / (count * batch)
                )
    finally:
        for graph in graphs.values():
            graph.reset()
    return {
        name: {"median_us": statistics.median(values), "samples_us": values}
        for name, values in samples.items()
    }


def benchmark():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        print("SM90 FP4 indexer benchmark requires a Hopper GPU")
        return
    output = os.getenv("SM90_INDEXER_BENCH_OUTPUT")
    if output and Path(output).exists():
        raise FileExistsError(f"Refusing to overwrite benchmark results: {output}")
    seed = int(os.getenv("SM90_INDEXER_BENCH_SEED", "11"))
    heads = int(os.getenv("SM90_INDEXER_BENCH_HEADS", "64"))
    ratio = int(os.getenv("SM90_INDEXER_BENCH_RATIO", "2"))
    if heads not in (32, 64) or ratio not in (1, 2):
        raise ValueError("HEADS must be 32 or 64 and RATIO must be 1 or 2")
    rounds = int(os.getenv("SM90_INDEXER_BENCH_ROUNDS", "9"))
    target_ms = float(os.getenv("SM90_INDEXER_BENCH_MS", "100"))
    cases = get_benchmark_range(
        full_range=[
            "6:512:1",
            "6:4096:1",
            "6:16384:1",
            "6:131072:1",
            "6:131072:0.5",
            "6:131072:0.125",
            "6:131072:0",
            "6:1048896:1",
            "24:512:1",
            "96:512:1",
            "384:512:1",
            "96:4096:1",
        ],
        ci_range=["6:512:1", "6:4096:1", "96:4096:1"],
    )
    if os.getenv("SM90_INDEXER_BENCH_CASES"):
        cases = os.environ["SM90_INDEXER_BENCH_CASES"].split(",")
    fixture = TestSm90Fp4GroupedIndexer("test_matches_triton")
    records = []
    for shape in cases:
        rows, width, fraction = shape.split(":")
        rows, width, fraction = int(rows), int(width), float(fraction)
        if rows <= 0 or rows % 6 or width <= 0 or not 0 <= fraction <= 1:
            raise ValueError(
                "Backend benchmark requires full groups of 6 and width > 0"
            )
        x = fixture.make_inputs(
            rows=rows, width=width, seed=seed, heads=heads, ratio=ratio
        )
        x.lens.copy_(
            (int(width * fraction) - torch.arange(rows, device="cuda") % 6).clamp_min(0)
        )
        case = fixture.make_backend_case(x, topk=512)

        def default_logits():
            position = torch.arange(width, device="cuda")
            slots = (
                x.mapping[x.req[:, None], (position * x.ratio)[None, :]].long()
                // x.ratio
            )
            slots = slots.masked_fill(position[None, :] >= x.lens[:, None], 0)
            return fp4_index_logits_decode(
                x.q, x.weights, slots, x.lens, x.table, x.page
            )

        def backend(enabled):
            with envs.SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER.override(enabled):
                case.run()

        torch.testing.assert_close(
            fixture.run_public(x), default_logits(), atol=0, rtol=0, equal_nan=False
        )
        backend(False)
        pages, raw = case.pages.clone(), case.raw.clone()
        backend(True)
        torch.testing.assert_close(case.pages, pages, atol=0, rtol=0)
        torch.testing.assert_close(case.raw, raw, atol=0, rtol=0)
        times = interleaved_times(
            {
                "default_logits": default_logits,
                "opt_in_logits": lambda: fixture.run_public(x),
                "default_backend": lambda: backend(False),
                "opt_in_backend": lambda: backend(True),
            },
            rounds,
            target_ms,
        )
        record = {
            "case": shape,
            "seed": seed,
            "group": 6,
            "heads": heads,
            "page": 64,
            "ratio": ratio,
            "topk": 512,
            "correct": True,
            "times": times,
            "gpu": torch.cuda.get_device_name(),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    benchmark()
