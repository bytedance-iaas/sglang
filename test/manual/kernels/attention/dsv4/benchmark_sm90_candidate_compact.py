"""Dense-output candidate compaction / same-layer K-reuse benchmark on SM90.

Match the PR1 protocol: BF16 Q, packed FP4 K, H32, page/tile 64, candidate
budget 2048 x 8, native TopK 512, paired AB/BA trials (20 ms warmup / 75 ms
measurement / seven trials by default). Report batch GPU latency, not TPOT.
Score includes dense initialization/scatter; group includes source publication
and three consumers. Q projection and initial slots construction are excluded.

Example:
  python test/manual/kernels/attention/dsv4/benchmark_sm90_candidate_compact.py \
    --rows 1 32 64 128 --widths 65536 131072 --output /tmp/compact.jsonl

Group fixtures share a synthetic K table to align with PR1. Use
--separate-layer-cache for a second run with independent layer buffers. Actual
K reuse is ONLY across verify queries within a layer, never across layers.
Optional --oracle-source points to an unmodified image/repo's fp4_indexer.py.
"""

import argparse
import importlib.util
import json
import statistics
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.benchmark.bench_utils import run_bench
from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_logits_decode
from sglang.kernels.ops.attention.dsv4.sm90_candidate_indexer import (
    compact_candidate_logits,
    paired_query_logits,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    sm90_index_logits_with_candidates,
)
from sglang.test.kernels.deepseek_v4.sm90_candidate_indexer import (
    make_candidate_case,
    select_candidate_topk,
)


def operands(case):
    return {k: case[k] for k in ("q", "weights", "slots", "lens", "table", "page_size")}


def check_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        if actual.dtype == torch.float32:
            assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        else:
            assert torch.equal(actual, expected)
    else:
        for new, old in zip(actual, expected, strict=True):
            check_equal(new, old)


def pack_fixture_mask(mask, budget):
    """Untimed for standalone score stages, as source publication was in PR1."""
    rows, width = mask.shape
    keep = torch.nn.functional.pad(mask, (0, -width % 8)).reshape(rows, -1, 8).any(-1)
    ids = torch.arange(keep.shape[1], device=mask.device, dtype=torch.int32)
    return (
        ids.expand_as(keep)
        .masked_fill(~keep, -1)
        .topk(min(budget, keep.shape[1]))
        .values
    )


def score(case, provider, blocks, request_ids=None):
    values = operands(case)
    mask = case["candidate_mask"]
    if provider in ("compact", "combined"):
        return compact_candidate_logits(**values, blocks=blocks, candidate_mask=mask)
    if provider == "paired":
        return paired_query_logits(
            **values, request_ids=request_ids, candidate_mask=mask
        )
    result = fp4_index_logits_decode(
        **values, candidate_mask=mask if provider == "premask" else None
    )
    return result.masked_fill(~mask, -torch.inf) if provider == "dense" else result


def consumer(case, provider, blocks, request_ids, topk):
    return select_candidate_topk(
        score(case, provider, blocks, request_ids), case["slots"], case["lens"], topk
    )


def group(source, consumers, request_ids, topk, budget):
    metadata = None
    outputs = []
    for is_source, case in [(True, source)] + [(False, c) for c in consumers]:
        indexer = SimpleNamespace(
            is_candidate_source=is_source,
            uses_candidates=not is_source,
            candidate_topk_blocks=budget,
            candidate_block_size=8,
        )
        logits, metadata = sm90_index_logits_with_candidates(
            **operands(case),
            indexer=indexer,
            candidate_metadata=metadata,
            request_ids=request_ids,
        )
        outputs.append(
            select_candidate_topk(
                logits,
                case["slots"],
                case["lens"],
                topk,
                filter_candidates=not is_source,
            )
        )
    return outputs


def provider_context(provider):
    stack = ExitStack()
    for flag, value in (
        (envs.SGLANG_DSV41_SM90_CANDIDATE_MASK, provider != "dense"),
        (envs.SGLANG_DSV41_SM90_CANDIDATE_MASK_MIN_WIDTH, 65536),
        (envs.SGLANG_DSV41_SM90_CANDIDATE_COMPACT, provider in ("compact", "combined")),
        (envs.SGLANG_DSV41_SM90_QUERY_K_REUSE, provider in ("paired", "combined")),
    ):
        stack.enter_context(flag.override(value))
    return stack


def measure(functions, graph, args):
    samples = {provider: [] for provider in functions}
    for trial in range(args.trials):
        order = list(functions)
        if trial % 2:
            order.reverse()
        for provider in order:
            with provider_context(provider):
                timing = run_bench(
                    functions[provider],
                    use_cuda_graph=graph,
                    warmup_ms=args.warmup_ms,
                    rep_ms=args.rep_ms,
                )
            samples[provider].append(float(timing[0]) * 1000)
    baseline = samples["premask"]
    return {
        name: dict(
            median_us=statistics.median(values),
            raw_us=values,
            paired_speedup=statistics.median(a / b for a, b in zip(baseline, values)),
            paired_speedup_min=min(a / b for a, b in zip(baseline, values)),
            paired_speedup_max=max(a / b for a, b in zip(baseline, values)),
        )
        for name, values in samples.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 32, 64, 128])
    parser.add_argument("--widths", nargs="+", type=int, default=[65536, 131072])
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=["clustered", "scattered"],
        default=["clustered", "scattered"],
    )
    parser.add_argument(
        "--execution", nargs="+", choices=["eager", "graph"], default=["eager", "graph"]
    )
    parser.add_argument("--verify-group-sizes", nargs="*", type=int, default=[4, 6])
    parser.add_argument("--verify-requests", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--candidate-blocks", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--warmup-ms", type=int, default=20)
    parser.add_argument("--rep-ms", type=int, default=75)
    parser.add_argument("--separate-layer-cache", action="store_true")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["score", "score_topk", "group", "source_score", "topk_only"],
        default=["score", "score_topk", "group", "source_score", "topk_only"],
    )
    parser.add_argument("--oracle-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        parser.error("requires SM90")
    if (
        min(
            args.rows
            + args.widths
            + args.verify_requests
            + [args.candidate_blocks, args.topk, args.trials]
        )
        <= 0
    ):
        parser.error("sizes and trials must be positive")
    if any(g < 2 for g in args.verify_group_sizes):
        parser.error("verify group sizes must be >= 2")
    oracle = fp4_index_logits_decode
    if args.oracle_source is not None:
        spec = importlib.util.spec_from_file_location(
            "candidate_original_fp4", args.oracle_source
        )
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        oracle = original.fp4_index_logits_decode
    args.output.parent.mkdir(parents=True, exist_ok=True)
    jobs = [("decode", b, width, 1) for b in args.rows for width in args.widths]
    jobs += [
        ("verify", n * g, width, g)
        for n in args.verify_requests
        for g in args.verify_group_sizes
        for width in args.widths
    ]
    with args.output.open("w") as output:
        for mode, rows, width, group_size in jobs:
            for pattern in args.patterns:
                case = make_candidate_case(
                    rows,
                    width,
                    candidate_blocks=args.candidate_blocks,
                    block_size=8,
                    pattern=pattern,
                )
                request_ids = None
                if mode == "verify":
                    request_ids = (
                        torch.arange(rows, device="cuda", dtype=torch.int64)
                        // group_size
                    )
                    case["slots"] = case["slots"][::group_size].repeat_interleave(
                        group_size, 0
                    )
                    # Query-local causal lengths remain distinct within a request.
                    case["lens"] -= (
                        group_size - 1 - torch.arange(rows, device="cuda") % group_size
                    )
                blocks = pack_fixture_mask(
                    case["candidate_mask"], args.candidate_blocks
                )
                providers = ["dense", "premask", "compact"]
                if mode == "verify":
                    providers += ["paired", "combined"]
                original_scores = oracle(**operands(case))
                check_equal(fp4_index_logits_decode(**operands(case)), original_scores)
                expected = original_scores.masked_fill(
                    ~case["candidate_mask"], -torch.inf
                )
                for provider in providers:
                    check_equal(score(case, provider, blocks, request_ids), expected)
                    check_equal(
                        consumer(case, provider, blocks, request_ids, args.topk),
                        select_candidate_topk(
                            expected, case["slots"], case["lens"], args.topk
                        ),
                    )
                stages = {
                    "score": {
                        p: partial(score, case, p, blocks, request_ids)
                        for p in providers
                    },
                    "score_topk": {
                        p: partial(consumer, case, p, blocks, request_ids, args.topk)
                        for p in providers
                    },
                }
                if mode == "verify":
                    stages["source_score"] = {
                        "premask": partial(fp4_index_logits_decode, **operands(case)),
                        "paired": partial(
                            paired_query_logits,
                            **operands(case),
                            request_ids=request_ids,
                        ),
                    }
                    check_equal(
                        stages["source_score"]["paired"](), oracle(**operands(case))
                    )
                if pattern == args.patterns[0]:
                    consumers = [
                        dict(
                            case,
                            q=case["q"].roll(i + 1, -1),
                            weights=case["weights"].roll(i + 1, 1),
                        )
                        for i in range(3)
                    ]
                    if args.separate_layer_cache:
                        consumers = [
                            dict(c, table=c["table"].clone()) for c in consumers
                        ]
                    stages["group"] = {
                        p: partial(
                            group,
                            case,
                            consumers,
                            request_ids,
                            args.topk,
                            args.candidate_blocks,
                        )
                        for p in providers
                    }
                    with provider_context("dense"):
                        expected_group = stages["group"]["dense"]()
                    for provider in providers:
                        with provider_context(provider):
                            check_equal(stages["group"][provider](), expected_group)
                for stage, functions in stages.items():
                    if stage not in args.stages:
                        continue
                    for execution in args.execution:
                        timing = measure(functions, execution == "graph", args)
                        record = dict(
                            mode=mode,
                            rows=rows,
                            requests=rows // group_size,
                            verify_group_size=group_size,
                            width=width,
                            heads=32,
                            topk=args.topk,
                            candidate_blocks=args.candidate_blocks,
                            block_size=8,
                            page_size=64,
                            dtype="bf16/fp4",
                            gpu=torch.cuda.get_device_name(),
                            torch=torch.__version__,
                            pattern="source_selected" if stage == "group" else pattern,
                            stage=stage,
                            execution=execution,
                            trials=args.trials,
                            warmup_ms=args.warmup_ms,
                            rep_ms=args.rep_ms,
                            separate_layer_cache=args.separate_layer_cache,
                            bitwise_equal=True,
                            exact_topk=True,
                            timing=timing,
                        )
                        output.write(json.dumps(record) + "\n")
                        output.flush()
                        print(json.dumps(record), flush=True)
                # Fixed logits, so this isolates the unchanged native selection.
                if (
                    pattern == "scattered"
                    and mode == "decode"
                    and "topk_only" in args.stages
                ):
                    for execution in args.execution:
                        functions = {
                            "premask": partial(
                                expected.topk,
                                min(args.topk, width),
                                dim=-1,
                                sorted=False,
                            )
                        }
                        record = dict(
                            mode=mode,
                            rows=rows,
                            width=width,
                            pattern=pattern,
                            stage="topk_only",
                            execution=execution,
                            timing=measure(functions, execution == "graph", args),
                        )
                        output.write(json.dumps(record) + "\n")
                        output.flush()
                        print(json.dumps(record), flush=True)
                del case, blocks, expected, stages
                torch.cuda.synchronize()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
