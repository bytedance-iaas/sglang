"""PR1: dense post-mask versus candidate-before-load SM90 indexer.

Run on one Hopper GPU, for example:
  python test/manual/kernels/attention/dsv4/benchmark_sm90_candidate_indexer.py \
    --rows 1 32 --widths 8192 65536 --patterns clustered scattered \
    --candidate-blocks 2048 --execution eager graph --output /tmp/sm90-candidates.jsonl

All providers use the same dense FP32 logits layout and original torch TopK.
candidate_auto uses the production width policy (default 65536); override it
with SGLANG_DSV41_SM90_CANDIDATE_MASK_MIN_WIDTH. The two forced providers are
independent of both feature flags so that the oracle remains available.
The score stage includes post-mask for the baseline; score_topk adds original
TopK, logical sorting, underfill masking and physical-slot conversion. Group
also includes dense source scoring, its token TopK and real candidate-block
publication followed by --consumers indexers. Q/weight projections and initial
request-to-slot construction are outside these microbenchmarks.

Clustered/scattered patterns apply to the first two stages. Group uses the
actual source-selected blocks and is measured once per shape/density, with its
own retained/tile fractions. No DeepGEMM or native FP4 instructions are needed.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from sglang.benchmark.bench_utils import run_bench
from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_logits_decode
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    select_candidate_blocks,
    should_move_sm90_candidate_mask,
)
from sglang.test.kernels.deepseek_v4.sm90_candidate_indexer import (
    make_candidate_case,
    select_candidate_topk,
)


def score(case, mask=None):
    return fp4_index_logits_decode(
        case["q"],
        case["weights"],
        case["slots"],
        case["lens"],
        case["table"],
        case["page_size"],
        candidate_mask=mask,
    )


def consumer(case, mask, masked, topk, *, select=True):
    # None selects the production width policy, independently of its master flag.
    if masked is None:
        masked = should_move_sm90_candidate_mask(case["slots"].shape[1])
    logits = score(case, mask if masked else None)
    if not masked:
        logits = logits.masked_fill(~mask, -torch.inf)
    if not select:
        return logits
    return select_candidate_topk(logits, case["slots"], case["lens"], topk)


def publish(case, topk, candidate_blocks, block_size):
    logits = score(case)
    mask = select_candidate_blocks(
        logits,
        case["lens"][:, None],
        topk_blocks=candidate_blocks,
        block_size=block_size,
    )
    indices = select_candidate_topk(
        logits, case["slots"], case["lens"], topk, filter_candidates=False
    )
    return mask, indices


def group(source, consumers, masked, topk, candidate_blocks, block_size):
    mask, indices = publish(source, topk, candidate_blocks, block_size)
    outputs = [indices]
    for case in consumers:
        outputs.append(consumer(case, mask, masked, topk))
    return outputs


def fractions(mask, lens):
    rows, width = mask.shape
    valid = torch.arange(width, device=mask.device)[None, :] < lens[:, None]
    retained = mask & valid
    padded = torch.nn.functional.pad(retained, (0, -width % 64))
    active = padded.reshape(rows, -1, 64).any(-1).float().mean().item()
    return dict(
        retained_fraction=retained.sum().item() / max(1, valid.sum().item()),
        active_tile_fraction=active,
    )


def check_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        if actual.dtype == torch.float32:
            assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        else:
            assert torch.equal(actual, expected)
    else:
        for new, old in zip(actual, expected):
            check_equal(new, old)


def measure_pair(functions, graph, args):
    times = {provider: [] for provider in functions}
    for trial in range(args.trials):
        # Alternate order to reduce clock/temperature bias between A and B.
        providers = list(functions)
        if trial % 2:
            providers.reverse()
        for provider in providers:
            times[provider].append(
                run_bench(
                    functions[provider],
                    use_cuda_graph=graph,
                    warmup_ms=args.warmup_ms,
                    rep_ms=args.rep_ms,
                )
            )
    return {
        provider: dict(
            median_us=1000 * statistics.median(t[0] for t in samples),
            p20_us=1000 * statistics.median(t[1] for t in samples),
            p80_us=1000 * statistics.median(t[2] for t in samples),
        )
        for provider, samples in times.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 32])
    parser.add_argument("--widths", nargs="+", type=int, default=[8192, 65536])
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--density", type=float, default=0.125)
    parser.add_argument(
        "--candidate-blocks",
        type=int,
        help="absolute candidate block budget (overrides --density); V4.1 Flash uses 2048",
    )
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--consumers", type=int, default=3)
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=["clustered", "scattered", "all", "none"],
        default=["clustered", "scattered"],
    )
    parser.add_argument(
        "--execution", nargs="+", choices=["eager", "graph"], default=["eager", "graph"]
    )
    parser.add_argument("--mixed-lengths", action="store_true")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        parser.error("this benchmark requires an SM90 CUDA GPU")
    if (
        min(args.rows + args.widths) <= 0
        or not 0 < args.density <= 1
        or min(args.heads, args.page_size, args.block_size, args.topk, args.trials) <= 0
        or args.consumers < 0
        or (args.candidate_blocks is not None and args.candidate_blocks <= 0)
    ):
        parser.error("sizes/trials must be positive, density in (0, 1], consumers >= 0")

    records = []
    gpu = torch.cuda.get_device_name()
    for rows in args.rows:
        for width in args.widths:
            for pattern in args.patterns:
                case = make_candidate_case(
                    rows,
                    width,
                    args.heads,
                    page_size=args.page_size,
                    block_size=args.block_size,
                    candidate_blocks=args.candidate_blocks,
                    density=args.density,
                    pattern=pattern,
                    mixed_lengths=args.mixed_lengths,
                )
                mask = case["candidate_mask"]
                check_equal(
                    consumer(case, mask, True, args.topk, select=False),
                    consumer(case, mask, False, args.topk, select=False),
                )
                check_equal(
                    consumer(case, mask, True, args.topk),
                    consumer(case, mask, False, args.topk),
                )
                check_equal(
                    consumer(case, mask, None, args.topk),
                    consumer(case, mask, False, args.topk),
                )
                stages = {
                    "score": {
                        "dense_postmask": lambda: consumer(
                            case, mask, False, args.topk, select=False
                        ),
                        "candidate_premask": lambda: consumer(
                            case, mask, True, args.topk, select=False
                        ),
                        "candidate_auto": lambda: consumer(
                            case, mask, None, args.topk, select=False
                        ),
                    },
                    "score_topk": {
                        "dense_postmask": lambda: consumer(
                            case, mask, False, args.topk
                        ),
                        "candidate_premask": lambda: consumer(
                            case, mask, True, args.topk
                        ),
                        "candidate_auto": lambda: consumer(case, mask, None, args.topk),
                    },
                }
                metadata = dict(
                    gpu=gpu,
                    torch=torch.__version__,
                    rows=rows,
                    width=width,
                    heads=args.heads,
                    dtype="bf16/fp4",
                    page_size=args.page_size,
                    block_size=args.block_size,
                    topk=args.topk,
                    requested_density=(
                        args.density if args.candidate_blocks is None else None
                    ),
                    candidate_topk_blocks=args.candidate_blocks,
                    candidate_mask_min_width=(
                        envs.SGLANG_DSV41_SM90_CANDIDATE_MASK_MIN_WIDTH.get()
                    ),
                    auto_uses_premask=should_move_sm90_candidate_mask(width),
                    pattern=pattern,
                    mixed_lengths=args.mixed_lengths,
                    **fractions(mask, case["lens"]),
                )
                if pattern == args.patterns[0] and args.consumers:
                    # Consumers share the source K/slots but have their own Q/weights.
                    consumers = [
                        dict(
                            case,
                            q=case["q"].roll(i + 1, dims=-1),
                            weights=case["weights"].roll(i + 1, dims=1),
                        )
                        for i in range(args.consumers)
                    ]
                    candidate_blocks = args.candidate_blocks or max(
                        1,
                        int(
                            ((width + args.block_size - 1) // args.block_size)
                            * args.density
                        ),
                    )
                    group_mask, _ = publish(
                        case, args.topk, candidate_blocks, args.block_size
                    )
                    check_equal(
                        group(
                            case,
                            consumers,
                            True,
                            args.topk,
                            candidate_blocks,
                            args.block_size,
                        ),
                        group(
                            case,
                            consumers,
                            False,
                            args.topk,
                            candidate_blocks,
                            args.block_size,
                        ),
                    )
                    check_equal(
                        group(
                            case,
                            consumers,
                            None,
                            args.topk,
                            candidate_blocks,
                            args.block_size,
                        ),
                        group(
                            case,
                            consumers,
                            False,
                            args.topk,
                            candidate_blocks,
                            args.block_size,
                        ),
                    )
                    stages["group"] = {
                        "dense_postmask": lambda: group(
                            case,
                            consumers,
                            False,
                            args.topk,
                            candidate_blocks,
                            args.block_size,
                        ),
                        "candidate_premask": lambda: group(
                            case,
                            consumers,
                            True,
                            args.topk,
                            candidate_blocks,
                            args.block_size,
                        ),
                        "candidate_auto": lambda: group(
                            case,
                            consumers,
                            None,
                            args.topk,
                            candidate_blocks,
                            args.block_size,
                        ),
                    }
                for stage, functions in stages.items():
                    stage_metadata = dict(metadata)
                    if stage == "group":
                        stage_metadata.update(
                            pattern="source_selected",
                            consumers=args.consumers,
                            **fractions(group_mask, case["lens"]),
                        )
                    for execution in args.execution:
                        result = measure_pair(functions, execution == "graph", args)
                        speedup = (
                            result["dense_postmask"]["median_us"]
                            / result["candidate_premask"]["median_us"]
                        )
                        record = dict(
                            stage_metadata,
                            stage=stage,
                            execution=execution,
                            speedup=speedup,
                            auto_speedup=(
                                result["dense_postmask"]["median_us"]
                                / result["candidate_auto"]["median_us"]
                            ),
                            timing=result,
                        )
                        records.append(record)
                        print(json.dumps(record), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(record) + "\n" for record in records))


if __name__ == "__main__":
    main()
