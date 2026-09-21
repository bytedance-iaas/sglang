"""Interleaved default / mapped / prefix indexer A/B on Hopper.

SM90_PREFIX_BENCH_CASES: rows:capacity:visible:ratio:role, comma separated.
Uses production candidate settings (2048 blocks x 8 positions). Timings include
candidate selection/consumption but exclude projection, RoPE and serving.
"""

import json
import os
from pathlib import Path

import torch
from bench_sm90_fp4_grouped_indexer import interleaved_times
from test_sm90_length_aware_indexer import TestSm90LengthAwareIndexer

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.candidate_indexer import select_candidate_blocks
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=90, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def benchmark():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        print("requires an SM90 GPU")
        return
    output = os.getenv("SM90_PREFIX_BENCH_OUTPUT")
    if output and Path(output).exists():
        raise FileExistsError(output)
    seed = int(os.getenv("SM90_PREFIX_BENCH_SEED", "11"))
    rounds = int(os.getenv("SM90_PREFIX_BENCH_ROUNDS", "9"))
    target_ms = float(os.getenv("SM90_PREFIX_BENCH_MS", "100"))
    cases = os.getenv(
        "SM90_PREFIX_BENCH_CASES",
        "12:524480:4096:2:none,48:1048896:8192:1:source,48:1048896:8192:1:consumer",
    ).split(",")
    check = TestSm90LengthAwareIndexer("test_backend_roles_and_layouts")
    check.setUp()
    records = []
    for shape in cases:
        rows, width, visible, ratio, role = shape.split(":")
        rows, width, visible, ratio = map(int, (rows, width, visible, ratio))
        if (
            rows <= 0
            or rows % 6
            or not 0 <= visible <= width
            or role not in ("none", "source", "consumer")
            or ratio not in (1, 2)
        ):
            raise ValueError(shape)
        x = check.fixture.make_inputs(
            rows=rows, width=width, heads=32, ratio=ratio, seed=seed
        )
        x.lens.copy_((visible - torch.arange(rows, device="cuda") % 6).clamp_min(0))
        case = check.fixture.make_backend_case(x, role, 512)
        case.layer.indexer.candidate_topk_blocks = 2048
        case.layer.indexer.candidate_block_size = 8
        if role == "consumer":
            mask = select_candidate_blocks(
                check.fixture.reference(x), x.lens[:, None], 2048, 8
            )
            case.state.forward_metadata.candidate_metadata.mask = mask
        else:
            mask = None

        def run(stage):
            with (
                envs.SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER.override(stage >= 1),
                envs.SGLANG_OPT_DSV41_SM90_LENGTH_AWARE_INDEXER.override(stage == 2),
            ):
                case.run()

        for stage in range(3):
            run(stage)
            check.check_outputs(x, case, role, mask)
        times = interleaved_times(
            {
                "default": lambda: run(0),
                "mapped": lambda: run(1),
                "prefix": lambda: run(2),
            },
            rounds,
            target_ms,
        )
        record = {
            "case": shape,
            "seed": seed,
            "correct_by_score": True,
            "gpu": torch.cuda.get_device_name(),
            "rounds": rounds,
            "target_ms": target_ms,
            "times": times,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        if output:
            Path(output).write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    benchmark()
