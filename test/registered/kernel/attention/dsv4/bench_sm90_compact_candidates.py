"""Interleaved full-mask / compact-candidate comparison on Hopper.

Uses the production 2048 x 8 candidate budget and exact-by-score checks. Query
projection, RoPE, attention and serving are excluded. Consumers use identical
source-selected blocks, so cutoff ties cannot change the compared workload.
"""

import json
import os
from pathlib import Path

import torch
from bench_sm90_fp4_grouped_indexer import interleaved_times
from test_sm90_length_aware_indexer import TestSm90LengthAwareIndexer

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    CandidateMasks,
    published_masks,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=90, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def benchmark():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        print("requires an SM90 GPU")
        return
    output = os.getenv("SM90_COMPACT_BENCH_OUTPUT")
    if output and Path(output).exists():
        raise FileExistsError(output)
    seed = int(os.getenv("SM90_COMPACT_BENCH_SEED", "11"))
    rounds = int(os.getenv("SM90_COMPACT_BENCH_ROUNDS", "7"))
    target_ms = float(os.getenv("SM90_COMPACT_BENCH_MS", "60"))
    cases = os.getenv(
        "SM90_COMPACT_BENCH_CASES",
        "48:1048896:8192:1:source,48:1048896:8192:1:consumer,"
        "48:1048896:131072:1:source,48:1048896:131072:1:consumer",
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
            or not 0 < visible <= width
            or ratio not in (1, 2)
            or role not in ("source", "consumer")
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
            source = check.fixture.make_backend_case(x, "source", 512)
            source.layer.indexer.candidate_topk_blocks = 2048
            source.layer.indexer.candidate_block_size = 8
            with envs.SGLANG_OPT_DSV41_SM90_COMPACT_CANDIDATES.override(True):
                check.run_case(source)
            candidates = source.state.forward_metadata.candidate_metadata
            mask = published_masks(candidates).mask
        else:
            mask = None

        def run(compact):
            if role == "consumer":
                case.state.forward_metadata.candidate_metadata = (
                    candidates if compact else CandidateMasks(mask=mask)
                )
            with envs.SGLANG_OPT_DSV41_SM90_COMPACT_CANDIDATES.override(compact):
                check.run_case(case)

        for compact in (False, True):
            run(compact)
            check.check_outputs(x, case, role, mask)
        times = interleaved_times(
            {"stage2": lambda: run(False), "compact": lambda: run(True)},
            rounds,
            target_ms,
            batch=10,
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
