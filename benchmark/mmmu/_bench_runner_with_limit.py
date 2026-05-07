"""A thin wrapper around bench_sglang.py that truncates prepare_samples().

Deterministic mode: we simply keep the FIRST SAMPLE_LIMIT samples in the
original order returned by prepare_samples(). This guarantees every variant
evaluates the exact same subset for apples-to-apples comparison.
"""
import os
import sys

SAMPLE_LIMIT = int(os.environ.get("SAMPLE_LIMIT", "0") or 0)

import eval_utils  # noqa: E402

_orig_prepare_samples = eval_utils.prepare_samples


def _limited_prepare_samples(eval_args):
    samples = _orig_prepare_samples(eval_args)
    total = len(samples)
    if SAMPLE_LIMIT and total > SAMPLE_LIMIT:
        # NOTE: deterministic first-N slice, no shuffle.
        samples = samples[:SAMPLE_LIMIT]
        first_id = getattr(samples[0], "id", "?") if samples else "?"
        last_id = getattr(samples[-1], "id", "?") if samples else "?"
        print(
            f"[bench_runner] deterministic first-{SAMPLE_LIMIT} slice of {total} "
            f"(first_id={first_id}, last_id={last_id})"
        )
    else:
        print(f"[bench_runner] using all {total} samples")
    return samples


eval_utils.prepare_samples = _limited_prepare_samples

# 让 bench_sglang 当做 __main__ 来跑
import runpy  # noqa: E402
sys.argv = ["bench_sglang.py", *sys.argv[1:]]
runpy.run_path(os.path.join(os.path.dirname(__file__), "bench_sglang.py"),
               run_name="__main__")
