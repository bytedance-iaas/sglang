"""
Usage:
python3 -m unittest test_pp_spec.TestPPSpecConsistency.test_pp_matches_non_pp
"""

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=900, stage="extra-b", runner_config="4-gpu-h100")

# topk=1 chains and a topk=2 tree: the relayed topology is constant for the
# former and data-dependent for the latter, so both shapes are covered.
SPEC_SHAPES = {
    "chain": ("2", "1", "3"),  # num_steps, eagle_topk, num_draft_tokens
    "tree": ("2", "2", "4"),
}


class TestPPSpecConsistency(CustomTestCase):
    """PP x speculative decoding must match plain speculative decoding.

    Every PP stage rebuilds the verify tree from state relayed by the last
    stage, so a mis-sized proxy buffer or a mis-rebuilt tree shows up as an
    accuracy drop or as speculation that never gets accepted -- not as a
    crash. Pin both.
    """

    def _run(self, pp_size: int, shape: str):
        num_steps, topk, num_draft_tokens = SPEC_SHAPES[shape]
        other_args = [
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model-path",
            DEFAULT_DRAFT_MODEL_EAGLE,
            "--speculative-num-steps",
            num_steps,
            "--speculative-eagle-topk",
            topk,
            "--speculative-num-draft-tokens",
            num_draft_tokens,
            "--mem-fraction-static",
            "0.7",
        ]
        if pp_size > 1:
            other_args += ["--pp-size", str(pp_size), "--disable-overlap-schedule"]

        process = popen_launch_server(
            DEFAULT_TARGET_MODEL_EAGLE,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={**os.environ, "SGLANG_ENABLE_PP_SPEC": "1"},
        )
        try:
            metrics = run_eval(
                SimpleNamespace(
                    base_url=DEFAULT_URL_FOR_TEST,
                    model=DEFAULT_TARGET_MODEL_EAGLE,
                    eval_name="gsm8k",
                    api="completion",
                    max_tokens=512,
                    num_examples=256,
                    num_threads=32,
                )
            )
            server_info = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info")
            accept_length = server_info.json()["internal_states"][0][
                "avg_spec_accept_length"
            ]
        finally:
            kill_process_tree(process.pid)
        return metrics["score"], accept_length

    def _assert_matches(self, shape: str):
        base_score, base_accept = self._run(pp_size=1, shape=shape)
        pp_score, pp_accept = self._run(pp_size=2, shape=shape)
        print(
            f"[PP spec {shape}] no-PP: score={base_score:.4f} accept={base_accept:.2f}"
            f" | PP2: score={pp_score:.4f} accept={pp_accept:.2f}"
        )

        self.assertGreaterEqual(
            pp_score,
            base_score - 0.02,
            msg=(
                f"PP+spec accuracy dropped more than 2% against plain spec. "
                f"no-PP: {base_score:.2%}, PP2: {pp_score:.2%}"
            ),
        )
        # A relay that loses the drafted tree still produces correct output --
        # the bonus token is force-accepted and the rest is rejected -- so
        # accuracy alone cannot tell drafting from a no-op. Acceptance can.
        self.assertGreaterEqual(
            pp_accept,
            base_accept - 0.2,
            msg=(
                f"PP+spec accept length collapsed against plain spec. "
                f"no-PP: {base_accept:.2f}, PP2: {pp_accept:.2f}"
            ),
        )

    def test_pp_matches_non_pp(self):
        self._assert_matches("chain")

    def test_pp_matches_non_pp_tree(self):
        self._assert_matches("tree")


class TestPPSpecGate(CustomTestCase):
    """The gate is off by default, and the combinations the relay cannot
    reproduce identically on every stage are rejected rather than silently
    mis-rebuilt."""

    def _check_pipeline_args(self, **overrides):
        from sglang.srt.arg_groups.validation_hook import check_pipeline_parallelism

        # Exercise the PP gate directly. Full ServerArgs resolution performs
        # model inspection and is unrelated to these pure validation cases.
        args = dict(
            pp_size=2,
            disable_overlap_schedule=True,
            speculative_algorithm="EAGLE",
            disaggregation_mode="null",
            enable_multi_layer_eagle=False,
            speculative_adaptive=False,
            min_free_slots_delay=None,
        )
        args.update(overrides)
        check_pipeline_parallelism(SimpleNamespace(**args))

    def test_gate_off_keeps_the_ban(self):
        os.environ.pop("SGLANG_ENABLE_PP_SPEC", None)
        with self.assertRaises(AssertionError):
            self._check_pipeline_args()

    def test_gate_on_allows_decode_dpa_and_rejects_unsupported_combinations(self):
        os.environ["SGLANG_ENABLE_PP_SPEC"] = "1"
        try:
            self._check_pipeline_args(disaggregation_mode="decode")
            self._check_pipeline_args(
                disaggregation_mode="decode",
            )
            # Adaptive spec changes num_draft_tokens at runtime, which the
            # relay slices results with.
            with self.assertRaises(AssertionError):
                self._check_pipeline_args(speculative_adaptive=True)
            # The relay carries an EAGLE-shaped tree; other algorithms would
            # be mis-rebuilt on the non-last stages.
            with self.assertRaises(AssertionError):
                self._check_pipeline_args(speculative_algorithm="NGRAM")
            # Prefill keeps its existing rank-uniform EAGLE path.
            self._check_pipeline_args(disaggregation_mode="prefill")
        finally:
            os.environ.pop("SGLANG_ENABLE_PP_SPEC", None)


if __name__ == "__main__":
    unittest.main()
