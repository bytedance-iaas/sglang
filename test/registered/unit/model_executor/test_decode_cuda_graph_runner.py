"""CPU contracts for decode CUDA-graph admission decisions."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDecodeCudaGraphRunnerLongContextGuard(unittest.TestCase):
    """Exercise ``can_run_graph`` without constructing or capturing a graph."""

    @staticmethod
    def _build_runner(*, disable_graph_max_seq_len: int):
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        # The production constructor owns mapping
        # SGLANG_DISABLE_CUDA_GRAPH_MAX_SEQ_LEN to this threshold.  This
        # white-box test isolates whether can_run_graph consumes it correctly.
        runner.disable_graph_max_seq_len = disable_graph_max_seq_len
        runner.require_mlp_tp_gather = False
        runner.disable_padding = False
        runner.max_bs = 4
        runner.enable_pdmux = False
        runner.require_mlp_sync = False
        runner.is_encoder_decoder = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.enable_two_batch_overlap = False
        runner.model_runner = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_ngram=lambda: False)
        )
        return runner

    @staticmethod
    def _build_forward_batch(*, sequence_depth: int):
        logical_depth = torch.tensor([sequence_depth], dtype=torch.int32)
        return SimpleNamespace(
            replace_embeds=None,
            batch_size=1,
            seq_lens_cpu=logical_depth,
            seq_lens=logical_depth.clone(),
            orig_seq_lens=logical_depth.clone(),
            # A KV slot is a physical address, not a logical sequence length.
            # Make this deliberately large so the guard must prefer the three
            # explicit runtime sequence-depth fields above.
            out_cache_loc=torch.tensor([999_999], dtype=torch.int64),
            capture_hidden_mode=CaptureHiddenMode.NULL,
            spec_info=SimpleNamespace(capture_hidden_mode=None),
            input_ids=torch.tensor([1], dtype=torch.int64),
        )

    def test_can_run_graph_rejects_only_depth_above_configured_limit(self):
        cases = (
            # An unset/disabled threshold preserves the existing graph path.
            (0, 1_000_000, True),
            # The limit itself is still supported; only greater depth exits.
            (128, 128, True),
            (128, 129, False),
        )

        for threshold, sequence_depth, expected in cases:
            with self.subTest(threshold=threshold, sequence_depth=sequence_depth):
                runner = self._build_runner(
                    disable_graph_max_seq_len=threshold
                )
                forward_batch = self._build_forward_batch(
                    sequence_depth=sequence_depth
                )

                self.assertIs(runner.can_run_graph(forward_batch), expected)


if __name__ == "__main__":
    unittest.main()
