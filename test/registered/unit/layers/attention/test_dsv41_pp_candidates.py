"""Candidates survive PP transfer while physical pages remain stage-local."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.dsv4.candidate_indexer import CandidateMasks
from sglang.srt.layers.attention.dsv4.pp import (
    export_candidate_metadata,
    restore_candidate_metadata,
)
from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import BlockIds
from sglang.srt.layers.attention.dsv4.v41_indexer.sparse_table import (
    _SparsePrefillTable,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPPIndexCandidates(CustomTestCase):
    def test_prefill_candidates_keep_empty_requests_and_tail_rows(self):
        candidates = [
            CandidateMasks(
                request_masks=[
                    torch.empty(0, 0, dtype=torch.bool),
                    torch.tensor([[True, False], [False, True]]),
                ]
            ),
            BlockIds(
                blocks=torch.tensor([[0], [1]], dtype=torch.int32),
                rows_per_request=[0, 2],
            ),
        ]
        for candidate in candidates:
            with self.subTest(kind=type(candidate).__name__):
                tensors = {}
                export_candidate_metadata(tensors, candidate, 1)
                received = restore_candidate_metadata(tensors, None, None)
                tail = received.tail([0, 1])
                if isinstance(candidate, CandidateMasks):
                    self.assertEqual(tail.request_masks[0].shape, (0, 0))
                    torch.testing.assert_close(
                        tail.request_masks[1], candidate.request_masks[1][-1:]
                    )
                else:
                    self.assertEqual(tail.rows_per_request, [0, 1])
                    torch.testing.assert_close(tail.blocks, candidate.blocks[-1:])

    def test_sparse_prefill_rebuilds_schedule_using_destination_pages(self):
        source = _SparsePrefillTable(
            blocks=torch.tensor([[0], [1]], dtype=torch.int32),
            schedule=torch.zeros(1, dtype=torch.uint8),
            phys_blocks=torch.tensor([[16], [17]], dtype=torch.int32),
            valid_lens=torch.tensor([8, 8], dtype=torch.int32),
            ready=Mock(),
            compress_lens=torch.tensor([8, 16], dtype=torch.int32),
            page_table=torch.tensor([[2], [2]], dtype=torch.int32),
            request_ids=torch.tensor([0, 0], dtype=torch.int32),
            q_dtype=torch.int8,
            page_size=64,
            rows_per_request=[0, 2],
        )
        tensors = {}
        with patch.object(torch.cuda, "current_stream", return_value=Mock()):
            export_candidate_metadata(tensors, source, 2)
        destination = SimpleNamespace(
            core_metadata=SimpleNamespace(
                page_size=256,
                page_table=torch.tensor([[9], [9]], dtype=torch.int32),
            )
        )
        with patch(
            "sglang.srt.layers.attention.dsv4.v41_indexer.sparse_table._build_prefill_table",
            side_effect=lambda **kwargs: kwargs,
        ):
            restored = restore_candidate_metadata(tensors, destination, None)
        torch.testing.assert_close(
            restored["page_table"],
            torch.tensor([[18, 19], [18, 19]], dtype=torch.int32),
        )
        torch.testing.assert_close(restored["blocks"], source.blocks)
        self.assertEqual(restored["rows_per_request"], [0, 2])
        self.assertNotIn("pp_candidate_schedule", tensors)


if __name__ == "__main__":
    unittest.main()
