import unittest

import torch

from sglang.srt.speculative.eagle_worker_v2 import (
    _slice_draft_output_to_local_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestEagleDraftDPPadding(unittest.TestCase):
    def test_discards_dp_padding_rows(self):
        logits = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        positions = torch.tensor([7, 100, 100])

        local_logits, local_hidden_states, local_positions = (
            _slice_draft_output_to_local_tokens(
                logits, hidden_states, positions, num_local_tokens=1
            )
        )

        self.assertEqual(local_logits.shape, (1, 8))
        self.assertEqual(local_hidden_states.shape, (1, 4))
        self.assertEqual(local_positions.tolist(), [7])
        local_positions.add_(1)
        self.assertEqual(positions.tolist(), [8, 100, 100])

    def test_idle_rank_discards_all_dp_padding_rows(self):
        local_logits, local_hidden_states, local_positions = (
            _slice_draft_output_to_local_tokens(
                torch.empty((2, 8)),
                torch.empty((2, 4)),
                torch.tensor([100, 100]),
                num_local_tokens=0,
            )
        )

        self.assertEqual(local_logits.shape, (0, 8))
        self.assertEqual(local_hidden_states.shape, (0, 4))
        self.assertEqual(local_positions.shape, (0,))

    def test_rejects_missing_local_rows(self):
        with self.assertRaisesRegex(RuntimeError, "next_token_logits has 0 rows"):
            _slice_draft_output_to_local_tokens(
                torch.empty((0, 8)),
                torch.empty((1, 4)),
                torch.tensor([7]),
                num_local_tokens=1,
            )


if __name__ == "__main__":
    unittest.main()
