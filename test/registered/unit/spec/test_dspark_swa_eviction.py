import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2


def _make_active_batch():
    req = SimpleNamespace(
        rid="req-0",
        decode_batch_idx=0,
        output_ids=[],
        origin_input_ids=[1],
        kv_committed_len=0,
        kv_allocated_len=12,
        sampling_params=SimpleNamespace(top_k=1),
    )
    batch = MagicMock()
    batch.batch_size.return_value = 1
    batch.reqs = [req]
    batch.device = torch.device("cpu")
    batch.sampling_info.penalizer_orchestrator.is_required = False
    batch.req_to_token_pool.req_to_token = torch.zeros(
        (1, 64), dtype=torch.int64
    )
    batch.req_pool_indices = torch.zeros((1,), dtype=torch.int64)
    batch.token_to_kv_pool_allocator.page_size = 1
    return batch, req


class TestDSparkSwaEviction(unittest.TestCase):
    def test_active_batch_evicts_before_ticking(self):
        draft_input = DFlashDraftInputV2.create_idle_input(torch.device("cpu"))
        batch, req = _make_active_batch()
        seen_decode_indices = []
        batch.maybe_evict_swa.side_effect = lambda: seen_decode_indices.append(
            req.decode_batch_idx
        )

        with patch(
            "sglang.srt.speculative.dflash_info_v2.get_global_server_args",
            return_value=SimpleNamespace(speculative_num_draft_tokens=6),
        ):
            draft_input.prepare_for_decode(batch)
            draft_input.prepare_for_decode(batch)

        self.assertEqual(seen_decode_indices, [0, 1])
        self.assertEqual(req.decode_batch_idx, 2)
        self.assertEqual(batch.maybe_evict_swa.call_count, 2)

    def test_idle_batch_does_not_advance_swa_clock(self):
        draft_input = DFlashDraftInputV2.create_idle_input(torch.device("cpu"))
        batch = MagicMock()
        batch.batch_size.return_value = 0
        batch.reqs = []

        draft_input.prepare_for_decode(batch)

        batch.maybe_evict_swa.assert_not_called()


if __name__ == "__main__":
    unittest.main()
