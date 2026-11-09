import random
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    _localize_dp_hidden,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    dp_global_verify_tier_num_tokens,
    local_verify_tier_num_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestLocalVerifyTierNumTokens(CustomTestCase):
    def test_no_budget_returns_sentinel(self):
        self.assertEqual(
            local_verify_tier_num_tokens(
                bs=8,
                verify_token_budget=None,
                verify_num_draft_tokens=6,
                min_verify_len=1,
            ),
            -1,
        )

    def test_budget_adds_to_anchor_floor(self):
        self.assertEqual(
            local_verify_tier_num_tokens(
                bs=8,
                verify_token_budget=10,
                verify_num_draft_tokens=6,
                min_verify_len=1,
            ),
            18,
        )

    # Clamp/floor variants (verify-all clamp, min_verify_len floor, min=0) are
    # covered by the TestBusyIdleGraphKeyIdentity sweep bounds.


class TestDpGlobalVerifyTierNumTokens(CustomTestCase):
    def test_any_sentinel_pins_everyone(self):
        # The sweep never emits a -1 contribution, so this is the only guard
        # on "any rank without a budget pins everyone"; losing it forks graph
        # keys across DP ranks.
        self.assertIsNone(
            dp_global_verify_tier_num_tokens(global_tier_num_tokens=[100, -1, 50, 0])
        )


class TestDraftDpSyncMetadata(CustomTestCase):
    def test_localizes_global_hidden_rows_by_dp_rank(self):
        hidden = torch.arange(10).view(10, 1)

        local = _localize_dp_hidden(
            hidden,
            local_bs=1,
            rows_per_req=5,
            global_num_reqs=[1, 1],
            dp_rank=1,
        )

        self.assertEqual(local.view(-1).tolist(), [5, 6, 7, 8, 9])

    def test_localizes_max_len_padded_hidden_rows(self):
        hidden = torch.arange(10).view(10, 1)

        local = _localize_dp_hidden(
            hidden,
            local_bs=1,
            rows_per_req=5,
            global_num_reqs=[1, 2],
            dp_rank=0,
        )

        self.assertEqual(local.view(-1).tolist(), [0, 1, 2, 3, 4])

    def test_preserves_unscaled_request_counts_for_cuda_graph_admission(self):
        proposer = DraftBlockProposer.__new__(DraftBlockProposer)
        proposer._dp_moe_sync = True
        proposer._draft_block_spec_info = SimpleNamespace(
            num_tokens_per_req=6,
            num_tokens_for_logprob_per_req=1,
        )
        proposer.draft_model_runner = SimpleNamespace(device="cpu")

        forward_batch = SimpleNamespace(input_ids=torch.arange(6))
        batch = SimpleNamespace(
            global_num_tokens=[1, 3, 0, 2],
            global_num_tokens_for_logprob=[1, 3, 0, 2],
            can_run_decode_cuda_graph=True,
        )

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_draft.enable_num_token_non_padded",
            return_value=True,
        ):
            proposer._fill_dp_moe_sync_metadata(forward_batch, batch)

        self.assertEqual(
            forward_batch.original_global_num_tokens_cpu,
            [1, 3, 0, 2],
        )
        self.assertEqual(forward_batch.global_num_tokens_cpu, [6, 18, 0, 12])
        self.assertEqual(forward_batch.num_token_non_padded.item(), 6)
        self.assertEqual(forward_batch.num_token_non_padded.dtype, torch.int32)
        self.assertEqual(forward_batch.num_token_non_padded_cpu, 6)
        self.assertTrue(forward_batch.can_run_decode_cuda_graph)

    def _make_idle_proposer(self, graph_runner=None):
        proposer = DraftBlockProposer.__new__(DraftBlockProposer)
        proposer._dp_moe_sync = True
        proposer.query_token_num = 6
        proposer._draft_block_spec_info = SimpleNamespace(
            num_tokens_per_req=6,
            num_tokens_for_logprob_per_req=1,
        )
        forward = Mock()
        proposer.draft_model_runner = SimpleNamespace(
            device="cpu",
            forward=forward,
            decode_cuda_graph_runner=graph_runner,
        )
        return proposer, forward

    def test_idle_participation_uses_one_shape_only_row_for_eager(self):
        proposer, forward = self._make_idle_proposer()
        batch = SimpleNamespace(
            global_num_tokens=[0, 2],
            global_num_tokens_for_logprob=[0, 2],
            can_run_decode_cuda_graph=False,
        )

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_draft.enable_num_token_non_padded",
            return_value=True,
        ):
            proposer.run_idle_participation(batch)

        idle_batch = forward.call_args.args[0]
        self.assertEqual(idle_batch.batch_size, 1)
        self.assertEqual(idle_batch.input_ids.numel(), 1)
        self.assertEqual(idle_batch.positions.numel(), 1)
        self.assertEqual(idle_batch.out_cache_loc.numel(), 1)
        self.assertEqual(idle_batch.original_global_num_tokens_cpu, [0, 2])
        self.assertEqual(idle_batch.num_token_non_padded.item(), 0)
        self.assertEqual(idle_batch.num_token_non_padded_cpu, 0)
        self.assertEqual(idle_batch.global_num_tokens_cpu, [0, 12])
        self.assertFalse(idle_batch.can_run_decode_cuda_graph)

    def test_idle_participation_uses_captured_width_for_graph(self):
        graph_runner = Mock()
        graph_runner.can_run_graph.return_value = True
        proposer, forward = self._make_idle_proposer(graph_runner)
        batch = SimpleNamespace(
            global_num_tokens=[0, 2],
            global_num_tokens_for_logprob=[0, 2],
            can_run_decode_cuda_graph=True,
        )

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_draft.enable_num_token_non_padded",
            return_value=True,
        ):
            proposer.run_idle_participation(batch)

        idle_batch = forward.call_args.args[0]
        graph_runner.can_run_graph.assert_called_once_with(idle_batch)
        self.assertEqual(idle_batch.input_ids.numel(), 6)
        self.assertEqual(idle_batch.positions.numel(), 6)
        self.assertEqual(idle_batch.out_cache_loc.numel(), 6)
        self.assertEqual(idle_batch.original_global_num_tokens_cpu, [0, 2])
        self.assertEqual(idle_batch.num_token_non_padded.item(), 0)
        self.assertEqual(idle_batch.num_token_non_padded_cpu, 0)
        self.assertTrue(idle_batch.can_run_decode_cuda_graph)


class TestBusyIdleGraphKeyIdentity(CustomTestCase):
    def test_busy_and_idle_floors_agree_on_random_topologies(self):
        rng = random.Random(20260703)
        for _ in range(2000):
            verify_num_draft_tokens = rng.randint(2, 8)
            min_verify_len = rng.randint(0, verify_num_draft_tokens - 1)
            effective_min = max(min_verify_len, 1)
            num_ranks = rng.randint(1, 8)
            contributions = []
            num_reqs_per_rank = []
            for _ in range(num_ranks):
                if rng.random() < 0.3:
                    num_reqs_per_rank.append(0)
                    contributions.append(0)
                    continue
                bs = rng.randint(1, 512)
                budget = rng.randint(0, bs * verify_num_draft_tokens)
                num_reqs_per_rank.append(bs)
                contributions.append(
                    local_verify_tier_num_tokens(
                        bs=bs,
                        verify_token_budget=budget,
                        verify_num_draft_tokens=verify_num_draft_tokens,
                        min_verify_len=min_verify_len,
                    )
                )
            tier_num_tokens = dp_global_verify_tier_num_tokens(
                global_tier_num_tokens=contributions
            )
            global_num_reqs = max(num_reqs_per_rank)
            if tier_num_tokens is None:
                self.assertEqual(global_num_reqs, 0)
                continue

            self.assertGreaterEqual(tier_num_tokens, global_num_reqs * effective_min)
            self.assertLessEqual(
                tier_num_tokens, global_num_reqs * verify_num_draft_tokens
            )

            busy_floor = min(tier_num_tokens, global_num_reqs * verify_num_draft_tokens)
            self.assertEqual(busy_floor, tier_num_tokens)

            idle_lens_total = global_num_reqs
            idle_bucket_input = max(idle_lens_total, tier_num_tokens)
            self.assertEqual(idle_bucket_input, tier_num_tokens)


if __name__ == "__main__":
    unittest.main()
