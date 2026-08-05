import random
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
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
            can_run_dp_cuda_graph=True,
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
        self.assertTrue(forward_batch.can_run_dp_cuda_graph)

    def test_target_verify_scales_request_counts_to_verify_tokens(self):
        forward_batch = ForwardBatch.__new__(ForwardBatch)
        forward_batch.spec_info = DFlashVerifyInput(
            draft_token=torch.empty(0, dtype=torch.int64),
            positions=torch.empty(0, dtype=torch.int64),
            draft_token_num=8,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        batch = SimpleNamespace(
            global_num_tokens=[1, 3, 0, 2],
            global_num_tokens_for_logprob=[1, 3, 0, 2],
            can_run_dp_cuda_graph=True,
        )

        forward_batch.init_mlp_sync_metadata(batch, device="cpu")

        self.assertEqual(forward_batch.original_global_num_tokens_cpu, [1, 3, 0, 2])
        self.assertEqual(forward_batch.global_num_tokens_cpu, [8, 24, 0, 16])
        self.assertEqual(
            forward_batch.global_num_tokens_for_logprob_cpu, [8, 24, 0, 16]
        )
        self.assertTrue(forward_batch.can_run_dp_cuda_graph)


class TestDenseDraftBackendIsolation(CustomTestCase):
    def test_dense_dp_draft_enters_tp_and_speculative_backend_contexts(self):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._draft_dp_context_enabled = True
        worker._draft_is_moe = False
        entered = []

        def recording_context(name):
            @contextmanager
            def context():
                entered.append(f"enter:{name}")
                try:
                    yield
                finally:
                    entered.append(f"exit:{name}")

            return context()

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_parallel",
            return_value=SimpleNamespace(attn_tp_group="attn-tp"),
        ), patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2.draft_tp_context",
            side_effect=lambda group: recording_context(f"tp:{group}"),
        ), patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2.speculative_moe_backend_context",
            side_effect=lambda: recording_context("moe"),
        ), patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2.speculative_moe_a2a_backend_context",
            side_effect=lambda: recording_context("a2a"),
        ):
            with worker._draft_context():
                self.assertEqual(
                    entered, ["enter:tp:attn-tp", "enter:moe", "enter:a2a"]
                )

        self.assertEqual(
            entered,
            [
                "enter:tp:attn-tp",
                "enter:moe",
                "enter:a2a",
                "exit:a2a",
                "exit:moe",
                "exit:tp:attn-tp",
            ],
        )


class TestPartialDpPrefillIdleNormalization(CustomTestCase):
    def test_empty_local_prefill_uses_idle_target_forward(self):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._target_worker = MagicMock()
        worker.device = "cpu"
        worker.verify_num_draft_tokens = 8

        batch = MagicMock()
        batch.batch_size.return_value = 0
        batch.forward_mode.is_idle.return_value = False
        on_publish = MagicMock()

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_parallel",
            return_value=SimpleNamespace(enable_dp_attention=True),
        ):
            result = worker._forward_prefill(batch, on_publish)

        batch.prepare_for_idle.assert_called_once_with()
        worker.target_worker.forward_batch_generation.assert_called_once_with(
            batch, capture_hidden_mode=CaptureHiddenMode.FULL
        )
        self.assertEqual(result.next_token_ids.numel(), 0)
        self.assertEqual(result.new_seq_lens.numel(), 0)
        on_publish.assert_called_once()

    def test_logits_restore_idle_semantics_after_dp_prefill_padding(self):
        forward_batch = MagicMock()
        forward_batch.forward_mode = ForwardMode.EXTEND
        forward_batch._original_forward_mode = ForwardMode.IDLE
        forward_batch.return_logprob = False
        forward_batch.spec_info = None

        metadata = LogitsMetadata.from_forward_batch(forward_batch)

        self.assertEqual(metadata.forward_mode, ForwardMode.IDLE)


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
