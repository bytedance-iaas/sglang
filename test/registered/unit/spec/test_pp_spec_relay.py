from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.pp_spec_relay import (
    PPSpecRelayInput,
    normalize_pp_spec_relay,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_pd_first_decode_normalizes_draft_input_to_degenerate_relay():
    draft = EagleDraftInput(
        bonus_tokens=torch.tensor([101, 202], dtype=torch.int64),
        topk_p=torch.ones((2, 1)),
        topk_index=torch.zeros((2, 1), dtype=torch.int64),
        hidden_states=torch.zeros((2, 4)),
    )

    relay = normalize_pp_spec_relay(
        draft, rids=["req-a", "req-b"], num_draft_tokens=3
    )

    assert relay.rids == ["req-a", "req-b"]
    torch.testing.assert_close(
        relay.tokens, torch.tensor([[101, 0, 0], [202, 0, 0]])
    )
    assert relay.parents is None
    assert relay.top_scores is None


def test_existing_relay_is_reindexed_to_live_request_order():
    relay = PPSpecRelayInput(
        rids=["req-a", "req-b"],
        tokens=torch.tensor([[101, 11], [202, 22]]),
    )

    normalized = normalize_pp_spec_relay(
        relay, rids=["req-b", "req-a"], num_draft_tokens=2
    )

    assert normalized.rids == ["req-b", "req-a"]
    torch.testing.assert_close(normalized.tokens, torch.tensor([[202, 22], [101, 11]]))


def test_first_decode_rejects_missing_or_misaligned_bonus_tokens():
    with pytest.raises(RuntimeError, match="bonus-token rows do not match"):
        normalize_pp_spec_relay(
            EagleDraftInput(bonus_tokens=torch.tensor([101])),
            rids=["req-a", "req-b"],
            num_draft_tokens=2,
        )

    with pytest.raises(TypeError, match="requires PPSpecRelayInput"):
        normalize_pp_spec_relay(
            SimpleNamespace(), rids=["req-a"], num_draft_tokens=2
        )


def test_running_relay_merges_a_newly_normalized_pd_decode_batch():
    running = PPSpecRelayInput(
        rids=["running"],
        tokens=torch.tensor([[101, 11]], dtype=torch.int64),
        parents=torch.tensor([[-1]], dtype=torch.int64),
        top_scores=torch.tensor([[0]], dtype=torch.int64),
    )
    admitted = normalize_pp_spec_relay(
        EagleDraftInput(bonus_tokens=torch.tensor([202], dtype=torch.int64)),
        rids=["admitted"],
        num_draft_tokens=2,
    )

    running.merge_batch(admitted)

    assert running.rids == ["running", "admitted"]
    torch.testing.assert_close(
        running.tokens, torch.tensor([[101, 11], [202, 0]], dtype=torch.int64)
    )
    torch.testing.assert_close(
        running.parents, torch.tensor([[-1], [-1]], dtype=torch.int64)
    )
    torch.testing.assert_close(
        running.top_scores, torch.tensor([[0], [0]], dtype=torch.int64)
    )
