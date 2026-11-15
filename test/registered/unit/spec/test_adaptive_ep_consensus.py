import unittest

from sglang.srt.speculative.adaptive_ep_consensus import (
    EpAdaptiveTransitionGate,
    build_ep_adaptive_observation,
    merge_ep_adaptive_observations,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAdaptiveEpConsensus(unittest.TestCase):
    def observation(
        self,
        rank,
        *,
        iteration=0,
        steps=3,
        accepted=(),
        proposed=0,
        committed=0,
        kv=1000,
    ):
        return build_ep_adaptive_observation(
            participant_id=rank,
            iteration=iteration,
            current_steps=steps,
            correct_drafts_per_request=accepted,
            proposed_draft_tokens=proposed,
            committed_tokens=committed,
            kv_headroom_tokens=kv,
            maximum_steps=3,
        )

    def test_all_eight_participants_include_idle_ranks(self):
        values = [
            self.observation(
                rank,
                accepted=(3, 2) if rank < 3 else (),
                proposed=6 if rank < 3 else 0,
                committed=7 if rank < 3 else 0,
                kv=1000 - rank,
            )
            for rank in range(8)
        ]
        merged = merge_ep_adaptive_observations(
            values,
            expected_participant_ids=range(8),
        )
        self.assertEqual(merged.participant_ids, tuple(range(8)))
        self.assertEqual(merged.active_participant_count, 3)
        self.assertEqual(merged.active_requests, 6)
        self.assertEqual(merged.correct_drafts_per_request, (2, 2, 2, 3, 3, 3))
        self.assertEqual(merged.acceptance_rate, 15 / 18)
        self.assertEqual(merged.committed_tokens, 21)
        self.assertEqual(merged.minimum_kv_headroom_tokens, 993)

    def test_missing_idle_rank_or_divergent_iteration_fails_closed(self):
        complete = [self.observation(rank) for rank in range(8)]
        with self.assertRaisesRegex(ValueError, "participant mismatch"):
            merge_ep_adaptive_observations(
                complete[:-1], expected_participant_ids=range(8)
            )
        complete[-1] = self.observation(7, iteration=1)
        with self.assertRaisesRegex(ValueError, "different adaptive iterations"):
            merge_ep_adaptive_observations(complete, expected_participant_ids=range(8))

    def test_divergent_current_tier_fails_before_decision(self):
        values = [self.observation(rank) for rank in range(8)]
        values[4] = self.observation(4, steps=1)
        with self.assertRaisesRegex(ValueError, "different adaptive tiers"):
            merge_ep_adaptive_observations(values, expected_participant_ids=range(8))

    def test_transition_is_registered_ordered_and_idempotent(self):
        merged0 = merge_ep_adaptive_observations(
            [self.observation(rank) for rank in range(8)],
            expected_participant_ids=range(8),
        )
        gate = EpAdaptiveTransitionGate(initial_steps=3)
        down = gate.apply(merged0, 1)
        self.assertEqual((down.from_steps, down.to_steps), (3, 1))
        self.assertIs(gate.apply(merged0, 1), down)
        with self.assertRaisesRegex(ValueError, "replay diverged"):
            gate.apply(merged0, 3)

        merged1 = merge_ep_adaptive_observations(
            [self.observation(rank, iteration=1, steps=1) for rank in range(8)],
            expected_participant_ids=range(8),
        )
        up = gate.apply(merged1, 3)
        self.assertEqual((up.from_steps, up.to_steps), (1, 3))
        with self.assertRaisesRegex(ValueError, "not pre-built"):
            gate.apply(merged1, 2)

    def test_skipped_iteration_and_bad_histogram_fail_closed(self):
        gate = EpAdaptiveTransitionGate(initial_steps=3)
        merged = merge_ep_adaptive_observations(
            [self.observation(rank, iteration=1) for rank in range(8)],
            expected_participant_ids=range(8),
        )
        with self.assertRaisesRegex(ValueError, "stale or skipped"):
            gate.apply(merged, 3)
        with self.assertRaisesRegex(ValueError, "accepted drafts exceed"):
            self.observation(0, accepted=(3,), proposed=2)


if __name__ == "__main__":
    unittest.main()
