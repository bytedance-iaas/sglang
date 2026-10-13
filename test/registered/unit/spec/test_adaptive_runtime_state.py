import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _make_state(steps: int) -> SpecRuntimeState:
    return SpecRuntimeState(
        speculative_num_steps=steps,
        speculative_num_draft_tokens=steps + 1,
        draft_attn_backend=None,
        cuda_graph_runner=None,
        target_attn_backend=None,
        target_graph_runner=None,
        draft_extend_attn_backend=None,
        cuda_graph_runner_for_draft_extend=None,
    )


class _FakeWorker:
    def __init__(self, initial_steps: int = 3):
        self.speculative_num_steps = initial_steps
        self.build_calls = []
        self.applied_steps = []

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ) -> SpecRuntimeState:
        self.build_calls.append(
            (speculative_num_steps, speculative_num_draft_tokens, cuda_graph_bs)
        )
        return _make_state(speculative_num_steps)

    def apply_runtime_state(self, state: SpecRuntimeState) -> None:
        self.speculative_num_steps = state.speculative_num_steps
        self.applied_steps.append(state.speculative_num_steps)


class _FakePolicy:
    def __init__(self):
        self.candidate_steps = [1, 3]
        self.cuda_graph_bs = None
        self.feedback_step = None
        self.aggregate_calls = []
        self.aggregate_selected_step = None

    def set_cuda_graph_bs(self, cuda_graph_bs: list[int] | None) -> None:
        self.cuda_graph_bs = cuda_graph_bs

    def initialize_synchronized_steps(self, steps: int) -> None:
        self.aggregate_selected_step = steps

    def get_steps_for_batch(self, batch_size: int) -> int:
        return 1 if batch_size >= 8 else 3

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> int | None:
        return self.feedback_step

    def observe_verify_complete_aggregates(
        self, observations: list[tuple[int, int, int, int]]
    ) -> None:
        self.aggregate_calls.append(list(observations))
        total_correct_drafts = sum(obs[0] for obs in observations)
        num_requests = sum(obs[1] for obs in observations)
        self.aggregate_selected_step = (
            1 if total_correct_drafts / num_requests < 1 else 3
        )

    def get_compatible_steps_for_batches(
        self, batch_sizes: list[int]
    ) -> int | None:
        if self.aggregate_selected_step is not None:
            return self.aggregate_selected_step
        selected = {self.get_steps_for_batch(bs) for bs in batch_sizes}
        return next(iter(selected)) if len(selected) == 1 else None

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None:
        if self.cuda_graph_bs is None:
            return None
        return [batch_size for batch_size in self.cuda_graph_bs if batch_size <= step]


class TestAdaptiveController(unittest.TestCase):
    def test_injected_policy_builds_pruned_states_and_applies_initial_state(self):
        worker = _FakeWorker(initial_steps=3)
        policy = _FakePolicy()
        controller = AdaptiveController(worker, policy)

        controller.init_states(cuda_graph_bs=[1, 2, 4])

        self.assertIs(controller.params, policy)
        self.assertEqual(policy.cuda_graph_bs, [1, 2, 4])
        self.assertEqual(
            worker.build_calls,
            [
                (1, 2, [1]),
                (3, 4, [1, 2]),
            ],
        )
        self.assertEqual(worker.applied_steps, [3])

    def test_registered_initial_state_is_reused(self):
        worker = _FakeWorker(initial_steps=3)
        controller = AdaptiveController(worker, _FakePolicy())
        controller.register(_make_state(3))

        controller.init_states()

        self.assertEqual(worker.build_calls, [(1, 2, None)])
        self.assertEqual(worker.applied_steps, [3])

    def test_batch_activation_is_idempotent(self):
        worker = _FakeWorker(initial_steps=3)
        controller = AdaptiveController(worker, _FakePolicy())
        controller.init_states()

        controller.activate_step_by_batch(batch_size=8)
        controller.activate_step_by_batch(batch_size=8)
        self.assertEqual(worker.applied_steps, [3, 1])

    def test_verify_feedback_can_activate_a_state(self):
        worker = _FakeWorker(initial_steps=3)
        policy = _FakePolicy()
        controller = AdaptiveController(worker, policy)
        controller.init_states()

        controller.on_verify_complete([1], batch_size=1)
        policy.feedback_step = 1
        controller.on_verify_complete([1], batch_size=1)
        self.assertEqual(worker.applied_steps, [3, 1])

    def test_missing_state_fails_loud(self):
        worker = _FakeWorker(initial_steps=3)
        controller = AdaptiveController(worker, _FakePolicy())
        controller.init_states()
        del controller._states[1]

        with self.assertRaisesRegex(
            ValueError, "Missing adaptive runtime state for steps=1"
        ):
            controller.activate_step_by_batch(batch_size=8)

    def test_homogeneous_low_rejection_keeps_fixed_tier(self):
        worker = _FakeWorker(initial_steps=3)
        policy = _FakePolicy()
        controller = AdaptiveController(worker, policy)
        controller.init_states()

        decision = controller.synchronize_dp_tier(
            feedback_observations=[(6, 2, 2, 3), (6, 2, 2, 3)],
            rank_batch_sizes=[2, 2],
        )

        self.assertEqual(decision.steps, 3)
        self.assertEqual(decision.saved_rows_sum_len, 0)
        self.assertIsNone(decision.fallback_reason)

    def test_controlled_high_rejection_downshifts_and_accounts_rows(self):
        worker = _FakeWorker(initial_steps=3)
        policy = _FakePolicy()
        controller = AdaptiveController(worker, policy)
        controller.init_states()

        decision = controller.synchronize_dp_tier(
            feedback_observations=[(0, 2, 2, 3)],
            rank_batch_sizes=[2, 0],
        )

        # accept counts exclude the two verifier bonus tokens.  The idle rank
        # contributes no requests, but MAX_LEN still materializes its rows.
        self.assertEqual(decision.feedback_total_correct_drafts, 0)
        self.assertEqual(decision.feedback_num_requests, 2)
        self.assertEqual(decision.steps, 1)
        self.assertEqual(decision.saved_rows_sum_len, 4)
        self.assertEqual(decision.saved_rows_max_len, 8)
        self.assertEqual(decision.rank_batch_sizes, (2, 0))

    def test_heterogeneous_ranks_force_fixed_tier(self):
        worker = _FakeWorker(initial_steps=3)
        controller = AdaptiveController(worker, _FakePolicy())
        controller.init_states()

        decision = controller.synchronize_dp_tier(
            feedback_observations=[],
            rank_batch_sizes=[1, 8],
        )

        self.assertEqual(decision.steps, 3)
        self.assertEqual(decision.fallback_reason, "heterogeneous_tier")

    def test_switching_tiers_reuses_prebuilt_state_objects(self):
        worker = _FakeWorker(initial_steps=3)
        policy = _FakePolicy()
        controller = AdaptiveController(worker, policy)
        controller.init_states()
        state_ids = {step: id(state) for step, state in controller._states.items()}

        controller.synchronize_dp_tier(
            feedback_observations=[(0, 1, 1, 3)],
            rank_batch_sizes=[1],
        )
        controller.synchronize_dp_tier(
            feedback_observations=[(3, 1, 1, 1)],
            rank_batch_sizes=[1],
        )

        self.assertEqual(
            {step: id(state) for step, state in controller._states.items()},
            state_ids,
        )


class TestEagleDpAdaptiveGlue(unittest.TestCase):
    @staticmethod
    def _worker():
        worker = object.__new__(EAGLEWorkerV2)
        worker.adaptive_controller = Mock()
        worker.speculative_num_steps = 5
        worker._adaptive_consumed_epochs = []
        return worker

    @staticmethod
    def _batch(rows, batch_sizes):
        return SimpleNamespace(
            global_adaptive_spec_sync=rows,
            adaptive_spec_rank_batch_sizes=batch_sizes,
            seq_lens=SimpleNamespace(shape=(batch_sizes[0],)),
        )

    @patch("sglang.srt.speculative.eagle_worker_v2.get_parallel")
    def test_idle_rank_does_not_contribute_feedback(self, get_parallel):
        get_parallel.return_value.attn_dp_size = 2
        worker = self._worker()
        batch = self._batch(
            [
                [1, 5, 5, 79, 16, 16, 1],
                [1, 5, -1, 0, 0, 0, 0],
            ],
            [16, 0],
        )

        worker.activate_step_by_batch(batch)

        worker.adaptive_controller.synchronize_dp_tier.assert_called_once_with(
            feedback_observations=[(79, 16, 16, 5)],
            rank_batch_sizes=[16, 0],
            fallback_reason=None,
        )

    @patch("sglang.srt.speculative.eagle_worker_v2.get_parallel")
    def test_active_tier_mismatch_falls_back_fixed(self, get_parallel):
        get_parallel.return_value.attn_dp_size = 2
        worker = self._worker()
        batch = self._batch(
            [
                [1, 5, 5, 16, 4, 4, 1],
                [1, 3, 3, 12, 4, 4, 1],
            ],
            [4, 4],
        )

        worker.activate_step_by_batch(batch)

        worker.adaptive_controller.activate_fixed.assert_called_once_with(
            "active_tier_mismatch"
        )

    @patch("sglang.srt.speculative.eagle_worker_v2.get_parallel")
    def test_stale_tier_feedback_is_consumed_without_fallback(self, get_parallel):
        get_parallel.return_value.attn_dp_size = 2
        worker = self._worker()
        rows = [
            [1, 5, 3, 4, 4, 4, 1],
            [1, 5, 3, 4, 4, 4, 1],
        ]
        batch = self._batch(rows, [4, 4])

        worker.activate_step_by_batch(batch)
        worker.activate_step_by_batch(batch)

        first, second = worker.adaptive_controller.synchronize_dp_tier.call_args_list
        self.assertIsNone(first.kwargs["fallback_reason"])
        self.assertEqual(
            first.kwargs["feedback_observations"],
            [(4, 4, 4, 3), (4, 4, 4, 3)],
        )
        self.assertIsNone(second.kwargs["fallback_reason"])
        self.assertEqual(second.kwargs["feedback_observations"], [])

    @patch("sglang.srt.speculative.eagle_worker_v2.get_parallel")
    def test_dp_width_change_rebases_epochs_before_fixed_fallback(self, get_parallel):
        get_parallel.return_value.attn_dp_size = 3
        worker = self._worker()
        worker._adaptive_consumed_epochs = [4, 7]
        rows = [
            [1, 5, 5, 0, 0, 0, 5],
            [1, 5, 5, 0, 0, 0, 8],
            [1, 5, 5, 0, 0, 0, 2],
        ]

        worker.activate_step_by_batch(self._batch(rows, [4, 4, 0]))

        self.assertEqual(worker._adaptive_consumed_epochs, [5, 8, 2])
        worker.adaptive_controller.activate_fixed.assert_called_once_with(
            "dp_width_changed"
        )

    @patch("sglang.srt.speculative.eagle_worker_v2.get_parallel")
    def test_epoch_regression_rebases_and_does_not_repeat(self, get_parallel):
        get_parallel.return_value.attn_dp_size = 2
        worker = self._worker()
        worker._adaptive_consumed_epochs = [5, 5]
        rows = [
            [1, 5, 5, 0, 0, 0, 1],
            [1, 5, 5, 0, 0, 0, 1],
        ]
        batch = self._batch(rows, [4, 4])

        worker.activate_step_by_batch(batch)
        worker.activate_step_by_batch(batch)

        self.assertEqual(worker._adaptive_consumed_epochs, [1, 1])
        worker.adaptive_controller.activate_fixed.assert_called_once_with(
            "feedback_epoch_regressed"
        )
        worker.adaptive_controller.synchronize_dp_tier.assert_called_once_with(
            feedback_observations=[],
            rank_batch_sizes=[4, 4],
            fallback_reason=None,
        )


if __name__ == "__main__":
    unittest.main()
