from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
    from sglang.srt.model_executor.runner import DecodeCudaGraphRunner


@dataclass
class SpecRuntimeState:
    """A complete set of runtime resources bound to a specific speculative
    decoding configuration.

    The draft and verify resources are required by adaptive workers. Algorithms
    with a draft-extend stage can also populate its optional resources.
    Switching adaptive steps swaps the entire state atomically.
    """

    # -- Configuration (determines shapes for all stages) --
    speculative_num_steps: int
    speculative_num_draft_tokens: int

    # -- Draft stage: draft model multi-step autoregressive generation --
    draft_attn_backend: "AttentionBackend | None"
    cuda_graph_runner: "DecodeCudaGraphRunner | None"

    # -- Verify stage: target model one-pass tree verification --
    target_attn_backend: "AttentionBackend"
    target_graph_runner: "DecodeCudaGraphRunner | CPUGraphRunner | None"

    # -- Extend stage: draft model KV cache catch-up after verify --
    draft_extend_attn_backend: "AttentionBackend | None"
    cuda_graph_runner_for_draft_extend: "DecodeCudaGraphRunner | None"


class AdaptiveSpecWorker(Protocol):
    """Protocol that a worker must implement to use AdaptiveController."""

    speculative_num_steps: int

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ) -> SpecRuntimeState: ...

    def apply_runtime_state(self, state: SpecRuntimeState) -> None: ...


class AdaptiveSpecPolicy(Protocol):
    """Policy interface used by AdaptiveController to select runtime states."""

    @property
    def candidate_steps(self) -> list[int]: ...

    def set_cuda_graph_bs(self, cuda_graph_bs: list[int] | None) -> None: ...

    def get_steps_for_batch(self, batch_size: int) -> int: ...

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> int | None: ...

    def initialize_synchronized_steps(self, steps: int) -> None: ...

    def observe_verify_complete_aggregates(
        self, observations: list[tuple[int, int, int, int]]
    ) -> None: ...

    def get_compatible_steps_for_batches(
        self, batch_sizes: list[int]
    ) -> int | None: ...

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None: ...


@dataclass(frozen=True)
class AdaptiveTierDecision:
    """One synchronized tier decision and its physical-row accounting."""

    steps: int
    draft_tokens: int
    fixed_steps: int
    fixed_draft_tokens: int
    rank_batch_sizes: tuple[int, ...]
    feedback_num_requests: int
    feedback_total_correct_drafts: int
    saved_rows_sum_len: int
    saved_rows_max_len: int
    fallback_reason: str | None = None


class AdaptiveController:
    """Facade that owns adaptive decision-making and runtime state switching.

    Works with any worker that implements AdaptiveSpecWorker protocol:
      - build_adaptive_runtime_state(steps, draft_tokens) → runtime state
      - apply_runtime_state(state) → apply it to the worker

    The worker only needs to:
      1. Call register() for the initial state, then init_states()
         once during startup.
      2. Call on_verify_complete(num_correct_drafts_per_req) after each decode verify.
    """

    def __init__(
        self,
        worker: AdaptiveSpecWorker,
        policy: AdaptiveSpecPolicy,
    ):
        self.worker = worker
        self.params: AdaptiveSpecPolicy = policy
        self._states: dict[int, SpecRuntimeState] = {}
        self.fixed_steps = worker.speculative_num_steps
        self.last_decision = AdaptiveTierDecision(
            steps=self.fixed_steps,
            draft_tokens=self.fixed_steps + 1,
            fixed_steps=self.fixed_steps,
            fixed_draft_tokens=self.fixed_steps + 1,
            rank_batch_sizes=(),
            feedback_num_requests=0,
            feedback_total_correct_drafts=0,
            saved_rows_sum_len=0,
            saved_rows_max_len=0,
            fallback_reason=None,
        )

    @property
    def candidate_steps(self) -> list[int]:
        return self.params.candidate_steps

    def register(self, state: SpecRuntimeState, steps: int | None = None) -> None:
        """Register a pre-built runtime state.

        *steps* defaults to state.speculative_num_steps when not given.
        """
        key = steps if steps is not None else state.speculative_num_steps
        self._states[key] = state

    def init_states(self, cuda_graph_bs: list[int] | None = None) -> None:
        """Build and register runtime states for all candidate steps."""
        self.params.set_cuda_graph_bs(cuda_graph_bs)

        for steps in self.candidate_steps:
            if steps in self._states:
                continue

            pruned_bs = self.params.cuda_graph_bs_for_step(steps)
            state = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
                cuda_graph_bs=pruned_bs,
            )
            self._states[steps] = state

        # Start on the initial step.
        self._activate(self.worker.speculative_num_steps)

    def activate_step_by_batch(self, batch_size: int) -> None:
        target = self.params.get_steps_for_batch(batch_size)
        if target != self.worker.speculative_num_steps:
            self._activate(target)

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> None:
        """Feed verify results; switch runtime state if the policy requests it."""
        new_step = self.params.on_verify_complete(
            num_correct_drafts_per_req, batch_size
        )
        if new_step is not None:
            self._activate(new_step)

    def synchronize_dp_tier(
        self,
        *,
        feedback_observations: list[tuple[int, int, int, int]],
        rank_batch_sizes: list[int],
        fallback_reason: str | None = None,
    ) -> AdaptiveTierDecision:
        """Apply one DP-wide observation and activate one compatible tier.

        IDLE ranks are represented by missing/zero batch sizes and do not add
        feedback.  A heterogeneous tier choice is fail-closed for this round:
        every rank uses the configured fixed tier, so the target/draft/EP
        collective geometry stays identical cluster-wide.
        """
        total_correct_drafts = sum(obs[0] for obs in feedback_observations)
        num_feedback_requests = sum(obs[1] for obs in feedback_observations)
        if num_feedback_requests > 0 and fallback_reason is None:
            self.params.observe_verify_complete_aggregates(feedback_observations)

        rank_batches = [int(bs) for bs in rank_batch_sizes]
        batches = [bs for bs in rank_batches if bs > 0]
        target = None
        if fallback_reason is None and batches:
            target = self.params.get_compatible_steps_for_batches(batches)
            if target is None:
                fallback_reason = "heterogeneous_tier"
        if target is None:
            target = self.fixed_steps
            self.params.initialize_synchronized_steps(self.fixed_steps)

        self._activate(target)
        decision = self._build_tier_decision(
            steps=target,
            rank_batch_sizes=rank_batches,
            feedback_num_requests=num_feedback_requests,
            feedback_total_correct_drafts=total_correct_drafts,
            fallback_reason=fallback_reason,
        )
        self.last_decision = decision
        return decision

    def activate_fixed(self, reason: str) -> AdaptiveTierDecision:
        self.params.initialize_synchronized_steps(self.fixed_steps)
        return self.synchronize_dp_tier(
            feedback_observations=[],
            rank_batch_sizes=[],
            fallback_reason=reason,
        )

    def _build_tier_decision(
        self,
        *,
        steps: int,
        rank_batch_sizes: list[int],
        feedback_num_requests: int,
        feedback_total_correct_drafts: int,
        fallback_reason: str | None,
    ) -> AdaptiveTierDecision:
        fixed_width = self.fixed_steps + 1
        width = steps + 1
        saved_width = max(0, fixed_width - width)
        sum_bs = sum(rank_batch_sizes)
        max_bs = max(rank_batch_sizes, default=0)
        num_ranks = len(rank_batch_sizes)
        return AdaptiveTierDecision(
            steps=steps,
            draft_tokens=width,
            fixed_steps=self.fixed_steps,
            fixed_draft_tokens=fixed_width,
            rank_batch_sizes=tuple(rank_batch_sizes),
            feedback_num_requests=feedback_num_requests,
            feedback_total_correct_drafts=feedback_total_correct_drafts,
            saved_rows_sum_len=saved_width * sum_bs,
            saved_rows_max_len=saved_width * max_bs * num_ranks,
            fallback_reason=fallback_reason,
        )

    def _activate(self, speculative_num_steps: int) -> None:
        state = self._states.get(speculative_num_steps)
        if state is None:
            raise ValueError(
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )
        self.worker.apply_runtime_state(state)
