"""CPU-testable EP consensus contract for adaptive speculative decoding.

The existing adaptive controller makes a process-local decision. This module
defines the fixed-shape observation, all-participant merge, and idempotent tier
transition that must surround it before adaptive MTP can be enabled with EP/DP.
It performs no distributed collective and does not switch runtime resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class EpAdaptiveObservation:
    participant_id: int
    iteration: int
    current_steps: int
    accepted_histogram: tuple[int, ...]
    proposed_draft_tokens: int
    committed_tokens: int
    kv_headroom_tokens: int

    @property
    def active_requests(self) -> int:
        return sum(self.accepted_histogram)

    @property
    def accepted_draft_tokens(self) -> int:
        return sum(
            accepted * count for accepted, count in enumerate(self.accepted_histogram)
        )


@dataclass(frozen=True)
class MergedEpAdaptiveObservation:
    participant_ids: tuple[int, ...]
    active_participant_count: int
    iteration: int
    current_steps: int
    accepted_histogram: tuple[int, ...]
    proposed_draft_tokens: int
    committed_tokens: int
    minimum_kv_headroom_tokens: int

    @property
    def active_requests(self) -> int:
        return sum(self.accepted_histogram)

    @property
    def accepted_draft_tokens(self) -> int:
        return sum(
            accepted * count for accepted, count in enumerate(self.accepted_histogram)
        )

    @property
    def acceptance_rate(self) -> float | None:
        if self.proposed_draft_tokens == 0:
            return None
        return self.accepted_draft_tokens / self.proposed_draft_tokens

    @property
    def correct_drafts_per_request(self) -> tuple[int, ...]:
        return tuple(
            accepted
            for accepted, count in enumerate(self.accepted_histogram)
            for _ in range(count)
        )


@dataclass(frozen=True)
class EpAdaptiveTransition:
    iteration: int
    from_steps: int
    to_steps: int
    observation: MergedEpAdaptiveObservation


def build_ep_adaptive_observation(
    *,
    participant_id: int,
    iteration: int,
    current_steps: int,
    correct_drafts_per_request: Sequence[int],
    proposed_draft_tokens: int,
    committed_tokens: int,
    kv_headroom_tokens: int,
    maximum_steps: int,
) -> EpAdaptiveObservation:
    """Build one fixed-width rank payload; idle ranks pass an empty sequence."""

    if participant_id < 0 or iteration < 0:
        raise ValueError("participant_id and iteration must be non-negative")
    if not 0 < current_steps <= maximum_steps:
        raise ValueError("current_steps must be in the fixed histogram domain")
    if any(
        value < 0
        for value in (
            proposed_draft_tokens,
            committed_tokens,
            kv_headroom_tokens,
        )
    ):
        raise ValueError("token counters must be non-negative")
    histogram = [0] * (maximum_steps + 1)
    for accepted in correct_drafts_per_request:
        accepted = int(accepted)
        if not 0 <= accepted <= current_steps:
            raise ValueError("accepted drafts exceed the active tier")
        histogram[accepted] += 1
    accepted_total = sum(accepted * count for accepted, count in enumerate(histogram))
    if accepted_total > proposed_draft_tokens:
        raise ValueError("accepted drafts exceed proposed draft tokens")
    return EpAdaptiveObservation(
        participant_id=participant_id,
        iteration=iteration,
        current_steps=current_steps,
        accepted_histogram=tuple(histogram),
        proposed_draft_tokens=proposed_draft_tokens,
        committed_tokens=committed_tokens,
        kv_headroom_tokens=kv_headroom_tokens,
    )


def merge_ep_adaptive_observations(
    observations: Iterable[EpAdaptiveObservation],
    *,
    expected_participant_ids: Sequence[int],
    candidate_steps: Sequence[int] = (1, 3),
) -> MergedEpAdaptiveObservation:
    """Validate and merge the payloads from one mandatory collective round."""

    values = tuple(observations)
    expected = tuple(sorted(int(value) for value in expected_participant_ids))
    if len(expected) == 0 or len(set(expected)) != len(expected):
        raise ValueError("expected participant IDs must be unique and non-empty")
    actual = tuple(sorted(value.participant_id for value in values))
    if actual != expected:
        raise ValueError(f"participant mismatch: expected {expected}, got {actual}")
    candidates = tuple(sorted(set(int(value) for value in candidate_steps)))
    if not candidates or candidates[0] <= 0:
        raise ValueError("candidate steps must be unique positive tiers")

    iterations = {value.iteration for value in values}
    current_steps = {value.current_steps for value in values}
    if len(iterations) != 1:
        raise ValueError("EP participants entered different adaptive iterations")
    if len(current_steps) != 1:
        raise ValueError("EP participants entered different adaptive tiers")
    current = next(iter(current_steps))
    if current not in candidates:
        raise ValueError("current tier is not a registered consensus candidate")

    width = max(candidates) + 1
    if any(len(value.accepted_histogram) != width for value in values):
        raise ValueError("adaptive histograms must have the same fixed width")
    if any(count < 0 for value in values for count in value.accepted_histogram):
        raise ValueError("adaptive histograms must be non-negative")
    histogram = tuple(
        sum(value.accepted_histogram[index] for value in values)
        for index in range(width)
    )
    proposed = sum(value.proposed_draft_tokens for value in values)
    accepted = sum(index * count for index, count in enumerate(histogram))
    if accepted > proposed:
        raise ValueError("merged accepted drafts exceed proposed drafts")

    return MergedEpAdaptiveObservation(
        participant_ids=actual,
        active_participant_count=sum(value.active_requests > 0 for value in values),
        iteration=next(iter(iterations)),
        current_steps=current,
        accepted_histogram=histogram,
        proposed_draft_tokens=proposed,
        committed_tokens=sum(value.committed_tokens for value in values),
        minimum_kv_headroom_tokens=min(value.kv_headroom_tokens for value in values),
    )


class EpAdaptiveTransitionGate:
    """Enforce lockstep, pre-built tier transitions after a merged decision."""

    def __init__(
        self,
        *,
        initial_steps: int,
        registered_steps: Sequence[int] = (1, 3),
        first_iteration: int = 0,
    ) -> None:
        self.registered_steps = tuple(sorted(set(int(v) for v in registered_steps)))
        if not self.registered_steps or self.registered_steps[0] <= 0:
            raise ValueError("registered steps must be unique positive tiers")
        if initial_steps not in self.registered_steps:
            raise ValueError("initial tier must be pre-built")
        if first_iteration < 0:
            raise ValueError("first_iteration must be non-negative")
        self.current_steps = initial_steps
        self.next_iteration = first_iteration
        self._last_transition: EpAdaptiveTransition | None = None

    def apply(
        self,
        observation: MergedEpAdaptiveObservation,
        target_steps: int,
    ) -> EpAdaptiveTransition:
        """Apply one coordinator-broadcast target or idempotently replay it."""

        if target_steps not in self.registered_steps:
            raise ValueError("target tier was not pre-built")
        if observation.iteration == self.next_iteration - 1:
            previous = self._last_transition
            if (
                previous is not None
                and previous.observation == observation
                and previous.to_steps == target_steps
            ):
                return previous
            raise ValueError("adaptive iteration replay diverged")
        if observation.iteration != self.next_iteration:
            raise ValueError("adaptive iteration is stale or skipped")
        if observation.current_steps != self.current_steps:
            raise ValueError("merged current tier disagrees with local runtime")
        transition = EpAdaptiveTransition(
            iteration=observation.iteration,
            from_steps=self.current_steps,
            to_steps=target_steps,
            observation=observation,
        )
        self.current_steps = target_steps
        self.next_iteration += 1
        self._last_transition = transition
        return transition
