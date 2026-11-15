"""Fail-closed contracts for a future P-only PIVOT candidate producer.

PIVOT is an approximate candidate-generation mechanism: it may share candidates
along the query axis, but the existing exact scorer must still rescore every
accepted candidate. This module owns the identity, causal filtering, budget,
padding, and quality-audit contracts. It deliberately does not implement or
dispatch a GPU candidate producer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Hashable, Mapping, Sequence


@dataclass(frozen=True)
class PivotShareIdentity:
    """State that must match before pivot candidates may be shared."""

    request_generation: int
    producer_layer_id: int
    index_share_id: Hashable
    cache_version: int
    key_representation_version: str


@dataclass(frozen=True)
class PivotQueryRow:
    """One original query row, independent of its rank-local packed position."""

    identity: PivotShareIdentity
    query_group: Hashable
    cp_rank: int
    absolute_position: int
    causal_start: int = 0

    @property
    def causal_end(self) -> int:
        return self.absolute_position + 1

    def __post_init__(self) -> None:
        if self.cp_rank < 0:
            raise ValueError("cp_rank must be non-negative")
        if self.causal_start < 0 or self.absolute_position < self.causal_start:
            raise ValueError("invalid causal domain")


@dataclass(frozen=True)
class PivotProposalKey:
    """The complete sharing key for a pivot proposal."""

    identity: PivotShareIdentity
    query_group: Hashable
    cp_rank: int


class PivotFallbackReason(str, Enum):
    FEATURE_DISABLED = "feature_disabled"
    UNSUPPORTED_PATH = "unsupported_path"
    SHORT_CONTEXT = "short_context"
    MISSING_PROPOSAL = "missing_proposal"
    INVALID_HEAD_COUNT = "invalid_head_count"
    INVALID_CANDIDATE = "invalid_candidate"
    CANDIDATE_OVERFLOW = "candidate_overflow"
    CANDIDATE_UNDERFILL = "candidate_underfill"


@dataclass(frozen=True)
class PivotPlannerConfig:
    topk: int = 2048
    candidate_budget: int = 24 * 1024
    scoring_heads: int = 32
    min_context_tokens: int = 24 * 1024

    def __post_init__(self) -> None:
        if self.topk <= 0:
            raise ValueError("topk must be positive")
        if self.candidate_budget < self.topk:
            raise ValueError("candidate_budget must be >= topk")
        if self.scoring_heads <= 0:
            raise ValueError("scoring_heads must be positive")
        if self.min_context_tokens < 0:
            raise ValueError("min_context_tokens must be non-negative")


@dataclass(frozen=True)
class PivotPlannerContext:
    """Runtime route gates for the first P-only implementation."""

    enabled: bool
    is_prefill: bool
    is_main_model: bool
    is_ordinary_extend: bool


@dataclass(frozen=True)
class PivotCandidatePlan:
    """Per-head logical candidate IDs for one original query row."""

    row: PivotQueryRow
    candidates: tuple[tuple[int, ...], ...]
    valid_counts: tuple[int, ...]
    fallback_reason: PivotFallbackReason | None = None

    @property
    def accepted(self) -> bool:
        return self.fallback_reason is None


def _fallback(
    row: PivotQueryRow,
    reason: PivotFallbackReason,
) -> PivotCandidatePlan:
    return PivotCandidatePlan(
        row=row,
        candidates=(),
        valid_counts=(),
        fallback_reason=reason,
    )


def _validate_head_candidates(
    candidates: Sequence[Sequence[int]] | None,
    expected_heads: int,
) -> bool:
    return candidates is not None and len(candidates) == expected_heads


def _merge_one_head(
    pivot: Sequence[int],
    refine: Sequence[int],
    *,
    causal_start: int,
    causal_end: int,
    budget: int,
) -> tuple[tuple[int, ...], PivotFallbackReason | None]:
    """Deduplicate in producer order and intersect with the row's causal domain."""

    seen: set[int] = set()
    output: list[int] = []
    for value in (*pivot, *refine):
        value = int(value)
        if value == -1:
            continue
        if value < -1:
            return (), PivotFallbackReason.INVALID_CANDIDATE
        if not causal_start <= value < causal_end or value in seen:
            continue
        seen.add(value)
        output.append(value)
        if len(output) > budget:
            return (), PivotFallbackReason.CANDIDATE_OVERFLOW
    return tuple(output), None


def plan_pivot_candidates(
    *,
    rows: Sequence[PivotQueryRow],
    pivot_proposals: Mapping[PivotProposalKey, Sequence[Sequence[int]]],
    row_refinements: Mapping[int, Sequence[Sequence[int]]] | None,
    config: PivotPlannerConfig,
    context: PivotPlannerContext,
) -> tuple[PivotCandidatePlan, ...]:
    """Validate and materialize candidate plans or atomically fall back per row.

    ``row_refinements`` is keyed by the row's index in ``rows``. Candidates are
    request-relative logical token IDs. A producer must supply exactly one
    sequence per scoring head. Proposals cannot cross request generations,
    query groups, IndexShare identities, cache generations, or K representation
    versions because all of those fields participate in ``PivotProposalKey``.
    """

    if not context.enabled:
        return tuple(
            _fallback(row, PivotFallbackReason.FEATURE_DISABLED) for row in rows
        )
    if not (
        context.is_prefill and context.is_main_model and context.is_ordinary_extend
    ):
        return tuple(
            _fallback(row, PivotFallbackReason.UNSUPPORTED_PATH) for row in rows
        )

    refinements = row_refinements or {}
    plans: list[PivotCandidatePlan] = []
    for row_index, row in enumerate(rows):
        if row.causal_end - row.causal_start < config.min_context_tokens:
            plans.append(_fallback(row, PivotFallbackReason.SHORT_CONTEXT))
            continue

        proposal = pivot_proposals.get(
            PivotProposalKey(
                identity=row.identity,
                query_group=row.query_group,
                cp_rank=row.cp_rank,
            )
        )
        refinement = refinements.get(row_index)
        if refinement is None:
            refinement = ((),) * config.scoring_heads
        if proposal is None:
            plans.append(_fallback(row, PivotFallbackReason.MISSING_PROPOSAL))
            continue
        if not _validate_head_candidates(
            proposal, config.scoring_heads
        ) or not _validate_head_candidates(refinement, config.scoring_heads):
            plans.append(_fallback(row, PivotFallbackReason.INVALID_HEAD_COUNT))
            continue

        merged_heads: list[tuple[int, ...]] = []
        fallback_reason = None
        for pivot, refine in zip(proposal, refinement):
            merged, fallback_reason = _merge_one_head(
                pivot,
                refine,
                causal_start=row.causal_start,
                causal_end=row.causal_end,
                budget=config.candidate_budget,
            )
            if fallback_reason is not None:
                break
            if len(merged) < config.topk:
                fallback_reason = PivotFallbackReason.CANDIDATE_UNDERFILL
                break
            merged_heads.append(
                merged + (-1,) * (config.candidate_budget - len(merged))
            )
        if fallback_reason is not None:
            plans.append(_fallback(row, fallback_reason))
            continue

        plans.append(
            PivotCandidatePlan(
                row=row,
                candidates=tuple(merged_heads),
                valid_counts=tuple(
                    sum(value >= 0 for value in head) for head in merged_heads
                ),
            )
        )
    return tuple(plans)


@dataclass(frozen=True)
class PivotRecallAudit:
    """Quality evidence for an approximate candidate plan."""

    accepted_rows: int
    fallback_rows: int
    evaluated_heads: int
    true_topk_items: int
    recovered_items: int
    severe_omissions: int

    @property
    def recall(self) -> float | None:
        if self.true_topk_items == 0:
            return None
        return self.recovered_items / self.true_topk_items


def audit_pivot_recall(
    plans: Sequence[PivotCandidatePlan],
    exact_topk: Mapping[int, Sequence[Sequence[int]]],
) -> PivotRecallAudit:
    """Compare accepted candidates with an exhaustive per-row/per-head TopK."""

    accepted_rows = fallback_rows = evaluated_heads = 0
    true_topk_items = recovered_items = severe_omissions = 0
    for row_index, plan in enumerate(plans):
        if not plan.accepted:
            fallback_rows += 1
            continue
        accepted_rows += 1
        reference = exact_topk.get(row_index)
        if reference is None or len(reference) != len(plan.candidates):
            raise ValueError(f"missing or malformed exact TopK for row {row_index}")
        for candidates, expected in zip(plan.candidates, reference):
            candidate_set = {value for value in candidates if value >= 0}
            expected_set = {int(value) for value in expected if int(value) >= 0}
            evaluated_heads += 1
            true_topk_items += len(expected_set)
            overlap = len(candidate_set & expected_set)
            recovered_items += overlap
            severe_omissions += len(expected_set) - overlap
    return PivotRecallAudit(
        accepted_rows=accepted_rows,
        fallback_rows=fallback_rows,
        evaluated_heads=evaluated_heads,
        true_topk_items=true_topk_items,
        recovered_items=recovered_items,
        severe_omissions=severe_omissions,
    )
