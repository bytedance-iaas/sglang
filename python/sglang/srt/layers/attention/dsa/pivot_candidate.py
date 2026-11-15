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

from sglang.srt.layers.attention.dsa.dosa_cp import DosaCPLayout


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
    INVALID_PROPOSAL = "invalid_proposal"
    INVALID_CANDIDATE = "invalid_candidate"
    CANDIDATE_OVERFLOW = "candidate_overflow"
    CANDIDATE_UNDERFILL = "candidate_underfill"


@dataclass(frozen=True)
class PivotPlannerConfig:
    topk: int = 2048
    candidate_budget: int = 4096
    scoring_heads: int = 32
    # PIVOT v1's short-group guard is t0 < c. Query rows use a causal end of
    # t0 + 1, so c + 1 is the equivalent row-local threshold.
    min_context_tokens: int = 4097

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
    """Logical token candidates for one original query row.

    PIVOT's proxy score already reduces all scoring heads to one scalar per
    token. The candidate set is therefore shared across heads; the existing
    exact indexer uses all ``scoring_heads`` only when it re-scores this set.
    """

    row: PivotQueryRow
    candidates: tuple[int, ...]
    valid_count: int
    scoring_heads: int
    fallback_reason: PivotFallbackReason | None = None

    @property
    def accepted(self) -> bool:
        return self.fallback_reason is None


def build_pivot_rows_from_cp_layout(
    layout: DosaCPLayout,
    *,
    group_size: int,
    producer_layer_id: int,
    index_share_id: Hashable,
    key_representation_version: str,
) -> tuple[PivotQueryRow, ...]:
    """Form consecutive PIVOT groups inside each local CP segment.

    The current CP packing order is all ``prev`` segments followed by all
    ``next`` segments. A flat rank-local grouping would therefore cross request
    and zigzag boundaries. This bridge groups each segment independently and
    preserves its local row order, so no query all-gather is implied.
    """

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if producer_layer_id < 0:
        raise ValueError("producer_layer_id must be non-negative")
    if not key_representation_version:
        raise ValueError("key_representation_version must be non-empty")

    result: list[PivotQueryRow] = []
    for segment in layout.segments:
        identity = PivotShareIdentity(
            request_generation=segment.request_generation,
            producer_layer_id=producer_layer_id,
            index_share_id=index_share_id,
            cache_version=segment.cache_version,
            key_representation_version=key_representation_version,
        )
        for row_offset, absolute_position in enumerate(segment.absolute_positions):
            group_index = row_offset // group_size
            result.append(
                PivotQueryRow(
                    identity=identity,
                    query_group=(
                        segment.rid,
                        segment.segment_kind,
                        segment.segment_index,
                        group_index,
                    ),
                    cp_rank=layout.cp_rank,
                    absolute_position=absolute_position,
                    causal_start=0,
                )
            )
    if len(result) != layout.local_row_count:
        raise ValueError("PIVOT rows do not cover the local CP layout")
    return tuple(result)


def _fallback(
    row: PivotQueryRow,
    reason: PivotFallbackReason,
    scoring_heads: int,
) -> PivotCandidatePlan:
    return PivotCandidatePlan(
        row=row,
        candidates=(),
        valid_count=0,
        scoring_heads=scoring_heads,
        fallback_reason=reason,
    )


def _merge_candidates(
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
    pivot_proposals: Mapping[PivotProposalKey, Sequence[int]],
    row_refinements: Mapping[int, Sequence[int]] | None,
    config: PivotPlannerConfig,
    context: PivotPlannerContext,
) -> tuple[PivotCandidatePlan, ...]:
    """Validate and materialize candidate plans or atomically fall back per row.

    ``row_refinements`` is keyed by the row's index in ``rows``. Candidates are
    request-relative logical token IDs. A producer supplies one shared sequence
    after its proxy score has summed across all scoring heads. Proposals cannot
    cross request generations, producer layers, query groups, CP ranks,
    IndexShare identities, cache generations, or K representation versions
    because all of those fields participate in ``PivotProposalKey``.
    """

    if not context.enabled:
        return tuple(
            _fallback(row, PivotFallbackReason.FEATURE_DISABLED, config.scoring_heads)
            for row in rows
        )
    if not (
        context.is_prefill and context.is_main_model and context.is_ordinary_extend
    ):
        return tuple(
            _fallback(row, PivotFallbackReason.UNSUPPORTED_PATH, config.scoring_heads)
            for row in rows
        )

    refinements = row_refinements or {}
    plans: list[PivotCandidatePlan] = []
    for row_index, row in enumerate(rows):
        if row.causal_end - row.causal_start < config.min_context_tokens:
            plans.append(
                _fallback(row, PivotFallbackReason.SHORT_CONTEXT, config.scoring_heads)
            )
            continue

        proposal = pivot_proposals.get(
            PivotProposalKey(
                identity=row.identity,
                query_group=row.query_group,
                cp_rank=row.cp_rank,
            )
        )
        refinement = refinements.get(row_index, ())
        if proposal is None:
            plans.append(
                _fallback(
                    row, PivotFallbackReason.MISSING_PROPOSAL, config.scoring_heads
                )
            )
            continue
        if isinstance(proposal, (str, bytes)) or isinstance(refinement, (str, bytes)):
            plans.append(
                _fallback(
                    row, PivotFallbackReason.INVALID_PROPOSAL, config.scoring_heads
                )
            )
            continue

        merged, fallback_reason = _merge_candidates(
            proposal,
            refinement,
            causal_start=row.causal_start,
            causal_end=row.causal_end,
            budget=config.candidate_budget,
        )
        if fallback_reason is None and len(merged) < config.topk:
            fallback_reason = PivotFallbackReason.CANDIDATE_UNDERFILL
        if fallback_reason is not None:
            plans.append(_fallback(row, fallback_reason, config.scoring_heads))
            continue

        padded = merged + (-1,) * (config.candidate_budget - len(merged))
        plans.append(
            PivotCandidatePlan(
                row=row,
                candidates=padded,
                valid_count=len(merged),
                scoring_heads=config.scoring_heads,
            )
        )
    return tuple(plans)


@dataclass(frozen=True)
class PivotRecallAudit:
    """Quality evidence for an approximate candidate plan."""

    accepted_rows: int
    fallback_rows: int
    evaluated_rows: int
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
    exact_topk: Mapping[int, Sequence[int]],
) -> PivotRecallAudit:
    """Compare accepted candidates with exhaustive per-row DSA TopK sets."""

    accepted_rows = fallback_rows = evaluated_rows = 0
    true_topk_items = recovered_items = severe_omissions = 0
    for row_index, plan in enumerate(plans):
        if not plan.accepted:
            fallback_rows += 1
            continue
        accepted_rows += 1
        reference = exact_topk.get(row_index)
        if reference is None:
            raise ValueError(f"missing or malformed exact TopK for row {row_index}")
        candidate_set = {value for value in plan.candidates if value >= 0}
        expected_set = {int(value) for value in reference if int(value) >= 0}
        evaluated_rows += 1
        true_topk_items += len(expected_set)
        overlap = len(candidate_set & expected_set)
        recovered_items += overlap
        severe_omissions += len(expected_set) - overlap
    return PivotRecallAudit(
        accepted_rows=accepted_rows,
        fallback_rows=fallback_rows,
        evaluated_rows=evaluated_rows,
        true_topk_items=true_topk_items,
        recovered_items=recovered_items,
        severe_omissions=severe_omissions,
    )
