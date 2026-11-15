"""CPU reference for PIVOT-Refine candidate generation and quality audit.

The implementation follows PIVOT v1's prefill algorithm: consecutive queries
form a group, their per-head queries and gates are mean pooled, one proxy scan
uses the first query's causal cutoff, and every query is exactly re-scored over
the shared candidate set. It is intentionally not wired into the serving path;
it is an offline oracle for real q/index-K dumps and future GPU kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from sglang.srt.layers.attention.dsa.pivot_candidate import (
    PivotCandidatePlan,
    PivotPlannerConfig,
    PivotPlannerContext,
    PivotProposalKey,
    PivotQueryRow,
    PivotRecallAudit,
    PivotShareIdentity,
    audit_pivot_recall,
    plan_pivot_candidates,
)


@dataclass(frozen=True)
class PivotReferenceConfig:
    group_size: int = 4
    topk: int = 2048
    candidate_budget: int = 4096
    local_window: int = 0
    scoring_heads: int = 32

    def __post_init__(self) -> None:
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.topk <= 0 or self.candidate_budget < self.topk:
            raise ValueError("expected 0 < topk <= candidate_budget")
        if self.local_window < 0:
            raise ValueError("local_window must be non-negative")
        if self.scoring_heads <= 0:
            raise ValueError("scoring_heads must be positive")
        reserved = self.local_window + self.group_size - 1
        if self.local_window and reserved >= self.candidate_budget:
            raise ValueError("local window union leaves no proxy candidate slots")


@dataclass(frozen=True)
class PivotReferenceResult:
    plans: tuple[PivotCandidatePlan, ...]
    selected_topk: tuple[tuple[int, ...], ...]
    exact_topk: tuple[tuple[int, ...], ...]
    recall: PivotRecallAudit
    full_score_items: int
    proxy_score_items: int
    refine_score_items: int

    @property
    def ideal_score_work_ratio(self) -> float:
        if self.full_score_items == 0:
            return 0.0
        return (self.proxy_score_items + self.refine_score_items) / float(
            self.full_score_items
        )


def _indexer_scores(
    queries: torch.Tensor,
    gates: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    """Apply the DSA indexer score formula in FP32 for the CPU oracle."""

    if queries.ndim != 3 or gates.ndim != 2 or keys.ndim != 2:
        raise ValueError(
            "expected queries [rows, heads, dim], gates [rows, heads], keys [tokens, dim]"
        )
    if queries.shape[:2] != gates.shape:
        raise ValueError("query and gate row/head dimensions must match")
    if queries.shape[2] != keys.shape[1]:
        raise ValueError("query/key dimensions must match")
    dots = torch.einsum("rhd,kd->rhk", queries.float(), keys.float())
    return (torch.relu(dots) * gates.float().unsqueeze(-1)).sum(dim=1)


def _stable_topk_ids(
    scores: torch.Tensor,
    token_ids: torch.Tensor,
    k: int,
) -> tuple[int, ...]:
    """Sort by score descending and token ID ascending for deterministic ties."""

    if scores.ndim != 1 or token_ids.ndim != 1 or scores.numel() != token_ids.numel():
        raise ValueError("scores and token_ids must be matching vectors")
    if k <= 0:
        raise ValueError("k must be positive")
    if scores.numel() == 0:
        return ()
    id_order = torch.argsort(token_ids, stable=True)
    score_order = torch.argsort(scores[id_order], descending=True, stable=True)
    chosen = token_ids[id_order[score_order[: min(k, scores.numel())]]]
    return tuple(int(value) for value in chosen.tolist())


def _group_indices(rows: Sequence[PivotQueryRow]) -> tuple[tuple[int, ...], ...]:
    groups: dict[PivotProposalKey, list[int]] = {}
    order: list[PivotProposalKey] = []
    for row_index, row in enumerate(rows):
        key = PivotProposalKey(row.identity, row.query_group, row.cp_rank)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row_index)
    return tuple(tuple(groups[key]) for key in order)


def pivot_refine_reference(
    *,
    queries: torch.Tensor,
    gates: torch.Tensor,
    keys: Mapping[PivotShareIdentity, torch.Tensor],
    rows: Sequence[PivotQueryRow],
    config: PivotReferenceConfig,
) -> PivotReferenceResult:
    """Run PIVOT-Refine and an exhaustive DSA comparison on CPU tensors.

    Query groups must stay within one request/layer/share/cache identity and one
    CP rank, and their absolute positions must be consecutive. Invalid groups
    receive no proposal and fail closed to exhaustive scoring through the
    planner. Groups whose first position is shorter than ``candidate_budget``
    follow the paper's dense short-group guardrail.
    """

    if queries.shape[0] != len(rows) or gates.shape[0] != len(rows):
        raise ValueError("one query/gate row is required per PivotQueryRow")
    if queries.shape[1] != config.scoring_heads:
        raise ValueError("query scoring-head count does not match config")

    proposals: dict[PivotProposalKey, tuple[int, ...]] = {}
    proxy_score_items = 0
    for group in _group_indices(rows):
        group_rows = [rows[index] for index in group]
        first = group_rows[0]
        positions = [row.absolute_position for row in group_rows]
        valid_group = (
            len(group) <= config.group_size
            and positions == list(range(positions[0], positions[0] + len(group)))
            and all(row.causal_start == first.causal_start for row in group_rows)
        )
        if not valid_group or first.absolute_position < config.candidate_budget:
            continue
        group_keys = keys.get(first.identity)
        if group_keys is None or group_keys.shape[0] < first.causal_end:
            continue

        mean_query = queries[list(group)].float().mean(dim=0, keepdim=True)
        mean_gate = gates[list(group)].float().mean(dim=0, keepdim=True)
        prefix_ids = torch.arange(
            first.causal_start,
            first.causal_end,
            dtype=torch.long,
            device=group_keys.device,
        )
        prefix_keys = group_keys[first.causal_start : first.causal_end]
        proxy_scores = _indexer_scores(mean_query, mean_gate, prefix_keys)[0]

        if config.local_window:
            reserved = config.local_window + len(group) - 1
            window_start = max(
                first.causal_start,
                first.absolute_position - config.local_window + 1,
            )
            window_end = group_rows[-1].causal_end
            window_ids = tuple(range(window_start, window_end))
        else:
            reserved = 0
            window_ids = ()
        proxy_budget = config.candidate_budget - reserved
        proxy_ids = _stable_topk_ids(proxy_scores, prefix_ids, proxy_budget)
        candidate_ids = tuple(dict.fromkeys((*proxy_ids, *window_ids)))
        proposals[
            PivotProposalKey(first.identity, first.query_group, first.cp_rank)
        ] = candidate_ids
        proxy_score_items += prefix_ids.numel()

    planner_config = PivotPlannerConfig(
        topk=config.topk,
        candidate_budget=config.candidate_budget,
        scoring_heads=config.scoring_heads,
        min_context_tokens=config.candidate_budget + 1,
    )
    plans = plan_pivot_candidates(
        rows=rows,
        pivot_proposals=proposals,
        row_refinements=None,
        config=planner_config,
        context=PivotPlannerContext(True, True, True, True),
    )

    selected: list[tuple[int, ...]] = []
    exact: list[tuple[int, ...]] = []
    full_score_items = refine_score_items = 0
    for row_index, (row, plan) in enumerate(zip(rows, plans)):
        row_keys = keys.get(row.identity)
        if row_keys is None or row_keys.shape[0] < row.causal_end:
            raise ValueError(f"missing key domain for row {row_index}")
        full_ids = torch.arange(
            row.causal_start,
            row.causal_end,
            dtype=torch.long,
            device=row_keys.device,
        )
        full_scores = _indexer_scores(
            queries[row_index : row_index + 1],
            gates[row_index : row_index + 1],
            row_keys[row.causal_start : row.causal_end],
        )[0]
        exact_ids = _stable_topk_ids(full_scores, full_ids, config.topk)
        exact.append(exact_ids)
        full_score_items += full_ids.numel()

        if not plan.accepted:
            selected.append(exact_ids)
            continue
        candidate_ids = torch.tensor(
            [value for value in plan.candidates if value >= 0],
            dtype=torch.long,
            device=row_keys.device,
        )
        candidate_scores = _indexer_scores(
            queries[row_index : row_index + 1],
            gates[row_index : row_index + 1],
            row_keys[candidate_ids],
        )[0]
        selected.append(_stable_topk_ids(candidate_scores, candidate_ids, config.topk))
        refine_score_items += candidate_ids.numel()

    exact_map = {index: value for index, value in enumerate(exact)}
    return PivotReferenceResult(
        plans=plans,
        selected_topk=tuple(selected),
        exact_topk=tuple(exact),
        recall=audit_pivot_recall(plans, exact_map),
        full_score_items=full_score_items,
        proxy_score_items=proxy_score_items,
        refine_score_items=refine_score_items,
    )
