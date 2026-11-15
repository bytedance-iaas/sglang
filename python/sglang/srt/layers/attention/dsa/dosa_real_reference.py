"""Real-arithmetic DoSA certificate and exact-selection CPU oracle.

This module implements the mathematical certificate from DoSA v1, not the
separate FP8/tcgen05 numerical certificate. It is an offline oracle for layout,
candidate, and recovery tests. Runtime pruning must remain disabled until an
upward-rounded bound dominates the actual GPU scorer arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class DosaRealConfig:
    topk: int
    retained_heads: int
    seed_size: int
    candidate_capacity: int | None = None

    def __post_init__(self) -> None:
        if self.topk <= 0:
            raise ValueError("topk must be positive")
        if self.retained_heads <= 0:
            raise ValueError("retained_heads must be positive")
        if self.seed_size < self.topk:
            raise ValueError("seed_size must be >= topk")
        if self.candidate_capacity is not None and self.candidate_capacity < 0:
            raise ValueError("candidate_capacity must be non-negative")


@dataclass(frozen=True)
class DosaRealResult:
    selected: tuple[int, ...]
    exact: tuple[int, ...]
    seed_ids: tuple[int, ...]
    candidate_ids: tuple[int, ...]
    retained_heads: tuple[int, ...]
    omitted_positive_heads: tuple[int, ...]
    negative_support_heads: tuple[int, ...]
    threshold: float
    minimum_bound_margin: float
    recovered_full_domain: bool
    arithmetic_contract: str = "real_arithmetic_only"

    @property
    def exact_match(self) -> bool:
        return self.selected == self.exact


def _stable_topk(scores: torch.Tensor, ids: torch.Tensor, k: int) -> tuple[int, ...]:
    if scores.ndim != 1 or ids.ndim != 1 or scores.numel() != ids.numel():
        raise ValueError("scores and ids must be matching vectors")
    if not 0 < k <= scores.numel():
        raise ValueError("topk exceeds the visible domain")
    id_order = torch.argsort(ids, stable=True)
    score_order = torch.argsort(scores[id_order], descending=True, stable=True)
    return tuple(int(value) for value in ids[id_order[score_order[:k]]].tolist())


def _scores(
    queries: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    dots = torch.einsum("hd,kd->hk", queries, keys)
    return (weights[:, None] * torch.relu(dots)).sum(dim=0)


def _select_retained_heads(
    queries: torch.Tensor,
    weights: torch.Tensor,
    seed_keys: torch.Tensor,
    retained_count: int,
) -> tuple[int, ...]:
    heads = queries.shape[0]
    nonzero = [head for head in range(heads) if float(weights[head]) != 0.0]
    if not 0 < retained_count < heads or retained_count > len(nonzero):
        raise ValueError("retained_heads must fit the nonzero scoring heads")
    positive = [head for head in nonzero if float(weights[head]) > 0.0]
    negative = [head for head in nonzero if float(weights[head]) < 0.0]
    strengths = {
        head: abs(float(weights[head])) * float(torch.linalg.vector_norm(queries[head]))
        for head in nonzero
    }
    if len(positive) > retained_count:
        return tuple(
            sorted(positive, key=lambda head: (-strengths[head], head))[:retained_count]
        )

    retained = list(positive)
    activations = (queries[negative] @ seed_keys.T) > 0 if negative else None
    frequencies = (
        activations.double().mean(dim=1).tolist() if activations is not None else []
    )
    priority = {
        head: strengths[head] * min(freq, 1.0 - freq)
        for head, freq in zip(negative, frequencies)
    }
    retained.extend(
        sorted(
            negative,
            key=lambda head: (-priority[head], -strengths[head], head),
        )[: retained_count - len(retained)]
    )
    return tuple(retained)


def dosa_real_reference(
    *,
    queries: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    config: DosaRealConfig,
    visible_start: int = 0,
    visible_end: int | None = None,
    seed_ids: Sequence[int] | None = None,
) -> DosaRealResult:
    """Evaluate the DoSA real-arithmetic bound and fixed-threshold algorithm."""

    queries = queries.double()
    weights = weights.double()
    keys = keys.double()
    if queries.ndim != 2 or weights.ndim != 1 or keys.ndim != 2:
        raise ValueError(
            "expected queries [heads, dim], weights [heads], keys [tokens, dim]"
        )
    if queries.shape[0] != weights.numel() or queries.shape[1] != keys.shape[1]:
        raise ValueError("query, weight, and key shapes do not match")
    if not torch.isfinite(queries).all() or not torch.isfinite(weights).all():
        raise ValueError("queries and weights must be finite")
    if not torch.isfinite(keys).all():
        raise ValueError("keys must be finite")

    end = keys.shape[0] if visible_end is None else int(visible_end)
    if not 0 <= visible_start < end <= keys.shape[0]:
        raise ValueError("invalid visible key interval")
    visible_ids = torch.arange(visible_start, end, dtype=torch.long)
    if visible_ids.numel() < config.topk:
        raise ValueError("visible domain is smaller than topk")

    if seed_ids is None:
        seed_count = min(config.seed_size, visible_ids.numel())
        chosen_seed_ids = visible_ids[:seed_count]
    else:
        chosen_seed_ids = torch.tensor(tuple(int(value) for value in seed_ids))
        if chosen_seed_ids.numel() < config.topk:
            raise ValueError("seed set is smaller than topk")
        if chosen_seed_ids.unique().numel() != chosen_seed_ids.numel():
            raise ValueError("seed IDs must be distinct")
        if (chosen_seed_ids < visible_start).any() or (chosen_seed_ids >= end).any():
            raise ValueError("seed IDs must lie in the visible domain")

    visible_keys = keys[visible_start:end]
    full_scores = _scores(queries, weights, visible_keys)
    exact = _stable_topk(full_scores, visible_ids, config.topk)
    seed_scores = _scores(queries, weights, keys[chosen_seed_ids])
    threshold = float(torch.topk(seed_scores, config.topk, sorted=True).values[-1])

    retained = _select_retained_heads(
        queries, weights, keys[chosen_seed_ids], config.retained_heads
    )
    retained_set = set(retained)
    positive = [head for head in range(queries.shape[0]) if weights[head] > 0]
    negative = [head for head in range(queries.shape[0]) if weights[head] < 0]
    omitted_positive = [head for head in positive if head not in retained_set]
    omitted_negative = [head for head in negative if head not in retained_set]

    folded = {head: float(weights[head]) for head in retained}
    residual_allowance = 0.0
    for head in omitted_positive:
        best_anchor = retained[0]
        best_similarity = -1.0
        best_beta = 0.0
        for anchor in retained:
            norm_sq = float(torch.dot(queries[anchor], queries[anchor]))
            if norm_sq == 0.0:
                similarity = beta = 0.0
            else:
                dot = float(torch.dot(queries[head], queries[anchor]))
                beta = max(dot, 0.0) / norm_sq
                similarity = max(dot, 0.0) / norm_sq**0.5
            if similarity > best_similarity:
                best_anchor = anchor
                best_similarity = similarity
                best_beta = beta
        residual = queries[head] - best_beta * queries[best_anchor]
        folded[best_anchor] += float(weights[head]) * best_beta
        residual_allowance += float(weights[head]) * float(
            torch.linalg.vector_norm(residual)
        )

    if omitted_negative:
        seed_activation = (queries[omitted_negative] @ keys[chosen_seed_ids].T) > 0
        frequencies = seed_activation.double().mean(dim=1)
        support = tuple(
            head
            for head, frequency in zip(omitted_negative, frequencies.tolist())
            if frequency >= 0.5
        )
    else:
        support = ()
    negative_aggregate = sum(
        (abs(float(weights[head])) * queries[head] for head in support),
        start=torch.zeros(queries.shape[1], dtype=torch.float64),
    )

    bound = torch.zeros(visible_keys.shape[0], dtype=torch.float64)
    for head in retained:
        bound += folded[head] * torch.relu(visible_keys @ queries[head])
    if residual_allowance:
        bound += residual_allowance * torch.linalg.vector_norm(visible_keys, dim=1)
    if support:
        bound -= torch.relu(visible_keys @ negative_aggregate)
    minimum_margin = float((bound - full_scores).min())
    tolerance = 1e-10 * max(1.0, float(full_scores.abs().max()))
    if minimum_margin < -tolerance:
        raise AssertionError("real-arithmetic DoSA bound does not dominate the scorer")

    seed_set = set(int(value) for value in chosen_seed_ids.tolist())
    survivors = tuple(
        int(token_id)
        for token_id, upper in zip(visible_ids.tolist(), bound.tolist())
        if token_id not in seed_set and upper >= threshold
    )
    recovered = (
        config.candidate_capacity is not None
        and len(survivors) > config.candidate_capacity
    )
    if recovered:
        selected = exact
        candidates = tuple(int(value) for value in visible_ids.tolist())
    else:
        candidate_scores = {
            token_id: float(full_scores[token_id - visible_start])
            for token_id in (*seed_set, *survivors)
        }
        kept = {
            token_id: score
            for token_id, score in candidate_scores.items()
            if token_id in seed_set or score >= threshold
        }
        ordered_ids = torch.tensor(sorted(kept), dtype=torch.long)
        ordered_scores = torch.tensor(
            [kept[int(token_id)] for token_id in ordered_ids], dtype=torch.float64
        )
        selected = _stable_topk(ordered_scores, ordered_ids, config.topk)
        candidates = tuple(sorted(kept))

    return DosaRealResult(
        selected=selected,
        exact=exact,
        seed_ids=tuple(int(value) for value in chosen_seed_ids.tolist()),
        candidate_ids=candidates,
        retained_heads=retained,
        omitted_positive_heads=tuple(omitted_positive),
        negative_support_heads=support,
        threshold=threshold,
        minimum_bound_margin=minimum_margin,
        recovered_full_domain=recovered,
    )
