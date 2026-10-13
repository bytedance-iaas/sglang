# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only SketchSSM KDA reference and transactional lifecycle prototype.

The numerical formulation follows the Apache-2.0 SketchSSM FP64 KDA oracle at
``SNU-ARC/SketchSSM@a149e5b629e242ff42003af9446540175301d68a``.  This module
does not register a serving backend or load calibration assets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import torch


class SketchStateError(ValueError):
    pass


class SketchLifecycle(str, Enum):
    LIVE = "live"
    CLOSED = "closed"
    RELEASED = "released"


@dataclass(frozen=True, slots=True)
class SketchCommitReceipt:
    start: int
    end: int
    accepted_drafts: int
    committed_bonus: int
    discarded_rows: int
    flushed: bool


@dataclass(frozen=True, slots=True)
class SketchStepOutput:
    approximate: torch.Tensor
    exact: torch.Tensor
    flushed: bool


@dataclass(slots=True)
class _Snapshot:
    state: torch.Tensor
    window_start: torch.Tensor
    transition: torch.Tensor
    replay: torch.Tensor
    position: int
    maps: list[torch.Tensor | None]

    def clone(self) -> _Snapshot:
        return _Snapshot(
            state=self.state.clone(),
            window_start=self.window_start.clone(),
            transition=self.transition.clone(),
            replay=self.replay.clone(),
            position=self.position,
            maps=[None if value is None else value.clone() for value in self.maps],
        )


def _coefficient_map(
    state: torch.Tensor,
    frame: torch.Tensor,
    rank: int,
    *,
    pivots: int,
    ridge: float,
) -> torch.Tensor:
    """The FP64 four-pivot residual-diagonal map used by SketchSSM KDA."""
    projected = state @ frame
    energy = projected.square().sum() / state.shape[-1]
    if energy > 0:
        projected = projected / energy.sqrt()

    basis: list[torch.Tensor] = []
    for index in range(min(rank, pivots)):
        original = projected[:, index]
        vector = original.clone()
        for _ in range(2):
            for previous in basis:
                vector = vector - previous * (previous @ vector)
        if vector.square().sum() > 1e-12 * original.square().sum():
            vector = vector / vector.norm()
        else:
            vector = torch.zeros_like(vector)
        basis.append(vector)

    if basis:
        q = torch.stack(basis)
        q_projected = q @ projected
        residual = (projected.square().sum(0) - q_projected.square().sum(0)).clamp_min(
            0
        )
        residual[: min(rank, pivots)] = 0
        metric = q_projected.T @ q_projected + torch.diag(residual)
    else:
        metric = torch.diag(projected.square().sum(0))

    selected = metric[:rank, :rank]
    regularized = selected + ridge * torch.eye(
        rank, dtype=state.dtype, device=state.device
    )
    weights = torch.linalg.solve(regularized, metric[:rank])
    return frame @ weights.T


class SketchKDAReference:
    """CPU reference and lifecycle contract for a full-write KDA sketch.

    The persistent recurrent state is always updated by the exact KDA
    recurrence.  Between flushes only the readout is approximated.  A
    speculative target pass is staged on a clone and becomes persistent only
    through :meth:`commit`, so physical verify rows never advance the window
    position by themselves.

    This is deliberately not a serving backend.  In particular, it consumes
    no calibration bundle and cannot establish model quality or performance.
    """

    def __init__(
        self,
        *,
        state: torch.Tensor,
        frames: torch.Tensor,
        ranks: Sequence[int],
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        window: int = 16,
        lower_bound: float = -5.0,
        pivots: int = 4,
        ridge: float = 0.003,
        slot_generation: int = 1,
    ) -> None:
        state = state.detach().to(device="cpu", dtype=torch.float64).contiguous()
        frames = frames.detach().to(device="cpu", dtype=torch.float64).contiguous()
        a_log = a_log.detach().to(device="cpu", dtype=torch.float64).contiguous()
        dt_bias = dt_bias.detach().to(device="cpu", dtype=torch.float64).contiguous()
        if state.ndim != 3 or state.shape[-1] != state.shape[-2]:
            raise SketchStateError("KDA reference requires square [H, V, K] state")
        heads, dim, _ = state.shape
        if frames.shape != (heads, dim, dim):
            raise SketchStateError("frames must have shape [H, K, K]")
        if a_log.shape != (heads,) or dt_bias.shape != (heads, dim):
            raise SketchStateError("a_log/dt_bias do not match state heads")
        if len(ranks) != heads or any(rank < 0 or rank > dim for rank in ranks):
            raise SketchStateError("ranks must contain one value in [0, K] per head")
        if window <= 0 or pivots <= 0 or ridge <= 0:
            raise SketchStateError("window, pivots, and ridge must be positive")
        if slot_generation <= 0:
            raise SketchStateError("slot_generation must be positive")

        self.frames = frames
        self.ranks = tuple(int(rank) for rank in ranks)
        self.a_log = a_log
        self.dt_bias = dt_bias
        self.window = int(window)
        self.lower_bound = float(lower_bound)
        self.pivots = int(pivots)
        self.ridge = float(ridge)
        self.slot_generation = int(slot_generation)
        self.committed_tokens = 0
        self.lifecycle = SketchLifecycle.LIVE
        identity = torch.eye(dim, dtype=torch.float64).expand(heads, dim, dim)
        self._snapshot = _Snapshot(
            state=state,
            window_start=state.clone(),
            transition=identity.clone(),
            replay=torch.zeros_like(state),
            position=0,
            maps=[None] * heads,
        )

    @property
    def state(self) -> torch.Tensor:
        return self._snapshot.state.clone()

    @property
    def position(self) -> int:
        return self._snapshot.position

    def _require_live(self) -> None:
        if self.lifecycle is not SketchLifecycle.LIVE:
            raise SketchStateError(
                f"state lifecycle is {self.lifecycle.value}, not live"
            )

    def _maps_at_window_start(self, snapshot: _Snapshot) -> None:
        if snapshot.position != 0 or any(value is not None for value in snapshot.maps):
            return
        maps: list[torch.Tensor | None] = []
        dim = snapshot.state.shape[-1]
        for head, rank in enumerate(self.ranks):
            if rank in (0, dim):
                maps.append(None)
            else:
                maps.append(
                    _coefficient_map(
                        snapshot.window_start[head],
                        self.frames[head],
                        rank,
                        pivots=self.pivots,
                        ridge=self.ridge,
                    )
                )
        snapshot.maps = maps

    def _step(
        self,
        snapshot: _Snapshot,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
    ) -> SketchStepOutput:
        heads, dim, _ = snapshot.state.shape
        tensors = []
        for name, value, shape in (
            ("q", q, (heads, dim)),
            ("k", k, (heads, dim)),
            ("v", v, (heads, dim)),
            ("gate", gate, (heads, dim)),
            ("beta", beta, (heads,)),
        ):
            value = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
            if value.shape != shape:
                raise SketchStateError(f"{name} must have shape {shape}")
            tensors.append(value)
        q, k, v, gate, beta = tensors

        self._maps_at_window_start(snapshot)
        q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() / dim**0.5
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        decay = torch.exp(
            self.lower_bound
            * torch.sigmoid(self.a_log.exp()[:, None] * (gate + self.dt_bias))
        )
        beta = beta.sigmoid()

        exact = torch.empty((heads, dim), dtype=torch.float64)
        approximate = torch.empty_like(exact)
        is_flush = snapshot.position == self.window - 1
        for head, rank in enumerate(self.ranks):
            key = k[head]
            value = v[head]
            weight = beta[head]
            head_decay = decay[head]
            for matrix, writes_value in (
                (snapshot.state, True),
                (snapshot.replay, True),
                (snapshot.transition, False),
            ):
                decayed = matrix[head] * head_decay[None]
                delta = -(decayed @ key) * weight
                if writes_value:
                    delta = delta + weight * value
                matrix[head] = decayed + delta[:, None] * key[None]

            exact[head] = snapshot.state[head] @ q[head]
            if rank in (0, dim) or is_flush:
                approximate[head] = exact[head]
            else:
                coefficient = snapshot.maps[head]
                if coefficient is None:
                    raise SketchStateError("missing coefficient map for sketch head")
                query_coefficients = coefficient.T @ (
                    snapshot.transition[head] @ q[head]
                )
                approximate[head] = (
                    snapshot.window_start[head]
                    @ self.frames[head, :, :rank]
                    @ query_coefficients
                    + snapshot.replay[head] @ q[head]
                )

        if is_flush:
            snapshot.window_start.copy_(snapshot.state)
            snapshot.transition.copy_(
                torch.eye(dim, dtype=torch.float64).expand(heads, dim, dim)
            )
            snapshot.replay.zero_()
            snapshot.position = 0
            snapshot.maps = [None] * heads
        else:
            snapshot.position += 1
        return SketchStepOutput(approximate=approximate, exact=exact, flushed=is_flush)

    def stage(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        proposed_drafts: int | None = None,
    ) -> SketchTransaction:
        self._require_live()
        if q.ndim != 3:
            raise SketchStateError("staged q must have shape [T, H, K]")
        rows = q.shape[0]
        if rows == 0 or any(value.shape[0] != rows for value in (k, v, gate, beta)):
            raise SketchStateError("all staged inputs need the same nonzero T")
        if proposed_drafts is None:
            proposed_drafts = rows - 1
        if proposed_drafts < 0 or proposed_drafts + 1 != rows:
            raise SketchStateError(
                "target verify must stage proposed drafts plus one bonus row"
            )

        working = self._snapshot.clone()
        snapshots: list[_Snapshot] = []
        outputs: list[SketchStepOutput] = []
        for index in range(rows):
            outputs.append(
                self._step(
                    working,
                    q=q[index],
                    k=k[index],
                    v=v[index],
                    gate=gate[index],
                    beta=beta[index],
                )
            )
            snapshots.append(working.clone())
        return SketchTransaction(
            owner=self,
            base_generation=self.slot_generation,
            base_committed_tokens=self.committed_tokens,
            proposed_drafts=proposed_drafts,
            snapshots=snapshots,
            outputs=tuple(outputs),
        )

    def close(self) -> None:
        self._require_live()
        self.lifecycle = SketchLifecycle.CLOSED

    def release(self) -> None:
        if self.lifecycle is SketchLifecycle.RELEASED:
            raise SketchStateError("state is already released")
        self.lifecycle = SketchLifecycle.RELEASED

    def replace_slot(self, *, state: torch.Tensor, slot_generation: int) -> None:
        if slot_generation <= self.slot_generation:
            raise SketchStateError("replacement slot generation must increase")
        state = state.detach().to(device="cpu", dtype=torch.float64).contiguous()
        if state.shape != self._snapshot.state.shape:
            raise SketchStateError("replacement state shape mismatch")
        heads, dim, _ = state.shape
        self.slot_generation = slot_generation
        self.committed_tokens = 0
        self.lifecycle = SketchLifecycle.LIVE
        self._snapshot = _Snapshot(
            state=state,
            window_start=state.clone(),
            transition=torch.eye(dim, dtype=torch.float64)
            .expand(heads, dim, dim)
            .clone(),
            replay=torch.zeros_like(state),
            position=0,
            maps=[None] * heads,
        )


@dataclass(slots=True)
class SketchTransaction:
    owner: SketchKDAReference
    base_generation: int
    base_committed_tokens: int
    proposed_drafts: int
    snapshots: list[_Snapshot]
    outputs: tuple[SketchStepOutput, ...]
    _finished: bool = False

    def _validate_open(self) -> None:
        if self._finished:
            raise SketchStateError("transaction is already finished")
        self.owner._require_live()
        if self.owner.slot_generation != self.base_generation:
            raise SketchStateError("staged state belongs to a stale slot generation")
        if self.owner.committed_tokens != self.base_committed_tokens:
            raise SketchStateError("persistent commit frontier changed after staging")

    def commit(
        self, *, accepted_drafts: int, committed_bonus: int
    ) -> SketchCommitReceipt:
        self._validate_open()
        if accepted_drafts < 0 or accepted_drafts > self.proposed_drafts:
            raise SketchStateError("accepted_drafts is outside the proposed range")
        if committed_bonus not in (0, 1):
            raise SketchStateError("committed_bonus must be zero or one")
        count = accepted_drafts + committed_bonus
        if count > len(self.snapshots):
            raise SketchStateError("commit exceeds staged physical rows")

        start = self.owner.committed_tokens
        flushed = False
        if count:
            self.owner._snapshot = self.snapshots[count - 1].clone()
            flushed = any(output.flushed for output in self.outputs[:count])
            self.owner.committed_tokens += count
        self._finished = True
        return SketchCommitReceipt(
            start=start,
            end=self.owner.committed_tokens,
            accepted_drafts=accepted_drafts,
            committed_bonus=committed_bonus,
            discarded_rows=len(self.snapshots) - count,
            flushed=flushed,
        )

    def abort(self) -> None:
        self._validate_open()
        self._finished = True
