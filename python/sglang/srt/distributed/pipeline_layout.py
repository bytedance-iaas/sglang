from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PipelineStage:
    stage_id: int
    physical_rank: int
    layer_ids: Tuple[int, ...]


@dataclass(frozen=True)
class PipelineWavefrontAction:
    tick: int
    batch_seq: int
    slot_id: int
    stage_id: int
    physical_rank: int


@dataclass(frozen=True)
class PipelineWavefrontSchedule:
    physical_size: int
    virtual_stages: int
    wave_size: int

    @classmethod
    def build(
        cls,
        physical_size: int,
        virtual_stages: int,
        wave_size: Optional[int] = None,
    ) -> "PipelineWavefrontSchedule":
        if physical_size < 1:
            raise ValueError("physical pipeline size must be positive")
        if virtual_stages != 2:
            raise ValueError("the deterministic wavefront supports VPP2 only")
        if wave_size is None:
            wave_size = physical_size
        if wave_size != physical_size:
            raise ValueError(
                "the deterministic wavefront requires one slot per PP rank"
            )
        return cls(
            physical_size=physical_size,
            virtual_stages=virtual_stages,
            wave_size=wave_size,
        )

    @property
    def logical_size(self) -> int:
        return self.physical_size * self.virtual_stages

    @property
    def num_ticks(self) -> int:
        return self.wave_size + self.logical_size - 1

    def batch_seqs(self, first_batch_seq: int) -> range:
        if first_batch_seq < 0:
            raise ValueError("first wavefront batch sequence must be non-negative")
        return range(first_batch_seq, first_batch_seq + self.wave_size)

    def completion_batch_seq(
        self, tick: int, first_batch_seq: int = 0
    ) -> Optional[int]:
        action = self.action(tick, self.physical_size - 1, first_batch_seq)
        if action is None or action.stage_id != self.logical_size - 1:
            return None
        return action.batch_seq

    def action(
        self,
        tick: int,
        physical_rank: int,
        first_batch_seq: int = 0,
    ) -> Optional[PipelineWavefrontAction]:
        if not 0 <= physical_rank < self.physical_size:
            raise ValueError(f"invalid physical pipeline rank {physical_rank}")
        if first_batch_seq < 0:
            raise ValueError("first wavefront batch sequence must be non-negative")
        if not 0 <= tick < self.num_ticks:
            raise ValueError(
                f"wavefront tick must be in [0, {self.num_ticks}), got {tick}"
            )

        batch_offset = tick - physical_rank
        stage_id = physical_rank
        if not 0 <= batch_offset < self.wave_size:
            batch_offset = tick - physical_rank - self.physical_size
            stage_id += self.physical_size
        if not 0 <= batch_offset < self.wave_size:
            return None
        batch_seq = first_batch_seq + batch_offset
        return PipelineWavefrontAction(
            tick=tick,
            batch_seq=batch_seq,
            slot_id=batch_seq % self.wave_size,
            stage_id=stage_id,
            physical_rank=physical_rank,
        )


class PipelineReadyQueueSchedule:
    def __init__(
        self,
        physical_size: int,
        virtual_stages: int,
        max_inflight: int,
    ):
        if physical_size < 1:
            raise ValueError("physical pipeline size must be positive")
        if virtual_stages != 2:
            raise ValueError("the ready-queue scheduler supports VPP2 only")
        if max_inflight < 1:
            raise ValueError("max inflight batches must be positive")
        self.physical_size = physical_size
        self.virtual_stages = virtual_stages
        self.max_inflight = max_inflight
        self._ready: list[list[tuple[int, int]]] = [[] for _ in range(physical_size)]
        self._slot_batch_seqs: list[Optional[int]] = [None] * max_inflight

    @property
    def logical_size(self) -> int:
        return self.physical_size * self.virtual_stages

    @property
    def inflight_count(self) -> int:
        return sum(batch_seq is not None for batch_seq in self._slot_batch_seqs)

    @property
    def slot_batch_seqs(self) -> Tuple[Optional[int], ...]:
        return tuple(self._slot_batch_seqs)

    def can_admit(self, batch_seq: int) -> bool:
        if batch_seq < 0:
            return False
        slot_id = batch_seq % self.max_inflight
        return (
            self.inflight_count < self.max_inflight
            and self._slot_batch_seqs[slot_id] is None
        )

    def admit(self, batch_seq: int) -> None:
        if not self.can_admit(batch_seq):
            raise RuntimeError(f"cannot admit VPP batch {batch_seq}")
        slot_id = batch_seq % self.max_inflight
        self._slot_batch_seqs[slot_id] = batch_seq
        self._ready[0].append((batch_seq, 0))

    def actions(self, tick: int) -> Tuple[Optional[PipelineWavefrontAction], ...]:
        if tick < 0:
            raise ValueError("ready-queue tick must be non-negative")
        actions = []
        for physical_rank, ready in enumerate(self._ready):
            if not ready:
                actions.append(None)
                continue
            decode = [item for item in ready if item[1] >= self.physical_size]
            candidates = decode if decode else ready
            batch_seq, stage_id = min(candidates)
            ready.remove((batch_seq, stage_id))
            actions.append(
                PipelineWavefrontAction(
                    tick=tick,
                    batch_seq=batch_seq,
                    slot_id=batch_seq % self.max_inflight,
                    stage_id=stage_id,
                    physical_rank=physical_rank,
                )
            )
        return tuple(actions)

    def complete(
        self,
        actions: Tuple[Optional[PipelineWavefrontAction], ...],
    ) -> Tuple[int, ...]:
        if len(actions) != self.physical_size:
            raise ValueError("one action is required for every physical PP rank")
        completed = []
        for physical_rank, action in enumerate(actions):
            if action is None:
                continue
            if action.physical_rank != physical_rank:
                raise ValueError(
                    f"action for PP rank {action.physical_rank} was submitted "
                    f"at position {physical_rank}"
                )
            if self._slot_batch_seqs[action.slot_id] != action.batch_seq:
                raise RuntimeError(
                    f"VPP batch {action.batch_seq} does not own slot {action.slot_id}"
                )
            next_stage_id = action.stage_id + 1
            if next_stage_id == self.logical_size:
                self._slot_batch_seqs[action.slot_id] = None
                completed.append(action.batch_seq)
                continue
            next_rank = next_stage_id % self.physical_size
            self._ready[next_rank].append((action.batch_seq, next_stage_id))
        return tuple(sorted(completed))


@dataclass(frozen=True)
class PipelineLayout:
    num_hidden_layers: int
    physical_size: int
    virtual_stages: int
    stages: Tuple[PipelineStage, ...]

    @classmethod
    def build(
        cls,
        num_hidden_layers: int,
        physical_size: int,
        virtual_stages: int = 1,
        partition: Optional[Tuple[int, ...]] = None,
    ) -> "PipelineLayout":
        if physical_size < 1 or virtual_stages < 1:
            raise ValueError("pipeline sizes must be positive")
        logical_size = physical_size * virtual_stages
        if num_hidden_layers < logical_size:
            raise ValueError(
                f"{num_hidden_layers=} must be >= logical pipeline size {logical_size}"
            )

        if partition is None:
            base, remainder = divmod(num_hidden_layers, logical_size)
            partition = tuple(
                base + (stage_id >= logical_size - remainder)
                for stage_id in range(logical_size)
            )
        if len(partition) != logical_size:
            raise ValueError(
                f"{len(partition)=} does not match logical pipeline size {logical_size}"
            )
        if sum(partition) != num_hidden_layers:
            raise ValueError(f"{sum(partition)=} does not match {num_hidden_layers=}")

        stages = []
        start = 0
        for stage_id, layer_count in enumerate(partition):
            end = start + layer_count
            stages.append(
                PipelineStage(
                    stage_id=stage_id,
                    physical_rank=stage_id % physical_size,
                    layer_ids=tuple(range(start, end)),
                )
            )
            start = end

        layout = cls(
            num_hidden_layers=num_hidden_layers,
            physical_size=physical_size,
            virtual_stages=virtual_stages,
            stages=tuple(stages),
        )
        layout.validate()
        return layout

    @property
    def logical_size(self) -> int:
        return len(self.stages)

    @property
    def is_interleaved(self) -> bool:
        return self.virtual_stages > 1

    @property
    def digest(self) -> str:
        manifest = [
            (stage.stage_id, stage.physical_rank, stage.layer_ids)
            for stage in self.stages
        ]
        return hashlib.sha256(
            json.dumps(manifest, separators=(",", ":")).encode()
        ).hexdigest()

    def stage(self, stage_id: int) -> PipelineStage:
        if not 0 <= stage_id < self.logical_size:
            raise ValueError(f"invalid logical pipeline stage {stage_id}")
        return self.stages[stage_id]

    def stages_for_rank(self, physical_rank: int) -> Tuple[PipelineStage, ...]:
        return tuple(
            stage for stage in self.stages if stage.physical_rank == physical_rank
        )

    def layer_ids_for_rank(self, physical_rank: int) -> Tuple[int, ...]:
        return tuple(
            layer_id
            for stage in self.stages_for_rank(physical_rank)
            for layer_id in stage.layer_ids
        )

    def validate(self) -> None:
        if len(self.stages) != self.physical_size * self.virtual_stages:
            raise ValueError("logical stage count does not match pipeline sizes")
        layer_ids = tuple(
            layer_id for stage in self.stages for layer_id in stage.layer_ids
        )
        if layer_ids != tuple(range(self.num_hidden_layers)):
            raise ValueError("logical stages must cover every layer exactly once")
        for stage in self.stages:
            if stage.physical_rank != stage.stage_id % self.physical_size:
                raise ValueError("interleaved stage owner does not match stage order")
