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
