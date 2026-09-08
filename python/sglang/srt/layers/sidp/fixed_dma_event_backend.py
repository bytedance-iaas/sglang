"""Fixed-order DMA slicing with legacy per-slot CUDA Event RAW/WAR."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.sidp.fixed_dma_slice_plan import SidpFixedDmaSlicePlan

if TYPE_CHECKING:
    from sglang.srt.layers.sidp.sidp_manager import SidpManager


class SidpFixedDmaEventBackend(SidpFixedDmaSlicePlan):
    """Submit one shared S/G copy plan using Event wait/record boundaries."""

    def __init__(self, manager: SidpManager):
        if manager.dma_slices <= 1:
            raise ValueError("SiDP fixed DMA Event slice backend requires slices > 1")
        super().__init__(manager)

    def enqueue_cycle(self, cycle: int, *, wait_for_consume: bool = True) -> None:
        with torch.cuda.stream(self.manager.comm_stream):
            for group in self._cycle_groups[cycle]:
                for slice_index in range(self.manager.dma_slices):
                    for layer_plan in group:
                        if slice_index == 0:
                            # WAR belongs to the reusable slot, not each byte
                            # fragment. One Event wait orders all later slices.
                            if wait_for_consume:
                                self.manager.comm_stream.wait_event(
                                    self.manager._consume_events[layer_plan.slot]
                                )
                            if self.manager._graph_profiler is not None:
                                self.manager._graph_profiler.record_copy_start(
                                    layer_plan.layer_id,
                                    layer_plan.total_nbytes,
                                    self.manager.comm_stream,
                                )
                        for fragment in layer_plan.slices[slice_index]:
                            self.manager.memcpy.async_copy(
                                fragment.destination_ptr,
                                fragment.source_ptr,
                                fragment.nbytes,
                                self.manager.comm_stream.cuda_stream,
                            )
                        if slice_index == self.manager.dma_slices - 1:
                            if self.manager._graph_profiler is not None:
                                self.manager._graph_profiler.record_copy_end(
                                    layer_plan.layer_id, self.manager.comm_stream
                                )
                            # RAW becomes visible only when this layer's main
                            # tensor and every extra component are complete.
                            self.manager._prefetch_events[layer_plan.slot].record(
                                self.manager.comm_stream
                            )

    def trace_snapshot(self) -> None:
        return None
