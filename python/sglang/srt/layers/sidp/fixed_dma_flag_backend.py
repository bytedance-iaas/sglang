"""Generation-flag RAW/WAR for fixed-order DMA without dynamic-owner state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.sidp import (
    load_sidp_sm_copy_module,
    publish_generation,
    reset_forward_state,
    wait_generation,
)
from sglang.srt.layers.sidp.fixed_dma_slice_plan import SidpFixedDmaSlicePlan
from sglang.srt.layers.sidp.scheduler import (
    cycle_fill_generation,
    cycle_reuse_requirement,
)

if TYPE_CHECKING:
    from sglang.srt.layers.sidp.sidp_manager import SidpManager


class SidpFixedDmaFlagBackend(SidpFixedDmaSlicePlan):
    """Keep fixed DMA copies while replacing per-slot Events with generations.

    This deliberately does not inherit ``SidpCycleBackend``: fixed DMA has no
    runtime-selected source, so it must not allocate owner-control state, raw
    IPC mappings, indirect copy descriptors, or dynamic profiling traces.
    """

    BACKOFF_NS = 500
    DEVICE_TIMEOUT_S = 30

    def __init__(self, manager: SidpManager):
        super().__init__(manager)
        num_slots = manager._cycle_cache_depth * len(manager._remote_positions)
        if num_slots <= 0:
            raise RuntimeError("SiDP fixed DMA flag backend requires remote slots")

        properties = torch.cuda.get_device_properties(self.device)
        # torch reports clock_rate in kHz, i.e. cycles per millisecond.
        self.timeout_clocks = int(
            properties.clock_rate * 1000 * self.DEVICE_TIMEOUT_S
        )
        self.fill_gen = torch.empty(num_slots, dtype=torch.int32, device=self.device)
        self.comp_gen = torch.empty(num_slots, dtype=torch.int32, device=self.device)
        self.error_state = torch.zeros(1, dtype=torch.int32, device=self.device)

        # Compile/load before model CUDA Graph capture starts.
        load_sidp_sm_copy_module()
        reset_forward_state(self.fill_gen, self.comp_gen, 0, self.error_state)
        torch.cuda.current_stream().synchronize()

    def reset_forward(self, *, cycle_zero_resident: bool) -> None:
        resident_slots = (
            len(self.manager._remote_positions) if cycle_zero_resident else 0
        )
        reset_forward_state(
            self.fill_gen, self.comp_gen, resident_slots, self.error_state
        )

    def wait_layer(self, layer_id: int) -> None:
        cycle = layer_id // self.manager.dp_size
        if cycle == 0:
            return
        wait_generation(
            self.fill_gen,
            self.manager._layer_to_slot[layer_id],
            cycle_fill_generation(cycle, self.manager._cycle_cache_depth),
            self.BACKOFF_NS,
            self.timeout_clocks,
            self.error_state,
        )

    def record_consumed(self, layer_id: int) -> None:
        cycle = layer_id // self.manager.dp_size
        publish_generation(
            self.comp_gen,
            self.manager._layer_to_slot[layer_id],
            cycle_fill_generation(cycle, self.manager._cycle_cache_depth),
        )

    def enqueue_cycle(
        self,
        cycle: int,
        *,
        wait_for_consume: bool = True,
        target_fill_gen: int | None = None,
        required_comp_gen: int | None = None,
    ) -> None:
        if target_fill_gen is None:
            target_fill_gen = cycle_fill_generation(
                cycle, self.manager._cycle_cache_depth
            )
        if required_comp_gen is None:
            required_comp_gen = cycle_reuse_requirement(
                cycle, self.manager._cycle_cache_depth
            )

        with torch.cuda.stream(self.manager.comm_stream):
            for group in self._cycle_groups[cycle]:
                for slice_index in range(self.manager.dma_slices):
                    for layer_plan in group:
                        if slice_index == 0:
                            # WAR is per slot generation, not per fragment.
                            # Once the old reader is done, every slice may
                            # overwrite its disjoint byte range in stream order.
                            # Cycle 1 is submitted at the forward boundary with
                            # fresh/safe slots, so it must not launch even an
                            # immediately-satisfied generation wait.
                            if wait_for_consume and required_comp_gen > 0:
                                wait_generation(
                                    self.comp_gen,
                                    layer_plan.slot,
                                    required_comp_gen,
                                    self.BACKOFF_NS,
                                    self.timeout_clocks,
                                    self.error_state,
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
                            # RAW is published only after every main/extra
                            # fragment of this layer. Earlier fragments may be
                            # interleaved with other owners in the same group.
                            publish_generation(
                                self.fill_gen,
                                layer_plan.slot,
                                target_fill_gen,
                            )

    def trace_snapshot(self) -> None:
        return None
