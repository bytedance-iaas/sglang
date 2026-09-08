"""SM-copy transport; descriptors and slot flags are shared with DMA graphs."""

from __future__ import annotations

import logging

import torch

from sglang.kernels.ops.sidp import (
    claim_owner,
    copy_selected,
    publish_selected_fill,
    record_trace,
    release_owner,
    reset_cycle_state,
    select_fixed,
    wait_generation,
)
from sglang.srt.layers.sidp.cycle_backend import SidpCycleBackend, SmCyclePlan
from sglang.srt.layers.sidp.scheduler import cycle_fill_generation, cycle_reuse_requirement

logger = logging.getLogger(__name__)


class SidpSmBackend(SidpCycleBackend):
    """Fixed/dynamic SM copy, including the fixed-order Event control path."""

    COPY_GRID_MULTIPLIER = 4
    COPY_BLOCK_SIZE = 512

    def __init__(self, manager):
        super().__init__(manager)
        properties = torch.cuda.get_device_properties(self.device)
        self.grid_blocks = manager.sm_copy_ctas or (
            properties.multi_processor_count * self.COPY_GRID_MULTIPLIER
        )
        logger.info(
            "[SiDP rank%d] SM copy launch: ctas=%d, block=%d, device_sms=%d%s",
            manager.dp_rank,
            self.grid_blocks,
            self.COPY_BLOCK_SIZE,
            properties.multi_processor_count,
            " (auto=4xSM)" if manager.sm_copy_ctas == 0 else " (throttled)",
        )

    def _copy_selected_components(
        self,
        plan: SmCyclePlan,
        step: int,
        scheduled_layer: int,
        *,
        selected: torch.Tensor | None = None,
    ) -> None:
        """Copy one already-selected fixed/dynamic candidate on comm_stream."""
        selected = self.selected if selected is None else selected
        if self.manager._graph_profiler is not None:
            self.manager._graph_profiler.record_copy_start(
                scheduled_layer,
                sum(
                    self.manager.peer_views[scheduled_layer][name].nbytes
                    for name in self.manager.peer_views[scheduled_layer]
                ),
                self.manager.comm_stream,
            )

        if self._selected_trace is not None:
            record_trace(
                selected,
                self.claim_spins,
                self.claim_collisions,
                self._selected_trace,
                self._spins_trace,
                self._collisions_trace,
                plan.trace_offset + step,
            )

        for component in plan.components:
            copy_selected(
                component.src_ptrs,
                component.dst_ptrs,
                component.sizes,
                selected,
                self.grid_blocks,
                self.COPY_BLOCK_SIZE,
                self.error_state,
            )
            if (
                self.manager.enable_debug_logging
                and not torch.cuda.is_current_stream_capturing()
            ):
                try:
                    self.manager.comm_stream.synchronize()
                except torch.AcceleratorError as error:
                    raise RuntimeError(
                        "SiDP SM copy failed: "
                        f"rank={self.manager.dp_rank}, cycle={plan.cycle}, "
                        f"step={step}, scheduled_layer={scheduled_layer}, "
                        f"component={component.key}"
                    ) from error

    def enqueue_fixed_layer_event(self, layer_id: int) -> None:
        """Enqueue only the fixed-order SM copy; Manager owns Event RAW/WAR.

        This validation path deliberately emits no fill/comp generation kernel.
        A setup-initialized one-element selection descriptor avoids adding a
        runtime selector kernel, so Event-vs-DMA differs only in copy backend.
        """
        cycle = layer_id // self.manager.dp_size
        plan = self._plans[cycle]
        try:
            step = plan.layers.index(layer_id)
        except ValueError as error:
            raise RuntimeError(
                f"SiDP SM event-copy layer is absent from cycle plan: {layer_id}"
            ) from error
        with torch.cuda.stream(self.manager.comm_stream):
            self._copy_selected_components(
                plan,
                step,
                layer_id,
                selected=plan.fixed_selections[step],
            )
            if self.manager._graph_profiler is not None:
                self.manager._graph_profiler.record_copy_end(
                    layer_id, self.manager.comm_stream
                )

    def enqueue_cycle(
        self,
        cycle: int,
        *,
        target_fill_gen: int | None = None,
        required_comp_gen: int | None = None,
    ) -> None:
        plan = self._plans[cycle]
        if target_fill_gen is None:
            target_fill_gen = cycle_fill_generation(
                cycle, self.manager._cycle_cache_depth
            )
        if required_comp_gen is None:
            required_comp_gen = cycle_reuse_requirement(
                cycle, self.manager._cycle_cache_depth
            )

        with torch.cuda.stream(self.manager.comm_stream):
            reset_cycle_state(
                plan.done,
                self.selected,
                self.claim_spins,
                self.claim_collisions,
            )
            for step, scheduled_layer in enumerate(plan.layers):
                if self.dynamic:
                    assert self.owner_state_ptrs is not None
                    claim_owner(
                        self.owner_state_ptrs,
                        plan.candidate_owners,
                        plan.candidate_slots,
                        plan.done,
                        self.comp_gen,
                        required_comp_gen,
                        self.claim_order,
                        self.probe_cursor,
                        self.selected,
                        self.claim_spins,
                        self.claim_collisions,
                        self.manager.dp_rank,
                        self.BACKOFF_NS,
                        self.timeout_clocks,
                        self.error_state,
                    )
                else:
                    # Fixed SM is the control path for the same flag protocol.
                    # Unlike DMA it has no consume Event, so the comm stream
                    # must wait for the slot's COMPUTED generation explicitly.
                    wait_generation(
                        self.comp_gen,
                        self.manager._layer_to_slot[scheduled_layer],
                        required_comp_gen,
                        self.BACKOFF_NS,
                        self.timeout_clocks,
                        self.error_state,
                    )
                    select_fixed(self.selected, step)

                self._copy_selected_components(plan, step, scheduled_layer)

                if self.dynamic:
                    release_owner(
                        self.owner_state_ptrs,
                        plan.candidate_owners,
                        self.selected,
                        self.manager.dp_rank,
                        self.error_state,
                    )
                publish_selected_fill(
                    self.fill_gen,
                    plan.candidate_slots,
                    self.selected,
                    target_fill_gen,
                    self.error_state,
                )
                if self.manager._graph_profiler is not None:
                    self.manager._graph_profiler.record_copy_end(
                        scheduled_layer, self.manager.comm_stream
                    )
