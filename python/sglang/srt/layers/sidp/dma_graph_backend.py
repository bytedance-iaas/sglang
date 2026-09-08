"""GPU-selected DMA for eager models and Full decode CUDA Graphs.

Each graph resets its cycle's done[] and unrolls candidate_count steps. Each
step claims one WAR-ready owner, executes its fixed-address DMA branch, then
releases the owner and publishes that slot's fill generation. Compute can use
each published slot before the rest of the graph finishes. During model
capture the same builder appends nodes directly to the current comm stream's
parent graph; it never nests/clones a standalone conditional graph.
"""

from __future__ import annotations

import logging

import torch

from sglang.kernels.ops.sidp.dma_graph import load_sidp_dma_graph_module
from sglang.srt.layers.sidp.cycle_backend import SidpCycleBackend
from sglang.srt.layers.sidp.scheduler import (
    cycle_fill_generation,
    cycle_reuse_requirement,
    next_forward_cycle_zero_generations,
)

logger = logging.getLogger(__name__)


class SidpDmaGraphBackend(SidpCycleBackend):
    def __init__(self, manager):
        if manager.prefetch_policy != "dynamic_owner" or manager.copy_backend != "dma":
            raise ValueError("conditional DMA requires dynamic_owner + dma")
        if manager.enable_graph_profiling and manager.config.disable_cuda_graph:
            raise ValueError("SiDP model-Graph profiling does not support eager DMA graphs")
        self._graphs: dict[tuple[int, int, int], int] = {}
        self._build_args: dict[tuple[int, int, int], tuple] = {}
        self._copies: dict[int, torch.Tensor] = {}
        self._module = load_sidp_dma_graph_module()
        super().__init__(manager)
        try:
            with torch.cuda.device(self.device), torch.cuda.stream(manager.comm_stream):
                for cycle in self._plans:
                    self._copies[cycle] = torch.tensor(
                        [
                            component.dma_copies
                            for component in self._plans[cycle].components
                        ],
                        dtype=torch.uint64,
                        device="cpu",
                    )
                    self._build_graph(
                        cycle,
                        cycle_reuse_requirement(cycle, manager._cycle_cache_depth),
                        cycle_fill_generation(cycle, manager._cycle_cache_depth),
                    )
                # c0's ordinary (0, 1) graph primes the first forward. Tail
                # refill uses the same addresses but a different generation;
                # prebuild it too, never mutate an in-flight graph's parameters.
                required, target = next_forward_cycle_zero_generations(
                    manager._num_cycles, manager._cycle_cache_depth
                )
                self._build_graph(0, required, target)
            manager.comm_stream.synchronize()  # setup-only graph upload join
        except Exception:
            self.close()
            raise
        logger.warning(
            "[SiDP rank%d] experimental dynamic_owner + DMA ready: %d eager "
            "cycle graphs (including c0 init/refill), model=%s, transport=%s, "
            "slot_sync=flag, claim_order=%s, dp_sync=none",
            manager.dp_rank,
            len(self._graphs),
            "eager" if manager.config.disable_cuda_graph else "full_graph",
            (
                "standalone_conditional_dma"
                if manager.config.disable_cuda_graph
                else "inline_conditional_dma (eager prefill/fallback: standalone)"
            ),
            manager.dynamic_claim_order,
        )

    def _build_graph(self, cycle: int, required: int, target: int) -> None:
        plan = self._plans[cycle]
        copies = self._copies[cycle]
        profiler = self.manager._graph_profiler
        if profiler is not None:
            events = profiler.prepare_dma_cycle_events(
                plan.layers,
                [
                    sum(c.dma_copies[i][2] for c in plan.components)
                    for i in range(plan.candidate_count)
                ],
                self.manager.comm_stream,
            )
            profile_args = (
                self._selected_trace,
                self._spins_trace,
                self._collisions_trace,
                events,
                plan.trace_offset,
            )
        else:
            # Unused aliases keep the FFI signature uniform without allocating
            # any profiling tensors/events or adding trace nodes when disabled.
            profile_args = (
                self.selected,
                self.claim_spins,
                self.claim_collisions,
                copies,
                -1,
            )
        key = (cycle, required, target)
        self._build_args[key] = (
            self.owner_state_ptrs,
            plan.candidate_owners,
            plan.candidate_slots,
            plan.done,
            self.fill_gen,
            self.comp_gen,
            self.probe_cursor,
            self.selected,
            self.claim_spins,
            self.claim_collisions,
            self.error_state,
            copies,
            required,
            target,
            self.claim_order,
            self.manager.dp_rank,
            self.BACKOFF_NS,
            self.timeout_clocks,
            *profile_args,
            # SMID/globaltimer tracing is owned by the standalone Nsight
            # pipeline tool.  These aliases are ignored while tag_base=-1,
            # preserving the production graph exactly.
            copies,
            self.selected,
            self.selected,
            -1,
        )
        self._graphs[key] = self._module.create(*self._build_args[key])

    def enqueue_cycle(
        self,
        cycle: int,
        *,
        target_fill_gen: int | None = None,
        required_comp_gen: int | None = None,
    ) -> None:
        if required_comp_gen is None:
            required_comp_gen = cycle_reuse_requirement(cycle, self.manager._cycle_cache_depth)
        if target_fill_gen is None:
            target_fill_gen = cycle_fill_generation(cycle, self.manager._cycle_cache_depth)
        key = (cycle, required_comp_gen, target_fill_gen)
        # Descriptors/JIT are setup-cached. Graph construction happens only
        # during capture; replay performs no Python dispatch or host selection.
        with torch.cuda.stream(self.manager.comm_stream):
            if torch.cuda.is_current_stream_capturing():
                self._module.append_cycle_to_capture(*self._build_args[key])
            else:
                self._module.launch(self._graphs[key], self.selected)

    def close(self) -> None:
        if not self._graphs:
            return
        # All referenced tensors and IPC mappings are still owned by this
        # backend/manager until outstanding graph work has completed.
        self.manager.comm_stream.synchronize()
        for handle in self._graphs.values():
            self._module.destroy(handle)
        self._graphs.clear()
        # Keep descriptors, IPC tensors and timing events alive with the
        # backend: captured model graphs also reference them. This method owns
        # only standalone execs, never the borrowed model parent graphs.

    def __del__(self):
        try:
            self.close()
        except Exception:
            # CUDA may already be torn down, or the group may have fail-stopped.
            # In either case process teardown owns the remaining CUDA resources.
            pass
