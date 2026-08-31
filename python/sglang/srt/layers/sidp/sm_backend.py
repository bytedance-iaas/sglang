"""SM-copy transport and dynamic-owner arbitration for SiDP."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.sidp import (
    claim_owner,
    copy_selected,
    load_sidp_sm_copy_module,
    native_peer_atomic_supported,
    publish_generation,
    publish_selected_fill,
    record_trace,
    release_owner,
    reset_cycle_state,
    reset_forward_state,
    select_fixed,
    wait_generation,
)
from sglang.srt.layers.sidp.scheduler import (
    cycle_fill_generation,
    cycle_reuse_requirement,
    owner_of,
)

if TYPE_CHECKING:
    from sglang.srt.layers.sidp.sidp_manager import SidpManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SmCopyComponentPlan:
    """Graph-stable pointer descriptors for one encoded component."""

    key: tuple[str, str]
    src_ptrs: torch.Tensor
    dst_ptrs: torch.Tensor
    sizes: torch.Tensor


@dataclass
class SmCyclePlan:
    cycle: int
    layers: tuple[int, ...]
    candidate_owners: torch.Tensor
    candidate_slots: torch.Tensor
    components: tuple[SmCopyComponentPlan, ...]
    done: torch.Tensor
    trace_offset: int = 0

    @property
    def candidate_count(self) -> int:
        return len(self.layers)


class SidpSmBackend:
    """Own the state used by both fixed-order and dynamic-owner SM copy."""

    BACKOFF_NS = 500
    DEVICE_TIMEOUT_S = 30
    COPY_GRID_MULTIPLIER = 4

    def __init__(self, manager: SidpManager):
        self.manager = manager
        self.dynamic = manager.prefetch_policy == "dynamic_owner"
        self.device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(self.device)
        self.grid_blocks = (
            properties.multi_processor_count * self.COPY_GRID_MULTIPLIER
        )
        # torch reports clock_rate in kHz, i.e. cycles per millisecond.
        self.timeout_clocks = int(
            properties.clock_rate * 1000 * self.DEVICE_TIMEOUT_S
        )

        num_slots = len(manager._layer_to_slot) and (
            manager._cycle_cache_depth * len(manager._remote_positions)
        )
        if not num_slots:
            raise RuntimeError("SiDP SM backend requires at least one remote slot")

        self.fill_gen = torch.empty(num_slots, dtype=torch.int32, device=self.device)
        self.comp_gen = torch.empty(num_slots, dtype=torch.int32, device=self.device)
        self.selected = torch.empty(1, dtype=torch.int32, device=self.device)
        self.probe_cursor = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.claim_spins = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.claim_collisions = torch.zeros(
            1, dtype=torch.int64, device=self.device
        )
        self.error_state = torch.zeros(1, dtype=torch.int32, device=self.device)

        self._local_owner_state: torch.Tensor | None = None
        self.owner_state_ptrs: torch.Tensor | None = None
        self._owner_state_ptr_values: list[int] = []
        self._owner_state_snapshot: torch.Tensor | None = None
        self._plans: dict[int, SmCyclePlan] = {}
        self._ipc_allocation_bases: dict[bytes, int] = {}

        self._selected_trace: torch.Tensor | None = None
        self._spins_trace: torch.Tensor | None = None
        self._collisions_trace: torch.Tensor | None = None

        # Force JIT compilation before any CUDA Graph capture begins.
        load_sidp_sm_copy_module()
        if self.dynamic:
            self._setup_distributed_owner_state()
        self._build_cycle_plans()
        self._allocate_trace_if_enabled()

        reset_forward_state(
            self.fill_gen, self.comp_gen, 0, self.error_state
        )
        torch.cuda.current_stream().synchronize()

    def _setup_distributed_owner_state(self) -> None:
        manager = self.manager
        support = []
        for owner in range(manager.dp_size):
            if owner == self.device:
                support.append(True)
                continue
            supported = native_peer_atomic_supported(self.device, owner)
            support.append(supported)
            if not supported:
                raise RuntimeError(
                    "SiDP dynamic_owner requires native peer atomics for every "
                    f"requester/owner pair; requester={self.device}, owner={owner}"
                )
        logger.warning(
            "[SiDP rank%d] dynamic_owner native peer atomics: %s",
            manager.dp_rank,
            support,
        )

        # Each owner exports a separate allocation. This avoids a central GPU0
        # control hot spot and prevents unrelated owners sharing one cache line.
        self._local_owner_state = torch.full(
            (1,), -1, dtype=torch.int32, device=self.device
        )
        torch.cuda.current_stream().synchronize()
        local_descriptor = manager.memcpy.export_ipc_pointer(
            self._local_owner_state.data_ptr(), self._local_owner_state.nbytes
        )
        manager.store.set(
            f"sidp/dynamic_owner/control/{manager.dp_rank}",
            pickle.dumps(local_descriptor),
        )

        owner_state_ptrs = []
        for owner in range(manager.dp_size):
            if owner == manager.dp_rank:
                pointer = self._local_owner_state.data_ptr()
            else:
                payload = manager.store.get(f"sidp/dynamic_owner/control/{owner}")
                pointer = self._resolve_source_pointer(pickle.loads(payload))
            owner_state_ptrs.append(pointer)

        self.owner_state_ptrs = torch.tensor(
            owner_state_ptrs,
            dtype=torch.uint64,
            device=self.device,
        )
        self._owner_state_ptr_values = owner_state_ptrs

    def _resolve_source_pointer(self, descriptor: dict) -> int:
        handle = descriptor["handle"]
        if handle not in self._ipc_allocation_bases:
            self._ipc_allocation_bases[handle] = (
                self.manager.memcpy.open_ipc_allocation(handle)
            )
        return self._ipc_allocation_bases[handle] + int(descriptor["offset"])

    def _flatten_components(self, manager: SidpManager, layer_id: int, slot: int):
        flattened = []
        for param_name, source in manager.peer_views[layer_id].items():
            destination = manager._transfer_buffers[slot][param_name]
            ipc_descriptors = manager._peer_sm_ipc[layer_id][param_name]
            flattened.append(
                (
                    (param_name, "<main>"),
                    self._resolve_source_pointer(ipc_descriptors["<main>"]),
                    int(ipc_descriptors["<main>"]["nbytes"]),
                    destination.tensor,
                )
            )
            if set(source.extra_tensors) != set(destination.extra_tensors):
                raise RuntimeError(
                    "SiDP encoded extra schema changed while building SM descriptors: "
                    f"layer={layer_id}, param={param_name}"
                )
            for extra_name in sorted(source.extra_tensors):
                flattened.append(
                    (
                        (param_name, extra_name),
                        self._resolve_source_pointer(ipc_descriptors[extra_name]),
                        int(ipc_descriptors[extra_name]["nbytes"]),
                        destination.extra_tensors[extra_name],
                    )
                )
        return flattened

    def _build_cycle_plans(self) -> None:
        manager = self.manager
        trace_offset = 0
        for cycle in range(manager._num_cycles):
            layers = tuple(manager._cycle_layers.get(cycle, ()))
            if not layers:
                continue
            owners = [owner_of(layer_id, manager.dp_size) for layer_id in layers]
            if any(
                manager._layers_ref[layer_id] is None
                or layer_id not in manager.peer_views
                for layer_id in layers
            ):
                raise RuntimeError(
                    f"SiDP SM cycle {cycle} contains a local or missing layer"
                )
            if len(set(owners)) != len(owners):
                raise RuntimeError(
                    f"SiDP SM cycle {cycle} contains duplicate candidate owners"
                )
            slots = [manager._layer_to_slot[layer_id] for layer_id in layers]

            per_layer = [
                self._flatten_components(manager, layer_id, slot)
                for layer_id, slot in zip(layers, slots)
            ]
            component_keys = [item[0] for item in per_layer[0]]
            for layer_id, components in zip(layers, per_layer):
                if [item[0] for item in components] != component_keys:
                    raise RuntimeError(
                        "SiDP encoded component schema must be graph-stable within "
                        f"a cycle: cycle={cycle}, layer={layer_id}"
                    )

            component_plans = []
            for component_index, key in enumerate(component_keys):
                source_ptrs = [items[component_index][1] for items in per_layer]
                source_sizes = [items[component_index][2] for items in per_layer]
                destinations = [items[component_index][3] for items in per_layer]
                for layer_id, source_ptr, source_size, destination in zip(
                    layers, source_ptrs, source_sizes, destinations
                ):
                    if not destination.is_contiguous():
                        raise RuntimeError(
                            "SiDP SM copy requires contiguous encoded components: "
                            f"cycle={cycle}, layer={layer_id}, component={key}, "
                            f"destination_contiguous={destination.is_contiguous()}"
                        )
                    if source_ptr % 16 or destination.data_ptr() % 16:
                        raise RuntimeError(
                            "SiDP SM copy requires 16-byte aligned pointers: "
                            f"cycle={cycle}, layer={layer_id}, component={key}"
                        )
                    if destination.nbytes < source_size:
                        raise RuntimeError(
                            "SiDP SM destination is smaller than its source: "
                            f"cycle={cycle}, layer={layer_id}, component={key}"
                        )
                component_plans.append(
                    SmCopyComponentPlan(
                        key=key,
                        src_ptrs=torch.tensor(
                            source_ptrs,
                            dtype=torch.uint64,
                            device=self.device,
                        ),
                        dst_ptrs=torch.tensor(
                            [tensor.data_ptr() for tensor in destinations],
                            dtype=torch.uint64,
                            device=self.device,
                        ),
                        sizes=torch.tensor(
                            source_sizes,
                            dtype=torch.int64,
                            device=self.device,
                        ),
                    )
                )

            plan = SmCyclePlan(
                cycle=cycle,
                layers=layers,
                candidate_owners=torch.tensor(
                    owners, dtype=torch.int32, device=self.device
                ),
                candidate_slots=torch.tensor(
                    slots, dtype=torch.int32, device=self.device
                ),
                components=tuple(component_plans),
                done=torch.empty(len(layers), dtype=torch.uint8, device=self.device),
                trace_offset=trace_offset,
            )
            self._plans[cycle] = plan
            trace_offset += len(layers)

        if set(self._plans) != set(manager._cycle_layers):
            raise RuntimeError("SiDP SM descriptor coverage does not match cycles")

    def _allocate_trace_if_enabled(self) -> None:
        if not self.manager.enable_graph_profiling:
            return
        trace_size = sum(plan.candidate_count for plan in self._plans.values())
        self._selected_trace = torch.empty(
            trace_size, dtype=torch.int32, device=self.device
        )
        self._spins_trace = torch.empty(
            trace_size, dtype=torch.int64, device=self.device
        )
        self._collisions_trace = torch.empty(
            trace_size, dtype=torch.int64, device=self.device
        )
        if self.dynamic:
            self._owner_state_snapshot = torch.empty(
                self.manager.dp_size, dtype=torch.int32, device=self.device
            )

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
                if self.manager._graph_profiler is not None:
                    self.manager._graph_profiler.record_copy_start(
                        scheduled_layer,
                        sum(
                            self.manager.peer_views[scheduled_layer][name].nbytes
                            for name in self.manager.peer_views[scheduled_layer]
                        ),
                        self.manager.comm_stream,
                    )

                if self.dynamic:
                    assert self.owner_state_ptrs is not None
                    claim_owner(
                        self.owner_state_ptrs,
                        plan.candidate_owners,
                        plan.candidate_slots,
                        plan.done,
                        self.comp_gen,
                        required_comp_gen,
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

                if self._selected_trace is not None:
                    record_trace(
                        self.selected,
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
                        self.selected,
                        self.grid_blocks,
                        self.error_state,
                    )
                    if self.manager.enable_debug_logging and not torch.cuda.is_current_stream_capturing():
                        try:
                            self.manager.comm_stream.synchronize()
                        except torch.AcceleratorError as error:
                            raise RuntimeError(
                                "SiDP SM copy failed: "
                                f"rank={self.manager.dp_rank}, cycle={cycle}, "
                                f"step={step}, scheduled_layer={scheduled_layer}, "
                                f"component={component.key}"
                            ) from error

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

    def trace_snapshot(self) -> dict | None:
        if self._selected_trace is None:
            return None
        selected = self._selected_trace.cpu().tolist()
        spins = self._spins_trace.cpu().tolist()
        collisions = self._collisions_trace.cpu().tolist()
        records = []
        for cycle, plan in self._plans.items():
            for step in range(plan.candidate_count):
                trace_index = plan.trace_offset + step
                candidate_index = selected[trace_index]
                layer_id = (
                    plan.layers[candidate_index]
                    if 0 <= candidate_index < plan.candidate_count
                    else -1
                )
                records.append(
                    {
                        "cycle": cycle,
                        "step": step,
                        "candidate_index": candidate_index,
                        "layer": layer_id,
                        "owner": (
                            owner_of(layer_id, self.manager.dp_size)
                            if layer_id >= 0
                            else -1
                        ),
                        "claim_spins": spins[trace_index],
                        "claim_collisions": collisions[trace_index],
                    }
                )
        owner_states = None
        if self._owner_state_snapshot is not None:
            stream = torch.cuda.current_stream()
            for owner, source_ptr in enumerate(self._owner_state_ptr_values):
                self.manager.memcpy.async_copy(
                    self._owner_state_snapshot[owner : owner + 1].data_ptr(),
                    source_ptr,
                    4,
                    stream.cuda_stream,
                )
            stream.synchronize()
            owner_states = self._owner_state_snapshot.cpu().tolist()
        return {
            "steps": records,
            "owner_states_after_replay": owner_states,
            "error_state": self.error_state.item(),
        }

    def debug_validate_cycle(self, cycle: int) -> None:
        """Sample-check one filled cycle against the DMA-capable peer views."""
        plan = self._plans[cycle]
        for layer_id in plan.layers:
            slot = self.manager._layer_to_slot[layer_id]
            for param_name, source in self.manager.peer_views[layer_id].items():
                destination = self.manager._transfer_buffers[slot][param_name]
                pairs = [("<main>", source.tensor, destination.tensor)]
                pairs.extend(
                    (
                        extra_name,
                        source.extra_tensors[extra_name],
                        destination.extra_tensors[extra_name],
                    )
                    for extra_name in sorted(source.extra_tensors)
                )
                for component_name, source_tensor, destination_tensor in pairs:
                    count = source_tensor.numel()
                    indices = sorted(
                        {0, min(1, count - 1), count // 2, max(0, count - 2), count - 1}
                    )
                    source_sample = source_tensor.flatten()[indices].to(self.device)
                    destination_sample = destination_tensor.flatten()[indices]
                    if not torch.equal(source_sample, destination_sample):
                        raise RuntimeError(
                            "SiDP SM setup validation mismatch: "
                            f"rank={self.manager.dp_rank}, cycle={cycle}, "
                            f"layer={layer_id}, param={param_name}, "
                            f"component={component_name}, "
                            f"expected={source_sample.cpu().tolist()}, "
                            f"actual={destination_sample.cpu().tolist()}"
                        )
