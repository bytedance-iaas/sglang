"""Graph-stable fixed-order DMA slice descriptors shared by Event and flag."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.sidp.scheduler import (
    contiguous_slice_groups,
    dma_slice_ranges,
)

if TYPE_CHECKING:
    from sglang.srt.layers.sidp.sidp_manager import SidpManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DmaCopyFragment:
    """One graph-stable contiguous memcpy node."""

    source_ptr: int
    destination_ptr: int
    nbytes: int


@dataclass(frozen=True)
class DmaLayerSlicePlan:
    """All encoded components of one layer, partitioned by logical slice."""

    layer_id: int
    slot: int
    total_nbytes: int
    slices: tuple[tuple[DmaCopyFragment, ...], ...]


class SidpFixedDmaSlicePlan:
    """Build and validate one immutable S/G plan for fixed DMA backends."""

    SMALL_DMA_FRAGMENT_BYTES = 32 * 1024 * 1024

    def __init__(self, manager: SidpManager):
        self.manager = manager
        self.device = torch.cuda.current_device()
        self._cycle_groups = self._build_cycle_groups()

    def _build_cycle_groups(
        self,
    ) -> dict[int, tuple[tuple[DmaLayerSlicePlan, ...], ...]]:
        manager = self.manager
        cycle_groups = {}
        small_fragments: list[tuple[int, str, int, int]] = []
        max_copy_nodes = 0

        for cycle, layer_ids in manager._cycle_layers.items():
            layer_plans = []
            copy_nodes = 0
            for layer_id in layer_ids:
                slot = manager._layer_to_slot[layer_id]
                per_slice: list[list[DmaCopyFragment]] = [
                    [] for _ in range(manager.dma_slices)
                ]
                total_nbytes = 0
                for param_name, source in manager.peer_views[layer_id].items():
                    destination = manager._transfer_buffers[slot][param_name]
                    if set(source.extra_tensors) != set(destination.extra_tensors):
                        raise RuntimeError(
                            "SiDP DMA slice extra-buffer schema mismatch: "
                            f"layer={layer_id}, param={param_name}"
                        )
                    components = [
                        ("<main>", source.tensor, destination.tensor),
                        *(
                            (
                                name,
                                source_tensor,
                                destination.extra_tensors[name],
                            )
                            for name, source_tensor in source.extra_tensors.items()
                        ),
                    ]
                    for component_name, source_tensor, destination_tensor in components:
                        if destination_tensor.nbytes < source_tensor.nbytes:
                            raise RuntimeError(
                                "SiDP DMA slice destination is smaller than source: "
                                f"layer={layer_id}, param={param_name}, "
                                f"component={component_name}, "
                                f"source={source_tensor.nbytes}, "
                                f"destination={destination_tensor.nbytes}"
                            )
                        total_nbytes += source_tensor.nbytes
                        ranges = dma_slice_ranges(
                            source_tensor.nbytes, manager.dma_slices
                        )
                        for slice_index, (start, end) in enumerate(ranges):
                            fragment_nbytes = end - start
                            if (
                                manager.dma_slices > 1
                                and fragment_nbytes < self.SMALL_DMA_FRAGMENT_BYTES
                            ):
                                small_fragments.append(
                                    (
                                        layer_id,
                                        f"{param_name}:{component_name}",
                                        slice_index,
                                        fragment_nbytes,
                                    )
                                )
                            # An overly large user-selected slice count may
                            # produce empty logical ranges. They need no memcpy
                            # node; layer ready is still recorded after the last
                            # logical slice in the ordered plan.
                            if fragment_nbytes == 0:
                                continue
                            per_slice[slice_index].append(
                                DmaCopyFragment(
                                    source_ptr=source_tensor.data_ptr() + start,
                                    destination_ptr=(
                                        destination_tensor.data_ptr() + start
                                    ),
                                    nbytes=fragment_nbytes,
                                )
                            )
                            copy_nodes += 1
                layer_plans.append(
                    DmaLayerSlicePlan(
                        layer_id=layer_id,
                        slot=slot,
                        total_nbytes=total_nbytes,
                        slices=tuple(tuple(items) for items in per_slice),
                    )
                )
            cycle_groups[cycle] = tuple(
                contiguous_slice_groups(layer_plans, manager.dma_slice_groups)
            )
            max_copy_nodes = max(max_copy_nodes, copy_nodes)

        if manager.dma_slices > 1 and manager.dp_rank == 0:
            if small_fragments:
                minimum = min(item[3] for item in small_fragments)
                examples = ", ".join(
                    f"layer={layer}/{component}/slice={slice_index}:"
                    f"{nbytes / (1024 * 1024):.2f}MiB"
                    for layer, component, slice_index, nbytes in small_fragments[:4]
                )
                logger.warning(
                    "SiDP DMA slicing produced %d logical component fragments "
                    "below 32 MiB (minimum=%.2f MiB; examples: %s). Small DMA "
                    "nodes may reduce copy-engine efficiency.",
                    len(small_fragments),
                    minimum / (1024 * 1024),
                    examples,
                )
            logger.warning(
                "SiDP fixed DMA slicing is enabled: slices=%d, groups=%d, "
                "slot_sync=%s, maximum copy nodes per cycle=%d. This is an "
                "opt-in communication-dominant workload optimization.",
                manager.dma_slices,
                manager.dma_slice_groups,
                manager.slot_sync,
                max_copy_nodes,
            )
        return cycle_groups

    def debug_validate_cycle(self, cycle: int) -> None:
        """Sample-check a completed setup-time cycle-zero DMA fill."""
        for layer_id in self.manager._cycle_layers[cycle]:
            slot = self.manager._layer_to_slot[layer_id]
            for param_name, source in self.manager.peer_views[layer_id].items():
                destination = self.manager._transfer_buffers[slot][param_name]
                pairs = [("<main>", source.tensor, destination.tensor)]
                pairs.extend(
                    (
                        name,
                        source.extra_tensors[name],
                        destination.extra_tensors[name],
                    )
                    for name in sorted(source.extra_tensors)
                )
                for component_name, source_tensor, destination_tensor in pairs:
                    count = source_tensor.numel()
                    indices = sorted(
                        {
                            0,
                            min(1, count - 1),
                            count // 2,
                            max(0, count - 2),
                            count - 1,
                        }
                    )
                    expected = source_tensor.flatten()[indices].to(self.device)
                    actual = destination_tensor.flatten()[indices]
                    if not torch.equal(expected, actual):
                        raise RuntimeError(
                            "SiDP fixed DMA slice setup validation mismatch: "
                            f"rank={self.manager.dp_rank}, cycle={cycle}, "
                            f"layer={layer_id}, param={param_name}, "
                            f"component={component_name}"
                        )
