from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator


def _debug_coordinator(*, free_slots, release_slots, slot_used):
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.debug_validate_lifecycle = True
    coordinator.mem_pool_host = SimpleNamespace(
        free_slots=torch.tensor(free_slots, dtype=torch.int64),
        release_slots=[
            torch.tensor(chunk, dtype=torch.int64) for chunk in release_slots
        ],
        slot_used=torch.tensor(slot_used, dtype=torch.bool),
        size=len(slot_used),
        available_size=Mock(
            return_value=len(free_slots) + sum(map(len, release_slots))
        ),
    )
    return coordinator


def test_host_lifecycle_probe_counts_deferred_release_slots():
    coordinator = _debug_coordinator(
        free_slots=[2, 3], release_slots=[[0, 1]], slot_used=[False] * 4
    )

    coordinator._debug_validate_host_allocator_after_free(
        SimpleNamespace(rid="deferred", req_pool_idx=1),
        released_host_slots=2,
        stage="finish",
    )


def test_host_lifecycle_probe_rejects_duplicate_across_free_queues():
    coordinator = _debug_coordinator(
        free_slots=[1, 2, 3], release_slots=[[0, 1]], slot_used=[False] * 4
    )

    with pytest.raises(RuntimeError, match="free-list contains duplicates"):
        coordinator._debug_validate_host_allocator_after_free(
            SimpleNamespace(rid="duplicate", req_pool_idx=1),
            released_host_slots=2,
            stage="finish",
        )
