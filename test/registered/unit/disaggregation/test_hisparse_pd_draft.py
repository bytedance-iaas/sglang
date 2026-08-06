from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.managers.hisparse_coordinator import (
    HiSparseCoordinator,
    HiSparseResidencyState,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _HostPool:
    def __init__(self, indices):
        self.indices = torch.tensor(indices, dtype=torch.int64)
        self.calls = []

    def alloc_paged_token_slots(
        self,
        req_to_host_pool,
        req_to_host_pool_allocated_len,
        req_pool_idx,
        start_pos,
        num_tokens,
    ):
        self.calls.append((req_pool_idx, start_pos, num_tokens))
        selected = self.indices[:num_tokens]
        req_to_host_pool[req_pool_idx, start_pos : start_pos + num_tokens].copy_(
            selected
        )
        req_to_host_pool_allocated_len[req_pool_idx] = start_pos + num_tokens
        return selected


def _coordinator(indices):
    indices = list(indices)
    request_capacity = max(16, len(indices))
    coordinator = SimpleNamespace(
        mem_pool_host=_HostPool(indices),
        req_to_host_pool=torch.full((4, request_capacity), -1, dtype=torch.int64),
        req_to_host_pool_allocated_len=torch.zeros(4, dtype=torch.int64),
        host_token_len=lambda length: length,
    )
    coordinator.mem_pool_host.size = len(indices)

    def mirror_host_slots_from(owner, req_pool_idx):
        allocated_len = int(owner.req_to_host_pool_allocated_len[req_pool_idx])
        coordinator.req_to_host_pool[req_pool_idx, :allocated_len].copy_(
            owner.req_to_host_pool[req_pool_idx, :allocated_len]
        )
        coordinator.req_to_host_pool_allocated_len[req_pool_idx] = allocated_len

    coordinator.mirror_host_slots_from = mirror_host_slots_from
    return coordinator


def test_hisparse_pd_draft_uses_host_pool():
    draft_device_pool = object()
    draft_host_pool = object()
    queue = SimpleNamespace(
        draft_token_to_kv_pool=draft_device_pool,
        scheduler=SimpleNamespace(
            enable_hisparse=True,
            draft_hisparse_coordinator=SimpleNamespace(mem_pool_host=draft_host_pool),
        ),
    )

    pool, kind = DecodePreallocQueue._draft_pd_transfer_pool(queue)

    assert pool is draft_host_pool
    assert kind == "DRAM"


def test_hisparse_pd_draft_requires_coordinator():
    queue = SimpleNamespace(
        draft_token_to_kv_pool=object(),
        scheduler=SimpleNamespace(
            enable_hisparse=True, draft_hisparse_coordinator=None
        ),
    )

    with pytest.raises(RuntimeError, match="draft HiSparse coordinator"):
        DecodePreallocQueue._draft_pd_transfer_pool(queue)


def test_hisparse_pd_mirrors_target_slots_without_draft_allocation():
    target = _coordinator(range(8))
    draft = _coordinator(range(8))
    queue = SimpleNamespace(
        scheduler=SimpleNamespace(
            hisparse_coordinator=target,
            draft_hisparse_coordinator=draft,
        )
    )
    req = SimpleNamespace(req_pool_idx=2, rid="req-1")

    indices = DecodePreallocQueue._allocate_hisparse_host_slots(queue, req, 8)

    torch.testing.assert_close(indices, torch.arange(8, dtype=torch.int64))
    assert target.mem_pool_host.calls == [(2, 0, 8)]
    assert draft.mem_pool_host.calls == []
    torch.testing.assert_close(draft.req_to_host_pool[2, :8], indices)


def test_hisparse_pd_reserves_complete_request_budget_but_returns_prompt_slice():
    target = _coordinator(range(520))
    draft = _coordinator(range(520))
    queue = SimpleNamespace(
        scheduler=SimpleNamespace(
            hisparse_coordinator=target,
            draft_hisparse_coordinator=draft,
            model_config=SimpleNamespace(context_len=1024),
        )
    )
    req = SimpleNamespace(
        req_pool_idx=2,
        rid="req-reserved",
        origin_input_ids=list(range(8)),
        sampling_params=SimpleNamespace(max_new_tokens=512),
    )

    indices = DecodePreallocQueue._allocate_hisparse_host_slots(queue, req, 8)

    torch.testing.assert_close(indices, torch.arange(8, dtype=torch.int64))
    assert target.mem_pool_host.calls == [(2, 0, 520)]
    assert draft.mem_pool_host.calls == []


def test_hisparse_pd_reserves_speculative_slack_past_validated_budget():
    target = _coordinator(range(523))
    draft = _coordinator(range(523))
    queue = SimpleNamespace(
        scheduler=SimpleNamespace(
            hisparse_coordinator=target,
            draft_hisparse_coordinator=draft,
            model_config=SimpleNamespace(context_len=520),
            server_args=SimpleNamespace(
                max_speculative_num_draft_tokens=3,
                speculative_num_draft_tokens=3,
            ),
        )
    )
    req = SimpleNamespace(
        req_pool_idx=2,
        rid="req-speculative-slack",
        origin_input_ids=list(range(8)),
        sampling_params=SimpleNamespace(max_new_tokens=512),
    )

    indices = DecodePreallocQueue._allocate_hisparse_host_slots(queue, req, 8)

    torch.testing.assert_close(indices, torch.arange(8, dtype=torch.int64))
    assert target.mem_pool_host.calls == [(2, 0, 523)]
    assert draft.mem_pool_host.calls == []
    assert int(target.req_to_host_pool_allocated_len[2]) == 523
    assert int(draft.req_to_host_pool_allocated_len[2]) == 523


def test_mirrored_host_growth_preserves_existing_slots_and_uses_owner_allocator():
    owner = object.__new__(HiSparseCoordinator)
    owner.device = "cpu"
    owner.page_size = 1
    owner.mem_pool_host = _HostPool([12])
    owner.mem_pool_host.size = 64
    owner.req_to_host_pool = torch.full((4, 16), -1, dtype=torch.int64)
    owner.req_to_host_pool[2, :2] = torch.tensor([10, 11])
    owner.req_to_host_pool_allocated_len = torch.zeros(4, dtype=torch.int64)
    owner.req_to_host_pool_allocated_len[2] = 2
    owner._host_slot_owner = owner

    mirror = object.__new__(HiSparseCoordinator)
    mirror.device = "cpu"
    mirror.page_size = 1
    mirror.mem_pool_host = _HostPool([40])
    mirror.mem_pool_host.size = 64
    mirror.req_to_host_pool = torch.full((4, 16), -1, dtype=torch.int64)
    mirror.req_to_host_pool[2, :2] = torch.tensor([10, 11])
    mirror.req_to_host_pool_allocated_len = torch.zeros(4, dtype=torch.int64)
    mirror.req_to_host_pool_allocated_len[2] = 2
    mirror._host_slot_owner = owner

    request_rows = torch.tensor([2, 2, 2], dtype=torch.int64)
    positions = torch.tensor([0, 1, 2], dtype=torch.int64)
    host_locs = mirror._ensure_host_slots_for_positions(request_rows, positions)

    torch.testing.assert_close(host_locs, torch.tensor([10, 11, 12]))
    torch.testing.assert_close(owner.req_to_host_pool[2, :3], host_locs)
    torch.testing.assert_close(mirror.req_to_host_pool[2, :3], host_locs)
    assert owner.mem_pool_host.calls == [(2, 2, 1)]
    assert mirror.mem_pool_host.calls == []
    assert int(owner.req_to_host_pool_allocated_len[2]) == 3
    assert int(mirror.req_to_host_pool_allocated_len[2]) == 3
    assert torch.unique(owner.req_to_host_pool[2, :3]).numel() == 3


def test_hisparse_pd_mirroring_ignores_draft_free_list_order():
    target = _coordinator(range(8))
    draft = _coordinator(range(1, 9))
    queue = SimpleNamespace(
        scheduler=SimpleNamespace(
            hisparse_coordinator=target,
            draft_hisparse_coordinator=draft,
        )
    )
    req = SimpleNamespace(req_pool_idx=1, rid="req-diverged")

    indices = DecodePreallocQueue._allocate_hisparse_host_slots(queue, req, 8)

    torch.testing.assert_close(indices, torch.arange(8, dtype=torch.int64))
    torch.testing.assert_close(draft.req_to_host_pool[1, :8], indices)
    assert draft.mem_pool_host.calls == []


def test_finished_request_releases_target_and_draft_hisparse_state():
    calls = []
    target = SimpleNamespace(request_finished=lambda req: calls.append(("target", req)))
    draft = SimpleNamespace(request_finished=lambda req: calls.append(("draft", req)))
    processor = SimpleNamespace(
        hisparse_coordinator=target,
        draft_hisparse_coordinator=draft,
    )
    req = object()

    SchedulerBatchResultProcessor._finish_hisparse_request(processor, req)

    assert calls == [("target", req), ("draft", req)]


def _device_coordinator(
    indices, *, buffer_capacity=None, device_capacity=64, num_layers=1
):
    buffer_capacity = buffer_capacity or len(indices)
    coordinator = object.__new__(HiSparseCoordinator)
    coordinator.page_size = 1
    coordinator.padded_buffer_size = buffer_capacity
    coordinator.mem_pool_device = SimpleNamespace(size=device_capacity)
    coordinator.req_to_device_buffer = torch.zeros(
        (4, buffer_capacity), dtype=torch.int64
    )
    if indices:
        coordinator.req_to_device_buffer[2, : len(indices)].copy_(
            torch.tensor(indices, dtype=torch.int64)
        )
    coordinator.req_device_buffer_size = torch.zeros(4, dtype=torch.int64)
    coordinator.req_device_buffer_size[2] = len(indices)
    coordinator.device_buffer_size = buffer_capacity
    coordinator._device_buffer_arange_i32 = torch.arange(
        buffer_capacity, dtype=torch.int32
    )
    coordinator.req_device_buffer_tokens = torch.full(
        (num_layers, 4, buffer_capacity), -1, dtype=torch.int32
    )
    if indices:
        coordinator.req_device_buffer_tokens[:, 2, : len(indices)] = torch.arange(
            len(indices), dtype=torch.int32
        )
    coordinator.req_device_buffer_token_locs = (
        coordinator.req_to_device_buffer.to(torch.int32)
        .unsqueeze(0)
        .repeat(num_layers, 1, 1)
    )
    coordinator._lru_init = torch.arange(buffer_capacity, dtype=torch.int16)
    coordinator.lru_slots = coordinator._lru_init.view(1, 1, -1).repeat(
        num_layers, 4, 1
    )
    coordinator._skip_first_backup = torch.zeros(4, dtype=torch.bool)
    coordinator._device_slot_owner = coordinator
    coordinator._residency_states = {}
    return coordinator


def test_hisparse_pd_draft_mirrors_target_device_slot_namespace():
    target = _device_coordinator([3, 5, 7, 9], num_layers=78)
    draft = _device_coordinator([], buffer_capacity=4, num_layers=1)

    draft.mirror_device_slots_from(target, req_pool_idx=2)

    torch.testing.assert_close(
        draft.req_to_device_buffer[2], target.req_to_device_buffer[2]
    )
    torch.testing.assert_close(
        draft.req_device_buffer_tokens[:, 2],
        torch.arange(4, dtype=torch.int32).view(1, -1),
    )
    torch.testing.assert_close(
        draft.req_device_buffer_token_locs[:, 2],
        torch.tensor([[3, 5, 7, 9]], dtype=torch.int32),
    )
    assert draft._device_slot_owner is target
    assert draft._skip_first_backup[2]


def test_scheduler_admits_draft_with_target_device_slot_owner():
    calls = []
    target = SimpleNamespace(
        admit_request_direct=lambda req, device_slot_owner=None: calls.append(
            ("target", req, device_slot_owner)
        )
    )
    draft = SimpleNamespace(
        admit_request_direct=lambda req, device_slot_owner=None: calls.append(
            ("draft", req, device_slot_owner)
        )
    )
    scheduler = SimpleNamespace(
        hisparse_coordinator=target,
        draft_hisparse_coordinator=draft,
    )
    req = object()

    Scheduler.admit_hisparse_request_direct(scheduler, req)

    assert calls == [("target", req, None), ("draft", req, target)]


def test_shared_hisparse_allocator_capacity_counts_one_mirrored_buffer():
    allocator = SimpleNamespace(available_size=lambda: 24)
    shared_pool_allocator = SimpleNamespace(hisparse_attn_allocator=allocator)
    target = SimpleNamespace(
        token_to_kv_pool_allocator=shared_pool_allocator,
        padded_buffer_size=8,
    )
    draft = SimpleNamespace(
        token_to_kv_pool_allocator=shared_pool_allocator,
        padded_buffer_size=8,
    )
    scheduler = SimpleNamespace(
        iter_hisparse_coordinators=lambda: iter((target, draft))
    )

    assert Scheduler.hisparse_direct_admission_capacity(scheduler) == 3


def _residency_coordinator():
    coordinator = object.__new__(HiSparseCoordinator)
    coordinator._residency_states = {}
    coordinator._last_residency_transition_step = {}
    coordinator._decode_step = 0
    coordinator._promotion_count = 0
    coordinator._demotion_count = 0
    coordinator._promotion_failure_count = 0
    coordinator._promotion_migration_acts = []
    coordinator._promotion_migrated_bytes = 0
    coordinator._promotion_migration_seconds = 0.0
    coordinator._demotion_reclaimed_bytes = 0
    coordinator._demotion_transition_seconds = 0.0
    return coordinator


def test_target_and_draft_residency_state_is_coordinator_local():
    target = _residency_coordinator()
    draft = _residency_coordinator()

    target._set_residency_state(2, HiSparseResidencyState.RESIDENT)
    draft._set_residency_state(2, HiSparseResidencyState.DEVICE_BUFFERED)

    assert target._is_resident(2)
    assert not draft._is_resident(2)
    assert target._promotion_count == 1
    assert draft._promotion_count == 0


def test_register_mirror_restores_shared_allocator_callbacks_to_owner():
    callback_updates = []
    allocator = SimpleNamespace(
        set_demote_until_hisparse_available=lambda callback: callback_updates.append(
            ("demote", callback)
        ),
        set_schedulable_hisparse_available=lambda callback: callback_updates.append(
            ("schedulable", callback)
        ),
    )
    target = object.__new__(HiSparseCoordinator)
    target.page_size = 64
    target._device_slot_mirrors = []
    target.token_to_kv_pool_allocator = allocator
    target.demote_until_hisparse_available = lambda need: True
    target.schedulable_hisparse_available = lambda: 1
    draft = object.__new__(HiSparseCoordinator)
    draft.page_size = 64
    draft._device_slot_owner = draft

    target.register_device_slot_mirror(draft)

    assert target._device_slot_mirrors == [draft]
    assert draft._device_slot_owner is target
    assert [kind for kind, _ in callback_updates] == ["demote", "schedulable"]
