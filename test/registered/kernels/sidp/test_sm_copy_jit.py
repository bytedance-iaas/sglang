import pytest
import torch

from sglang.kernels.ops.sidp import (
    claim_owner,
    copy_selected,
    copy_selected_traced,
    publish_generation,
    publish_selected_fill,
    reset_cycle_state,
    reset_forward_state,
    reset_sm_trace,
    release_owner,
    select_fixed,
    wait_generation,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sidp_fixed_sm_copy_main_extra_flag_protocol_and_tail_bytes():
    device = torch.cuda.current_device()
    sources = [
        torch.arange(1027, dtype=torch.int32, device=device),
        torch.arange(1027, dtype=torch.int32, device=device) + 17,
    ]
    destinations = [
        torch.zeros_like(sources[0]),
        torch.zeros_like(sources[1]),
    ]
    extra_sources = [
        torch.arange(35, dtype=torch.uint8, device=device),
        torch.arange(35, dtype=torch.uint8, device=device) + 23,
    ]
    extra_destinations = [
        torch.zeros_like(extra_sources[0]),
        torch.zeros_like(extra_sources[1]),
    ]
    src_ptrs = torch.tensor(
        [tensor.data_ptr() for tensor in sources],
        dtype=torch.uint64,
        device=device,
    )
    dst_ptrs = torch.tensor(
        [tensor.data_ptr() for tensor in destinations],
        dtype=torch.uint64,
        device=device,
    )
    sizes = torch.tensor(
        [sources[0].nbytes - 3, sources[1].nbytes - 3],
        dtype=torch.int64,
        device=device,
    )
    extra_src_ptrs = torch.tensor(
        [tensor.data_ptr() for tensor in extra_sources],
        dtype=torch.uint64,
        device=device,
    )
    extra_dst_ptrs = torch.tensor(
        [tensor.data_ptr() for tensor in extra_destinations],
        dtype=torch.uint64,
        device=device,
    )
    extra_sizes = torch.tensor(
        [extra_sources[0].nbytes, extra_sources[1].nbytes - 1],
        dtype=torch.int64,
        device=device,
    )
    selected = torch.empty(1, dtype=torch.int32, device=device)
    done = torch.empty(2, dtype=torch.uint8, device=device)
    spins = torch.empty(1, dtype=torch.int64, device=device)
    collisions = torch.empty(1, dtype=torch.int64, device=device)
    fill_gen = torch.empty(2, dtype=torch.int32, device=device)
    comp_gen = torch.empty(2, dtype=torch.int32, device=device)
    error = torch.empty(1, dtype=torch.int32, device=device)
    candidate_slots = torch.tensor([0, 1], dtype=torch.int32, device=device)

    reset_forward_state(fill_gen, comp_gen, 0, error)
    reset_cycle_state(done, selected, spins, collisions)
    select_fixed(selected, 1)
    copy_selected(src_ptrs, dst_ptrs, sizes, selected, 4, 128, error)
    copy_selected(
        extra_src_ptrs,
        extra_dst_ptrs,
        extra_sizes,
        selected,
        4,
        512,
        error,
    )
    publish_selected_fill(fill_gen, candidate_slots, selected, 1, error)
    wait_generation(fill_gen, 1, 1, 100, 10**12, error)
    publish_generation(comp_gen, 1, 1)
    torch.cuda.synchronize()

    source_bytes = sources[1].view(torch.uint8)
    destination_bytes = destinations[1].view(torch.uint8)
    assert torch.equal(destination_bytes[:-3], source_bytes[:-3])
    assert torch.all(destination_bytes[-3:] == 0)
    assert torch.equal(extra_destinations[1][:-1], extra_sources[1][:-1])
    assert extra_destinations[1][-1].item() == 0
    assert fill_gen.cpu().tolist() == [0, 1]
    assert comp_gen.cpu().tolist() == [0, 1]
    assert error.item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sidp_dynamic_claim_and_release_protocol():
    device = torch.cuda.current_device()
    owner_states = [
        torch.full((1,), -1, dtype=torch.int32, device=device),
        torch.full((1,), -1, dtype=torch.int32, device=device),
    ]
    owner_ptrs = torch.tensor(
        [state.data_ptr() for state in owner_states],
        dtype=torch.uint64,
        device=device,
    )
    owners = torch.tensor([1], dtype=torch.int32, device=device)
    slots = torch.tensor([0], dtype=torch.int32, device=device)
    done = torch.empty(1, dtype=torch.uint8, device=device)
    selected = torch.empty(1, dtype=torch.int32, device=device)
    cursor = torch.zeros(1, dtype=torch.int32, device=device)
    spins = torch.empty(1, dtype=torch.int64, device=device)
    collisions = torch.empty(1, dtype=torch.int64, device=device)
    fill_gen = torch.empty(1, dtype=torch.int32, device=device)
    comp_gen = torch.empty(1, dtype=torch.int32, device=device)
    error = torch.empty(1, dtype=torch.int32, device=device)

    reset_forward_state(fill_gen, comp_gen, 0, error)
    reset_cycle_state(done, selected, spins, collisions)
    claim_owner(
        owner_ptrs,
        owners,
        slots,
        done,
        comp_gen,
        0,
        0,
        cursor,
        selected,
        spins,
        collisions,
        0,
        100,
        10**12,
        error,
    )
    release_owner(owner_ptrs, owners, selected, 0, error)
    torch.cuda.synchronize()

    assert selected.item() == 0
    assert done.item() == 1
    assert owner_states[1].item() == -1
    assert error.item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sidp_dynamic_claim_compute_priority_ignores_rotating_cursor():
    device = torch.cuda.current_device()
    owner_states = [
        torch.full((1,), -1, dtype=torch.int32, device=device) for _ in range(2)
    ]
    owner_ptrs = torch.tensor(
        [state.data_ptr() for state in owner_states],
        dtype=torch.uint64,
        device=device,
    )
    owners = torch.tensor([0, 1], dtype=torch.int32, device=device)
    slots = torch.tensor([0, 1], dtype=torch.int32, device=device)
    done = torch.empty(2, dtype=torch.uint8, device=device)
    selected = torch.empty(1, dtype=torch.int32, device=device)
    cursor = torch.ones(1, dtype=torch.int32, device=device)
    spins = torch.empty(1, dtype=torch.int64, device=device)
    collisions = torch.empty(1, dtype=torch.int64, device=device)
    fill_gen = torch.empty(2, dtype=torch.int32, device=device)
    comp_gen = torch.empty(2, dtype=torch.int32, device=device)
    error = torch.empty(1, dtype=torch.int32, device=device)

    reset_forward_state(fill_gen, comp_gen, 0, error)
    reset_cycle_state(done, selected, spins, collisions)
    claim_owner(
        owner_ptrs,
        owners,
        slots,
        done,
        comp_gen,
        0,
        1,
        cursor,
        selected,
        spins,
        collisions,
        0,
        100,
        10**12,
        error,
    )
    release_owner(owner_ptrs, owners, selected, 0, error)
    torch.cuda.synchronize()

    assert selected.item() == 0
    assert cursor.item() == 1
    assert owner_states[0].item() == -1
    assert error.item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sidp_sm_copy_execution_trace_records_real_cta_smids():
    device = torch.cuda.current_device()
    source = torch.arange(4096, dtype=torch.uint8, device=device)
    destination = torch.zeros_like(source)
    src_ptrs = torch.tensor([source.data_ptr()], dtype=torch.uint64, device=device)
    dst_ptrs = torch.tensor(
        [destination.data_ptr()], dtype=torch.uint64, device=device
    )
    sizes = torch.tensor([source.nbytes], dtype=torch.int64, device=device)
    selected = torch.zeros(1, dtype=torch.int32, device=device)
    error = torch.zeros(1, dtype=torch.int32, device=device)
    rows = torch.full((16, 6), -1, dtype=torch.int64, device=device)
    count = torch.zeros(1, dtype=torch.int32, device=device)
    overflow = torch.zeros_like(count)

    reset_sm_trace(count, overflow)
    copy_selected_traced(
        src_ptrs,
        dst_ptrs,
        sizes,
        selected,
        4,
        128,
        error,
        rows,
        count,
        overflow,
        17,
    )
    torch.cuda.synchronize()

    assert torch.equal(source, destination)
    assert count.item() == 4
    assert overflow.item() == 0
    actual = rows[:4].cpu()
    assert torch.all(actual[:, 0] == 17)
    assert sorted(actual[:, 1].tolist()) == [0, 1, 2, 3]
    assert torch.all(actual[:, 2] >= 0)
    assert torch.all(actual[:, 3] >= 0)
    assert torch.all(actual[:, 4] > 0)
    assert torch.all(actual[:, 5] >= actual[:, 4])
