import pytest
import torch

from sglang.kernels.ops.sidp import (
    claim_owner,
    copy_selected,
    publish_generation,
    publish_selected_fill,
    reset_cycle_state,
    reset_forward_state,
    release_owner,
    select_fixed,
    wait_generation,
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sidp_fixed_sm_copy_flag_protocol_and_tail_bytes():
    device = torch.cuda.current_device()
    sources = [
        torch.arange(1027, dtype=torch.int32, device=device),
        torch.arange(1027, dtype=torch.int32, device=device) + 17,
    ]
    destinations = [
        torch.zeros_like(sources[0]),
        torch.zeros_like(sources[1]),
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
    copy_selected(src_ptrs, dst_ptrs, sizes, selected, 4, error)
    publish_selected_fill(fill_gen, candidate_slots, selected, 1, error)
    wait_generation(fill_gen, 1, 1, 100, 10**12, error)
    publish_generation(comp_gen, 1, 1)
    torch.cuda.synchronize()

    source_bytes = sources[1].view(torch.uint8)
    destination_bytes = destinations[1].view(torch.uint8)
    assert torch.equal(destination_bytes[:-3], source_bytes[:-3])
    assert torch.all(destination_bytes[-3:] == 0)
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
