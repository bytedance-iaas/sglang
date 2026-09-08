"""Eight-GPU SiDP raw-IPC SM-copy and owner-arbitration regression."""

from __future__ import annotations

import pickle
import socket
from datetime import timedelta

import pytest
import torch

from sglang.kernels.ops.sidp import (
    claim_owner,
    copy_selected,
    native_peer_atomic_supported,
    publish_selected_fill,
    release_owner,
    reset_cycle_state,
    reset_forward_state,
    wait_generation,
)
from sglang.srt.layers.sidp.cuda_memcpy import SidpCudaMemcpy
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="extra-b", runner_config="8-gpu-h200")


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _open_descriptor(
    copier: SidpCudaMemcpy, descriptor: dict, bases: dict[bytes, int]
) -> int:
    handle = descriptor["handle"]
    if handle not in bases:
        bases[handle] = copier.open_ipc_allocation(handle)
    return bases[handle] + int(descriptor["offset"])


def _distributed_owner_worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    owner = world_size - 1
    copier = SidpCudaMemcpy()
    store = torch.distributed.TCPStore(
        "127.0.0.1",
        port,
        world_size,
        rank == 0,
        timeout=timedelta(seconds=120),
        wait_for_workers=True,
    )

    owner_state = torch.full((1,), -1, dtype=torch.int32, device=rank)
    store.set(
        f"control/{rank}",
        pickle.dumps(
            copier.export_ipc_pointer(owner_state.data_ptr(), owner_state.nbytes)
        ),
    )

    source = None
    if rank == owner:
        source = torch.arange(1_048_579, dtype=torch.int32, device=rank)
        store.set(
            "source",
            pickle.dumps(copier.export_ipc_pointer(source.data_ptr(), source.nbytes)),
        )

    if rank == owner:
        for requester in range(owner):
            store.get(f"done/{requester}")
        torch.cuda.synchronize()
        assert owner_state.item() == -1
        store.set("verified", b"1")
        return

    assert native_peer_atomic_supported(rank, owner)
    copier.enable_peer_access(owner)
    allocation_bases: dict[bytes, int] = {}
    control_descriptor = pickle.loads(store.get(f"control/{owner}"))
    source_descriptor = pickle.loads(store.get("source"))
    control_ptr = _open_descriptor(copier, control_descriptor, allocation_bases)
    source_ptr = _open_descriptor(copier, source_descriptor, allocation_bases)

    owner_state_ptrs = torch.tensor(
        [control_ptr] * world_size, dtype=torch.uint64, device=rank
    )
    candidate_owners = torch.tensor([owner], dtype=torch.int32, device=rank)
    candidate_slots = torch.tensor([0], dtype=torch.int32, device=rank)
    destination = torch.zeros(1_048_579, dtype=torch.int32, device=rank)
    source_ptrs = torch.tensor([source_ptr], dtype=torch.uint64, device=rank)
    destination_ptrs = torch.tensor(
        [destination.data_ptr()], dtype=torch.uint64, device=rank
    )
    sizes = torch.tensor(
        [destination.nbytes - 3], dtype=torch.int64, device=rank
    )
    fill_gen = torch.empty(1, dtype=torch.int32, device=rank)
    comp_gen = torch.empty(1, dtype=torch.int32, device=rank)
    done = torch.empty(1, dtype=torch.uint8, device=rank)
    selected = torch.empty(1, dtype=torch.int32, device=rank)
    cursor = torch.zeros(1, dtype=torch.int32, device=rank)
    spins = torch.empty(1, dtype=torch.int64, device=rank)
    collisions = torch.empty(1, dtype=torch.int64, device=rank)
    error = torch.empty(1, dtype=torch.int32, device=rank)

    reset_forward_state(fill_gen, comp_gen, 0, error)
    reset_cycle_state(done, selected, spins, collisions)
    claim_owner(
        owner_state_ptrs,
        candidate_owners,
        candidate_slots,
        done,
        comp_gen,
        0,
        0,
        cursor,
        selected,
        spins,
        collisions,
        rank,
        500,
        10**12,
        error,
    )
    copy_selected(
        source_ptrs,
        destination_ptrs,
        sizes,
        selected,
        4,
        512,
        error,
    )
    release_owner(owner_state_ptrs, candidate_owners, selected, rank, error)
    publish_selected_fill(fill_gen, candidate_slots, selected, 1, error)
    wait_generation(fill_gen, 0, 1, 500, 10**12, error)
    torch.cuda.synchronize()

    expected = torch.arange(1_048_579, dtype=torch.int32, device=rank)
    assert torch.equal(
        destination.view(torch.uint8)[:-3], expected.view(torch.uint8)[:-3]
    )
    assert torch.all(destination.view(torch.uint8)[-3:] == 0)
    assert error.item() == 0
    store.set(f"done/{rank}", b"1")
    store.get("verified")


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="eight CUDA GPUs required")
def test_sidp_distributed_owner_raw_ipc_sm_copy():
    torch.multiprocessing.spawn(
        _distributed_owner_worker,
        args=(8, _free_tcp_port()),
        nprocs=8,
        join=True,
    )
