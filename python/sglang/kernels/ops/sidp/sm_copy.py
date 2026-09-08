"""JIT CUDA kernels for SiDP's SM-copy and dynamic-owner backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def load_sidp_sm_copy_module() -> Module:
    return load_jit(
        "sidp_sm_copy",
        cuda_files=["sidp/sm_copy.cuh"],
        cuda_wrappers=[
            ("reset_sm_trace", "&sidp::SidpSmCopyKernels::reset_sm_trace"),
            ("reset_forward_state", "&sidp::SidpSmCopyKernels::reset_forward_state"),
            (
                "reset_forward_state_traced",
                "&sidp::SidpSmCopyKernels::reset_forward_state_traced",
            ),
            ("reset_cycle_state", "&sidp::SidpSmCopyKernels::reset_cycle_state"),
            (
                "reset_cycle_state_traced",
                "&sidp::SidpSmCopyKernels::reset_cycle_state_traced",
            ),
            ("select_fixed", "&sidp::SidpSmCopyKernels::select_fixed"),
            (
                "select_fixed_traced",
                "&sidp::SidpSmCopyKernels::select_fixed_traced",
            ),
            ("claim_owner", "&sidp::SidpSmCopyKernels::claim_owner"),
            (
                "claim_owner_traced",
                "&sidp::SidpSmCopyKernels::claim_owner_traced",
            ),
            ("copy_selected", "&sidp::SidpSmCopyKernels::copy_selected"),
            (
                "copy_selected_traced",
                "&sidp::SidpSmCopyKernels::copy_selected_traced",
            ),
            ("release_owner", "&sidp::SidpSmCopyKernels::release_owner"),
            (
                "release_owner_traced",
                "&sidp::SidpSmCopyKernels::release_owner_traced",
            ),
            (
                "publish_selected_fill",
                "&sidp::SidpSmCopyKernels::publish_selected_fill",
            ),
            (
                "publish_selected_fill_traced",
                "&sidp::SidpSmCopyKernels::publish_selected_fill_traced",
            ),
            ("wait_generation", "&sidp::SidpSmCopyKernels::wait_generation"),
            (
                "wait_generation_traced",
                "&sidp::SidpSmCopyKernels::wait_generation_traced",
            ),
            (
                "publish_generation",
                "&sidp::SidpSmCopyKernels::publish_generation",
            ),
            (
                "publish_generation_traced",
                "&sidp::SidpSmCopyKernels::publish_generation_traced",
            ),
            ("record_trace", "&sidp::SidpSmCopyKernels::record_trace"),
            (
                "native_peer_atomic_supported",
                "&sidp::SidpSmCopyKernels::native_peer_atomic_supported",
            ),
        ],
    )


def reset_sm_trace(count: torch.Tensor, overflow: torch.Tensor) -> None:
    load_sidp_sm_copy_module().reset_sm_trace(count, overflow)


def reset_forward_state(
    fill_gen: torch.Tensor,
    comp_gen: torch.Tensor,
    resident_slots: int,
    error_state: torch.Tensor,
) -> None:
    load_sidp_sm_copy_module().reset_forward_state(
        fill_gen, comp_gen, resident_slots, error_state
    )


def reset_forward_state_traced(
    fill_gen: torch.Tensor,
    comp_gen: torch.Tensor,
    resident_slots: int,
    error_state: torch.Tensor,
    trace_rows: torch.Tensor,
    trace_count: torch.Tensor,
    trace_overflow: torch.Tensor,
    trace_tag: int,
) -> None:
    load_sidp_sm_copy_module().reset_forward_state_traced(
        fill_gen,
        comp_gen,
        resident_slots,
        error_state,
        trace_rows,
        trace_count,
        trace_overflow,
        trace_tag,
    )


def reset_cycle_state(
    done: torch.Tensor,
    selected: torch.Tensor,
    claim_spins: torch.Tensor,
    claim_collisions: torch.Tensor,
) -> None:
    load_sidp_sm_copy_module().reset_cycle_state(
        done, selected, claim_spins, claim_collisions
    )


def reset_cycle_state_traced(
    done: torch.Tensor,
    selected: torch.Tensor,
    claim_spins: torch.Tensor,
    claim_collisions: torch.Tensor,
    trace_rows: torch.Tensor,
    trace_count: torch.Tensor,
    trace_overflow: torch.Tensor,
    trace_tag: int,
) -> None:
    load_sidp_sm_copy_module().reset_cycle_state_traced(
        done,
        selected,
        claim_spins,
        claim_collisions,
        trace_rows,
        trace_count,
        trace_overflow,
        trace_tag,
    )


def select_fixed(selected: torch.Tensor, index: int) -> None:
    load_sidp_sm_copy_module().select_fixed(selected, index)


def select_fixed_traced(
    selected: torch.Tensor,
    index: int,
    trace_rows: torch.Tensor,
    trace_count: torch.Tensor,
    trace_overflow: torch.Tensor,
    trace_tag: int,
) -> None:
    load_sidp_sm_copy_module().select_fixed_traced(
        selected,
        index,
        trace_rows,
        trace_count,
        trace_overflow,
        trace_tag,
    )


def claim_owner(
    owner_state_ptrs: torch.Tensor,
    candidate_owners: torch.Tensor,
    candidate_slots: torch.Tensor,
    done: torch.Tensor,
    comp_gen: torch.Tensor,
    required_comp_gen: int,
    claim_order: int,
    probe_cursor: torch.Tensor,
    selected: torch.Tensor,
    claim_spins: torch.Tensor,
    claim_collisions: torch.Tensor,
    requester_rank: int,
    backoff_ns: int,
    timeout_clocks: int,
    error_state: torch.Tensor,
) -> None:
    load_sidp_sm_copy_module().claim_owner(
        owner_state_ptrs,
        candidate_owners,
        candidate_slots,
        done,
        comp_gen,
        required_comp_gen,
        claim_order,
        probe_cursor,
        selected,
        claim_spins,
        claim_collisions,
        requester_rank,
        backoff_ns,
        timeout_clocks,
        error_state,
    )


def claim_owner_traced(
    owner_state_ptrs: torch.Tensor,
    candidate_owners: torch.Tensor,
    candidate_slots: torch.Tensor,
    done: torch.Tensor,
    comp_gen: torch.Tensor,
    required_comp_gen: int,
    claim_order: int,
    probe_cursor: torch.Tensor,
    selected: torch.Tensor,
    claim_spins: torch.Tensor,
    claim_collisions: torch.Tensor,
    requester_rank: int,
    backoff_ns: int,
    timeout_clocks: int,
    error_state: torch.Tensor,
    trace_rows: torch.Tensor,
    trace_count: torch.Tensor,
    trace_overflow: torch.Tensor,
    trace_tag: int,
) -> None:
    load_sidp_sm_copy_module().claim_owner_traced(
        owner_state_ptrs,
        candidate_owners,
        candidate_slots,
        done,
        comp_gen,
        required_comp_gen,
        claim_order,
        probe_cursor,
        selected,
        claim_spins,
        claim_collisions,
        requester_rank,
        backoff_ns,
        timeout_clocks,
        error_state,
        trace_rows,
        trace_count,
        trace_overflow,
        trace_tag,
    )


def copy_selected(
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    sizes: torch.Tensor,
    selected: torch.Tensor,
    grid_blocks: int,
    block_size: int,
    error_state: torch.Tensor,
) -> None:
    load_sidp_sm_copy_module().copy_selected(
        src_ptrs,
        dst_ptrs,
        sizes,
        selected,
        grid_blocks,
        block_size,
        error_state,
    )


def copy_selected_traced(
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    sizes: torch.Tensor,
    selected: torch.Tensor,
    grid_blocks: int,
    block_size: int,
    error_state: torch.Tensor,
    trace_rows: torch.Tensor,
    trace_count: torch.Tensor,
    trace_overflow: torch.Tensor,
    trace_tag: int,
) -> None:
    load_sidp_sm_copy_module().copy_selected_traced(
        src_ptrs,
        dst_ptrs,
        sizes,
        selected,
        grid_blocks,
        block_size,
        error_state,
        trace_rows,
        trace_count,
        trace_overflow,
        trace_tag,
    )


def release_owner(
    owner_state_ptrs: torch.Tensor,
    candidate_owners: torch.Tensor,
    selected: torch.Tensor,
    requester_rank: int,
    error_state: torch.Tensor,
) -> None:
    load_sidp_sm_copy_module().release_owner(
        owner_state_ptrs,
        candidate_owners,
        selected,
        requester_rank,
        error_state,
    )


def release_owner_traced(
    owner_state_ptrs: torch.Tensor,
    candidate_owners: torch.Tensor,
    selected: torch.Tensor,
    requester_rank: int,
    error_state: torch.Tensor,
    trace_rows: torch.Tensor,
    trace_count: torch.Tensor,
    trace_overflow: torch.Tensor,
    trace_tag: int,
) -> None:
    load_sidp_sm_copy_module().release_owner_traced(
        owner_state_ptrs,
        candidate_owners,
        selected,
        requester_rank,
        error_state,
        trace_rows,
        trace_count,
        trace_overflow,
        trace_tag,
    )


def publish_selected_fill(
    fill_gen: torch.Tensor,
    candidate_slots: torch.Tensor,
    selected: torch.Tensor,
    target_gen: int,
    error_state: torch.Tensor,
) -> None:
    load_sidp_sm_copy_module().publish_selected_fill(
        fill_gen, candidate_slots, selected, target_gen, error_state
    )


def publish_selected_fill_traced(
    fill_gen: torch.Tensor,
    candidate_slots: torch.Tensor,
    selected: torch.Tensor,
    target_gen: int,
    error_state: torch.Tensor,
    trace_rows: torch.Tensor,
    trace_count: torch.Tensor,
    trace_overflow: torch.Tensor,
    trace_tag: int,
) -> None:
    load_sidp_sm_copy_module().publish_selected_fill_traced(
        fill_gen,
        candidate_slots,
        selected,
        target_gen,
        error_state,
        trace_rows,
        trace_count,
        trace_overflow,
        trace_tag,
    )


def wait_generation(
    generations: torch.Tensor,
    slot: int,
    target: int,
    backoff_ns: int,
    timeout_clocks: int,
    error_state: torch.Tensor,
) -> None:
    load_sidp_sm_copy_module().wait_generation(
        generations, slot, target, backoff_ns, timeout_clocks, error_state
    )


def wait_generation_traced(
    generations: torch.Tensor,
    slot: int,
    target: int,
    backoff_ns: int,
    timeout_clocks: int,
    error_state: torch.Tensor,
    trace_rows: torch.Tensor,
    trace_count: torch.Tensor,
    trace_overflow: torch.Tensor,
    trace_tag: int,
) -> None:
    load_sidp_sm_copy_module().wait_generation_traced(
        generations,
        slot,
        target,
        backoff_ns,
        timeout_clocks,
        error_state,
        trace_rows,
        trace_count,
        trace_overflow,
        trace_tag,
    )


def publish_generation(
    generations: torch.Tensor, slot: int, target: int
) -> None:
    load_sidp_sm_copy_module().publish_generation(generations, slot, target)


def publish_generation_traced(
    generations: torch.Tensor,
    slot: int,
    target: int,
    trace_rows: torch.Tensor,
    trace_count: torch.Tensor,
    trace_overflow: torch.Tensor,
    trace_tag: int,
) -> None:
    load_sidp_sm_copy_module().publish_generation_traced(
        generations,
        slot,
        target,
        trace_rows,
        trace_count,
        trace_overflow,
        trace_tag,
    )


def record_trace(
    selected: torch.Tensor,
    claim_spins: torch.Tensor,
    claim_collisions: torch.Tensor,
    selected_trace: torch.Tensor,
    spins_trace: torch.Tensor,
    collisions_trace: torch.Tensor,
    trace_index: int,
) -> None:
    load_sidp_sm_copy_module().record_trace(
        selected,
        claim_spins,
        claim_collisions,
        selected_trace,
        spins_trace,
        collisions_trace,
        trace_index,
    )


def native_peer_atomic_supported(device: int, peer: int) -> bool:
    return bool(
        load_sidp_sm_copy_module().native_peer_atomic_supported(device, peer)
    )
