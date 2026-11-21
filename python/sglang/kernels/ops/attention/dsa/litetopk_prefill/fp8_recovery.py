"""FP8 LiteTopK prefill indexer with GPU-only exact recovery on SM100.

The device implementation is adapted from vLLM PR #65920 at
833266447780a74713c574e67f3b3bf9caa868eb, itself based on SGLang #47469.
Rows whose sampled threshold underfills, whose candidate slab overflows, or
whose selector sees a wide cutoff tie are repaired on device without a host
synchronization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

SUPPORTED_HEADS = (32, 64)
SUPPORTED_TOPK = (512, 1024, 2048)
HEAD_DIM = 128
CANDIDATE_CAP = 12288
WINDOW_ROWS = 4096
CODES_PER_CTA = 32768
RECOVERY_FLOATS = 1 << 24
RECOVERY_SLOTS = 32
UNIT_OVERHEAD_TILES = 10
MAX_KEYS = 1 << 20


class RecoveryWorkspaces(NamedTuple):
    counter: torch.Tensor
    codes: torch.Tensor
    values: torch.Tensor
    indices: torch.Tensor
    row_state: torch.Tensor
    recovery: torch.Tensor


@cache_once
def is_fp8_recovery_supported() -> bool:
    return (
        torch.version.cuda is not None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
    )


@cache_once
def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def pick_tail_chunks(num_blocks: int, num_ctas: int, block_tiles: int) -> int:
    unit_cost = UNIT_OVERHEAD_TILES / max(block_tiles, 1)

    def launch_time(chunks: int) -> float:
        grid = min(num_ctas, num_blocks * chunks)
        tail = num_blocks % grid
        units = -(-tail * chunks // grid)
        return num_blocks // grid * (1 + unit_cost) + (
            units / chunks + units * unit_cost if tail else 0.0
        )

    return min((1, 2, 4, 8, 16, 32), key=launch_time)


def min_profitable_keys(topk: int, heads: int, rows: int) -> int:
    """Pinned vLLM #65920 SM100 crossover; final service A/B still decides."""
    if rows < 1024:
        return 1 << 16
    return 16384 + 4096 * (topk.bit_length() - 10 + heads // 32)


@cache_once
def _jit_module(num_heads: int, topk: int) -> Module:
    if not is_fp8_recovery_supported():
        raise RuntimeError("FP8 LiteTopK recovery requires CUDA SM100")
    if num_heads not in SUPPORTED_HEADS:
        raise RuntimeError(f"unsupported index heads: {num_heads}")
    if topk not in SUPPORTED_TOPK:
        raise RuntimeError(f"unsupported top-k: {topk}")
    args = make_cpp_args(num_heads, topk)
    return load_jit(
        "dsa_prefill_fp8_litetopk_recovery",
        *args,
        cuda_files=["dsa/litetopk_prefill/recovery/fp8_indexer_topk.cuh"],
        cuda_wrappers=[("run", f"sglang::Fp8IndexerTopKRecoveryKernel<{args}>::run")],
        extra_cuda_cflags=["-Xcicc", "-disable-new-nvvm-remat"],
        extra_dependencies=["cutlass"],
    )


_WORKSPACES: dict[tuple[int, int, int], RecoveryWorkspaces] = {}
_CAPTURE_WORKSPACES: dict[tuple[int, int], RecoveryWorkspaces] = {}


def _workspaces(device: torch.device, heads: int) -> RecoveryWorkspaces:
    rows = WINDOW_ROWS
    keys = MAX_KEYS
    block_rows = 256 // heads
    blocks = -(-rows // block_rows)
    ctas = min(_num_sms(device), 32 * blocks)
    recovery_slots = min(
        -(-rows // 8), RECOVERY_SLOTS, RECOVERY_FLOATS // (8 * max(keys, 1))
    )
    recovery_slots = max(1, recovery_slots)
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    base_key = (device_index, heads)
    if torch.cuda.is_current_stream_capturing():
        cached = _CAPTURE_WORKSPACES.get(base_key)
        if cached is None:
            raise RuntimeError(
                "FP8 LiteTopK CUDA Graph capture requires one eager warmup call "
                "with the same device and head count"
            )
        return cached
    stream_id = torch.cuda.current_stream(device).cuda_stream
    cache_key = (*base_key, stream_id)
    cached = _WORKSPACES.get(cache_key)
    if cached is not None:
        return cached
    required = RecoveryWorkspaces(
        counter=torch.zeros(2 * blocks, dtype=torch.int32, device=device),
        codes=torch.empty(ctas * CODES_PER_CTA, dtype=torch.int32, device=device),
        values=torch.empty((rows, CANDIDATE_CAP), dtype=torch.float32, device=device),
        indices=torch.empty((rows, CANDIDATE_CAP), dtype=torch.int32, device=device),
        row_state=torch.empty(3 * rows, dtype=torch.int32, device=device),
        recovery=torch.empty(
            recovery_slots * 8 * keys, dtype=torch.float32, device=device
        ),
    )
    _WORKSPACES[cache_key] = required
    # Graph capture runs on a private capture stream and cannot allocate its
    # large scratch buffers. Reserve the first eager workspace for capture;
    # SGLang's scheduler serializes graph replays that share this shape.
    _CAPTURE_WORKSPACES.setdefault(base_key, required)
    return required


def fp8_indexer_topk_recovery(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scale: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    topk: int,
    *,
    out_offsets: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    return_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return exact top-k relative indices, repairing exceptional rows on GPU.

    ``ks`` / ``ke`` are absolute bounds in the gathered key buffer. Output is
    relative to ``ks`` until ``out_offsets`` is supplied. ``return_state`` is a
    diagnostic-only GPU tensor ``[3, rows]`` containing candidate counts,
    threshold bits and overflow flags; reading it on the host synchronizes.
    """
    rows, heads = q_fp8.shape[:2]
    if out is None:
        out = torch.empty((rows, topk), dtype=torch.int32, device=q_fp8.device)
    if rows == 0:
        state = torch.empty((3, 0), dtype=torch.int32, device=q_fp8.device)
        return (out, state) if return_state else out

    module = _jit_module(heads, topk)
    state_out = (
        torch.empty((3, rows), dtype=torch.int32, device=q_fp8.device)
        if return_state
        else None
    )
    for start in range(0, rows, WINDOW_ROWS):
        stop = min(rows, start + WINDOW_ROWS)
        n = stop - start
        workspace = _workspaces(q_fp8.device, heads)
        block_rows = 256 // heads
        blocks = -(-n // block_rows)
        num_ctas = min(_num_sms(q_fp8.device), 32 * blocks)
        tail_chunks = pick_tail_chunks(blocks, num_ctas, max(1, kv_fp8.shape[0] // 256))
        row_state = workspace.row_state[: 3 * n].view(3, n)
        module.run(
            q_fp8[start:stop],
            weights[start:stop],
            kv_fp8,
            kv_scale,
            ks[start:stop],
            ke[start:stop],
            workspace.counter,
            workspace.codes,
            workspace.values[:n],
            workspace.indices[:n],
            row_state,
            workspace.recovery,
            out[start:stop],
            num_ctas,
            tail_chunks,
        )
        if state_out is not None:
            state_out[:, start:stop].copy_(row_state)

    if out_offsets is not None:
        offsets = out_offsets[:rows, None]
        out.copy_(torch.where(out >= 0, out + offsets, out))
    return (out, state_out) if state_out is not None else out


__all__ = [
    "CANDIDATE_CAP",
    "MAX_KEYS",
    "SUPPORTED_HEADS",
    "SUPPORTED_TOPK",
    "fp8_indexer_topk_recovery",
    "is_fp8_recovery_supported",
    "min_profitable_keys",
]
