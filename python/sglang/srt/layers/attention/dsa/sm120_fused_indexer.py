"""Opt-in exact SM120 prefill selection using FlashInfer's paged interface.

Adapted from the owning interfaces in SGLang #49098 and FlashInfer #7085.
CP query rows have independent causal ends: a ragged compression ratio is
not a query stride. Per-row paged lengths retain the existing ownership.
"""

import logging
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from threading import Lock
from typing import Optional

import torch

from sglang.kernels.ops.attention.topk_page_table import topk_page_table_transform_
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)
_locks = {}
_last_use = {}
_SCRATCH_LIMIT = 1 << 30
# FlashInfer 0.7.4 decode_common.cuh: histogram, lists, counters and ties.
_LONG_BYTES_PER_ROW = (5 * 4096 + 32 + 4 * 2048 + 256) * 4


@lru_cache(maxsize=1)
def runtime_ok() -> bool:
    # gfx950 retains its auto policy. SM120 remains explicitly selected until
    # serving accuracy, capacity and same-load performance have been measured.
    if get_exec().kernel.enable_dsa_fused_indexer is not True or not is_cuda():
        return False
    if torch.cuda.get_device_capability()[0] != 12:
        return False
    try:
        import flashinfer
        from flashinfer.dsa_indexer import dsa_indexer_det_topk  # noqa: F401
    except ImportError:
        logger.warning("SM120 exact prefill disabled: missing FlashInfer indexer API")
        return False
    if version("flashinfer-python").split("+")[0] != "0.7.4":
        logger.warning(
            "SM120 exact prefill disabled: unvalidated experimental API version"
        )
        return False
    source = (
        Path(flashinfer.__file__).parent
        / "experimental/sm120_dsa_indexer/csrc/binding.cu"
    )
    if not source.is_file():
        logger.warning(
            "SM120 exact prefill disabled: install the full-source FlashInfer wheel"
        )
        return False
    return True


def shape_supported(heads: int, dim: int, topk: int, initial: int, local: int) -> bool:
    return (
        heads in (32, 64)
        and dim == 128
        and 0 < topk <= 4096
        and topk % 4 == 0
        and initial == local == 0
    )


def prefill_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    cache: torch.Tensor,
    index_table: torch.Tensor,
    output_table: torch.Tensor,
    row_batch: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    out: torch.Tensor,
    page_size: int,
    scratch_budget: int,
) -> tuple[Optional[torch.Tensor], Optional[str]]:
    """Write native physical slots, preserving padded rows and current streams.

    The source index table and destination MLA table may differ; only the
    request-local positions cross that boundary. Unsupported formats decline
    before touching output. Kernel/runtime failures propagate normally.
    """
    if torch.cuda.is_current_stream_capturing():
        return None, "prefill CUDA capture"
    if page_size != 64 or q.dtype != torch.float8_e4m3fn or cache.dtype != torch.uint8:
        return None, "unsupported cache/query format"
    if q.ndim != 3 or q.shape[1] not in (32, 64) or q.shape[2] != 128:
        return None, "unsupported query geometry"
    rows = starts.shape[0]
    if rows > q.shape[0] or ends.shape != starts.shape or row_batch.shape[0] != rows:
        return None, "incomplete row ownership"
    if cache.ndim != 2 or cache.shape[1] != page_size * 132:
        return None, "unsupported packed key layout"
    if rows == 0:
        return out, None
    width = index_table.shape[1] * page_size
    bytes_per_row = max(1, width * 4)
    limit = min(scratch_budget, _SCRATCH_LIMIT)
    # 0.7.4 grows each shared buffer geometrically and retains replaced
    # buffers for captured graphs. Current plus retired storage is less than
    # four times the largest requested size. Split the resident ceiling
    # equally between scores and long-row selection, reserving that factor.
    # Resident storage follows the process high-water mark, not a later,
    # smaller per-call budget; neither buffer is mutated behind FlashInfer.
    chunk_rows = min(limit // (8 * bytes_per_row), limit // (8 * _LONG_BYTES_PER_ROW))
    if chunk_rows < 1:
        return None, "native score scratch budget"
    chunk_rows = min(rows, chunk_rows)
    if chunk_rows >= 16:
        chunk_rows = chunk_rows // 16 * 16
    from flashinfer.dsa_indexer import dsa_indexer_det_topk

    weights = weights.squeeze(-1)
    if weights.dtype != torch.float32 or weights.shape != q.shape[:2]:
        return None, "unsupported score weights"
    lengths = (ends - starts).clamp_min(0).to(torch.int32)
    row_batch = row_batch.to(torch.int32)
    device = q.device.index
    stream = torch.cuda.current_stream(q.device)
    lock = _locks.setdefault(device, Lock())
    # FlashInfer's paged score buffer belongs to a device in this process.
    # A different stream waits on its last use; same-stream calls already order.
    with lock:
        previous = _last_use.get(device)
        if previous is not None and previous[0] != stream.cuda_stream:
            stream.wait_event(previous[1])
        for begin in range(0, rows, chunk_rows):
            end = min(begin + chunk_rows, rows)
            count = end - begin
            if index_table.shape[0] == 1:
                next_n = 16 if count % 16 == 0 else 1
                table = index_table.expand(count // next_n, -1)
            else:
                next_n = 1
                table = index_table.index_select(
                    0, row_batch[begin:end].to(torch.int64)
                )
            dsa_indexer_det_topk(
                q[begin:end].reshape(count // next_n, next_n, q.shape[1], 128),
                cache.view(-1, page_size, 132),
                weights[begin:end],
                None,
                None,
                top_k=out.shape[1],
                indices=out[begin:end],
                return_scores=False,
                backend="sm120",
                block_table=table,
                context_lens=lengths[begin:end].reshape(-1, next_n),
            )
            topk_page_table_transform_(
                out[begin:end], output_table, row_batch[begin:end], page_size
            )
        event = torch.cuda.Event()
        event.record(stream)
        _last_use[device] = (stream.cuda_stream, event)
    return out, None
