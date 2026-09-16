"""Exact-request numerical fingerprints for EAGLE and PD handoff localization.

The probes are diagnostic-only and default-off. ``EaglePDHandoffProbe`` records
the physical Prefill-to-Decode token boundary without depending on a draft
worker; ``EagleNumericalProbe`` records the first Decode target-verify and
draft-extend sequence. Those probes synchronize and emit each selected stage
immediately. ``EaglePPSenderProbe`` instead snapshots asynchronously after the
proxy send is enqueued and finalizes at the scheduler's next safe boundary.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import itertools
import json
import logging
import os
import stat
import struct
import zlib
from typing import Optional

import torch

from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

_ATOMIC_LOG_FD = 2
_COMPRESSED_RECORD_ENCODING = "zlib+base64"
_CHUNKED_RECORD_ENCODING = "zlib+base64-chunk-v1"
_COMPRESSED_RECORD_KEY = "__eagle_probe_encoding__"
_CHUNKED_RECORD_SEQUENCE = itertools.count()
_MAX_RECORD_BYTES = 1 << 20


def _emit_json_record(marker: str, payload: dict) -> None:
    """Emit a probe result as checksum-bound atomic container-log chunks.

    Multiple TP workers share the container stderr pipe.  Python logging can
    split a record across writes, allowing two workers' JSON payloads to
    interleave even when the serialized record itself fits in ``PIPE_BUF``.
    Emit every line in one syscall, compressing and chunking oversized records.
    The comparator reassembles arbitrarily interleaved chunks and verifies their
    lengths and checksums before accepting a payload.
    """

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if len(raw) > _MAX_RECORD_BYTES:
        raise ValueError(
            f"encoded {marker} raw record is {len(raw)} bytes, exceeding "
            f"safety limit {_MAX_RECORD_BYTES}"
        )
    plain_line = marker.encode() + b" " + raw + b"\n"
    try:
        fd_mode = os.fstat(_ATOMIC_LOG_FD).st_mode
        pipe_buf = int(os.fpathconf(_ATOMIC_LOG_FD, "PC_PIPE_BUF"))
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"cannot establish atomic {marker} stderr pipe limit"
        ) from exc
    if not stat.S_ISFIFO(fd_mode) or pipe_buf <= 0:
        raise RuntimeError(
            f"cannot establish atomic {marker} stderr pipe: "
            f"mode={fd_mode:o}, pipe_buf={pipe_buf}"
        )
    output_line = plain_line
    packed = None
    raw_sha256 = None
    if len(output_line) > pipe_buf:
        packed = zlib.compress(raw, level=9)
        if len(packed) > _MAX_RECORD_BYTES:
            raise ValueError(
                f"encoded {marker} packed record is {len(packed)} bytes, "
                f"exceeding safety limit {_MAX_RECORD_BYTES}"
            )
        raw_sha256 = hashlib.sha256(raw).hexdigest()
        envelope = {
            _COMPRESSED_RECORD_KEY: _COMPRESSED_RECORD_ENCODING,
            "payload": base64.b64encode(packed).decode("ascii"),
            "raw_bytes": len(raw),
            "sha256": raw_sha256,
        }
        encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
        output_line = marker.encode() + b" " + encoded + b"\n"
    if len(output_line) > pipe_buf:
        assert packed is not None and raw_sha256 is not None
        encoded_payload = base64.b64encode(packed).decode("ascii")
        packed_sha256 = hashlib.sha256(packed).hexdigest()
        record_id = f"{os.getpid()}-{next(_CHUNKED_RECORD_SEQUENCE)}-{raw_sha256[:16]}"

        def build_chunk(index: int, chunks: int, value: str) -> bytes:
            chunk = {
                _COMPRESSED_RECORD_KEY: _CHUNKED_RECORD_ENCODING,
                "chunk_index": index,
                "chunks": chunks,
                "packed_bytes": len(packed),
                "packed_sha256": packed_sha256,
                "payload": value,
                "raw_bytes": len(raw),
                "record_id": record_id,
                "sha256": raw_sha256,
            }
            return (
                marker.encode()
                + b" "
                + json.dumps(chunk, sort_keys=True, separators=(",", ":")).encode()
                + b"\n"
            )

        # Size against the conservative one-character-per-chunk count. Actual
        # index/count fields can only be shorter, so every resulting line is
        # guaranteed to fit without depending on a platform-specific margin.
        max_chunks = len(encoded_payload)
        metadata_bytes = len(build_chunk(max_chunks - 1, max_chunks, ""))
        chunk_chars = pipe_buf - metadata_bytes
        if chunk_chars <= 0:
            raise ValueError(
                f"encoded {marker} chunk metadata exceeds atomic pipe limit "
                f"{pipe_buf}"
            )
        chunks = (len(encoded_payload) + chunk_chars - 1) // chunk_chars
        output_lines = [
            build_chunk(index, chunks, encoded_payload[start : start + chunk_chars])
            for index, start in enumerate(range(0, len(encoded_payload), chunk_chars))
        ]
    else:
        output_lines = [output_line]

    for output_line in output_lines:
        written = os.write(_ATOMIC_LOG_FD, output_line)
        if written != len(output_line):
            raise RuntimeError(
                f"short atomic {marker} write: {written}/{len(output_line)} bytes"
            )


_BASE_REQUIRED_STAGES = (
    "target_verify_input",
    "target_verify_output",
    "target_verify_sample",
    "target_verify_accept",
    "target_verify_handoff",
    "draft_extend_input",
    "nextn_embed",
    "nextn_decoder",
    "nextn_norm",
    "nextn_logits",
    "proposed_token",
)
_REQUIRED_TENSORS = {
    "target_verify_input": frozenset(
        {
            "draft_token",
            "positions",
            "retrieve_index",
            "retrieve_next_token",
            "retrieve_next_sibling",
        }
    ),
    "target_verify_pp_input": frozenset({"hidden_states", "residual"}),
    "target_verify_output": frozenset({"logits", "hidden_states"}),
    "target_verify_sample": frozenset({"predict"}),
    "target_verify_accept": frozenset({"accept_lens", "accept_index"}),
    "target_verify_handoff": frozenset({"predict", "hidden_states"}),
    "draft_extend_input": frozenset({"input_ids", "positions", "target_hidden_states"}),
    "nextn_embed": frozenset({"hidden_states"}),
    "nextn_decoder": frozenset({"hidden_states"}),
    "nextn_norm": frozenset({"hidden_states"}),
    "nextn_logits": frozenset({"logits"}),
    "proposed_token": frozenset({"topk_index", "topk_probability"}),
}
_DENSE_ROW_DOMAIN = "dense_request_major_prefix"
_PROPOSAL_ROW_DOMAIN = "request_terminal"
_TARGET_TREE_ROW_DOMAIN = "target_verify_tree_node"
_PP_RANK_LOCAL_TARGET_TREE_ROW_DOMAIN = "pp_rank_local_target_verify_tree_node"
_PP_SCATTERED_TARGET_TREE_ROW_DOMAIN = "pp_scattered_target_verify_tree_node"
_PP_ATTN_GROUP_TARGET_TREE_ROW_DOMAIN = "pp_attn_group_target_verify_tree_node"
_PP_FULL_TARGET_TREE_ROW_DOMAIN = "pp_full_target_verify_tree_node"
_REQUEST_ROW_DOMAIN = "request"
_PD_HANDOFF_STAGES = {
    "prefill": ("pp_output", "metadata_write"),
    "decode": ("metadata_read", "prebuilt_bonus"),
}
_PD_HANDOFF_REQUIRED_TENSORS = {
    "pp_output": frozenset({"next_token_ids", "serialized_next_token_ids"}),
    "metadata_write": frozenset({"sampled_token", "wire_output_id"}),
    "metadata_read": frozenset({"wire_output_id", "committed_output_id"}),
    "prebuilt_bonus": frozenset({"committed_output_id", "bonus_tokens"}),
}
_PP_SENDER_STAGE = "target_verify_pp_output"
_PP_SENDER_REQUIRED_TENSORS = frozenset({"hidden_states", "residual"})
_PP_TARGET_FORWARD_BOUNDARIES = (
    "attn_input",
    "attn_output",
    "mlp_input",
    "layer_return",
)
_PP_TARGET_FORWARD_TENSORS = ("hidden_states", "residual")
_PP_TARGET_ATTENTION_OUTPUT_BOUNDARIES = (
    "flashmla_raw_output",
    "v_projection_output",
)
_PP_TARGET_ATTENTION_TENSOR = "output"
_PP_INDEXER_INPUT_TENSORS = ("q_fp8", "weights", "block_tables", "seq_lens")
_PP_INDEXER_LOGITS_TENSORS = ("logits", "seq_lens")
_PP_FLASHMLA_INPUT_TENSORS = (
    "q_nope",
    "q_rope",
    "q_input",
    "topk_indices",
    "indices",
    "logical_topk_indices",
    "cache_seqlens",
    "num_splits",
    "tile_scheduler_metadata",
)
_FLASHMLA_QUERY_SPLIT_INDPTR_ROW_DOMAIN = "flashmla_query_split_indptr"
_FLASHMLA_SCHEDULER_ROW_DOMAIN = "flashmla_scheduler_partition"


class _PPTargetForwardDeviceObserver:
    """Fixed CUDA-graph side buffer for PP0 target-layer boundaries.

    The buffer is allocated before CUDA graph capture.  Selected decoder layers
    only enqueue fixed device-to-device copies into it; request selection, host
    copies, hashing, and logging stay outside capture/replay. Once captured,
    every target-verify replay executes the fixed copies; exact-RID selection
    applies only to the host snapshot. This is a diagnostic-only observer and
    must not be used for performance evidence.
    """

    def __init__(
        self,
        *,
        layer_ids: tuple[int, ...],
        max_rows: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if not layer_ids:
            raise ValueError("PP target-forward observer requires layer anchors")
        if max_rows <= 0 or hidden_size <= 0:
            raise ValueError(
                "PP target-forward observer dimensions must be positive: "
                f"max_rows={max_rows}, hidden_size={hidden_size}"
            )
        self.layer_ids = layer_ids
        self.max_rows = int(max_rows)
        self.hidden_size = int(hidden_size)
        self.dtype = dtype
        self.stage_names = tuple(
            f"target_verify_layer_{layer_id:02d}_{boundary}"
            for layer_id in layer_ids
            for boundary in _PP_TARGET_FORWARD_BOUNDARIES
        )
        self._stage_index = {name: index for index, name in enumerate(self.stage_names)}
        self.buffer = torch.empty(
            (
                len(self.stage_names),
                len(_PP_TARGET_FORWARD_TENSORS),
                self.max_rows,
                self.hidden_size,
            ),
            dtype=dtype,
            device=device,
        )
        # Zero is the fail-closed sentinel if a stage is not executed by a
        # captured path; successful capture/replay overwrites it in-graph.
        self.row_counts = torch.zeros(
            (len(self.stage_names), len(_PP_TARGET_FORWARD_TENSORS)),
            dtype=torch.int32,
            device=device,
        )
        self._row_domains: dict[str, dict[str, str]] = {}
        self._attention_buffers: dict[str, torch.Tensor] = {}
        self._attention_row_counts: dict[str, torch.Tensor] = {}
        self._attention_widths: dict[str, int] = {}
        self._flashmla_input_buffers: dict[int, dict[str, torch.Tensor]] = {}
        self._flashmla_input_row_counts: dict[int, dict[str, torch.Tensor]] = {}
        self._logical_topk_kernel_rows: dict[int, int] = {}
        self._indexer_input_buffers: dict[int, dict[str, torch.Tensor]] = {}
        self._indexer_input_row_counts: dict[int, dict[str, torch.Tensor]] = {}
        self._indexer_input_page_sizes: dict[int, int] = {}
        self._indexer_page_columns: dict[int, torch.Tensor] = {}
        self._indexer_logits_buffers: dict[int, dict[str, torch.Tensor]] = {}
        self._indexer_logits_row_counts: dict[int, torch.Tensor] = {}
        self.device = self.buffer.device

    def install_attention_boundaries(
        self,
        *,
        layer_id: int,
        raw_output_width: int,
        v_projection_width: int,
        row_domain: str,
        num_q_heads: int,
        padded_num_q_heads: int,
        q_nope_head_dim: int,
        q_rope_head_dim: int,
        topk_width: int,
        max_scheduler_rows: int,
    ) -> None:
        """Allocate first-attention cut buffers before CUDA graph capture."""
        widths = {
            "flashmla_raw_output": raw_output_width,
            "v_projection_output": v_projection_width,
        }
        for boundary in _PP_TARGET_ATTENTION_OUTPUT_BOUNDARIES:
            width = int(widths[boundary])
            if width <= 0:
                raise ValueError(
                    f"PP target-attention observer width must be positive: "
                    f"boundary={boundary}, width={width}"
                )
            stage = f"target_verify_layer_{layer_id:02d}_{boundary}"
            if stage in self._attention_buffers:
                raise RuntimeError(
                    f"target-attention observer already installed: {stage}"
                )
            self._attention_buffers[stage] = torch.empty(
                (self.max_rows, width), dtype=self.dtype, device=self.device
            )
            self._attention_row_counts[stage] = torch.zeros(
                (1,), dtype=torch.int32, device=self.device
            )
            self._attention_widths[stage] = width
            self._row_domains[stage] = {_PP_TARGET_ATTENTION_TENSOR: row_domain}

        input_stage = f"target_verify_layer_{layer_id:02d}_flashmla_inputs"
        if layer_id in self._flashmla_input_buffers:
            raise RuntimeError(
                f"FlashMLA input observer is already installed for layer {layer_id}"
            )
        shapes_and_dtypes = {
            "q_nope": (
                (self.max_rows, num_q_heads, q_nope_head_dim),
                self.dtype,
            ),
            "q_rope": (
                (self.max_rows, num_q_heads, q_rope_head_dim),
                self.dtype,
            ),
            "q_input": (
                (
                    self.max_rows,
                    1,
                    padded_num_q_heads,
                    q_nope_head_dim + q_rope_head_dim,
                ),
                self.dtype,
            ),
            "topk_indices": ((self.max_rows, topk_width), torch.int32),
            "indices": ((self.max_rows, 1, topk_width), torch.int32),
            "logical_topk_indices": (
                (self.max_rows, topk_width),
                torch.int32,
            ),
            "cache_seqlens": ((self.max_rows,), torch.int32),
            "num_splits": ((self.max_rows + 1,), torch.int32),
            "tile_scheduler_metadata": (
                (max_scheduler_rows, 8),
                torch.int32,
            ),
        }
        input_buffers = {}
        input_row_counts = {}
        for name, (shape, tensor_dtype) in shapes_and_dtypes.items():
            if any(dim <= 0 for dim in shape):
                raise ValueError(
                    "FlashMLA input observer dimensions must be positive: "
                    f"tensor={name}, shape={shape}"
                )
            input_buffers[name] = torch.empty(
                shape, dtype=tensor_dtype, device=self.device
            )
            input_row_counts[name] = torch.zeros(
                (1,), dtype=torch.int32, device=self.device
            )
        self._flashmla_input_buffers[layer_id] = input_buffers
        self._flashmla_input_row_counts[layer_id] = input_row_counts
        self._row_domains[input_stage] = {
            **{
                name: row_domain
                for name in (
                    "q_nope",
                    "q_rope",
                    "q_input",
                    "topk_indices",
                    "indices",
                    "logical_topk_indices",
                    "cache_seqlens",
                )
            },
            "num_splits": _FLASHMLA_QUERY_SPLIT_INDPTR_ROW_DOMAIN,
            "tile_scheduler_metadata": _FLASHMLA_SCHEDULER_ROW_DOMAIN,
        }

    def install_indexer_inputs(
        self,
        *,
        layer_id: int,
        num_heads: int,
        head_dim: int,
        max_page_table_columns: int,
        page_size: int,
    ) -> None:
        """Allocate graph-stable copies of paged-MQA score inputs."""
        if layer_id in self._indexer_input_buffers:
            raise RuntimeError(
                f"indexer-input observer is already installed for layer {layer_id}"
            )
        dimensions = (num_heads, head_dim, max_page_table_columns, page_size)
        if any(value <= 0 for value in dimensions):
            raise ValueError(f"indexer-input dimensions must be positive: {dimensions}")
        stage = f"target_verify_layer_{layer_id:02d}_indexer_inputs"
        self._indexer_input_buffers[layer_id] = {
            "q_fp8": torch.zeros(
                (self.max_rows, num_heads, head_dim),
                dtype=torch.uint8,
                device=self.device,
            ),
            "weights": torch.zeros(
                (self.max_rows, num_heads),
                dtype=torch.float32,
                device=self.device,
            ),
            "block_tables": torch.zeros(
                (self.max_rows, max_page_table_columns),
                dtype=torch.int32,
                device=self.device,
            ),
            "seq_lens": torch.zeros(
                (self.max_rows * self.max_rows,),
                dtype=torch.int32,
                device=self.device,
            ),
        }
        self._indexer_input_row_counts[layer_id] = {
            name: torch.zeros((1,), dtype=torch.int32, device=self.device)
            for name in _PP_INDEXER_INPUT_TENSORS
        }
        self._indexer_input_page_sizes[layer_id] = page_size
        self._indexer_page_columns[layer_id] = torch.arange(
            max_page_table_columns, dtype=torch.int32, device=self.device
        )
        self._row_domains[stage] = {
            "q_fp8": _PP_ATTN_GROUP_TARGET_TREE_ROW_DOMAIN,
            "weights": _PP_ATTN_GROUP_TARGET_TREE_ROW_DOMAIN,
            "block_tables": "paged_mqa_block_table",
            "seq_lens": "paged_mqa_context_length",
        }

    def capture_indexer_inputs(
        self,
        *,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        """Copy the exact paged-MQA inputs used by one graph replay."""
        buffers = self._indexer_input_buffers.get(layer_id)
        if buffers is None:
            return
        stage = f"target_verify_layer_{layer_id:02d}_indexer_inputs"
        q_bytes = q_fp8.view(torch.uint8)
        if block_tables.ndim != 2 or seq_lens.numel() % block_tables.shape[0] != 0:
            raise ValueError(
                f"{stage} invalid block-table/context layout: "
                f"block_tables={tuple(block_tables.shape)}, "
                f"seq_lens={tuple(seq_lens.shape)}"
            )
        if any(
            actual > limit
            for actual, limit in zip(block_tables.shape, buffers["block_tables"].shape)
        ):
            raise ValueError(
                f"{stage}.block_tables shape {tuple(block_tables.shape)} does not fit "
                f"fixed slot {tuple(buffers['block_tables'].shape)}"
            )
        page_size = self._indexer_input_page_sizes[layer_id]
        context_lens = seq_lens.reshape(block_tables.shape[0], -1)
        valid_pages = (context_lens.max(dim=1).values + page_size - 1) // page_size
        valid_page_mask = self._indexer_page_columns[layer_id][
            : block_tables.shape[1]
        ].unsqueeze(0) < valid_pages.unsqueeze(1)
        logical_block_tables = torch.where(
            valid_page_mask, block_tables, torch.zeros_like(block_tables)
        )
        values = {
            "q_fp8": q_bytes,
            "weights": weights,
            "block_tables": logical_block_tables,
            "seq_lens": seq_lens.reshape(-1),
        }
        expected_dtypes = {
            "q_fp8": torch.uint8,
            "weights": torch.float32,
            "block_tables": torch.int32,
            "seq_lens": torch.int32,
        }
        normalized = {}
        for name, value in values.items():
            if value.device != self.device or value.dtype != expected_dtypes[name]:
                raise ValueError(
                    f"{stage}.{name} identity changed: shape={tuple(value.shape)}, "
                    f"dtype={value.dtype}, device={value.device}"
                )
            if name == "weights" and value.ndim == 3 and value.shape[-1] == 1:
                value = value.squeeze(-1)
            if value.ndim != buffers[name].ndim:
                raise ValueError(
                    f"{stage}.{name} rank changed: shape={tuple(value.shape)}, "
                    f"slot={tuple(buffers[name].shape)}"
                )
            if any(
                actual > limit
                for actual, limit in zip(value.shape, buffers[name].shape)
            ):
                raise ValueError(
                    f"{stage}.{name} shape {tuple(value.shape)} does not fit "
                    f"fixed slot {tuple(buffers[name].shape)}"
                )
            normalized[name] = value
        if q_fp8.shape[0] <= 0 or q_fp8.shape[0] != weights.shape[0]:
            raise ValueError(f"{stage} invalid q/weights rows")
        for name, value in normalized.items():
            slices = tuple(slice(0, int(size)) for size in value.shape)
            buffers[name][slices].copy_(value)
            self._indexer_input_row_counts[layer_id][name].fill_(value.shape[0])

    def install_indexer_logits(self, *, layer_id: int, max_columns: int) -> None:
        """Allocate a graph-stable copy of one layer's direct TopK inputs."""
        if layer_id in self._indexer_logits_buffers:
            raise RuntimeError(
                f"indexer-logits observer is already installed for layer {layer_id}"
            )
        if max_columns <= 0:
            raise ValueError(
                f"indexer-logits observer width must be positive: {max_columns}"
            )
        stage = f"target_verify_layer_{layer_id:02d}_indexer_logits"
        self._indexer_logits_buffers[layer_id] = {
            "logits": torch.empty(
                (self.max_rows, max_columns),
                dtype=torch.float32,
                device=self.device,
            ),
            "seq_lens": torch.empty(
                (self.max_rows,), dtype=torch.int32, device=self.device
            ),
        }
        self._indexer_logits_row_counts[layer_id] = torch.zeros(
            (1,), dtype=torch.int32, device=self.device
        )
        self._row_domains[stage] = {
            name: _PP_ATTN_GROUP_TARGET_TREE_ROW_DOMAIN
            for name in _PP_INDEXER_LOGITS_TENSORS
        }

    def capture_indexer_logits(
        self, *, layer_id: int, logits: torch.Tensor, seq_lens: torch.Tensor
    ) -> None:
        """Copy post-mask logits and their valid per-row lengths in-graph."""
        buffers = self._indexer_logits_buffers.get(layer_id)
        if buffers is None:
            return
        stage = f"target_verify_layer_{layer_id:02d}_indexer_logits"
        if (
            logits.ndim != 2
            or logits.dtype != torch.float32
            or logits.device != self.device
        ):
            raise ValueError(
                f"{stage}.logits identity changed: shape={tuple(logits.shape)}, "
                f"dtype={logits.dtype}, device={logits.device}"
            )
        rows, columns = map(int, logits.shape)
        if rows <= 0 or rows > self.max_rows or columns > buffers["logits"].shape[1]:
            raise ValueError(
                f"{stage}.logits shape {tuple(logits.shape)} does not fit "
                f"fixed slot {tuple(buffers['logits'].shape)}"
            )
        if (
            seq_lens.shape != (rows,)
            or seq_lens.dtype != torch.int32
            or seq_lens.device != self.device
        ):
            raise ValueError(
                f"{stage}.seq_lens identity changed: shape={tuple(seq_lens.shape)}, "
                f"dtype={seq_lens.dtype}, device={seq_lens.device}"
            )
        buffers["logits"][:rows, :columns].copy_(logits)
        buffers["seq_lens"][:rows].copy_(seq_lens)
        self._indexer_logits_row_counts[layer_id].fill_(rows)

    def logical_topk_kernel_output(
        self,
        *,
        layer_id: int,
        rows: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the preallocated output written by fused TopK v2."""
        output = self._logical_topk_buffer_view(
            layer_id=layer_id, rows=rows, dtype=dtype, device=device
        )
        # This is capture-time host state only. Each graph records a fixed-shape
        # TopK write to the shared slot; replay does not execute this assignment.
        self._logical_topk_kernel_rows[layer_id] = rows
        return output

    def logical_topk_flashmla_input(
        self,
        *,
        layer_id: int,
        rows: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the same logical rows previously written by fused TopK v2."""
        if self._logical_topk_kernel_rows.get(layer_id) != rows:
            raise RuntimeError(
                "logical TopK producer/consumer row mismatch: "
                f"kernel_rows={self._logical_topk_kernel_rows.get(layer_id)}, "
                f"flashmla_rows={rows}"
            )
        return self._logical_topk_buffer_view(
            layer_id=layer_id, rows=rows, dtype=dtype, device=device
        )

    def _logical_topk_buffer_view(
        self,
        *,
        layer_id: int,
        rows: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        input_buffers = self._flashmla_input_buffers.get(layer_id)
        if input_buffers is None:
            raise ValueError(
                "logical TopK output requested for an uninstalled layer: "
                f"requested={layer_id}, installed={sorted(self._flashmla_input_buffers)}"
            )
        buffer = input_buffers.get("logical_topk_indices")
        if buffer is None:
            raise RuntimeError("logical TopK output requested before observer install")
        if rows <= 0 or rows > buffer.shape[0]:
            raise ValueError(
                f"logical TopK rows {rows} are outside fixed capacity "
                f"1..{buffer.shape[0]}"
            )
        if dtype != buffer.dtype or device != buffer.device:
            raise ValueError(
                "logical TopK output identity changed: "
                f"dtype={dtype}, device={device}, expected dtype={buffer.dtype}, "
                f"device={buffer.device}"
            )
        return buffer[:rows]

    def capture_flashmla_inputs(
        self,
        *,
        layer_id: int,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        q_input: torch.Tensor,
        topk_indices: torch.Tensor,
        indices: torch.Tensor,
        logical_topk_indices: torch.Tensor,
        cache_seqlens: torch.Tensor,
        num_splits: torch.Tensor,
        tile_scheduler_metadata: torch.Tensor,
    ) -> None:
        """Copy the exact FlashMLA inputs into preallocated graph slots."""
        stage = f"target_verify_layer_{layer_id:02d}_flashmla_inputs"
        values = {
            "q_nope": q_nope,
            "q_rope": q_rope,
            "q_input": q_input,
            "topk_indices": topk_indices,
            "indices": indices,
            "logical_topk_indices": logical_topk_indices,
            "cache_seqlens": cache_seqlens,
            "num_splits": num_splits,
            "tile_scheduler_metadata": tile_scheduler_metadata,
        }
        input_buffers = self._flashmla_input_buffers.get(layer_id)
        input_row_counts = self._flashmla_input_row_counts.get(layer_id)
        if input_buffers is None or input_row_counts is None:
            return
        if set(values) != set(input_buffers):
            raise RuntimeError("FlashMLA input observer schema drift")
        # Validate the complete schema before recording any copy/fill. A shape
        # or dtype mismatch therefore cannot leave a partially credible stage.
        for name in _PP_FLASHMLA_INPUT_TENSORS:
            value = values[name]
            buffer = input_buffers[name]
            if not isinstance(value, torch.Tensor) or value.ndim < 1:
                raise TypeError(f"{stage}.{name} must be a rank >= 1 tensor")
            rows = int(value.shape[0])
            if (
                rows <= 0
                or rows > buffer.shape[0]
                or tuple(value.shape[1:]) != tuple(buffer.shape[1:])
            ):
                raise ValueError(
                    f"{stage}.{name} shape {tuple(value.shape)} does not fit "
                    f"fixed slot {tuple(buffer.shape)}"
                )
            if value.dtype != buffer.dtype or value.device != buffer.device:
                raise ValueError(
                    f"{stage}.{name} identity changed: dtype={value.dtype}, "
                    f"device={value.device}, expected dtype={buffer.dtype}, "
                    f"device={buffer.device}"
                )
        for name in _PP_FLASHMLA_INPUT_TENSORS:
            value = values[name]
            buffer = input_buffers[name]
            rows = int(value.shape[0])
            # The fused v2 TopK kernel writes the logical selection directly to
            # this preallocated slot. Do not add a redundant graph node when
            # capture receives that exact view back from the attention path.
            if value.data_ptr() != buffer.data_ptr():
                buffer[:rows].copy_(value)
            input_row_counts[name].fill_(rows)

    def capture_attention(
        self,
        *,
        layer_id: int,
        boundary: str,
        output: torch.Tensor,
    ) -> None:
        """Copy one fixed attention intermediate into its graph-side slot."""
        stage = f"target_verify_layer_{layer_id:02d}_{boundary}"
        buffer = self._attention_buffers.get(stage)
        if buffer is None:
            return
        if not isinstance(output, torch.Tensor) or output.ndim < 2:
            raise TypeError(f"{stage}.output must be a rank >= 2 tensor")
        rows = int(output.shape[0])
        flattened = output.reshape(rows, -1)
        expected_width = self._attention_widths[stage]
        if flattened.shape[1] != expected_width:
            raise ValueError(
                f"{stage}.output width {flattened.shape[1]} does not match "
                f"fixed width {expected_width}"
            )
        if rows <= 0 or rows > self.max_rows:
            raise ValueError(
                f"{stage}.output rows {rows} outside fixed capacity "
                f"[1, {self.max_rows}]"
            )
        if output.dtype != self.dtype or output.device != self.device:
            raise ValueError(
                f"{stage}.output identity changed: dtype={output.dtype}, "
                f"device={output.device}, expected dtype={self.dtype}, "
                f"device={self.device}"
            )
        buffer[:rows].copy_(flattened)
        self._attention_row_counts[stage].fill_(rows)

    def capture(
        self,
        *,
        layer_id: int,
        boundary: str,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        hidden_row_domain: str,
        residual_row_domain: str,
    ) -> None:
        stage = f"target_verify_layer_{layer_id:02d}_{boundary}"
        stage_index = self._stage_index.get(stage)
        if stage_index is None:
            return
        tensors = (hidden_states, residual)
        for tensor_index, (tensor_name, tensor) in enumerate(
            zip(_PP_TARGET_FORWARD_TENSORS, tensors)
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{stage}.{tensor_name} must be a tensor")
            if tensor.ndim != 2 or tensor.shape[1] != self.hidden_size:
                raise ValueError(
                    f"{stage}.{tensor_name} shape {tuple(tensor.shape)} does not "
                    f"match [rows, {self.hidden_size}]"
                )
            if tensor.shape[0] > self.max_rows:
                raise ValueError(
                    f"{stage}.{tensor_name} rows {tensor.shape[0]} exceed fixed "
                    f"capacity {self.max_rows}"
                )
            if tensor.dtype != self.dtype or tensor.device != self.device:
                raise ValueError(
                    f"{stage}.{tensor_name} identity changed: "
                    f"dtype={tensor.dtype}, device={tensor.device}, "
                    f"expected dtype={self.dtype}, device={self.device}"
                )
        row_domains = {
            "hidden_states": hidden_row_domain,
            "residual": residual_row_domain,
        }
        self._validate_row_domains(stage, tensors, row_domains)
        previous_domains = self._row_domains.setdefault(stage, row_domains)
        if previous_domains != row_domains:
            raise ValueError(
                f"{stage} row-domain drift: {previous_domains} -> {row_domains}"
            )
        for tensor_index, tensor in enumerate(tensors):
            self.buffer[stage_index, tensor_index, : tensor.shape[0]].copy_(tensor)
            # Each graph shape writes each tensor's valid-row count during
            # replay. The communicator may keep residual in TP_ATTN_FULL while
            # hidden_states use FULL layout, so counts are not stage-level.
            self.row_counts[stage_index, tensor_index].fill_(tensor.shape[0])

    def _validate_row_domains(
        self,
        stage: str,
        tensors: tuple[torch.Tensor, torch.Tensor],
        row_domains: dict[str, str],
    ) -> None:
        allowed = {
            _PP_RANK_LOCAL_TARGET_TREE_ROW_DOMAIN,
            _PP_SCATTERED_TARGET_TREE_ROW_DOMAIN,
            _PP_ATTN_GROUP_TARGET_TREE_ROW_DOMAIN,
            _PP_FULL_TARGET_TREE_ROW_DOMAIN,
        }
        for name, tensor in zip(_PP_TARGET_FORWARD_TENSORS, tensors):
            domain = row_domains[name]
            rows = int(tensor.shape[0])
            if domain not in allowed:
                raise ValueError(f"{stage} unsupported {name} row domain: {domain!r}")
            if rows <= 0:
                raise ValueError(
                    f"{stage} invalid {name} rows={rows} for " f"row_domain={domain}"
                )

    def snapshot_stages(self) -> dict[str, dict]:
        stages = {
            stage: {
                "tensor_metadata": {
                    tensor_name: {
                        "logical_rows": self.row_counts[
                            stage_index, tensor_index : tensor_index + 1
                        ],
                        "row_domain": self._row_domains.get(stage, {}).get(
                            tensor_name, _PP_RANK_LOCAL_TARGET_TREE_ROW_DOMAIN
                        ),
                    }
                    for tensor_index, tensor_name in enumerate(
                        _PP_TARGET_FORWARD_TENSORS
                    )
                },
                "tensors": {
                    tensor_name: self.buffer[stage_index, tensor_index]
                    for tensor_index, tensor_name in enumerate(
                        _PP_TARGET_FORWARD_TENSORS
                    )
                },
            }
            for stage, stage_index in self._stage_index.items()
        }
        stages.update(
            {
                stage: {
                    "tensor_metadata": {
                        _PP_TARGET_ATTENTION_TENSOR: {
                            "logical_rows": self._attention_row_counts[stage],
                            "row_domain": self._row_domains[stage][
                                _PP_TARGET_ATTENTION_TENSOR
                            ],
                        }
                    },
                    "tensors": {
                        _PP_TARGET_ATTENTION_TENSOR: buffer,
                    },
                }
                for stage, buffer in self._attention_buffers.items()
            }
        )
        for layer_id, input_buffers in self._flashmla_input_buffers.items():
            stage = f"target_verify_layer_{layer_id:02d}_flashmla_inputs"
            stages[stage] = {
                "tensor_metadata": {
                    name: {
                        "logical_rows": self._flashmla_input_row_counts[layer_id][name],
                        "row_domain": self._row_domains[stage][name],
                    }
                    for name in _PP_FLASHMLA_INPUT_TENSORS
                },
                "tensors": {
                    name: input_buffers[name] for name in _PP_FLASHMLA_INPUT_TENSORS
                },
            }
        for layer_id, buffers in self._indexer_logits_buffers.items():
            stage = f"target_verify_layer_{layer_id:02d}_indexer_logits"
            stages[stage] = {
                "tensor_metadata": {
                    name: {
                        "logical_rows": self._indexer_logits_row_counts[layer_id],
                        "row_domain": self._row_domains[stage][name],
                    }
                    for name in _PP_INDEXER_LOGITS_TENSORS
                },
                "tensors": buffers,
            }
        for layer_id, buffers in self._indexer_input_buffers.items():
            stage = f"target_verify_layer_{layer_id:02d}_indexer_inputs"
            stages[stage] = {
                "tensor_metadata": {
                    name: {
                        "logical_rows": self._indexer_input_row_counts[layer_id][name],
                        "row_domain": self._row_domains[stage][name],
                    }
                    for name in _PP_INDEXER_INPUT_TENSORS
                },
                "tensors": buffers,
            }
        return stages


class EaglePPSenderProbe:
    """Capture one PP-non-last target output before the async proxy send.

    The sender cannot complete the PP-last EAGLE numerical state machine, so
    this observer owns an independent one-stage record. The worker only attaches
    the observer to an exact-RID result. The scheduler first enqueues the proxy
    send, then starts a non-blocking host snapshot, and finalizes the hash after
    the existing next-slot send commit.
    """

    def __init__(
        self,
        expected_rid: Optional[str],
        *,
        capture_id: Optional[str] = None,
        pod_name: Optional[str] = None,
        pod_uid: Optional[str] = None,
    ) -> None:
        self.expected_rid = expected_rid or None
        capture_values = (capture_id, pod_name, pod_uid)
        if self.expected_rid is None and any(capture_values):
            raise ValueError(
                "PP sender probe capture identity requires an exact request id"
            )
        if self.expected_rid is not None and not all(capture_values):
            raise ValueError(
                "PP sender probe requires capture id and Downward API Pod name/UID"
            )
        self.capture = (
            {"id": capture_id, "pod_name": pod_name, "pod_uid": pod_uid}
            if self.expected_rid is not None
            else None
        )
        self._sealed = False
        self._pending: Optional[dict] = None
        self._target_forward_observer: Optional[_PPTargetForwardDeviceObserver] = None

    @property
    def target_forward_observer(self) -> Optional[_PPTargetForwardDeviceObserver]:
        return self._target_forward_observer

    def install_target_forward_observer(
        self,
        *,
        model,
        max_rows: int,
        max_indexer_columns: int,
        indexer_page_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        """Install the fixed observer before target CUDA graphs are captured."""
        if not self.can_probe:
            return
        body = getattr(model, "model", None)
        layers = getattr(body, "layers", None)
        start_layer = getattr(body, "start_layer", None)
        end_layer = getattr(body, "end_layer", None)
        if (
            layers is None
            or not isinstance(start_layer, int)
            or not isinstance(end_layer, int)
        ):
            raise TypeError(
                "PP target-forward observer requires a model body with local layers"
            )
        if model.__class__.__name__ not in {
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
            "DeepseekV32ForCausalLM",
            "GlmMoeDsaForCausalLM",
        }:
            raise TypeError(
                "PP target-forward observer requires a DeepseekV2 decoder body, "
                f"got {model.__class__.__name__}"
            )
        num_local_layers = end_layer - start_layer
        if start_layer != 0 or num_local_layers <= 0:
            raise ValueError(
                "PP target-forward observer is restricted to PP0, got "
                f"layer range [{start_layer}, {end_layer})"
            )
        layer_ids = tuple(
            dict.fromkeys(
                (
                    start_layer,
                    min(start_layer + 1, end_layer - 1),
                    start_layer + num_local_layers // 4,
                    start_layer + num_local_layers // 2,
                    end_layer - 1,
                )
            )
        )
        observer_device = torch.device(device)
        observer = _PPTargetForwardDeviceObserver(
            layer_ids=layer_ids,
            max_rows=max_rows,
            hidden_size=int(model.config.hidden_size),
            dtype=dtype,
            device=observer_device,
        )
        for layer_id in layer_ids:
            layer = layers[layer_id]
            if getattr(layer, "target_forward_probe", None) is not None:
                raise RuntimeError(
                    f"target-forward observer already installed on layer {layer_id}"
                )
            layer.target_forward_probe = observer
        for layer_id in layer_ids[:2]:
            layer = layers[layer_id]
            attention = getattr(layer, "self_attn", None)
            if attention is None:
                raise TypeError(
                    f"PP target-attention observer requires self_attn on layer {layer_id}"
                )
            if getattr(attention, "target_forward_probe", None) is not None:
                raise RuntimeError(
                    f"target-attention observer already installed on layer {layer_id}"
                )
            indexer = getattr(attention, "indexer", None)
            if indexer is None:
                raise TypeError(
                    f"PP target-attention observer requires an indexer on layer {layer_id}"
                )
            if getattr(indexer, "target_forward_probe", None) is not None:
                raise RuntimeError(
                    f"TopK observer already installed on layer {layer_id}"
                )
            radix_attention = getattr(attention, "attn_mqa", None)
            if radix_attention is None:
                raise TypeError(
                    f"PP target-attention observer requires attn_mqa on layer {layer_id}"
                )
            if getattr(radix_attention, "target_forward_probe", None) is not None:
                raise RuntimeError(
                    f"FlashMLA observer already installed on layer {layer_id}"
                )
            observer.install_attention_boundaries(
                layer_id=layer_id,
                raw_output_width=(
                    int(attention.num_local_heads) * int(attention.kv_lora_rank)
                ),
                v_projection_width=(
                    int(attention.num_local_heads) * int(attention.v_head_dim)
                ),
                # Both cuts precede the rank-local o_proj partial and therefore
                # retain the attention-group target-tree row layout.
                row_domain=_PP_ATTN_GROUP_TARGET_TREE_ROW_DOMAIN,
                num_q_heads=int(attention.num_local_heads),
                padded_num_q_heads=(
                    64
                    if int(attention.num_local_heads) <= 64
                    else (
                        128
                        if int(attention.num_local_heads) <= 128
                        else int(attention.num_local_heads)
                    )
                ),
                q_nope_head_dim=int(attention.kv_lora_rank),
                q_rope_head_dim=int(attention.qk_rope_head_dim),
                topk_width=int(model.config.index_topk),
                max_scheduler_rows=(
                    int(
                        torch.cuda.get_device_properties(
                            observer_device
                        ).multi_processor_count
                    )
                    if observer_device.type == "cuda"
                    else 1
                ),
            )
            observer.install_indexer_logits(
                layer_id=layer_id,
                max_columns=max_indexer_columns,
            )
            observer.install_indexer_inputs(
                layer_id=layer_id,
                num_heads=int(indexer.n_heads),
                head_dim=int(indexer.head_dim),
                max_page_table_columns=(max_indexer_columns + indexer_page_size - 1)
                // indexer_page_size,
                page_size=indexer_page_size,
            )
            attention.target_forward_probe = observer
            indexer.target_forward_probe = observer
            radix_attention.target_forward_probe = observer
        self._target_forward_observer = observer

    @property
    def can_probe(self) -> bool:
        return self.expected_rid is not None and not self._sealed

    def matches_schedule_batch(self, batch) -> bool:
        reqs = getattr(batch, "reqs", ())
        return bool(
            self.can_probe
            and [getattr(req, "rid", None) for req in reqs] == [self.expected_rid]
        )

    def begin_target_verify_pp_output(
        self,
        *,
        pp_proxy_tensors: Optional[dict[str, object]],
        target_world_rank: int,
        require_attn_tp_allgather: bool,
    ) -> bool:
        """Queue a snapshot after the proxy send without waiting for it."""
        if not self.can_probe:
            return False
        self._sealed = True
        proxy = pp_proxy_tensors or {}
        tensors = {
            name: proxy.get(name)
            for name in _PP_SENDER_REQUIRED_TENSORS
            if isinstance(proxy.get(name), torch.Tensor)
        }
        missing = sorted(_PP_SENDER_REQUIRED_TENSORS - tensors.keys())
        if missing:
            self._emit(
                rejection=f"stage decode.{_PP_SENDER_STAGE} missing tensors {missing}"
            )
            return False
        hidden_rows = tensors["hidden_states"].shape[0]
        residual_rows = tensors["residual"].shape[0]
        if hidden_rows != residual_rows:
            self._emit(
                rejection=(
                    f"stage decode.{_PP_SENDER_STAGE} row mismatch "
                    f"hidden_states={hidden_rows}, residual={residual_rows}"
                )
            )
            return False

        try:
            stage_sources = {
                _PP_SENDER_STAGE: {
                    "tensor_metadata": {
                        name: {
                            "logical_rows": hidden_rows,
                            "row_domain": _PP_RANK_LOCAL_TARGET_TREE_ROW_DOMAIN,
                        }
                        for name in tensors
                    },
                    "tensors": tensors,
                }
            }
            if self._target_forward_observer is not None:
                stage_sources.update(self._target_forward_observer.snapshot_stages())
            snapshots = {}
            cuda_devices = set()
            for stage_name, stage_source in stage_sources.items():
                snapshots[stage_name] = {
                    "tensor_metadata": {},
                    "tensors": {},
                }
                for name, metadata in stage_source["tensor_metadata"].items():
                    logical_rows = metadata["logical_rows"]
                    if isinstance(logical_rows, torch.Tensor):
                        row_count = logical_rows.detach()
                        if row_count.is_cuda:
                            host_row_count = torch.empty(
                                row_count.shape,
                                dtype=row_count.dtype,
                                device="cpu",
                                pin_memory=True,
                            )
                            host_row_count.copy_(row_count, non_blocking=True)
                            row_count.record_stream(
                                torch.cuda.current_stream(row_count.device)
                            )
                            cuda_devices.add(row_count.device)
                        else:
                            host_row_count = row_count.clone()
                        logical_rows = host_row_count
                    else:
                        logical_rows = int(logical_rows)
                    snapshots[stage_name]["tensor_metadata"][name] = {
                        "logical_rows": logical_rows,
                        "row_domain": metadata["row_domain"],
                    }
                for name, tensor in stage_source["tensors"].items():
                    value = tensor.detach()
                    if value.is_cuda:
                        snapshot = torch.empty(
                            value.shape,
                            dtype=value.dtype,
                            device="cpu",
                            pin_memory=True,
                        )
                        snapshot.copy_(value, non_blocking=True)
                        value.record_stream(torch.cuda.current_stream(value.device))
                        cuda_devices.add(value.device)
                    else:
                        snapshot = value.clone()
                    snapshots[stage_name]["tensors"][name] = snapshot
            if len(cuda_devices) > 1:
                raise RuntimeError(
                    "PP sender tensors span multiple CUDA devices: "
                    f"{sorted(map(str, cuda_devices))}"
                )
            completion_event = None
            if cuda_devices:
                device = next(iter(cuda_devices))
                completion_event = torch.cuda.Event()
                completion_event.record(torch.cuda.current_stream(device=device))
        except (RuntimeError, TypeError, ValueError) as exc:
            self._emit(
                rejection=(
                    f"failed to queue snapshot for decode.{_PP_SENDER_STAGE}: {exc}"
                )
            )
            raise

        parallel = get_parallel()
        rank = _rank_payload() or {}
        for name in (
            "pp_size",
            "tp_rank",
            "tp_size",
            "attn_tp_rank",
            "attn_tp_size",
            "attn_dp_size",
        ):
            value = getattr(parallel, name, None)
            if value is not None:
                rank[name] = value
        self._pending = {
            "snapshots": snapshots,
            "completion_event": completion_event,
            "rank": rank,
            "transport": {
                "target_world_rank": target_world_rank,
                "mode": (
                    "send_slice_recv_allgather"
                    if require_attn_tp_allgather
                    else "direct_rank_local"
                ),
                "require_attn_tp_allgather": require_attn_tp_allgather,
            },
        }
        return True

    def finalize_target_verify_pp_output(self) -> None:
        """Finalize only after the existing proxy-send commit."""
        if self._pending is None:
            return
        pending, self._pending = self._pending, None
        try:
            if pending["completion_event"] is not None:
                pending["completion_event"].synchronize()
            stages = {
                stage_name: self._finalize_stage_snapshot(stage_snapshot)
                for stage_name, stage_snapshot in pending["snapshots"].items()
            }
        except (RuntimeError, TypeError, ValueError) as exc:
            self._emit(
                rejection=(
                    f"failed to finalize snapshot for decode.{_PP_SENDER_STAGE}: {exc}"
                ),
                rank=pending["rank"],
                transport=pending["transport"],
            )
            raise
        self._emit(
            stages=stages,
            rank=pending["rank"],
            transport=pending["transport"],
        )

    def _finalize_stage_snapshot(self, stage_snapshot: dict) -> dict:
        tensor_metadata = stage_snapshot["tensor_metadata"]
        if set(tensor_metadata) != set(stage_snapshot["tensors"]):
            raise ValueError("target-forward tensor metadata does not match tensors")
        fingerprints = {}
        stage_metadata = set()
        indexer_lengths = (
            stage_snapshot["tensors"].get("seq_lens")
            if set(stage_snapshot["tensors"]) == set(_PP_INDEXER_LOGITS_TENSORS)
            else None
        )
        for name, tensor in sorted(stage_snapshot["tensors"].items()):
            metadata = tensor_metadata[name]
            logical_rows = metadata["logical_rows"]
            if isinstance(logical_rows, torch.Tensor):
                if logical_rows.numel() != 1:
                    raise ValueError("target-forward row count must be scalar")
                logical_rows = int(logical_rows.item())
            row_domain = metadata.get("row_domain")
            if (
                not isinstance(row_domain, str)
                or not row_domain
                or logical_rows <= 0
                or logical_rows > int(tensor.shape[0])
            ):
                raise ValueError(
                    f"invalid target-forward metadata for {name}: "
                    f"row_domain={row_domain!r}, logical_rows={logical_rows}"
                )
            fingerprint = (
                _ragged_rows_fingerprint(tensor, indexer_lengths, logical_rows)
                if name == "logits" and indexer_lengths is not None
                else _tensor_fingerprint(
                    tensor,
                    logical_rows,
                    include_row_multiset=(name == "logical_topk_indices"),
                )
            )
            fingerprints[name] = {
                "row_domain": row_domain,
                "logical_rows": logical_rows,
                **fingerprint,
            }
            stage_metadata.add((row_domain, logical_rows))
        result = {"tensors": fingerprints}
        # Preserve the stage-level fields for the ordinary PP sender boundary,
        # while mixed-layout target-forward stages use tensor-level metadata.
        if len(stage_metadata) == 1:
            result["row_domain"], result["logical_rows"] = next(iter(stage_metadata))
        return result

    def _emit(
        self,
        *,
        stage: Optional[dict] = None,
        stages: Optional[dict[str, dict]] = None,
        rejection: Optional[str] = None,
        rank: Optional[dict] = None,
        transport: Optional[dict] = None,
    ) -> None:
        payload = {
            "rid": self.expected_rid,
            "capture": self.capture,
            "phase": "decode",
            "role": "decode",
            "status": "rejected" if rejection else "complete",
            "rejection": rejection,
            "stages": (
                stages
                if stages is not None
                else ({_PP_SENDER_STAGE: stage} if stage is not None else {})
            ),
        }
        rank = _rank_payload() if rank is None else rank
        if rank is not None:
            payload["rank"] = rank
        if transport is not None:
            payload["transport"] = transport
        _emit_json_record("EAGLE_PP_SENDER_PROBE_RESULT", payload)


def _rank_payload() -> Optional[dict[str, int]]:
    try:
        parallel = get_parallel()
        return {
            "world": parallel.world_rank,
            "pp": parallel.pp_rank,
            "attn_dp": parallel.attn_dp_rank,
        }
    except (AssertionError, AttributeError, RuntimeError):
        return None


def _synchronize_cuda_tensors(
    tensors: dict[str, Optional[torch.Tensor]],
) -> None:
    """Expose asynchronous device faults at the owning stage boundary."""

    devices = {
        tensor.device
        for tensor in tensors.values()
        if tensor is not None and tensor.is_cuda
    }
    for device in sorted(devices, key=str):
        torch.cuda.current_stream(device=device).synchronize()


def _tensor_fingerprint(
    tensor: torch.Tensor,
    logical_rows: int,
    *,
    include_row_multiset: bool = False,
) -> dict:
    value = tensor.detach()
    if logical_rows <= 0:
        raise ValueError(f"logical_rows must be positive, got {logical_rows}")
    if value.ndim == 0:
        raise ValueError("row-domain tensor must have at least one dimension")
    if value.shape[0] < logical_rows:
        raise ValueError(
            f"tensor has {value.shape[0]} rows, expected at least {logical_rows}"
        )
    value = value[:logical_rows]
    cpu = value.contiguous().cpu()
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    result = {
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if include_row_multiset:
        if cpu.ndim != 2 or cpu.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError(
                "row-multiset fingerprint requires a rank-2 integer tensor, "
                f"got shape={tuple(cpu.shape)}, dtype={cpu.dtype}"
            )
        canonical = torch.sort(cpu, dim=1).values.contiguous()
        canonical_raw = canonical.reshape(-1).view(torch.uint8).numpy().tobytes()
        result["row_multiset_sha256"] = hashlib.sha256(canonical_raw).hexdigest()
    if cpu.numel() == 0:
        result.update({"finite": True, "sum": 0.0, "abs_max": 0.0})
        return result

    numeric = cpu.to(torch.float64)
    finite = torch.isfinite(numeric)
    result["finite"] = bool(finite.all())
    if bool(finite.any()):
        finite_values = numeric[finite]
        result["sum"] = float(finite_values.sum())
        result["abs_max"] = float(finite_values.abs().max())
    else:
        result["sum"] = None
        result["abs_max"] = None
    if cpu.numel() <= 16:
        result["values"] = cpu.reshape(-1).tolist()
    return result


def _ragged_rows_fingerprint(
    tensor: torch.Tensor, row_lengths: torch.Tensor, logical_rows: int
) -> dict:
    """Fingerprint only each row's valid prefix, excluding uninitialized tail."""
    if tensor.ndim != 2 or tensor.dtype != torch.float32:
        raise ValueError(
            "ragged-row fingerprint requires a rank-2 float32 tensor, "
            f"got shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
        )
    lengths = row_lengths.detach()[:logical_rows].contiguous().cpu()
    if (
        lengths.ndim != 1
        or lengths.dtype != torch.int32
        or len(lengths) != logical_rows
    ):
        raise ValueError(
            "ragged-row fingerprint lengths must be int32 with one value per row"
        )
    cpu = tensor.detach()[:logical_rows].contiguous().cpu()
    digest_state = hashlib.sha256()
    finite = True
    total = 0.0
    abs_max = 0.0
    valid_elements = 0
    for row, length_value in enumerate(lengths.tolist()):
        length = int(length_value)
        if length < 0 or length > cpu.shape[1]:
            raise ValueError(f"ragged-row length {length} is outside 0..{cpu.shape[1]}")
        values = cpu[row, :length]
        digest_state.update(struct.pack("<Q", length))
        digest_state.update(values.view(torch.uint8).numpy().tobytes())
        finite_values = values.to(torch.float64)
        finite_mask = torch.isfinite(finite_values)
        finite = finite and bool(finite_mask.all())
        if bool(finite_mask.any()):
            selected = finite_values[finite_mask]
            total += float(selected.sum())
            abs_max = max(abs_max, float(selected.abs().max()))
        valid_elements += length
    return {
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "sha256": digest_state.hexdigest(),
        "valid_elements": valid_elements,
        "finite": finite,
        "sum": total,
        "abs_max": abs_max,
    }


class EaglePDHandoffProbe:
    """Capture the real two-process PD token handoff for one request."""

    def __init__(
        self,
        expected_rid: Optional[str],
        *,
        role: str,
        capture_id: Optional[str] = None,
        pod_name: Optional[str] = None,
        pod_uid: Optional[str] = None,
    ) -> None:
        if role not in _PD_HANDOFF_STAGES:
            raise ValueError(f"unsupported PD handoff role {role!r}")
        self.expected_rid = expected_rid or None
        self.role = role
        capture_values = (capture_id, pod_name, pod_uid)
        if self.expected_rid is None and any(capture_values):
            raise ValueError(
                "PD handoff probe capture identity requires an exact request id"
            )
        if self.expected_rid is not None and not all(capture_values):
            raise ValueError(
                "PD handoff probe requires capture id and Downward API Pod name/UID"
            )
        self.capture = (
            {"id": capture_id, "pod_name": pod_name, "pod_uid": pod_uid}
            if self.expected_rid is not None
            else None
        )
        self._records: dict[str, dict] = {}
        self._rejection: Optional[str] = None
        self._sealed = False
        self._seen = False

    @property
    def can_probe(self) -> bool:
        return self.expected_rid is not None and not self._sealed

    def _matches(self, rids: Optional[list[str]]) -> bool:
        return bool(self.can_probe and rids == [self.expected_rid])

    def _emit(
        self,
        marker: str,
        stage: str,
        logical_rows: int,
        fingerprints: dict[str, dict],
        error: Optional[BaseException] = None,
    ) -> None:
        payload = {
            "rid": self.expected_rid,
            "capture": self.capture,
            "phase": "pd_handoff",
            "role": self.role,
            "stage": stage,
            # Successful stages are inserted before emission, while a stage
            # error is emitted before insertion. Keep both paths one-based.
            "ordinal": len(self._records) + (stage not in self._records),
            "row_domain": _REQUEST_ROW_DOMAIN,
            "logical_rows": logical_rows,
            "fingerprints": fingerprints,
            "rank": _rank_payload(),
        }
        if error is not None:
            payload.update({"error_type": type(error).__name__, "error": str(error)})
        logger.warning(
            "%s %s", marker, json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )

    def _reject(self, reason: str) -> None:
        if self._rejection is None:
            self._rejection = reason
            logger.error(
                "EAGLE PD handoff diagnostic failed closed for rid=%s role=%s: %s",
                self.expected_rid,
                self.role,
                reason,
            )

    def _record(
        self,
        stage: str,
        tensors: dict[str, Optional[torch.Tensor]],
        *,
        logical_rows: int,
    ) -> None:
        if not self.can_probe:
            return
        self._seen = True
        expected_stages = _PD_HANDOFF_STAGES[self.role]
        if self._rejection is not None:
            if stage == expected_stages[-1]:
                self._seal()
            return
        expected_stage = expected_stages[len(self._records)]
        if stage != expected_stage:
            self._reject(f"out-of-order stage {stage}, expected {expected_stage}")
            self._seal()
            return
        required = _PD_HANDOFF_REQUIRED_TENSORS[stage]
        missing = sorted(name for name in required if tensors.get(name) is None)
        if logical_rows <= 0 or missing:
            self._reject(
                f"stage {stage} invalid logical_rows={logical_rows} missing={missing}"
            )
            self._seal()
            return
        try:
            _synchronize_cuda_tensors(tensors)
            fingerprints = {
                name: _tensor_fingerprint(tensor, logical_rows)
                for name, tensor in sorted(tensors.items())
                if tensor is not None
            }
        except (RuntimeError, TypeError, ValueError) as exc:
            self._emit(
                "EAGLE_PD_HANDOFF_PROBE_STAGE_ERROR",
                stage,
                logical_rows,
                {},
                error=exc,
            )
            self._reject(f"failed to fingerprint {stage}: {exc}")
            self._seal()
            if isinstance(exc, RuntimeError) and any(
                tensor is not None and tensor.is_cuda for tensor in tensors.values()
            ):
                raise
            return
        self._records[stage] = {
            "row_domain": _REQUEST_ROW_DOMAIN,
            "logical_rows": logical_rows,
            "tensors": fingerprints,
        }
        self._emit("EAGLE_PD_HANDOFF_PROBE_STAGE", stage, logical_rows, fingerprints)
        if stage == expected_stages[-1]:
            self._seal()

    def _seal(self) -> None:
        if self._sealed:
            return
        self._sealed = True
        expected_stages = _PD_HANDOFF_STAGES[self.role]
        missing = [stage for stage in expected_stages if stage not in self._records]
        if self._rejection is None and missing:
            self._rejection = f"missing stages {missing}"
        payload = {
            "rid": self.expected_rid,
            "capture": self.capture,
            "phase": "pd_handoff",
            "role": self.role,
            "status": "rejected" if self._rejection else "complete",
            "rejection": self._rejection,
            "seen": self._seen,
            "stages": self._records,
            "rank": _rank_payload(),
        }
        logger.warning(
            "EAGLE_PD_HANDOFF_PROBE_RESULT %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )

    def record_prefill_pp_output(
        self,
        *,
        rids: Optional[list[str]],
        next_token_ids: torch.Tensor,
        serialized_next_token_ids: torch.Tensor,
    ) -> None:
        if self.role == "prefill" and self._matches(rids):
            self._record(
                "pp_output",
                {
                    "next_token_ids": next_token_ids,
                    "serialized_next_token_ids": serialized_next_token_ids,
                },
                logical_rows=len(rids),
            )

    def record_prefill_metadata_write(
        self, *, rid: str, sampled_token: int, wire_output_id: torch.Tensor
    ) -> None:
        if self.role == "prefill" and self._matches([rid]):
            sampled_token_tensor = torch.as_tensor(
                [sampled_token],
                dtype=wire_output_id.dtype,
                device=wire_output_id.device,
            )
            self._record(
                "metadata_write",
                {
                    "sampled_token": sampled_token_tensor,
                    "wire_output_id": wire_output_id,
                },
                logical_rows=1,
            )

    def record_decode_metadata_read(
        self, *, rid: str, wire_output_id: torch.Tensor, committed_output_id: int
    ) -> None:
        if self.role == "decode" and self._matches([rid]):
            committed_output_id_tensor = torch.as_tensor(
                [committed_output_id],
                dtype=wire_output_id.dtype,
                device=wire_output_id.device,
            )
            self._record(
                "metadata_read",
                {
                    "wire_output_id": wire_output_id,
                    "committed_output_id": committed_output_id_tensor,
                },
                logical_rows=1,
            )

    def record_decode_prebuilt_bonus(
        self,
        *,
        rids: Optional[list[str]],
        committed_output_id: torch.Tensor,
        bonus_tokens: torch.Tensor,
    ) -> None:
        if self.role == "decode" and self._matches(rids):
            self._record(
                "prebuilt_bonus",
                {
                    "committed_output_id": committed_output_id,
                    "bonus_tokens": bonus_tokens,
                },
                logical_rows=len(rids),
            )


class EagleNumericalProbe:
    """Capture one Decode numerical fingerprint."""

    def __init__(
        self,
        expected_rid: Optional[str],
        *,
        capture_id: Optional[str] = None,
        pod_name: Optional[str] = None,
        pod_uid: Optional[str] = None,
        require_target_verify_pp_input: bool = False,
    ) -> None:
        self.expected_rid = expected_rid or None
        capture_values = (capture_id, pod_name, pod_uid)
        if self.expected_rid is None and any(capture_values):
            raise ValueError(
                "numerical probe capture identity requires an exact request id"
            )
        if self.expected_rid is not None and not all(capture_values):
            raise ValueError(
                "numerical probe requires capture id and Downward API Pod name/UID"
            )
        self.capture = (
            {"id": capture_id, "pod_name": pod_name, "pod_uid": pod_uid}
            if self.expected_rid is not None
            else None
        )
        self._rejection: Optional[str] = None
        self._sealed = False
        self._seen = False
        self._active_phase: Optional[str] = None
        self._active_rows = 0
        self._records: dict[str, dict[str, dict]] = {}
        self._proposal_rows: dict[str, int] = {}
        self._next_ordinal = 1
        self._required_stages = list(_BASE_REQUIRED_STAGES)
        if require_target_verify_pp_input:
            self._required_stages.insert(1, "target_verify_pp_input")
        self._required_stage_set = frozenset(self._required_stages)

    @property
    def can_probe(self) -> bool:
        return (
            self.expected_rid is not None
            and self._rejection is None
            and not self._sealed
        )

    def matches_rids(self, rids: Optional[list[str]]) -> bool:
        # This diagnostic must not perturb unrelated requests.  A co-batched
        # occurrence is deliberately deferred instead of forcing the whole DP
        # batch eager and then permanently rejecting the one-shot capture.
        return bool(self.can_probe and rids == [self.expected_rid])

    def matches_schedule_batch(self, batch) -> bool:
        return self.matches_rids([req.rid for req in batch.reqs])

    def needs_eager_for_schedule_batch(self, batch) -> bool:
        return self.matches_schedule_batch(batch) and "draft_extend_input" not in (
            self._records.get("decode", {})
        )

    def _reject(self, reason: str) -> None:
        if self._rejection is None:
            self._rejection = reason
            logger.error(
                "EAGLE numerical diagnostic failed closed for rid=%s: %s",
                self.expected_rid,
                reason,
            )

    def _emit_stage_event(
        self,
        *,
        prefix: str,
        phase: str,
        stage: str,
        row_domain: str,
        fingerprints: dict[str, dict],
        error: Optional[BaseException] = None,
    ) -> None:
        payload = {
            "rid": self.expected_rid,
            "capture": self.capture,
            "phase": phase,
            "stage": stage,
            "ordinal": self._next_ordinal,
            "row_domain": row_domain,
            "logical_rows": self._active_rows,
            "fingerprints": fingerprints,
            "rank": _rank_payload(),
        }
        if error is not None:
            payload.update(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        logger.warning(
            "%s %s",
            prefix,
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )

    def _handle_stage_error(
        self,
        *,
        phase: str,
        stage: str,
        row_domain: str,
        error: BaseException,
    ) -> None:
        self._emit_stage_event(
            prefix="EAGLE_NUMERICAL_PROBE_STAGE_ERROR",
            phase=phase,
            stage=stage,
            row_domain=row_domain,
            fingerprints={},
            error=error,
        )
        self._reject(f"failed to synchronize/fingerprint {phase}.{stage}: {error}")

    def _record(
        self,
        stage: str,
        tensors: dict[str, Optional[torch.Tensor]],
        *,
        row_domain: str = _DENSE_ROW_DOMAIN,
    ) -> None:
        phase = self._active_phase
        if not self.can_probe or phase is None:
            return
        bucket = self._records.setdefault(phase, {})
        if stage in bucket:
            self._reject(f"duplicate stage {phase}.{stage}")
            return
        required = _REQUIRED_TENSORS.get(stage)
        if required is None:
            self._reject(f"unexpected stage {phase}.{stage}")
            return
        expected_stage = self._required_stages[len(bucket)]
        if stage != expected_stage:
            self._reject(
                f"out-of-order stage {phase}.{stage}, expected {expected_stage}"
            )
            return
        missing = sorted(name for name in required if tensors.get(name) is None)
        if missing:
            self._reject(f"stage {phase}.{stage} missing tensors {missing}")
            return
        try:
            _synchronize_cuda_tensors(tensors)
        except RuntimeError as exc:
            self._handle_stage_error(
                phase=phase, stage=stage, row_domain=row_domain, error=exc
            )
            raise

        try:
            fingerprints = {
                name: _tensor_fingerprint(tensor, self._active_rows)
                for name, tensor in sorted(tensors.items())
                if tensor is not None
            }
            bucket[stage] = {
                "row_domain": row_domain,
                "logical_rows": self._active_rows,
                "tensors": fingerprints,
            }
            self._emit_stage_event(
                prefix="EAGLE_NUMERICAL_PROBE_STAGE",
                phase=phase,
                stage=stage,
                row_domain=row_domain,
                fingerprints=fingerprints,
            )
            self._next_ordinal += 1
        except (RuntimeError, TypeError, ValueError) as exc:
            self._handle_stage_error(
                phase=phase, stage=stage, row_domain=row_domain, error=exc
            )
            if isinstance(exc, RuntimeError) and any(
                tensor is not None and tensor.is_cuda for tensor in tensors.values()
            ):
                raise

    def _record_explicit(
        self,
        stage: str,
        tensors: dict[str, Optional[torch.Tensor]],
        *,
        logical_rows: int,
        row_domain: str,
    ) -> None:
        previous_phase = self._active_phase
        previous_rows = self._active_rows
        self._active_phase = "decode"
        self._active_rows = logical_rows
        try:
            self._record(stage, tensors, row_domain=row_domain)
        finally:
            self._active_phase = previous_phase
            self._active_rows = previous_rows

    def record_target_verify_input(
        self,
        *,
        rids: Optional[list[str]],
        draft_token: torch.Tensor,
        positions: torch.Tensor,
        retrieve_index: torch.Tensor,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        batch_size: int,
        draft_token_num: int,
    ) -> bool:
        """Start the one-shot capture at the target-verify input tree."""
        if not self.matches_rids(rids):
            return False
        self._seen = True
        if "decode" in self._records:
            return False
        logical_rows = batch_size * draft_token_num
        self._proposal_rows["decode"] = batch_size
        self._record_explicit(
            "target_verify_input",
            {
                "draft_token": draft_token,
                "positions": positions,
                "retrieve_index": retrieve_index.reshape(-1),
                "retrieve_next_token": retrieve_next_token.reshape(-1),
                "retrieve_next_sibling": retrieve_next_sibling.reshape(-1),
            },
            logical_rows=logical_rows,
            row_domain=_TARGET_TREE_ROW_DOMAIN,
        )
        return self.can_probe

    def record_target_verify_output(
        self,
        *,
        logits: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        logical_rows: int,
    ) -> None:
        self._record_explicit(
            "target_verify_output",
            {"logits": logits, "hidden_states": hidden_states},
            logical_rows=logical_rows,
            row_domain=_TARGET_TREE_ROW_DOMAIN,
        )

    def record_target_verify_pp_input(
        self,
        *,
        pp_proxy_tensors: Optional[dict[str, object]],
    ) -> None:
        """Fingerprint the PP-last target input before graph-buffer staging.

        This runs outside CUDA Graph replay.  It observes the real PP proxy
        tensors before ``DecodeCudaGraphRunner.load_batch`` copies them into
        pointer-stable graph buffers, so it does not add request-conditioned
        Python work to the captured graph or alter graph admission.
        """
        proxy = pp_proxy_tensors or {}
        # The transport dictionary also carries non-tensor metadata such as
        # ``__msg_type__``.  Keep that protocol envelope out of the numerical
        # observer and admit only model inputs understood by this stage.
        tensors = {
            name: proxy.get(name)
            for name in ("hidden_states", "residual")
            if isinstance(proxy.get(name), torch.Tensor)
        }
        missing = sorted(
            name
            for name in _REQUIRED_TENSORS["target_verify_pp_input"]
            if name not in tensors
        )
        if missing:
            self._reject(
                f"stage decode.target_verify_pp_input missing tensors {missing}"
            )
            return
        hidden_rows = tensors["hidden_states"].shape[0]
        residual_rows = tensors["residual"].shape[0]
        if hidden_rows != residual_rows:
            self._reject(
                "stage decode.target_verify_pp_input row mismatch "
                f"hidden_states={hidden_rows}, residual={residual_rows}"
            )
            return
        self._record_explicit(
            "target_verify_pp_input",
            tensors,
            logical_rows=hidden_rows,
            row_domain=_PP_RANK_LOCAL_TARGET_TREE_ROW_DOMAIN,
        )

    def record_target_verify_sample(
        self, *, predict: torch.Tensor, logical_rows: int
    ) -> None:
        self._record_explicit(
            "target_verify_sample",
            {"predict": predict},
            logical_rows=logical_rows,
            row_domain=_TARGET_TREE_ROW_DOMAIN,
        )

    def record_target_verify_accept(
        self, *, accept_lens: torch.Tensor, accept_index: torch.Tensor, batch_size: int
    ) -> None:
        self._record_explicit(
            "target_verify_accept",
            {"accept_lens": accept_lens, "accept_index": accept_index},
            logical_rows=batch_size,
            row_domain=_REQUEST_ROW_DOMAIN,
        )

    def record_target_verify_handoff(
        self,
        *,
        predict: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        logical_rows: int,
    ) -> None:
        self._record_explicit(
            "target_verify_handoff",
            {"predict": predict, "hidden_states": hidden_states},
            logical_rows=logical_rows,
            row_domain=_TARGET_TREE_ROW_DOMAIN,
        )

    @contextlib.contextmanager
    def forward_scope(
        self,
        forward_batch,
        *,
        phase: str,
        logical_rows: int,
        using_cuda_graph: bool,
        input_ids: torch.Tensor,
        target_hidden_states: Optional[torch.Tensor],
        positions: torch.Tensor,
    ):
        if not self.matches_rids(forward_batch.rids):
            yield False
            return

        self._seen = True
        if not self.can_probe or "draft_extend_input" in self._records.get(phase, {}):
            yield False
            return
        if phase != "decode":
            self._reject(f"unexpected phase {phase}")
            yield False
            return
        if using_cuda_graph:
            self._reject("CUDA graph capture/replay is unsupported")
            yield False
            return

        previous_callback = forward_batch._eagle_numerical_probe_callback
        previous_phase = forward_batch._eagle_numerical_probe_phase
        self._active_phase = phase
        self._active_rows = logical_rows
        self._proposal_rows[phase] = len(forward_batch.rids)
        forward_batch._eagle_numerical_probe_callback = self._record
        forward_batch._eagle_numerical_probe_phase = phase
        self._record(
            "draft_extend_input",
            {
                "input_ids": input_ids,
                "positions": positions,
                "target_hidden_states": target_hidden_states,
            },
        )
        try:
            yield True
        finally:
            forward_batch._eagle_numerical_probe_callback = previous_callback
            forward_batch._eagle_numerical_probe_phase = previous_phase
            self._active_phase = None
            self._active_rows = 0

    def record_proposal(
        self,
        *,
        phase: str,
        logical_rows: int,
        topk_index: torch.Tensor,
        topk_probability: torch.Tensor,
    ) -> None:
        if (
            not self.can_probe
            or phase not in self._records
            or "proposed_token" in self._records[phase]
        ):
            return
        expected_rows = self._proposal_rows.get(phase)
        if logical_rows != expected_rows:
            self._reject(
                f"phase {phase} proposal has {logical_rows} rows, "
                f"expected {expected_rows}"
            )
            return
        previous_phase = self._active_phase
        previous_rows = self._active_rows
        self._active_phase = phase
        self._active_rows = logical_rows
        try:
            self._record(
                "proposed_token",
                {
                    "topk_index": topk_index,
                    "topk_probability": topk_probability,
                },
                row_domain=_PROPOSAL_ROW_DOMAIN,
            )
        finally:
            self._active_phase = previous_phase
            self._active_rows = previous_rows

    def finish(self, *, rid: str, natural_stop: bool, normal_completion: bool) -> None:
        if rid != self.expected_rid or self._sealed:
            return
        self._sealed = True
        if self._rejection is None and not normal_completion:
            self._rejection = "request did not complete normally"
        if self._rejection is None:
            for phase in ("decode",):
                missing = self._required_stage_set - self._records.get(phase, {}).keys()
                if missing:
                    self._rejection = f"phase {phase} missing stages {sorted(missing)}"
                    break
        payload = {
            "rid": rid,
            "capture": self.capture,
            "natural_stop": natural_stop,
            "normal_completion": normal_completion,
            "status": "rejected" if self._rejection else "complete",
            "rejection": self._rejection,
            "seen": self._seen,
            "phases": self._records,
        }
        rank = _rank_payload()
        if rank is not None:
            payload["rank"] = rank
        logger.warning(
            "EAGLE_NUMERICAL_PROBE_RESULT %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
        self._records.clear()
        self._proposal_rows.clear()


def maybe_record_eagle_numerical_stage(
    forward_batch, stage: str, **tensors: Optional[torch.Tensor]
) -> None:
    callback = getattr(forward_batch, "_eagle_numerical_probe_callback", None)
    if callback is not None:
        callback(stage, tensors)
