"""Extension points for compressed SiDP weight storage and compute.

Only the identity codec is implemented today. A future compressed-storage
codec must preserve this fixed memory topology:

1. Every weight retained locally by the rank (including the ``k > 1`` local
   replicas) is stored canonically in compressed form; there is no persistent
   BF16 copy plus compressed transport duplicate. Only the canonical owner's
   encoded main/extra tensors are exported to peers through IPC.
2. An encoded weight consists of one main tensor and zero or more named extra
   tensors, such as scale, zero-point, codebook, or indices. Extra tensors are
   device data, not metadata: canonical owners export them through IPC and the
   two-cycle cache pulls them together with the main tensor.
3. The two-cycle rolling cache stores the main encoded tensor and all extras.
   The prefetch RAW event is recorded only after every component is resident.
4. BF16 HBM materialization is codec-dependent, not mandatory. ``direct``
   codecs expose an already compute-compatible cache tensor; ``materialize``
   codecs decode into at most one rank-shared layer-sized HBM buffer; ``fused``
   codecs let a custom GEMM consume the encoded tensors and dequantize on chip.
   Local layers follow the same compute mode using their persistent encoding.

The identity codec is an optimized ``direct`` compatibility path: its encoded
cache is the existing compute-dtype cache, it has no extras, and decode is a
no-op. Non-identity materialized/fused compute integration is deliberately not
implemented yet.

Non-identity codecs must not allocate memory or synchronize the host from the
runtime decode hook. Their decode kernel must be CUDA Graph capture-safe.
Codec metadata must be pickle-serializable and must not contain CUDA tensors;
all auxiliary device data belongs in ``extra_tensors`` so it receives an
explicit IPC/cache/copy lifecycle.

The setup-time encode assumes inference weights are immutable.  A codec used
with online weight updates must add an explicit refresh/re-encode protocol
before the next prefetch; that lifecycle is intentionally left as future work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

import torch


class WeightComputeMode(str, Enum):
    """How a codec's encoded representation reaches the weight consumer."""

    DIRECT = "direct"
    MATERIALIZE = "materialize"
    FUSED = "fused"


@dataclass(frozen=True)
class MaterializationSpec:
    """Optional rank-shared HBM output required by a materializing codec."""

    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class EncodedWeight:
    """One encoded main tensor, named device extras, and host metadata."""

    tensor: torch.Tensor
    extra_tensors: Mapping[str, torch.Tensor] = field(default_factory=dict)
    metadata: Any = None

    @property
    def nbytes(self) -> int:
        """Total bytes pulled for the main tensor and every device extra."""
        return self.tensor.nbytes + sum(t.nbytes for t in self.extra_tensors.values())


class SidpWeightCodec(Protocol):
    """Lifecycle contract for an SiDP weight transport representation."""

    name: str
    compute_mode: WeightComputeMode

    def encode_for_storage(
        self,
        *,
        layer_id: int,
        param_name: str,
        weight: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> EncodedWeight:
        """Encode one locally retained weight once during setup.

        The returned main/extra tensors are canonical local storage. The
        manager exports them through IPC only when this rank is also the
        layer's canonical owner.
        """
        ...

    def allocate_cycle_buffers(
        self,
        *,
        param_name: str,
        encoded_weights: Sequence[EncodedWeight],
        direct_compute_buffer: torch.Tensor | None,
    ) -> EncodedWeight:
        """Allocate one graph-stable encoded slot, including all extras.

        ``direct_compute_buffer`` exists only for the identity fast path, where
        encoded and compute representations alias. Compressed codecs allocate
        only their encoded main/extra tensors here.
        """
        ...

    def materialization_spec(
        self,
        *,
        param_name: str,
        original_shape: Sequence[int],
        original_dtype: torch.dtype,
    ) -> MaterializationSpec | None:
        """Request optional shared HBM output; return ``None`` otherwise.

        ``MATERIALIZE`` codecs return the buffer they need. ``DIRECT`` and
        ``FUSED`` codecs return ``None`` and must not cause the manager to
        allocate a speculative BF16 copy.
        """
        ...

    def decode_before_compute(
        self,
        *,
        layer_id: int,
        param_name: str,
        encoded: EncodedWeight,
        compute_buffer: torch.Tensor | None,
        metadata: Any,
        stream: torch.cuda.Stream,
    ) -> None:
        """Prepare one RAW-ready encoded weight for its configured consumer.

        A ``MATERIALIZE`` codec decodes into ``compute_buffer``. A ``FUSED``
        codec normally records no standalone decode here; its model/operator
        adapter must feed ``encoded.tensor`` and ``encoded.extra_tensors`` to
        the fused GEMM. Runtime implementations must be CUDA Graph safe.
        """
        ...


class IdentityWeightCodec:
    """Current BF16/FP16 path: copy directly into the model weight buffer."""

    name = "identity"
    compute_mode = WeightComputeMode.DIRECT

    def encode_for_storage(
        self,
        *,
        layer_id: int,
        param_name: str,
        weight: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> EncodedWeight:
        del layer_id, param_name, stream
        return EncodedWeight(tensor=weight)

    def allocate_cycle_buffers(
        self,
        *,
        param_name: str,
        encoded_weights: Sequence[EncodedWeight],
        direct_compute_buffer: torch.Tensor | None,
    ) -> EncodedWeight:
        if direct_compute_buffer is None:
            raise ValueError("SiDP identity codec requires a direct compute buffer")
        for encoded in encoded_weights:
            if encoded.extra_tensors:
                raise ValueError("SiDP identity codec does not accept extra tensors")
            if (
                encoded.tensor.shape != direct_compute_buffer.shape
                or encoded.tensor.dtype != direct_compute_buffer.dtype
            ):
                raise ValueError(
                    f"SiDP identity codec layout mismatch for {param_name}: "
                    f"encoded={tuple(encoded.tensor.shape)}/{encoded.tensor.dtype}, "
                    "compute="
                    f"{tuple(direct_compute_buffer.shape)}/"
                    f"{direct_compute_buffer.dtype}"
                )
        return EncodedWeight(tensor=direct_compute_buffer)

    def materialization_spec(
        self,
        *,
        param_name: str,
        original_shape: Sequence[int],
        original_dtype: torch.dtype,
    ) -> MaterializationSpec | None:
        del param_name, original_shape, original_dtype
        return None

    def decode_before_compute(
        self,
        *,
        layer_id: int,
        param_name: str,
        encoded: EncodedWeight,
        compute_buffer: torch.Tensor | None,
        metadata: Any,
        stream: torch.cuda.Stream,
    ) -> None:
        del layer_id, param_name, encoded
        del compute_buffer, metadata, stream


def build_weight_codec(transfer_dtype: str) -> SidpWeightCodec:
    """Build the configured codec; compressed implementations are future work."""
    if transfer_dtype in ("same", "identity"):
        return IdentityWeightCodec()
    raise NotImplementedError(
        f"SiDP transfer codec {transfer_dtype!r} is not implemented; "
        "implement SidpWeightCodec and register it in build_weight_codec"
    )
