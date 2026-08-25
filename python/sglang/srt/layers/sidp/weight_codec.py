"""Extension points for compressed SiDP weight transport.

Only the identity codec is implemented today. A future compressed-storage
codec must preserve this fixed memory topology:

1. Every weight retained locally by the rank (including the ``k > 1`` local
   replicas) is stored canonically in compressed form; there is no persistent
   BF16 copy plus compressed transport duplicate. Only the canonical owner's
   encoded tensor is exported to peers through IPC.
2. The two-cycle rolling cache stores compressed prefetched weights. Its size
   is therefore ``cache_cycles * remote_positions`` compressed layers.
3. Each rank owns exactly one layer-sized BF16 materialization buffer (one
   shared set of FFN parameter buffers, not one buffer per cache slot). After a
   compressed cache slot becomes RAW-ready, ``decode_before_compute`` restores
   that layer into the shared BF16 buffers immediately before its serial GEMM.
   A local layer follows the same rule, but decodes directly from its persistent
   compressed owner tensor without a prefetch.

The identity codec is an optimized compatibility path: its "compressed" cache
is the existing compute-dtype cache and decode is a no-op. A non-identity codec
will require the manager to bind both local and remote layers to the single
shared compute buffer; that integration is deliberately not implemented yet.

Non-identity codecs must not allocate memory or synchronize the host from the
runtime decode hook. Their decode kernel must be CUDA Graph capture-safe.
Codec metadata must be pickle-serializable and must not contain CUDA tensors;
auxiliary device data needs an explicit IPC lifecycle.

The setup-time encode assumes inference weights are immutable.  A codec used
with online weight updates must add an explicit refresh/re-encode protocol
before the next prefetch; that lifecycle is intentionally left as future work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch


@dataclass(frozen=True)
class EncodedWeight:
    """Canonical local encoded tensor plus decode metadata."""

    tensor: torch.Tensor
    metadata: Any = None


class SidpWeightCodec(Protocol):
    """Lifecycle contract for an SiDP weight transport representation."""

    name: str

    def encode_for_storage(
        self,
        *,
        layer_id: int,
        param_name: str,
        weight: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> EncodedWeight:
        """Encode one locally retained weight once during setup.

        The returned tensor is canonical local storage. The manager exports it
        through IPC only when this rank is also the layer's canonical owner.
        """
        ...

    def allocate_cycle_buffer(
        self,
        *,
        param_name: str,
        encoded_views: Sequence[torch.Tensor],
        encoded_metadata: Sequence[Any],
        direct_compute_buffer: torch.Tensor,
    ) -> torch.Tensor:
        """Allocate one graph-stable compressed slot in the two-cycle cache.

        ``direct_compute_buffer`` exists only for the identity fast path, where
        the compressed representation and compute representation are identical
        and can alias. A real compressed codec must allocate only its encoded
        representation here; the manager will provide one rank-global BF16
        materialization buffer separately.
        """
        ...

    def decode_before_compute(
        self,
        *,
        layer_id: int,
        param_name: str,
        encoded: torch.Tensor,
        encoded_nbytes: int,
        compute_buffer: torch.Tensor,
        metadata: Any,
        stream: torch.cuda.Stream,
    ) -> None:
        """Decode one RAW-ready weight into the rank-shared BF16 buffer."""
        ...


class IdentityWeightCodec:
    """Current BF16/FP16 path: copy directly into the model weight buffer."""

    name = "identity"

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

    def allocate_cycle_buffer(
        self,
        *,
        param_name: str,
        encoded_views: Sequence[torch.Tensor],
        encoded_metadata: Sequence[Any],
        direct_compute_buffer: torch.Tensor,
    ) -> torch.Tensor:
        del encoded_metadata
        for encoded in encoded_views:
            if (
                encoded.shape != direct_compute_buffer.shape
                or encoded.dtype != direct_compute_buffer.dtype
            ):
                raise ValueError(
                    f"SiDP identity codec layout mismatch for {param_name}: "
                    f"encoded={tuple(encoded.shape)}/{encoded.dtype}, "
                    "compute="
                    f"{tuple(direct_compute_buffer.shape)}/"
                    f"{direct_compute_buffer.dtype}"
                )
        return direct_compute_buffer

    def decode_before_compute(
        self,
        *,
        layer_id: int,
        param_name: str,
        encoded: torch.Tensor,
        encoded_nbytes: int,
        compute_buffer: torch.Tensor,
        metadata: Any,
        stream: torch.cuda.Stream,
    ) -> None:
        del layer_id, param_name, encoded, encoded_nbytes
        del compute_buffer, metadata, stream


def build_weight_codec(transfer_dtype: str) -> SidpWeightCodec:
    """Build the configured codec; compressed implementations are future work."""
    if transfer_dtype in ("same", "identity"):
        return IdentityWeightCodec()
    raise NotImplementedError(
        f"SiDP transfer codec {transfer_dtype!r} is not implemented; "
        "implement SidpWeightCodec and register it in build_weight_codec"
    )
