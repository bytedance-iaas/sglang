"""Coordinate one image encoder owner across replicated prefill ranks."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import msgspec
import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    disable_symmetric_memory_context,
    restore_symmetric_memory_context,
)
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache

logger = logging.getLogger(__name__)

SpanKey = Tuple[Optional[int], int]
SpanEncoder = Callable[[List[Any]], torch.Tensor | List[torch.Tensor]]

LOCAL_HIT = 0
OWNER_CACHE_BROADCAST = 1
OWNER_ENCODE_BROADCAST = 2


class MmOwnerProtocolError(RuntimeError):
    pass


class ImageSpanRequest(msgspec.Struct, frozen=True):
    hash: Optional[int]
    span_len: int
    item: Any
    duplicates: List[Any] = []


class ImageSpanKey(msgspec.Struct, frozen=True):
    hash: Optional[int]
    span_len: int
    geometry: Tuple[Any, ...]


class RankManifest(msgspec.Struct, frozen=True):
    rank: int
    keys: List[ImageSpanKey]
    cached: List[bool]
    dtype: str
    width: int
    rids: List[str]
    error: Optional[str] = None


class OwnerPlan(msgspec.Struct, frozen=True):
    actions: List[int] = []
    owners: List[int] = []
    error: Optional[str] = None


class RankStatus(msgspec.Struct, frozen=True):
    error: Optional[str] = None


def select_owner_group(parallel) -> Optional[Any]:
    """Return the group whose ranks execute identical image chunks."""
    replication = parallel.tp_size // parallel.attn_dp_size
    if replication <= 1:
        return None
    if parallel.attn_cp_size == 1:
        group = parallel.attn_tp_group
    elif parallel.attn_dp_size == 1 and parallel.attn_cp_size == parallel.tp_size:
        group = parallel.attn_cp_group
    else:
        return None
    return group if group.world_size == replication else None


def has_owner_span_work(
    mm_inputs: Sequence[Any],
    extend_prefix_lens: Sequence[int],
    extend_seq_lens: Sequence[int],
) -> bool:
    for mm_input, prefix_len, extend_len in zip(
        mm_inputs, extend_prefix_lens, extend_seq_lens
    ):
        if mm_input is None or extend_len <= 0:
            continue
        items = [item for item in mm_input.mm_items if item is not None]
        if any(
            item.precomputed_embeddings is not None or len(item.offsets) != 1
            for item in items
        ):
            continue
        chunk_end = prefix_len + extend_len
        if any(
            end >= prefix_len and start < chunk_end
            for item in items
            for start, end in item.offsets
        ):
            return True
    return False


class MmOwnerSession(msgspec.Struct):
    group: Any
    device: torch.device
    dtype: torch.dtype
    width: int
    rids: List[str]
    signature: Any
    phase: str = "prepare"
    in_collective: bool = False

    def resolve(
        self,
        requests: Sequence[ImageSpanRequest],
        cache: MultiModalStaticCache,
        encode: SpanEncoder,
    ) -> Dict[SpanKey, torch.Tensor]:
        saved_context = disable_symmetric_memory_context()
        try:
            return _resolve_owner_features(self, requests, cache, encode)
        finally:
            restore_symmetric_memory_context(saved_context)

    def features_ready(self) -> None:
        self._complete()
        self.phase = "finalize"

    @contextmanager
    def uncaptured(self) -> Iterator[None]:
        self.in_collective = True
        try:
            yield
        finally:
            self.in_collective = False

    @contextmanager
    def fence(self) -> Iterator[None]:
        try:
            yield
        except Exception as exc:
            self._fail(exc)
            raise
        self._complete()

    def _fail(self, exc: BaseException) -> None:
        if self.in_collective or isinstance(exc, MmOwnerProtocolError):
            raise exc
        text = _describe(self, self.phase, exc)
        if self.phase == "prepare":
            _exchange_manifest(self, _manifest(self, [], [], text))
        _exchange_status(self, text, exc)

    def _complete(self) -> None:
        if self.phase == "prepare":
            raise RuntimeError("owner protocol completed without a manifest exchange")
        _drain_and_agree(self, self.phase)


def _manifest(session, keys, cached, error=None) -> RankManifest:
    return RankManifest(
        rank=session.group.rank_in_group,
        keys=keys,
        cached=cached,
        dtype=str(session.dtype),
        width=session.width,
        rids=list(session.rids),
        error=error,
    )


def _exchange_manifest(session: MmOwnerSession, manifest: RankManifest) -> OwnerPlan:
    with session.uncaptured():
        manifests = session.group.all_gather_object(manifest)
        plan = None
        if session.group.rank_in_group == 0:
            try:
                plan = _make_plan(manifests)
            except Exception as exc:
                plan = OwnerPlan(error=_describe(session, "plan", exc))
        plan = session.group.broadcast_object(plan, src=0)
    session.phase = "features"
    if plan.error is not None:
        raise MmOwnerProtocolError(plan.error)
    return plan


def _exchange_status(
    session: MmOwnerSession,
    error: Optional[str],
    cause: Optional[BaseException],
) -> None:
    with session.uncaptured():
        statuses = session.group.all_gather_object(RankStatus(error=error))
    for status in statuses:
        if status.error is not None:
            raise MmOwnerProtocolError(status.error) from cause


def _drain_and_agree(
    session: MmOwnerSession,
    stage: str,
    step: Optional[Callable[[], Any]] = None,
) -> Any:
    result = error = cause = None
    try:
        result = step() if step is not None else None
        if session.device.type == "cuda":
            torch.cuda.current_stream(session.device).synchronize()
    except Exception as exc:
        cause = exc
        error = _describe(session, stage, exc)
    _exchange_status(session, error, cause)
    return result


def _resolve_owner_features(
    session: MmOwnerSession,
    requests: Sequence[ImageSpanRequest],
    cache: MultiModalStaticCache,
    encode: SpanEncoder,
) -> Dict[SpanKey, torch.Tensor]:
    features: Dict[SpanKey, torch.Tensor] = {}
    keys: List[ImageSpanKey] = []
    cached: List[bool] = []
    error = None
    try:
        keys, cached = _pin_local_cache(session, requests, cache, features)
    except Exception as exc:
        error = _describe(session, "manifest", exc)
    plan = _exchange_manifest(session, _manifest(session, keys, cached, error))
    if all(action == LOCAL_HIT for action in plan.actions):
        return features

    buffers = _drain_and_agree(
        session,
        "encode",
        lambda: _prepare_transfers(session, requests, keys, plan, features, encode),
    )
    with session.uncaptured():
        for index, (action, owner) in enumerate(zip(plan.actions, plan.owners)):
            if action != LOCAL_HIT:
                session.group.broadcast(buffers[index], src=owner)

    for index, key in enumerate(keys):
        if plan.actions[index] == LOCAL_HIT:
            continue
        span = buffers[index]
        features[(key.hash, key.span_len)] = span
        cache.set(key.hash, EmbeddingResult(embedding=span))
    return features


def _pin_local_cache(session, requests, cache, features):
    keys, cached = [], []
    for request in requests:
        if request.hash is None:
            raise ValueError(
                f"image span of {request.span_len} tokens has no content hash"
            )
        geometry = session.signature(request.item, request.span_len)
        for duplicate in request.duplicates:
            other = session.signature(duplicate, request.span_len)
            if other != geometry:
                raise ValueError(
                    f"image hash {request.hash} occurs with different geometry"
                )
        key = ImageSpanKey(request.hash, request.span_len, geometry)
        keys.append(key)
        entry = cache.get_single(request.hash)
        span = None if entry is None else entry.embedding
        valid = (
            isinstance(span, torch.Tensor)
            and span.dim() == 2
            and tuple(span.shape) == (request.span_len, session.width)
            and span.dtype == session.dtype
            and span.device == session.device
        )
        if valid:
            features[(request.hash, request.span_len)] = span
        elif entry is not None:
            logger.warning("Discarding incompatible cached image span %s", request.hash)
            cache.free(request.hash, None)
        cached.append(valid)
    return keys, cached


def _make_plan(manifests: List[RankManifest]) -> OwnerPlan:
    for manifest in manifests:
        if manifest.error is not None:
            return OwnerPlan(error=manifest.error)
    lead = manifests[0]
    for manifest in manifests[1:]:
        if (manifest.keys, manifest.dtype, manifest.width, manifest.rids) != (
            lead.keys,
            lead.dtype,
            lead.width,
            lead.rids,
        ):
            return OwnerPlan(
                error=(
                    f"image manifest mismatch between group ranks 0 and {manifest.rank}"
                )
            )
    actions, owners = [], []
    for index, key in enumerate(lead.keys):
        owner = key.hash % len(manifests)
        if all(manifest.cached[index] for manifest in manifests):
            action = LOCAL_HIT
        elif manifests[owner].cached[index]:
            action = OWNER_CACHE_BROADCAST
        else:
            action = OWNER_ENCODE_BROADCAST
        actions.append(action)
        owners.append(owner)
    return OwnerPlan(actions=actions, owners=owners)


def _prepare_transfers(session, requests, keys, plan, features, encode):
    buffers, owned = {}, []
    rank = session.group.rank_in_group
    for index, (action, owner) in enumerate(zip(plan.actions, plan.owners)):
        if action == LOCAL_HIT:
            continue
        if owner != rank:
            buffers[index] = torch.empty(
                (keys[index].span_len, session.width),
                device=session.device,
                dtype=session.dtype,
            )
        elif action == OWNER_CACHE_BROADCAST:
            buffers[index] = features[
                (keys[index].hash, keys[index].span_len)
            ].contiguous()
        else:
            owned.append(index)
    if owned:
        encoded = encode([requests[index].item for index in owned])
        spans = _split_spans(encoded, [keys[index].span_len for index in owned])
        for index, span in zip(owned, spans):
            expected = (keys[index].span_len, session.width)
            span = span.reshape(-1, span.shape[-1])
            if tuple(span.shape) != expected or span.dtype != session.dtype:
                raise ValueError(
                    f"encoded image span has {tuple(span.shape)} {span.dtype}; "
                    f"expected {expected} {session.dtype}"
                )
            buffers[index] = span.to(session.device).contiguous()
    return buffers


def _split_spans(encoded, lengths):
    if isinstance(encoded, list):
        if len(encoded) != len(lengths):
            raise ValueError("encoder result count does not match image count")
        return encoded
    encoded = encoded.reshape(-1, encoded.shape[-1])
    if encoded.shape[0] != sum(lengths):
        raise ValueError("encoder row count does not match image spans")
    return list(torch.split(encoded, lengths, dim=0))


def _describe(session, stage, exc):
    rank = session.group.rank_in_group
    return (
        f"multimodal owner protocol failed during {stage} on group rank {rank} "
        f"(global rank {session.group.ranks[rank]}, rids={session.rids}): "
        f"{type(exc).__name__}: {exc}"
    )
