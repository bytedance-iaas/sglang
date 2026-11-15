"""CPU-testable contract for decode-side target/draft prompt-prefix reuse.

This module deliberately does not alias or copy KV payloads.  It defines the
identity, committed-frontier, transfer-ACK, and lifetime conditions that a
future decode runtime implementation must prove before reusing a paired
target/draft prompt prefix.  Every mismatch fails closed to the existing full
target/draft preparation path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Hashable, Protocol, TypeVar


@dataclass(frozen=True)
class PairedPrefixIdentity:
    """Identity of one verifier-owned prompt frontier.

    ``token_digest`` covers prompt tokens only.  Generated suffixes are never
    part of the reusable frontier in v1.  Target and draft cache/layout versions
    stay separate because their pools have independent storage and invalidation.
    """

    rid: Hashable
    request_generation: int
    owner_epoch: int
    room_id: Hashable
    endpoint_id: Hashable
    token_digest: str
    target_cache_version: int
    draft_cache_version: int
    target_layout_version: str
    draft_layout_version: str

    def __post_init__(self) -> None:
        if self.request_generation < 0 or self.owner_epoch < 0:
            raise ValueError("generation and owner_epoch must be non-negative")
        if not self.token_digest:
            raise ValueError("token_digest must be non-empty")
        if not self.target_layout_version or not self.draft_layout_version:
            raise ValueError("layout versions must be non-empty")


@dataclass(frozen=True)
class PairedPrefixState:
    """Committed target/draft state for one request at one instant."""

    identity: PairedPrefixIdentity
    prompt_len: int
    target_committed_len: int
    draft_committed_len: int
    target_ack_generation: int
    draft_ack_generation: int
    target_ack_complete: bool
    draft_ack_complete: bool
    target_page_generation: int
    draft_page_generation: int

    def __post_init__(self) -> None:
        lengths = (self.prompt_len, self.target_committed_len, self.draft_committed_len)
        if any(length < 0 for length in lengths):
            raise ValueError("prefix lengths must be non-negative")
        if self.target_ack_generation < 0 or self.draft_ack_generation < 0:
            raise ValueError("ACK generations must be non-negative")
        if self.target_page_generation < 0 or self.draft_page_generation < 0:
            raise ValueError("page generations must be non-negative")


class PairedPrefixFallback(str, Enum):
    FEATURE_DISABLED = "feature_disabled"
    UNSUPPORTED_PATH = "unsupported_path"
    IDENTITY_MISMATCH = "identity_mismatch"
    ACK_INCOMPLETE = "ack_incomplete"
    ACK_GENERATION_MISMATCH = "ack_generation_mismatch"
    PAGE_GENERATION_MISMATCH = "page_generation_mismatch"
    EMPTY_FRONTIER = "empty_frontier"


@dataclass(frozen=True)
class PairedPrefixContext:
    enabled: bool
    is_decode: bool
    is_target_draft_pair: bool
    is_prompt_phase: bool


@dataclass(frozen=True)
class PairedPrefixPlan:
    """A reuse plan expressed in request-relative prompt-token positions."""

    reusable_tokens: int
    target_start: int = 0
    draft_start: int = 0
    fallback_reason: PairedPrefixFallback | None = None

    @property
    def accepted(self) -> bool:
        return self.fallback_reason is None


def _fallback(reason: PairedPrefixFallback) -> PairedPrefixPlan:
    return PairedPrefixPlan(reusable_tokens=0, fallback_reason=reason)


def plan_paired_prompt_prefix(
    *,
    source: PairedPrefixState,
    destination: PairedPrefixState,
    context: PairedPrefixContext,
) -> PairedPrefixPlan:
    """Return a shared prompt frontier or fail closed.

    The frontier is capped by both prompt lengths and both pools' committed
    lengths.  It can never include a generated suffix, even when a pool has
    committed more tokens.  v1 requires exact identity, ACK generation, and page
    generation equality; a runtime may rebuild one side and call this planner
    again, but must not partially accept a mismatched state.
    """

    if not context.enabled:
        return _fallback(PairedPrefixFallback.FEATURE_DISABLED)
    if not (
        context.is_decode and context.is_target_draft_pair and context.is_prompt_phase
    ):
        return _fallback(PairedPrefixFallback.UNSUPPORTED_PATH)
    if source.identity != destination.identity:
        return _fallback(PairedPrefixFallback.IDENTITY_MISMATCH)
    if not (
        source.target_ack_complete
        and source.draft_ack_complete
        and destination.target_ack_complete
        and destination.draft_ack_complete
    ):
        return _fallback(PairedPrefixFallback.ACK_INCOMPLETE)
    if (
        source.target_ack_generation != destination.target_ack_generation
        or source.draft_ack_generation != destination.draft_ack_generation
    ):
        return _fallback(PairedPrefixFallback.ACK_GENERATION_MISMATCH)
    if (
        source.target_page_generation != destination.target_page_generation
        or source.draft_page_generation != destination.draft_page_generation
    ):
        return _fallback(PairedPrefixFallback.PAGE_GENERATION_MISMATCH)

    frontier = min(
        source.prompt_len,
        destination.prompt_len,
        source.target_committed_len,
        source.draft_committed_len,
        destination.target_committed_len,
        destination.draft_committed_len,
    )
    if frontier <= 0:
        return _fallback(PairedPrefixFallback.EMPTY_FRONTIER)
    return PairedPrefixPlan(reusable_tokens=frontier)


class CompletionEvent(Protocol):
    def query(self) -> bool: ...


ReuseValue = TypeVar("ReuseValue")


@dataclass
class _Entry(Generic[ReuseValue]):
    value: ReuseValue
    page_generations: tuple[int, int]
    events: list[CompletionEvent] = field(default_factory=list)
    retired: bool = False


class PairedPrefixRegistry(Generic[ReuseValue]):
    """Event-safe ownership for a future paired-prefix sidecar/index plan."""

    def __init__(self) -> None:
        self._entries: dict[PairedPrefixIdentity, _Entry[ReuseValue]] = {}

    def put(
        self,
        identity: PairedPrefixIdentity,
        value: ReuseValue,
        *,
        target_page_generation: int,
        draft_page_generation: int,
    ) -> None:
        if target_page_generation < 0 or draft_page_generation < 0:
            raise ValueError("page generations must be non-negative")
        if identity in self._entries:
            raise ValueError(f"paired prefix identity already exists: {identity}")
        self._entries[identity] = _Entry(
            value=value,
            page_generations=(target_page_generation, draft_page_generation),
        )

    def lookup(
        self,
        identity: PairedPrefixIdentity,
        *,
        target_page_generation: int,
        draft_page_generation: int,
    ) -> ReuseValue | None:
        entry = self._entries.get(identity)
        if (
            entry is None
            or entry.retired
            or entry.page_generations != (target_page_generation, draft_page_generation)
        ):
            return None
        return entry.value

    def record_use(
        self, identity: PairedPrefixIdentity, event: CompletionEvent
    ) -> None:
        entry = self._entries.get(identity)
        if entry is None or entry.retired:
            raise KeyError(f"paired prefix is not active: {identity}")
        entry.events.append(event)

    def release_request(self, rid: Hashable, request_generation: int) -> None:
        for identity, entry in self._entries.items():
            if (
                identity.rid == rid
                and identity.request_generation == request_generation
            ):
                entry.retired = True

    def invalidate_page_generations(
        self, target_page_generation: int, draft_page_generation: int
    ) -> None:
        for entry in self._entries.values():
            if entry.page_generations != (
                target_page_generation,
                draft_page_generation,
            ):
                entry.retired = True

    def collect_retired(self) -> tuple[PairedPrefixIdentity, ...]:
        ready = tuple(
            identity
            for identity, entry in self._entries.items()
            if entry.retired and all(event.query() for event in entry.events)
        )
        for identity in ready:
            del self._entries[identity]
        return ready

    @property
    def active_count(self) -> int:
        return sum(not entry.retired for entry in self._entries.values())

    @property
    def retired_count(self) -> int:
        return sum(entry.retired for entry in self._entries.values())
