"""CPU-testable contracts for a future DoSA context-parallel prefill path.

This module deliberately does not dispatch a DoSA kernel.  It defines the
request/segment layout, the PAGED output mapping, and the carry-state lifetime
rules that a kernel implementation must consume.  Keeping these contracts
independent from CUDA lets us validate multi-request CP layouts and stale-state
handling before enabling the optimized path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Hashable, Protocol, Sequence, TypeVar


@dataclass(frozen=True)
class DosaCPSegment:
    """One request segment owned by one attention-CP rank.

    ``absolute_positions`` and ``visible_ends`` are expressed in the request's
    logical token domain, including any cached prefix.  ``output_rows`` are in
    the packed extend-token output domain shared by all requests in the batch.
    Local rows follow the runtime's current CP layout: all ``prev`` segments in
    batch order, followed by all ``next`` segments in batch order.
    """

    request_generation: int
    rid: Hashable
    batch_idx: int
    cp_rank: int
    segment_kind: str
    segment_index: int
    local_row_start: int
    local_row_end: int
    absolute_positions: tuple[int, ...]
    visible_ends: tuple[int, ...]
    prefix_len: int
    cache_version: int
    query_layout_version: str
    page_table_row: int
    output_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.segment_kind not in ("prev", "next"):
            raise ValueError(f"invalid segment kind: {self.segment_kind!r}")
        lengths = {
            self.local_row_end - self.local_row_start,
            len(self.absolute_positions),
            len(self.visible_ends),
            len(self.output_rows),
        }
        if len(lengths) != 1 or next(iter(lengths)) < 0:
            raise ValueError("segment row domains have inconsistent lengths")
        if self.local_row_start < 0 or self.local_row_end < self.local_row_start:
            raise ValueError("invalid local row slice")
        if any(
            end != position + 1
            for position, end in zip(self.absolute_positions, self.visible_ends)
        ):
            raise ValueError("visible_end must equal absolute_position + 1")


@dataclass(frozen=True)
class DosaCPLayout:
    """The complete local DoSA query layout for one CP rank."""

    cp_rank: int
    cp_size: int
    query_layout_version: str
    extend_seq_lens: tuple[int, ...]
    prefix_lens: tuple[int, ...]
    segments: tuple[DosaCPSegment, ...]
    local_row_count: int

    def __post_init__(self) -> None:
        if self.cp_size <= 0 or not 0 <= self.cp_rank < self.cp_size:
            raise ValueError("invalid CP rank or size")
        cursor = 0
        for segment in self.segments:
            if segment.cp_rank != self.cp_rank:
                raise ValueError("layout contains a segment for another CP rank")
            if segment.query_layout_version != self.query_layout_version:
                raise ValueError("segment layout version mismatch")
            if segment.local_row_start != cursor:
                raise ValueError("local rows are not contiguous")
            cursor = segment.local_row_end
        if cursor != self.local_row_count:
            raise ValueError("local row count does not match segment slices")


def _split_sizes(length: int, pieces: int) -> tuple[int, ...]:
    if length < 0:
        raise ValueError("sequence lengths must be non-negative")
    base, remainder = divmod(length, pieces)
    return tuple(base + (index < remainder) for index in range(pieces))


def build_dosa_cp_layout(
    *,
    cp_rank: int,
    cp_size: int,
    request_generations: Sequence[int],
    rids: Sequence[Hashable],
    seq_lens: Sequence[int],
    extend_seq_lens: Sequence[int],
    query_layout_version: str,
    cache_versions: Sequence[int] | None = None,
    page_table_rows: Sequence[int] | None = None,
) -> DosaCPLayout:
    """Build the explicit zigzag CP query contract for one rank.

    Every request is split into ``2 * cp_size`` blocks.  Rank ``r`` owns block
    ``r`` and block ``2 * cp_size - 1 - r``.  Cached-prefix offsets are added to
    the owned extend-token positions rather than inferred from rank-local row
    numbers.
    """

    if cp_size <= 0 or not 0 <= cp_rank < cp_size:
        raise ValueError("invalid CP rank or size")
    if not query_layout_version:
        raise ValueError("query_layout_version must be non-empty")

    batch_size = len(extend_seq_lens)
    values = (request_generations, rids, seq_lens)
    if any(len(value) != batch_size for value in values):
        raise ValueError("request metadata lengths must match extend_seq_lens")
    if cache_versions is None:
        cache_versions = (0,) * batch_size
    if page_table_rows is None:
        page_table_rows = tuple(range(batch_size))
    if len(cache_versions) != batch_size or len(page_table_rows) != batch_size:
        raise ValueError("cache/page-table metadata lengths must match batch size")

    pieces = 2 * cp_size
    blocks: list[tuple[int, ...]] = []
    prefixes: list[int] = []
    output_bases: list[int] = []
    output_cursor = 0
    for seq_len, extend_len in zip(seq_lens, extend_seq_lens):
        seq_len = int(seq_len)
        extend_len = int(extend_len)
        if extend_len < 0 or seq_len < extend_len:
            raise ValueError("expected 0 <= extend_seq_len <= seq_len")
        blocks.append(_split_sizes(extend_len, pieces))
        prefixes.append(seq_len - extend_len)
        output_bases.append(output_cursor)
        output_cursor += extend_len

    descriptors: list[tuple[str, int, int]] = []
    descriptors.extend(("prev", batch_idx, cp_rank) for batch_idx in range(batch_size))
    descriptors.extend(
        ("next", batch_idx, pieces - 1 - cp_rank) for batch_idx in range(batch_size)
    )

    segments: list[DosaCPSegment] = []
    local_cursor = 0
    for segment_kind, batch_idx, segment_index in descriptors:
        block_sizes = blocks[batch_idx]
        relative_start = sum(block_sizes[:segment_index])
        length = block_sizes[segment_index]
        prefix_len = prefixes[batch_idx]
        absolute_positions = tuple(
            range(prefix_len + relative_start, prefix_len + relative_start + length)
        )
        output_rows = tuple(
            range(
                output_bases[batch_idx] + relative_start,
                output_bases[batch_idx] + relative_start + length,
            )
        )
        segment = DosaCPSegment(
            request_generation=int(request_generations[batch_idx]),
            rid=rids[batch_idx],
            batch_idx=batch_idx,
            cp_rank=cp_rank,
            segment_kind=segment_kind,
            segment_index=segment_index,
            local_row_start=local_cursor,
            local_row_end=local_cursor + length,
            absolute_positions=absolute_positions,
            visible_ends=tuple(position + 1 for position in absolute_positions),
            prefix_len=prefix_len,
            cache_version=int(cache_versions[batch_idx]),
            query_layout_version=query_layout_version,
            page_table_row=int(page_table_rows[batch_idx]),
            output_rows=output_rows,
        )
        segments.append(segment)
        local_cursor += length

    return DosaCPLayout(
        cp_rank=cp_rank,
        cp_size=cp_size,
        query_layout_version=query_layout_version,
        extend_seq_lens=tuple(int(value) for value in extend_seq_lens),
        prefix_lens=tuple(prefixes),
        segments=tuple(segments),
        local_row_count=local_cursor,
    )


def validate_dosa_cp_layouts(layouts: Sequence[DosaCPLayout]) -> None:
    """Validate exact query coverage across every rank of one CP group."""

    if not layouts:
        raise ValueError("at least one layout is required")
    cp_size = layouts[0].cp_size
    if len(layouts) != cp_size:
        raise ValueError("one layout per CP rank is required")
    if {layout.cp_rank for layout in layouts} != set(range(cp_size)):
        raise ValueError("CP ranks must be unique and complete")

    reference = layouts[0]
    if any(
        layout.cp_size != cp_size
        or layout.query_layout_version != reference.query_layout_version
        or layout.extend_seq_lens != reference.extend_seq_lens
        or layout.prefix_lens != reference.prefix_lens
        for layout in layouts
    ):
        raise ValueError("layouts do not describe the same batch")

    output_base = 0
    for batch_idx, extend_len in enumerate(reference.extend_seq_lens):
        expected_rows = list(range(output_base, output_base + extend_len))
        segments = [
            segment
            for layout in layouts
            for segment in layout.segments
            if segment.batch_idx == batch_idx
        ]
        actual_rows = [row for segment in segments for row in segment.output_rows]
        if sorted(actual_rows) != expected_rows or len(actual_rows) != len(
            set(actual_rows)
        ):
            raise ValueError(f"request {batch_idx} query coverage is not exact")
        for segment in segments:
            expected_positions = tuple(
                reference.prefix_lens[batch_idx] + row - output_base
                for row in segment.output_rows
            )
            if segment.absolute_positions != expected_positions:
                raise ValueError(f"request {batch_idx} absolute positions are invalid")
        output_base += extend_len


def map_logical_topk_to_physical(
    logical_indices: Sequence[Sequence[int]],
    *,
    page_table: Sequence[Sequence[int]],
    row_batch: Sequence[int],
    page_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Apply the runtime's PAGED TopK output transform without CUDA.

    A non-negative request-relative token position ``p`` becomes
    ``page_table[batch][p // page_size] * page_size + p % page_size``.  ``-1``
    remains padding.  This is the same contract used by
    ``topk_page_table_transform_`` and the fused v2 paged transform.
    """

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if len(logical_indices) != len(row_batch):
        raise ValueError("row_batch must contain one entry per TopK row")
    result: list[tuple[int, ...]] = []
    for indices, batch_idx in zip(logical_indices, row_batch):
        if not 0 <= batch_idx < len(page_table):
            raise ValueError(f"invalid page-table row: {batch_idx}")
        table = page_table[batch_idx]
        physical: list[int] = []
        for position in indices:
            if position == -1:
                physical.append(-1)
                continue
            if position < -1:
                raise ValueError(f"invalid logical TopK position: {position}")
            page_index, page_offset = divmod(position, page_size)
            if page_index >= len(table):
                raise ValueError(
                    f"logical position {position} exceeds page-table row {batch_idx}"
                )
            physical_page = int(table[page_index])
            if physical_page < 0:
                raise ValueError("page-table entries must be non-negative")
            physical.append(physical_page * page_size + page_offset)
        result.append(tuple(physical))
    return tuple(result)


@dataclass(frozen=True)
class DosaCarryKey:
    """Identity required before a DoSA carry value may be reused."""

    rid: Hashable
    request_generation: int
    layer_id: int
    cp_rank: int
    query_layout_version: str
    cache_version: int


class DosaCompletionEvent(Protocol):
    def query(self) -> bool: ...


CarryValue = TypeVar("CarryValue")


@dataclass
class _CarryEntry(Generic[CarryValue]):
    value: CarryValue
    events: list[DosaCompletionEvent] = field(default_factory=list)
    retired: bool = False


class DosaCarryRegistry(Generic[CarryValue]):
    """Fail-closed carry storage with deferred event-safe retirement.

    Entries become unavailable to readers as soon as they are retired, but are
    retained until every recorded completion event reports ready.  This models
    the calling-stream and DoSA-side-stream fences required before request/page
    state can be reused.
    """

    def __init__(self) -> None:
        self._entries: dict[DosaCarryKey, _CarryEntry[CarryValue]] = {}

    def put(self, key: DosaCarryKey, value: CarryValue) -> None:
        entry = self._entries.get(key)
        if entry is not None and not entry.retired:
            raise ValueError(f"active carry already exists for {key}")
        if entry is not None:
            raise ValueError(f"retired carry is still awaiting completion for {key}")
        self._entries[key] = _CarryEntry(value=value)

    def lookup(self, key: DosaCarryKey) -> CarryValue | None:
        entry = self._entries.get(key)
        return None if entry is None or entry.retired else entry.value

    def record_use(self, key: DosaCarryKey, event: DosaCompletionEvent) -> None:
        entry = self._entries.get(key)
        if entry is None or entry.retired:
            raise KeyError(f"carry is not active: {key}")
        entry.events.append(event)

    def release_request(self, rid: Hashable, request_generation: int) -> None:
        for key, entry in self._entries.items():
            if key.rid == rid and key.request_generation == request_generation:
                entry.retired = True

    def invalidate_cache_version(self, current_cache_version: int) -> None:
        for key, entry in self._entries.items():
            if key.cache_version != current_cache_version:
                entry.retired = True

    def collect_retired(self) -> tuple[DosaCarryKey, ...]:
        ready = tuple(
            key
            for key, entry in self._entries.items()
            if entry.retired and all(event.query() for event in entry.events)
        )
        for key in ready:
            del self._entries[key]
        return ready

    @property
    def active_count(self) -> int:
        return sum(not entry.retired for entry in self._entries.values())

    @property
    def retired_count(self) -> int:
        return sum(entry.retired for entry in self._entries.values())
