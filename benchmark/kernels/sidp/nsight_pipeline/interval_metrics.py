"""Pure interval algorithms used by the Nsight Systems parser.

All timestamps are integer nanoseconds.  Intervals are half-open ``[start,end)``
so adjacent GPU activities do not create a fake bubble.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Interval:
    start: int
    end: int
    role: str = "unknown"
    name: str = ""
    device: int = -1
    stream: int = -1
    pid: int = -1

    def __post_init__(self):
        if self.end < self.start:
            raise ValueError(f"invalid interval [{self.start}, {self.end})")

    @property
    def duration(self) -> int:
        return self.end - self.start


def merge_intervals(intervals: Iterable[Interval | tuple[int, int]]):
    points = sorted(
        (item.start, item.end) if isinstance(item, Interval) else item
        for item in intervals
        if ((item.end - item.start) if isinstance(item, Interval) else item[1] - item[0])
        > 0
    )
    merged: list[tuple[int, int]] = []
    for start, end in points:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def interval_total(intervals: Iterable[Interval | tuple[int, int]]) -> int:
    return sum(end - start for start, end in merge_intervals(intervals))


def intersect_intervals(left, right):
    left = merge_intervals(left)
    right = merge_intervals(right)
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def complement_intervals(intervals, window_start: int, window_end: int):
    if window_end < window_start:
        raise ValueError("window end precedes start")
    clipped = [
        (max(start, window_start), min(end, window_end))
        for start, end in merge_intervals(intervals)
        if end > window_start and start < window_end
    ]
    gaps = []
    cursor = window_start
    for start, end in merge_intervals(clipped):
        if cursor < start:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < window_end:
        gaps.append((cursor, window_end))
    return gaps


def percentile(values, q: float):
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def describe_ns(values):
    values = list(values)
    if not values:
        return {
            "count": 0,
            "total_ms": 0.0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "std_ms": 0.0,
            "cv": 0.0,
        }
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    to_ms = 1e-6
    return {
        "count": len(values),
        "total_ms": sum(values) * to_ms,
        "mean_ms": mean * to_ms,
        "median_ms": percentile(values, 0.5) * to_ms,
        "p95_ms": percentile(values, 0.95) * to_ms,
        "min_ms": min(values) * to_ms,
        "max_ms": max(values) * to_ms,
        "std_ms": math.sqrt(variance) * to_ms,
        "cv": math.sqrt(variance) / mean if mean else 0.0,
    }


def bubbles_with_neighbors(intervals: list[Interval], window_start, window_end):
    merged = merge_intervals(intervals)
    gaps = complement_intervals(merged, window_start, window_end)
    result = []
    ordered = sorted(intervals, key=lambda item: (item.start, item.end))
    for start, end in gaps:
        before = max(
            (item for item in ordered if item.end <= start),
            key=lambda item: item.end,
            default=None,
        )
        after = min(
            (item for item in ordered if item.start >= end),
            key=lambda item: item.start,
            default=None,
        )
        result.append(
            {
                "start_ns": start,
                "end_ns": end,
                "duration_ms": (end - start) / 1e6,
                "before": (
                    {"role": before.role, "name": before.name} if before else None
                ),
                "after": {"role": after.role, "name": after.name} if after else None,
            }
        )
    return result

