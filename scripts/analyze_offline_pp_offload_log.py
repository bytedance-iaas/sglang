#!/usr/bin/env python3
"""Analyze Offline PP Offload scheduler logs.

The script is intentionally log-format tolerant: it reconstructs rank-local
epoch/wave state from lines like:

  [.. PP0] OfflinePP PREFETCH_START epoch=0 wave=3 ...

It checks the invariants we care about while validating the feature manually:

  - no prefill is inserted during an epoch drain;
  - only one wave decodes at a time per rank;
  - only one wave prefetches at a time per rank;
  - prefetch can overlap the current decode wave;
  - rollback/block/hard-timeout events are visible;
  - PP ranks have roughly comparable epoch/wave progress.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RANK_RE = re.compile(r"\bPP(?P<rank>\d+)\]")
OFFLINE_RE = re.compile(r"OfflinePP\s+(?P<action>[A-Z_]+)\s*(?P<rest>.*)$")
KV_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^\s,]+)")


@dataclass
class Event:
    line_no: int
    rank: int
    action: str
    fields: dict[str, Any]
    raw: str


@dataclass
class RankState:
    events: list[Event] = field(default_factory=list)
    action_counts: Counter[str] = field(default_factory=Counter)
    limited_by: Counter[str] = field(default_factory=Counter)
    fill_stop_reasons: Counter[str] = field(default_factory=Counter)
    wave_budget_reqs: list[int] = field(default_factory=list)
    wave_budget_tokens: list[int] = field(default_factory=list)
    wave_admit_host_bytes: list[int] = field(default_factory=list)
    max_total_host_bytes: int = 0
    prefill_lines: list[int] = field(default_factory=list)
    prefill_line_epochs: list[tuple[int, int | None]] = field(default_factory=list)


@dataclass
class Finding:
    level: str
    message: str
    line_no: int | None = None


def _parse_value(value: str) -> Any:
    if value == "None":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_fields(rest: str) -> dict[str, Any]:
    return {m.group("key"): _parse_value(m.group("value")) for m in KV_RE.finditer(rest)}


def parse_log(path: Path) -> tuple[dict[int, RankState], list[str]]:
    ranks: dict[int, RankState] = defaultdict(RankState)
    global_lines: list[str] = []
    current_epoch: dict[int, int | None] = defaultdict(lambda: None)

    for line_no, raw_line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        line = raw_line.rstrip("\n")
        global_lines.append(line)
        rank_match = RANK_RE.search(line)
        if rank_match is None:
            continue
        rank = int(rank_match.group("rank"))
        state = ranks[rank]

        if "Prefill batch" in line:
            state.prefill_lines.append(line_no)
            state.prefill_line_epochs.append((line_no, current_epoch[rank]))

        offline_match = OFFLINE_RE.search(line)
        if offline_match is None:
            continue

        action = offline_match.group("action")
        fields = _parse_fields(offline_match.group("rest"))
        event = Event(line_no=line_no, rank=rank, action=action, fields=fields, raw=line)
        state.events.append(event)
        state.action_counts[action] += 1
        if isinstance(fields.get("epoch"), int):
            current_epoch[rank] = fields["epoch"]

        limited_by = fields.get("limited_by")
        if limited_by is not None:
            state.limited_by[str(limited_by)] += 1
        reason = fields.get("reason")
        if action == "EPOCH_FILL_STOP" and reason is not None:
            state.fill_stop_reasons[str(reason)] += 1
        if action == "WAVE_BUDGET":
            reqs = fields.get("prefill_max_requests")
            tokens = fields.get("max_prefill_tokens")
            if isinstance(reqs, int):
                state.wave_budget_reqs.append(reqs)
            if isinstance(tokens, int):
                state.wave_budget_tokens.append(tokens)
        if action == "WAVE_ADMIT":
            exact = fields.get("host_bytes_exact")
            total = fields.get("total_host_bytes")
            if isinstance(exact, int):
                state.wave_admit_host_bytes.append(exact)
            if isinstance(total, int):
                state.max_total_host_bytes = max(state.max_total_host_bytes, total)

    return dict(ranks), global_lines


def _event_key(event: Event) -> tuple[int | None, int | None]:
    epoch = event.fields.get("epoch")
    wave = event.fields.get("wave")
    return (
        epoch if isinstance(epoch, int) else None,
        wave if isinstance(wave, int) else None,
    )


def check_rank(rank: int, state: RankState) -> list[Finding]:
    findings: list[Finding] = []
    draining = False
    draining_epoch: int | None = None
    active_decode: tuple[int | None, int | None] | None = None
    active_prefetch: tuple[int | None, int | None] | None = None
    decode_start_line: int | None = None
    decode_overlap_seen: dict[tuple[int | None, int | None], bool] = {}

    drain_intervals: list[tuple[int, int | None, int | None]] = []
    open_drain: tuple[int, int | None] | None = None

    for event in state.events:
        key = _event_key(event)
        if event.action == "EPOCH_DRAIN_START":
            draining = True
            draining_epoch = event.fields.get("epoch")
            open_drain = (event.line_no, draining_epoch)
        elif event.action == "EPOCH_DRAIN_DONE":
            if open_drain is not None:
                drain_intervals.append((open_drain[0], event.line_no, open_drain[1]))
                open_drain = None
            draining = False
            draining_epoch = None

        is_prefill_admit = (
            event.action == "WAVE_ADMIT" and event.fields.get("state") == "prefilling"
        )
        if (
            draining
            and (event.action == "OFFLOAD_ENQUEUE" or is_prefill_admit)
            and event.fields.get("epoch") == draining_epoch
        ):
            findings.append(
                Finding(
                    "WARN",
                    f"PP{rank}: {event.action} for wave {key} appeared after EPOCH_DRAIN_START.",
                    event.line_no,
                )
            )

        if event.action == "DECODE_START":
            if active_decode is not None:
                findings.append(
                    Finding(
                        "FAIL",
                        f"PP{rank}: DECODE_START {key} before DECODE_RETIRE {active_decode}.",
                        event.line_no,
                    )
                )
            active_decode = key
            decode_start_line = event.line_no
            decode_overlap_seen.setdefault(key, False)
        elif event.action == "DECODE_RETIRE":
            if active_decode is None:
                findings.append(
                    Finding("WARN", f"PP{rank}: DECODE_RETIRE {key} without active decode.", event.line_no)
                )
            elif key != active_decode:
                findings.append(
                    Finding(
                        "FAIL",
                        f"PP{rank}: DECODE_RETIRE {key} does not match active decode {active_decode}.",
                        event.line_no,
                    )
                )
            active_decode = None
            decode_start_line = None

        if event.action == "PREFETCH_START":
            if active_prefetch is not None:
                findings.append(
                    Finding(
                        "FAIL",
                        f"PP{rank}: PREFETCH_START {key} before previous prefetch {active_prefetch} reached ready/rollback.",
                        event.line_no,
                    )
                )
            active_prefetch = key
            if active_decode is not None and decode_start_line is not None:
                decode_overlap_seen[active_decode] = True
        elif event.action in ("DECODE_READY", "ROLLBACK", "PREFETCH_HARD_TIMEOUT"):
            if active_prefetch == key or event.action in ("ROLLBACK", "PREFETCH_HARD_TIMEOUT"):
                active_prefetch = None

    if active_decode is not None:
        findings.append(Finding("WARN", f"PP{rank}: log ends with active decode {active_decode}."))
    if active_prefetch is not None:
        findings.append(Finding("WARN", f"PP{rank}: log ends with active prefetch {active_prefetch}."))

    if open_drain is not None:
        drain_intervals.append((open_drain[0], None, open_drain[1]))

    for line_no, epoch in state.prefill_line_epochs:
        for start, end, drain_epoch in drain_intervals:
            if line_no <= start:
                continue
            if end is not None and line_no >= end:
                continue
            findings.append(
                Finding(
                    "WARN",
                    f"PP{rank}: Prefill batch appeared while epoch {drain_epoch} is draining.",
                    line_no,
                )
            )
            break

    decode_count = state.action_counts["DECODE_START"]
    overlap_count = sum(1 for seen in decode_overlap_seen.values() if seen)
    if decode_count > 1 and overlap_count == 0:
        findings.append(
            Finding(
                "WARN",
                f"PP{rank}: no prefetch-over-decode overlap observed across {decode_count} decode waves.",
            )
        )
    return findings


def check_cross_rank(ranks: dict[int, RankState], strict: bool) -> list[Finding]:
    findings: list[Finding] = []
    if len(ranks) <= 1:
        return findings

    rank_ids = sorted(ranks)
    baseline = ranks[rank_ids[0]]
    keys = [
        "EPOCH_START",
        "EPOCH_DRAIN_START",
        "EPOCH_DRAIN_DONE",
        "OFFLOAD_ENQUEUE",
        "PREFETCH_START",
        "DECODE_START",
        "DECODE_RETIRE",
        "ROLLBACK",
        "PREFETCH_HARD_TIMEOUT",
    ]
    for rank in rank_ids[1:]:
        state = ranks[rank]
        for key in keys:
            left = baseline.action_counts[key]
            right = state.action_counts[key]
            if left != right:
                level = "FAIL" if strict else "WARN"
                findings.append(
                    Finding(
                        level,
                        f"Cross-rank count mismatch for {key}: PP{rank_ids[0]}={left}, PP{rank}={right}.",
                    )
                )

    for rank in rank_ids[1:]:
        base_decode = [
            _event_key(e) for e in baseline.events if e.action == "DECODE_START"
        ]
        rank_decode = [
            _event_key(e) for e in ranks[rank].events if e.action == "DECODE_START"
        ]
        if base_decode != rank_decode:
            level = "FAIL" if strict else "WARN"
            findings.append(
                Finding(
                    level,
                    f"Decode wave order differs between PP{rank_ids[0]} and PP{rank}.",
                )
            )
    return findings


def summarize(ranks: dict[int, RankState], findings: list[Finding]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ranks": {},
        "findings": [
            {"level": f.level, "line_no": f.line_no, "message": f.message}
            for f in findings
        ],
    }
    for rank, state in sorted(ranks.items()):
        reqs = state.wave_budget_reqs
        tokens = state.wave_budget_tokens
        host = state.wave_admit_host_bytes
        summary["ranks"][f"PP{rank}"] = {
            "actions": dict(state.action_counts),
            "limited_by": dict(state.limited_by),
            "fill_stop_reasons": dict(state.fill_stop_reasons),
            "wave_budget_prefill_max_requests_min": min(reqs) if reqs else None,
            "wave_budget_prefill_max_requests_max": max(reqs) if reqs else None,
            "wave_budget_max_prefill_tokens_min": min(tokens) if tokens else None,
            "wave_budget_max_prefill_tokens_max": max(tokens) if tokens else None,
            "wave_admit_host_bytes_min": min(host) if host else None,
            "wave_admit_host_bytes_max": max(host) if host else None,
            "max_total_host_bytes": state.max_total_host_bytes,
        }
    return summary


def print_text(summary: dict[str, Any]) -> None:
    findings = summary["findings"]
    fail_count = sum(1 for item in findings if item["level"] == "FAIL")
    warn_count = sum(1 for item in findings if item["level"] == "WARN")
    status = "PASS" if fail_count == 0 else "FAIL"
    if fail_count == 0 and warn_count:
        status = "PASS_WITH_WARNINGS"

    print(f"Offline PP log analysis: {status}")
    print(f"Findings: {fail_count} fail, {warn_count} warn")
    print()

    for rank, info in summary["ranks"].items():
        actions = info["actions"]
        print(f"{rank}:")
        print(
            "  waves: offload={offload} prefetch={prefetch} decode_start={decode} "
            "decode_retire={retire} rollback={rollback} blocked={blocked}".format(
                offload=actions.get("OFFLOAD_ENQUEUE", 0),
                prefetch=actions.get("PREFETCH_START", 0),
                decode=actions.get("DECODE_START", 0),
                retire=actions.get("DECODE_RETIRE", 0),
                rollback=actions.get("ROLLBACK", 0),
                blocked=actions.get("PREFETCH_BLOCKED", 0),
            )
        )
        print(
            "  epochs: start={start} drain_start={drain_start} drain_done={drain_done}".format(
                start=actions.get("EPOCH_START", 0),
                drain_start=actions.get("EPOCH_DRAIN_START", 0),
                drain_done=actions.get("EPOCH_DRAIN_DONE", 0),
            )
        )
        print(f"  limited_by: {info['limited_by'] or {}}")
        print(f"  fill_stop_reasons: {info['fill_stop_reasons'] or {}}")
        print(
            "  wave_budget: prefill_max_requests=[{req_min},{req_max}] "
            "max_prefill_tokens=[{tok_min},{tok_max}]".format(
                req_min=info["wave_budget_prefill_max_requests_min"],
                req_max=info["wave_budget_prefill_max_requests_max"],
                tok_min=info["wave_budget_max_prefill_tokens_min"],
                tok_max=info["wave_budget_max_prefill_tokens_max"],
            )
        )
        print(
            "  host_bytes: wave_exact=[{host_min},{host_max}] max_total={host_total}".format(
                host_min=info["wave_admit_host_bytes_min"],
                host_max=info["wave_admit_host_bytes_max"],
                host_total=info["max_total_host_bytes"],
            )
        )
        print()

    if findings:
        print("Findings:")
        for item in findings:
            line = f":{item['line_no']}" if item["line_no"] is not None else ""
            print(f"  [{item['level']}]{line} {item['message']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Path to server log.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text.",
    )
    parser.add_argument(
        "--strict-pp-consistency",
        action="store_true",
        help="Treat cross-rank count/order mismatches as failures instead of warnings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranks, _lines = parse_log(args.log)
    findings: list[Finding] = []
    for rank, state in sorted(ranks.items()):
        findings.extend(check_rank(rank, state))
    findings.extend(check_cross_rank(ranks, strict=args.strict_pp_consistency))
    summary = summarize(ranks, findings)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_text(summary)


if __name__ == "__main__":
    main()
