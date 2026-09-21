#!/usr/bin/env python3
"""Measure exact Chat Completions JSONL requests with request-level SSE evidence."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import requests


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = q * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def request_once(
    url: str,
    request_index: int,
    raw: bytes,
    connect_timeout: float,
    read_timeout: float,
) -> dict:
    started_ns = time.perf_counter_ns()
    status = None
    headers: dict[str, str] = {}
    events: list[dict] = []
    transport_error = None
    try:
        with requests.post(
            url,
            data=raw,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=(connect_timeout, read_timeout),
        ) as response:
            status = response.status_code
            headers = dict(response.headers)
            for raw_line in response.iter_lines():
                elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
                line = raw_line.decode("utf-8", errors="replace")
                event = {"elapsed_ms": elapsed_ms, "line": line}
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        event["json"] = json.loads(line[6:])
                    except json.JSONDecodeError as exc:
                        event["json_error"] = str(exc)
                events.append(event)
    except Exception as exc:
        transport_error = f"{type(exc).__name__}: {exc}"

    first_reasoning_ms = None
    first_content_ms = None
    first_effective_ms = None
    last_token_event_ms = None
    usage = None
    finish_reasons = []
    request_ids = []
    content_parts = []
    reasoning_parts = []
    tool_call_deltas = []
    for event in events:
        payload = event.get("json")
        if not isinstance(payload, dict):
            continue
        request_id = payload.get("id") or payload.get("request_id")
        if request_id and request_id not in request_ids:
            request_ids.append(request_id)
        if isinstance(payload.get("usage"), dict):
            usage = payload["usage"]
        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            content = delta.get("content")
            tool_calls = delta.get("tool_calls") or []
            if reasoning:
                reasoning_parts.append(reasoning)
                if first_reasoning_ms is None:
                    first_reasoning_ms = event["elapsed_ms"]
                if first_effective_ms is None:
                    first_effective_ms = event["elapsed_ms"]
                last_token_event_ms = event["elapsed_ms"]
            if content:
                content_parts.append(content)
                if first_content_ms is None:
                    first_content_ms = event["elapsed_ms"]
                if first_effective_ms is None:
                    first_effective_ms = event["elapsed_ms"]
                last_token_event_ms = event["elapsed_ms"]
            if tool_calls:
                tool_call_deltas.extend(tool_calls)
                if first_effective_ms is None:
                    first_effective_ms = event["elapsed_ms"]
                last_token_event_ms = event["elapsed_ms"]
            if choice.get("finish_reason") is not None:
                finish_reasons.append(choice["finish_reason"])

    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    tpot_ms = None
    if (
        isinstance(completion_tokens, int)
        and completion_tokens >= 2
        and first_effective_ms is not None
        and last_token_event_ms is not None
    ):
        tpot_ms = (last_token_event_ms - first_effective_ms) / (completion_tokens - 1)
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
    success = (
        status is not None
        and 200 <= status < 300
        and transport_error is None
        and bool(finish_reasons)
        and isinstance(usage, dict)
    )
    return {
        "request_index": request_index,
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "status_code": status,
        "response_headers": headers,
        "events": events,
        "elapsed_ms": elapsed_ms,
        "first_effective_ms": first_effective_ms,
        "first_reasoning_ms": first_reasoning_ms,
        "first_content_ms": first_content_ms,
        "last_token_event_ms": last_token_event_ms,
        "tpot_ms": tpot_ms,
        "usage": usage,
        "finish_reasons": finish_reasons,
        "request_ids": request_ids,
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "tool_call_deltas": tool_call_deltas,
        "transport_error": transport_error,
        "success": success,
    }


def summarize(results: list[dict], field: str) -> dict:
    values = [r[field] for r in results if r.get(field) is not None]
    return {
        "count": len(values),
        "avg_ms": statistics.fmean(values) if values else None,
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--connect-timeout", type=float, default=10)
    parser.add_argument("--read-timeout", type=float, default=600)
    args = parser.parse_args()

    raw_requests = [line for line in args.requests.read_bytes().splitlines() if line.strip()]
    if args.limit is not None:
        raw_requests = raw_requests[: args.limit]
    if not raw_requests:
        raise ValueError("request file is empty")
    for raw in raw_requests:
        if not isinstance(json.loads(raw), dict):
            raise ValueError("each JSONL line must be a complete request object")

    wall_started_ns = time.perf_counter_ns()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                request_once,
                args.url,
                index,
                raw,
                args.connect_timeout,
                args.read_timeout,
            )
            for index, raw in enumerate(raw_requests)
        ]
        results = [future.result() for future in futures]
    wall_elapsed_s = (time.perf_counter_ns() - wall_started_ns) / 1_000_000_000
    results.sort(key=lambda item: item["request_index"])
    successes = [result for result in results if result["success"]]
    summary = {
        "url": args.url,
        "request_file": str(args.requests),
        "request_file_sha256": hashlib.sha256(args.requests.read_bytes()).hexdigest(),
        "concurrency": args.concurrency,
        "requests": len(results),
        "successful_requests": len(successes),
        "failed_requests": len(results) - len(successes),
        "wall_elapsed_s": wall_elapsed_s,
        "successful_requests_per_second": len(successes) / wall_elapsed_s,
        "first_effective": summarize(successes, "first_effective_ms"),
        "first_reasoning": summarize(successes, "first_reasoning_ms"),
        "first_content": summarize(successes, "first_content_ms"),
        "tpot": summarize(successes, "tpot_ms"),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False))
    return 0 if len(successes) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
