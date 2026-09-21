#!/usr/bin/env python3
"""Send an exact Chat Completions request and preserve raw SSE evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--connect-timeout", type=float, default=10)
    parser.add_argument("--read-timeout", type=float, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request_bytes = args.request.read_bytes()
    request_body = json.loads(request_bytes)
    if not isinstance(request_body, dict):
        raise ValueError("request JSON must be an object")

    started_ns = time.perf_counter_ns()
    status = None
    response_headers: dict[str, str] = {}
    chunks: list[dict[str, object]] = []
    transport_error = None
    try:
        with requests.post(
            args.url,
            data=request_bytes,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=(args.connect_timeout, args.read_timeout),
        ) as response:
            status = response.status_code
            response_headers = dict(response.headers)
            for raw_line in response.iter_lines():
                received_ns = time.perf_counter_ns()
                line = raw_line.decode("utf-8", errors="replace")
                event: dict[str, object] = {
                    "elapsed_ms": (received_ns - started_ns) / 1_000_000,
                    "line": line,
                }
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        event["json"] = json.loads(line[6:])
                    except json.JSONDecodeError as exc:
                        event["json_error"] = str(exc)
                chunks.append(event)
    except Exception as exc:  # preserve the exact failed attempt
        transport_error = f"{type(exc).__name__}: {exc}"

    finished_ns = time.perf_counter_ns()
    first_event_ms = None
    first_reasoning_ms = None
    first_content_ms = None
    finish_reasons = []
    usage = None
    response_request_ids = []
    content_parts = []
    reasoning_parts = []
    tool_call_deltas = []
    for event in chunks:
        payload = event.get("json")
        if not isinstance(payload, dict):
            continue
        if first_event_ms is None:
            first_event_ms = event["elapsed_ms"]
        request_id = payload.get("id") or payload.get("request_id")
        if request_id and request_id not in response_request_ids:
            response_request_ids.append(request_id)
        if isinstance(payload.get("usage"), dict):
            usage = payload["usage"]
        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            content = delta.get("content")
            if reasoning and first_reasoning_ms is None:
                first_reasoning_ms = event["elapsed_ms"]
            if reasoning:
                reasoning_parts.append(reasoning)
            if content and first_content_ms is None:
                first_content_ms = event["elapsed_ms"]
            if content:
                content_parts.append(content)
            if delta.get("tool_calls"):
                tool_call_deltas.extend(delta["tool_calls"])
            if choice.get("finish_reason") is not None:
                finish_reasons.append(choice["finish_reason"])

    evidence = {
        "url": args.url,
        "request_path": str(args.request),
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "request": request_body,
        "status_code": status,
        "response_headers": response_headers,
        "chunks": chunks,
        "summary": {
            "first_json_event_ms": first_event_ms,
            "first_reasoning_ms": first_reasoning_ms,
            "first_content_ms": first_content_ms,
            "finish_reasons": finish_reasons,
            "usage": usage,
            "response_request_ids": response_request_ids,
            "content": "".join(content_parts),
            "reasoning_content": "".join(reasoning_parts),
            "tool_call_deltas": tool_call_deltas,
        },
        "elapsed_ms": (finished_ns - started_ns) / 1_000_000,
        "transport_error": transport_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "status_code": status,
        "chunks": len(chunks),
        "elapsed_ms": evidence["elapsed_ms"],
        "transport_error": transport_error,
    }, ensure_ascii=False))
    return 0 if status is not None and 200 <= status < 300 and transport_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
