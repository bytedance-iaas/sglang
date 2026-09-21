#!/usr/bin/env python3
"""Verify shared-prefix cache hits at bounded delays with raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests


def send(url: str, request_path: Path, connect_timeout: float, read_timeout: float) -> dict:
    raw = request_path.read_bytes()
    started_ns = time.perf_counter_ns()
    status = None
    headers: dict[str, str] = {}
    events: list[dict] = []
    error = None
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
        error = f"{type(exc).__name__}: {exc}"

    usage = None
    for event in events:
        payload = event.get("json")
        if isinstance(payload, dict) and isinstance(payload.get("usage"), dict):
            usage = payload["usage"]
    return {
        "request_path": str(request_path),
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "status_code": status,
        "headers": headers,
        "events": events,
        "usage": usage,
        "elapsed_ms": (time.perf_counter_ns() - started_ns) / 1_000_000,
        "transport_error": error,
    }


def cache_counts(usage: dict | None) -> tuple[int | None, int | None, str | None]:
    if not isinstance(usage, dict):
        return None, None, None
    prompt_tokens = usage.get("prompt_tokens")
    if usage.get("cached_tokens") is not None:
        return prompt_tokens, usage["cached_tokens"], "usage.cached_tokens"
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        return prompt_tokens, details["cached_tokens"], "usage.prompt_tokens_details.cached_tokens"
    return prompt_tokens, None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--delay-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-ratio", type=float, default=0.8)
    parser.add_argument("--connect-timeout", type=float, default=10)
    parser.add_argument("--read-timeout", type=float, default=600)
    args = parser.parse_args()

    warm = send(args.url, args.warm, args.connect_timeout, args.read_timeout)
    if warm["status_code"] is not None and 200 <= warm["status_code"] < 300:
        time.sleep(args.delay_seconds)
        probe = send(args.url, args.probe, args.connect_timeout, args.read_timeout)
    else:
        probe = {"skipped": "warm request failed"}

    prompt_tokens, cached_tokens, source = cache_counts(probe.get("usage"))
    ratio = (
        cached_tokens / prompt_tokens
        if isinstance(cached_tokens, int) and isinstance(prompt_tokens, int) and prompt_tokens
        else None
    )
    passed = (
        warm.get("status_code") is not None
        and 200 <= warm["status_code"] < 300
        and probe.get("status_code") is not None
        and 200 <= probe["status_code"] < 300
        and ratio is not None
        and ratio >= args.minimum_ratio
    )
    result = {
        "delay_seconds": args.delay_seconds,
        "minimum_ratio": args.minimum_ratio,
        "warm": warm,
        "probe": probe,
        "cached_tokens_source": source,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cached_ratio": ratio,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "delay_seconds": args.delay_seconds,
        "cached_tokens_source": source,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cached_ratio": ratio,
        "passed": passed,
    }, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
