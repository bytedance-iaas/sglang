#!/usr/bin/env python3
"""Benchmark DeepSeek-V4 long-context agent cache scenarios.

The script uses SGLang's native /generate endpoint with synthetic input_ids so
token counts are exact and no external dataset download is needed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from typing import Any

import requests


PREFILL_RE = re.compile(
    r"#cached-token: (?P<cached>\d+), full token usage: (?P<full>[0-9.]+), "
    r"swa token usage: (?P<swa>[0-9.]+).*#hit_rate\(total/gpu/eic\): "
    r"(?P<total_hit>[0-9.]+)/(?P<gpu_hit>[0-9.]+)/(?P<eic_hit>[0-9.]+)"
)


def make_input_ids(length: int, seed: int) -> list[int]:
    base = 1000 + (seed % 997)
    return [base + ((i * 131 + seed) % 20000) for i in range(length)]


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int) -> Any:
    resp = requests.post(f"{base_url}{path}", json=payload, timeout=timeout)
    resp.raise_for_status()
    if resp.content:
        return resp.json()
    return None


def get(base_url: str, path: str, timeout: int = 30) -> requests.Response:
    resp = requests.get(f"{base_url}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp


def flush_cache(base_url: str) -> None:
    resp = requests.post(f"{base_url}/flush_cache", timeout=300)
    if resp.status_code >= 400:
        resp = requests.get(f"{base_url}/flush_cache", timeout=300)
    resp.raise_for_status()


def log_offset(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def read_new_log(path: str, offset: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            return f.read()
    except OSError:
        return ""


def extract_prefill_stats(text: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if "Prefill batch" in line]
    parsed: list[dict[str, Any]] = []
    for line in lines:
        match = PREFILL_RE.search(line)
        if not match:
            continue
        item: dict[str, Any] = {
            "cached_tokens_log": int(match.group("cached")),
            "full_token_usage": float(match.group("full")),
            "swa_token_usage": float(match.group("swa")),
            "hit_rate_total": float(match.group("total_hit")),
            "hit_rate_gpu": float(match.group("gpu_hit")),
            "hit_rate_eic": float(match.group("eic_hit")),
            "line": line,
        }
        parsed.append(item)
    if not parsed:
        return {"prefill_lines": lines[-8:]}
    latest = parsed[-1]
    latest["prefill_lines"] = lines[-8:]
    return latest


def stream_generate(
    base_url: str,
    input_ids: list[int],
    output_len: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "input_ids": input_ids,
        "stream": True,
        "sampling_params": {
            "max_new_tokens": output_len,
            "temperature": 0,
            "ignore_eos": True,
        },
    }
    start = time.perf_counter()
    first_token_time: float | None = None
    event_times: list[float] = []
    last_obj: dict[str, Any] | None = None

    with requests.post(
        f"{base_url}/generate", json=payload, stream=True, timeout=(30, timeout)
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            data = raw_line[len("data: ") :]
            if data == "[DONE]":
                break
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            event_times.append(now)
            last_obj = json.loads(data)

    end = time.perf_counter()
    meta = (last_obj or {}).get("meta_info", {})
    completion_tokens = int(meta.get("completion_tokens") or len(event_times))
    prompt_tokens = int(meta.get("prompt_tokens") or len(input_ids))
    ttft = (first_token_time - start) if first_token_time is not None else None
    if completion_tokens > 1 and first_token_time is not None:
        tpot = (end - first_token_time) / (completion_tokens - 1)
    else:
        tpot = None

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": meta.get("cached_tokens"),
        "ttft_s": ttft,
        "tpot_s": tpot,
        "e2e_latency_s": end - start,
        "output_tok_s": completion_tokens / (end - start) if end > start else None,
        "total_tok_s": (prompt_tokens + completion_tokens) / (end - start)
        if end > start
        else None,
        "events": len(event_times),
        "finish_reason": meta.get("finish_reason"),
        "last_meta_info": meta,
    }


def run_case(
    base_url: str,
    log_path: str,
    mode: str,
    input_len: int,
    output_len: int,
    trial: int,
    timeout: int,
    warmup_output_len: int,
) -> dict[str, Any]:
    input_ids = make_input_ids(input_len, seed=input_len + trial * 17)
    flush_cache(base_url)

    warm_start = time.perf_counter()
    post_json(
        base_url,
        "/generate",
        {
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": warmup_output_len,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
        timeout=timeout,
    )
    warm_latency = time.perf_counter() - warm_start

    start_offset = log_offset(log_path)
    result = stream_generate(base_url, input_ids, output_len, timeout)
    time.sleep(0.5)
    result.update(extract_prefill_stats(read_new_log(log_path, start_offset)))
    result.update(
        {
            "mode": mode,
            "input_len": input_len,
            "output_len": output_len,
            "trial": trial,
            "warmup_output_len": warmup_output_len,
            "warmup_latency_s": warm_latency,
        }
    )
    return result


def summarize(results: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    rows.append(
        "| mode | input/output | trials | TTFT s | TPOT ms | E2E s | output tok/s | total tok/s | cached | hit total/gpu/eic | SLO |"
    )
    rows.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for item in results:
        grouped.setdefault((item["mode"], item["input_len"], item["output_len"]), []).append(
            item
        )

    for (mode, input_len, output_len), items in sorted(grouped.items()):
        ttfts = [x["ttft_s"] for x in items if x.get("ttft_s") is not None]
        tpots = [x["tpot_s"] for x in items if x.get("tpot_s") is not None]
        e2es = [x["e2e_latency_s"] for x in items if x.get("e2e_latency_s") is not None]
        outs = [x["output_tok_s"] for x in items if x.get("output_tok_s") is not None]
        totals = [x["total_tok_s"] for x in items if x.get("total_tok_s") is not None]
        cached = [x.get("cached_tokens") or 0 for x in items]
        latest = items[-1]
        ttft = statistics.median(ttfts) if ttfts else float("nan")
        tpot = statistics.median(tpots) * 1000 if tpots else float("nan")
        slo_ttft = 8 if input_len == 65536 else 10 if input_len == 131072 else 20
        slo = "PASS" if ttft < slo_ttft and tpot < 50 else "FAIL"
        rows.append(
            "| {mode} | {inp}/{out} | {n} | {ttft:.3f} | {tpot:.2f} | {e2e:.3f} | "
            "{outps:.2f} | {totalps:.2f} | {cached:.0f} | {hit:.2f}/{gpu:.2f}/{eic:.2f} | {slo} |".format(
                mode=mode,
                inp=input_len,
                out=output_len,
                n=len(items),
                ttft=ttft,
                tpot=tpot,
                e2e=statistics.median(e2es) if e2es else float("nan"),
                outps=statistics.median(outs) if outs else float("nan"),
                totalps=statistics.median(totals) if totals else float("nan"),
                cached=statistics.median(cached) if cached else 0,
                hit=latest.get("hit_rate_total", float("nan")),
                gpu=latest.get("hit_rate_gpu", float("nan")),
                eic=latest.get("hit_rate_eic", float("nan")),
                slo=slo,
            )
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--lengths", default="65536,131072,262144")
    parser.add_argument("--output-len", type=int, default=1500)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--warmup-output-len", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--log-path", default="/tmp/dsv4-eic-server.log")
    parser.add_argument("--output-jsonl", default="/tmp/dsv4_agent_cache_bench.jsonl")
    parser.add_argument("--output-md", default="/tmp/dsv4_agent_cache_bench.md")
    args = parser.parse_args()

    get(args.base_url, "/health", timeout=30)

    lengths = [int(x) for x in args.lengths.split(",") if x]
    results: list[dict[str, Any]] = []
    for input_len in lengths:
        for trial in range(args.trials):
            result = run_case(
                args.base_url,
                args.log_path,
                args.mode,
                input_len,
                args.output_len,
                trial,
                args.timeout,
                args.warmup_output_len,
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
            results.append(result)
            with open(args.output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    with open(args.output_md, "a", encoding="utf-8") as f:
        f.write(f"\n## {args.mode}\n\n")
        f.write(summarize(results))


if __name__ == "__main__":
    main()
