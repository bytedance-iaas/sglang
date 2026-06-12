#!/usr/bin/env python3
"""DSv4 DCP numerical-equivalence regression script.

Usage:
    1. Launch a baseline sglang server with --dcp-size 1 on :30000.
    2. Launch a candidate sglang server with --dcp-size 2 (and
       SGLANG_DSV4_ENABLE_DCP=1) on :30001.
    3. Run:
         python scripts/playground/dcp_equivalence_check.py \\
             --baseline-url http://127.0.0.1:30000 \\
             --candidate-url http://127.0.0.1:30001 \\
             --model-path /path/to/dsv4 \\
             --num-prompts 8 --max-tokens 64

The script issues the same prompt set to both endpoints with
``temperature=0, top_p=1, top_k=1`` and compares:
  1. exact decoded text
  2. completion token id sequence
  3. per-token chosen-token logprob (max abs delta)

Exits non-zero on any divergence so it can be wired into CI.

This script intentionally has no torch / sglang internal dependency; it only
needs ``requests`` so it can run from any Python 3.8+ environment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import requests

DEFAULT_PROMPTS: List[str] = [
    # Short / boilerplate
    "Hello, world!",
    # English Q&A
    "Question: What is the capital of France?\nAnswer:",
    # Code completion
    "def fibonacci(n):\n    if n < 2:\n        return n\n    ",
    # Chinese
    "请用一句话总结相对论的核心思想：",
    # Long context (~256 tokens)
    "The following is a verbose technical specification of a hypothetical "
    "machine learning system. Please continue writing in the same style. "
    "The system shall implement decode context parallelism over a sharded "
    "key-value cache. " * 8 + "\n\nContinuing from the above:\n",
    # Multi-turn chat-style
    "User: Explain quantum entanglement in one paragraph.\nAssistant:",
    # Math
    "Compute step by step: 17 * 23 + 41.",
    # Edge: empty-ish (single token prompt)
    "1, 2, 3,",
]


@dataclass
class CompletionResult:
    text: str
    token_ids: List[int]
    chosen_logprobs: List[float]  # logprob of each chosen token

    def fingerprint(self) -> str:
        # Stable repr for diffing.
        return json.dumps(
            {"token_ids": self.token_ids, "text": self.text},
            ensure_ascii=False,
        )


def call_completion(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> CompletionResult:
    """Call the OpenAI-compatible /v1/completions endpoint."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        # sglang accepts top_k via extra body; ignore if unsupported
        "top_k": 1,
        "logprobs": 1,
        "echo": False,
        "stream": False,
        "seed": 42,
    }
    resp = requests.post(
        f"{url.rstrip('/')}/v1/completions",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    text = choice["text"]
    lp = choice.get("logprobs") or {}
    token_logprobs = lp.get("token_logprobs") or []
    # Token ids are not in OpenAI logprobs; fall back to per-token strings.
    # We hash strings as a stand-in for token ids — sufficient for divergence
    # detection on identical tokenizers.
    tokens = lp.get("tokens") or []
    token_ids = [hash(t) & 0xFFFFFFFF for t in tokens]
    chosen_logprobs = [float(x) if x is not None else 0.0 for x in token_logprobs]
    return CompletionResult(
        text=text, token_ids=token_ids, chosen_logprobs=chosen_logprobs
    )


def compare_one(
    idx: int,
    prompt: str,
    base: CompletionResult,
    cand: CompletionResult,
    logprob_atol: float,
) -> bool:
    """Return True if results match within tolerance."""
    ok = True
    if base.text != cand.text:
        ok = False
        print(f"[prompt #{idx}] TEXT MISMATCH")
        print(f"  baseline:  {base.text!r}")
        print(f"  candidate: {cand.text!r}")
    if base.token_ids != cand.token_ids:
        ok = False
        # Find first diverging position
        n = min(len(base.token_ids), len(cand.token_ids))
        first_div = next(
            (i for i in range(n) if base.token_ids[i] != cand.token_ids[i]),
            n,
        )
        print(
            f"[prompt #{idx}] TOKEN MISMATCH at position {first_div} "
            f"(baseline_len={len(base.token_ids)}, "
            f"candidate_len={len(cand.token_ids)})"
        )
    # Logprob comparison only meaningful when token ids match
    if base.token_ids == cand.token_ids and base.chosen_logprobs and cand.chosen_logprobs:
        deltas = [
            abs(b - c) for b, c in zip(base.chosen_logprobs, cand.chosen_logprobs)
        ]
        max_delta = max(deltas) if deltas else 0.0
        if max_delta > logprob_atol:
            ok = False
            print(
                f"[prompt #{idx}] LOGPROB DELTA exceeds atol={logprob_atol}: "
                f"max={max_delta:.6f}"
            )
        else:
            print(
                f"[prompt #{idx}] OK (max_logprob_delta={max_delta:.6f}, "
                f"len={len(base.token_ids)})"
            )
    elif ok:
        print(f"[prompt #{idx}] OK (text+tokens identical, no logprob compare)")
    return ok


def wait_for_health(url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{url.rstrip('/')}/health", timeout=5.0)
            if r.status_code == 200:
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(2.0)
    raise RuntimeError(f"server {url} not healthy within {timeout}s: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-url", required=True, help="dcp_size=1 server URL")
    ap.add_argument("--candidate-url", required=True, help="dcp_size>1 server URL")
    ap.add_argument(
        "--model-path",
        required=True,
        help="model path / name as registered with the sglang server",
    )
    ap.add_argument("--num-prompts", type=int, default=len(DEFAULT_PROMPTS))
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument(
        "--logprob-atol",
        type=float,
        default=1e-2,
        help="Max abs delta of chosen-token logprob between baseline/candidate",
    )
    ap.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Optional path to a newline-delimited prompts file",
    )
    ap.add_argument(
        "--skip-health-check", action="store_true", help="skip /health probe"
    )
    args = ap.parse_args()

    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS[: args.num_prompts]

    if not prompts:
        print("No prompts to evaluate.", file=sys.stderr)
        return 2

    if not args.skip_health_check:
        for url in (args.baseline_url, args.candidate_url):
            print(f"Waiting for {url} ...")
            wait_for_health(url)

    failures = 0
    for i, prompt in enumerate(prompts):
        try:
            base = call_completion(
                args.baseline_url,
                args.model_path,
                prompt,
                args.max_tokens,
                args.timeout,
            )
            cand = call_completion(
                args.candidate_url,
                args.model_path,
                prompt,
                args.max_tokens,
                args.timeout,
            )
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"[prompt #{i}] REQUEST FAILED: {e}")
            continue

        if not compare_one(i, prompt, base, cand, args.logprob_atol):
            failures += 1

    print()
    print(f"Total prompts: {len(prompts)}, failures: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
