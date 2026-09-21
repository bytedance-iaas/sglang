#!/usr/bin/env python3
"""Generate deterministic warm/probe requests with a measured shared prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer, __version__ as transformers_version

from generate_project_requests import content_for_exact_length, token_ids


def chat_ids(tokenizer, content: str) -> list[int]:
    return token_ids(tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
    ))


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/models/GLM-5.3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=71680)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    seed = content_for_exact_length(tokenizer, args.target_tokens, "缓存验证公共文档。\n")
    raw_ids = tokenizer.encode(seed, add_special_tokens=False)
    split = int(len(raw_ids) * 0.9)
    common = tokenizer.decode(raw_ids[:split], skip_special_tokens=True)
    tail_len = len(raw_ids) - split
    warm_tail_ids = tokenizer.encode("暖色海风记录。", add_special_tokens=False)
    probe_tail_ids = tokenizer.encode("冷色山谷记录。", add_special_tokens=False)
    warm = common + tokenizer.decode(
        (warm_tail_ids * (tail_len // len(warm_tail_ids) + 2))[:tail_len],
        skip_special_tokens=True,
    )
    probe = common + tokenizer.decode(
        (probe_tail_ids * (tail_len // len(probe_tail_ids) + 2))[:tail_len],
        skip_special_tokens=True,
    )

    warm_ids = chat_ids(tokenizer, warm)
    probe_ids = chat_ids(tokenizer, probe)
    shared = longest_common_prefix(warm_ids, probe_ids)
    shared_ratio = shared / len(probe_ids)
    if not (68000 <= len(warm_ids) <= 75000 and 68000 <= len(probe_ids) <= 75000):
        raise RuntimeError((len(warm_ids), len(probe_ids)))
    if shared_ratio < 0.89:
        raise RuntimeError((shared, len(probe_ids), shared_ratio))

    def request(content: str) -> dict:
        return {
            "model": "glm-5.3",
            "messages": [{"role": "user", "content": content}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 512,
            "temperature": 1.0,
            "top_p": 1.0,
            "thinking": {"type": "disabled"},
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, content in (("warm", warm), ("probe", probe)):
        path = args.output_dir / f"cache-{name}.json"
        path.write_text(
            json.dumps(request(content), ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        paths[name] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "prompt_tokens_with_template": len(chat_ids(tokenizer, content)),
        }
    manifest = {
        "model_path": args.model_path,
        "transformers_version": transformers_version,
        "chat_template_sha256": hashlib.sha256(
            (tokenizer.chat_template or "").encode()
        ).hexdigest(),
        "warm": paths["warm"],
        "probe": paths["probe"],
        "longest_common_prefix_tokens": shared,
        "shared_prefix_ratio_of_probe": shared_ratio,
    }
    (args.output_dir / "cache-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
