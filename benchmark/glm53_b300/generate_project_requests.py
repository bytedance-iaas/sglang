#!/usr/bin/env python3
"""Generate deterministic project requests at exact chat-template token lengths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer, __version__ as transformers_version


BASE_PARAGRAPH = (
    "海风观测站按小时记录温度、湿度、气压和风速。值班员先核对传感器时间戳，"
    "再比较相邻记录，发现异常时保留原始值并写明原因。所有记录只用于本次确定性"
    "长上下文性能测试，不包含真实业务数据。\n"
)
QUESTION = "请仅用一句中文概括以上文档的记录流程。"


def token_ids(encoded) -> list[int]:
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("expected an unbatched chat template result")
        encoded = encoded[0]
    return list(encoded)


def chat_tokens(tokenizer, content: str) -> list[int]:
    messages = [{"role": "user", "content": content + "\n\n" + QUESTION}]
    return token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )


def content_for_exact_length(tokenizer, target: int, prefix: str) -> str:
    base_ids = tokenizer.encode(BASE_PARAGRAPH, add_special_tokens=False)
    if not base_ids:
        raise RuntimeError("base paragraph produced no tokens")

    # Keep a unique marker at the start so requests in a cold-load file do not
    # share the long repeated body in the server's radix cache.
    fixed_overhead = len(chat_tokens(tokenizer, ""))
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    wanted_content_tokens = max(0, target - fixed_overhead)
    repeated = base_ids * (wanted_content_tokens // len(base_ids) + 2)
    body_token_count = max(0, wanted_content_tokens - len(prefix_ids))
    previous_counts = set()
    closest_below = None
    for _ in range(16):
        content = tokenizer.decode(
            prefix_ids + repeated[:body_token_count],
            skip_special_tokens=True,
        )
        actual = len(chat_tokens(tokenizer, content))
        if actual == target:
            return content
        if actual < target and (
            closest_below is None or actual > closest_below[0]
        ):
            closest_below = (actual, content)
        if actual in previous_counts:
            break
        previous_counts.add(actual)
        body_token_count += target - actual
        if body_token_count < 0:
            break

    # Decode/re-encode can skip one length at a token boundary. Search a small
    # deterministic suffix space without weakening the exact-length invariant.
    if closest_below is not None:
        _, content = closest_below
        for suffix in ["a", " b", ".", "。", " 1", "\na", " aa", "abc"]:
            candidate = content + suffix
            actual = len(chat_tokens(tokenizer, candidate))
            if actual == target:
                return candidate
    raise RuntimeError(f"could not construct {target} tokens; nearest got {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/models/GLM-5.3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[10240, 16384, 32768, 65536, 102400, 131072, 204800, 229376],
    )
    parser.add_argument("--copies", type=int, default=100)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for length in args.lengths:
        path = args.output_dir / f"project-{length}-c{args.copies}.jsonl"
        with path.open("w", encoding="utf-8") as output:
            for request_index in range(args.copies):
                prefix = f"唯一请求编号：{length}-{request_index:04d}。\n"
                content = content_for_exact_length(tokenizer, length, prefix)
                body = {
                    "model": "glm-5.3",
                    "messages": [{"role": "user", "content": content + "\n\n" + QUESTION}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "max_tokens": 512,
                    # Freeze the fixed SGLang image's observed sampling defaults.
                    # EvalScope otherwise injects its own temperature=0.0 default.
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "thinking": {"type": "disabled"},
                }
                actual = len(token_ids(tokenizer.apply_chat_template(
                    body["messages"], tokenize=True, add_generation_prompt=True
                )))
                if actual != length:
                    raise AssertionError((length, request_index, actual))
                output.write(json.dumps(body, ensure_ascii=False, separators=(",", ":")) + "\n")
        entries.append({
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "prompt_tokens_with_template": actual,
            "requests": args.copies,
            "max_tokens": 512,
            "temperature": 1.0,
            "top_p": 1.0,
            "stream": True,
            "thinking": {"type": "disabled"},
            "cache_condition": "unique marker at start; only chat-template prefix shared",
            "source": "synthetic/project deterministic unique marker plus repeated coherent paragraph",
        })

    manifest = {
        "model_path": args.model_path,
        "transformers_version": transformers_version,
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": hashlib.sha256(
            (tokenizer.chat_template or "").encode("utf-8")
        ).hexdigest(),
        "k": 1024,
        "entries": entries,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
