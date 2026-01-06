#!/usr/bin/env python3
import argparse
import dataclasses
import json
import sys
from typing import Any, Dict, List, Optional

import sglang as sgl
from sglang.srt.server_args import ServerArgs


def _normalize_argv(argv: List[str]) -> List[str]:
    """Keep compatibility with common shorthand flags."""
    out = list(argv)

    def replace_flag(old: str, new: str):
        if old in out and new not in out:
            out[out.index(old)] = new

    # Your command used these:
    replace_flag("--model", "--model-path")
    replace_flag("--tp", "--tp-size")

    return out


def _build_prompt(prompt: str, system: Optional[str]) -> str:
    # Keep it simple and robust. If you want model-specific chat templates,
    # you can replace this with tokenizer.apply_chat_template(...).
    if system and system.strip():
        return f"[SYSTEM]\n{system.strip()}\n\n[USER]\n{prompt}"
    return prompt


def main():
    argv = _normalize_argv(sys.argv[1:])

    parser = argparse.ArgumentParser(
        description="SGLang local/offline prompt demo (no HTTP server)."
    )

    # This is the key: pull in *all* SGLang runtime/server flags.
    ServerArgs.add_cli_args(parser)

    # Add local-demo-only flags.
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text.")
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt.")
    parser.add_argument("--stream", action="store_true", help="Stream tokens to stdout.")
    parser.add_argument(
        "--sampling-json",
        type=str,
        default=None,
        help='Optional JSON dict to merge into sampling_params, e.g. \'{"max_new_tokens": 64}\'',
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate (used if not provided in --sampling-json).",
    )

    # Parse known args; if your branch has extra flags not in ServerArgs,
    # we won't crash — we’ll show them so you can add them properly.
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"[warn] Ignoring unknown args (not defined in ServerArgs on this branch): {unknown}")

    server_args = ServerArgs.from_cli_args(args)

    prompt_text = _build_prompt(args.prompt, args.system)

    sampling_params: Dict[str, Any] = {
        "temperature": getattr(args, "temperature", 0.0),
        "top_p": getattr(args, "top_p", 1.0),
        "max_new_tokens": args.max_new_tokens,
    }
    if args.sampling_json:
        sampling_params.update(json.loads(args.sampling_json))

    llm = None
    try:
        # Engine accepts ServerArgs as kwargs (same pattern as official offline_batch_inference.py)
        llm = sgl.Engine(**dataclasses.asdict(server_args))

        if args.stream:
            # Stream yields chunks; print incrementally.
            for chunk in llm.generate(prompt_text, sampling_params, stream=True):
                sys.stdout.write(chunk.get("text", ""))
                sys.stdout.flush()
            sys.stdout.write("\n")
        else:
            out = llm.generate([prompt_text], sampling_params)[0]
            print(out["text"])

    finally:
        if llm is not None:
            llm.shutdown()


if __name__ == "__main__":
    main()