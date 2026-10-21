"""Build the PR #44612 LongBench-v2 serving workload reproducibly.

Each record contains the first ``context_tokens`` tokens of the document plus
the original multiple-choice question.  The output uses the two-turn ShareGPT
shape expected by ``sglang.benchmark.serving --dataset-name sharegpt``.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer


DATASET_ID = "THUDM/LongBench-v2"


def format_prompt(example: dict, context: str) -> str:
    return (
        f"{context}\n\n"
        f"Question: {example['question']}\n"
        f"A. {example['choice_A']}\n"
        f"B. {example['choice_B']}\n"
        f"C. {example['choice_C']}\n"
        f"D. {example['choice_D']}\n"
        "Answer:"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, default=8000)
    parser.add_argument("--expected-count", type=int, default=503)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )
    dataset_info = HfApi().dataset_info(DATASET_ID)
    dataset = load_dataset(DATASET_ID, split="train")
    if len(dataset) != args.expected_count:
        raise RuntimeError(
            f"expected {args.expected_count} LongBench-v2 records, got {len(dataset)}"
        )

    output = []
    context_source_lengths = []
    context_kept_lengths = []
    prompt_lengths = []
    for example in dataset:
        context_ids = tokenizer.encode(
            example["context"],
            add_special_tokens=False,
        )
        kept_ids = context_ids[: args.context_tokens]
        context = tokenizer.decode(kept_ids, skip_special_tokens=False)
        prompt = format_prompt(example, context)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
            return_dict=False,
        )
        if tokenizer.bos_token:
            rendered = rendered.replace(tokenizer.bos_token, "")

        context_source_lengths.append(len(context_ids))
        context_kept_lengths.append(len(kept_ids))
        prompt_lengths.append(len(tokenizer.encode(rendered)))
        output.append(
            {
                "id": example.get("_id"),
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": str(example.get("answer", "A"))},
                ],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        output,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    args.output.write_bytes(encoded)
    metadata = {
        "dataset_id": DATASET_ID,
        "dataset_revision": dataset_info.sha,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "record_count": len(output),
        "tokenizer": args.tokenizer,
        "context_token_limit": args.context_tokens,
        "context_source_tokens": {
            "min": min(context_source_lengths),
            "median": statistics.median(context_source_lengths),
            "max": max(context_source_lengths),
        },
        "context_kept_tokens": {
            "min": min(context_kept_lengths),
            "median": statistics.median(context_kept_lengths),
            "max": max(context_kept_lengths),
        },
        "rendered_prompt_tokens": {
            "min": min(prompt_lengths),
            "median": statistics.median(prompt_lengths),
            "max": max(prompt_lengths),
        },
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
        "output_bytes": len(encoded),
        "note": "Derived from the PR description; the author's exact JSON was unavailable.",
    }
    args.metadata_file.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_file.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
