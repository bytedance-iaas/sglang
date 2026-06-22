import argparse
import ast
import json
import os
import re
import time

import numpy as np
from datasets import load_dataset

from sglang.lang.api import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    dump_bench_raw_result,
    select_sglang_backend,
)
from sglang.utils import download_and_cache_file, dump_state_text, read_jsonl

INVALID = -9999999


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def main(args):
    # Select backend
    set_default_backend(select_sglang_backend(args))

    # Load tokenizer if enable_thinking is set
    tokenizer = None
    if args.enable_thinking:
        from transformers import AutoTokenizer

        assert (
            args.tokenizer_path is not None
        ), "--tokenizer-path is required when --enable-thinking is set"
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )

    # Read data
    if args.platinum:
        print("Loading GSM8K Platinum dataset from HuggingFace...")
        dataset = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
        lines = [
            {"question": item["question"], "answer": item["answer"]} for item in dataset
        ]
    else:
        data_path = args.data_path
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        if not os.path.isfile(data_path):
            data_path = download_and_cache_file(url)
        lines = list(read_jsonl(data_path))

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    if args.question_index is not None:
        target_indices = [args.question_index]
        if args.question_index < num_shots:
            print(
                f"WARNING: --question-index {args.question_index} overlaps the "
                f"first {num_shots} few-shot examples; the model will see this "
                "question's answer in its prompt."
            )
    else:
        target_indices = list(range(min(num_questions, len(lines))))

    questions = []
    labels = []
    for i in target_indices:
        raw_question = few_shot_examples + get_one_example(lines, i, False)
        if tokenizer is not None:
            messages = [{"role": "user", "content": raw_question}]
            raw_question = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        questions.append(raw_question)
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    gen_kwargs = dict(
        max_tokens=args.max_new_tokens,
        stop=["Question", "Assistant:", "<|separator|>"],
    )
    if args.dump_logprobs:
        gen_kwargs.update(
            return_logprob=True,
            top_logprobs_num=args.top_logprobs,
            return_text_in_logprobs=True,
        )

    @sgl.function
    def few_shot_gsm8k(s, question):
        s += question
        s += sgl.gen("answer", **gen_kwargs)

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    tic = time.perf_counter()
    states = few_shot_gsm8k.run_batch(
        arguments,
        temperature=args.temperature,
        top_p=args.top_p,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    preds = []
    for i in range(len(states)):
        preds.append(get_answer_value(states[i]["answer"]))

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    # Compute speed
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    output_throughput = num_output_tokens / latency

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")

    # Dump results
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    dump_state_text(f"tmp_output_{args.backend}{suffix}.txt", states)

    if args.dump_answers:
        # One-line-per-question dump for easy diff between two runs (e.g.
        # CP=1 vs CP=2). Includes the question index, the predicted final
        # numeric answer, the full generated answer text, and a stable
        # hash of the answer text so a quick `diff -y` shows divergence
        # immediately even when --question-index is used.
        import hashlib

        answers_path = f"tmp_answers_{args.backend}{suffix}.txt"
        with open(answers_path, "w") as fout:
            for idx, state, pred, label in zip(
                target_indices, states, preds, labels
            ):
                ans_text = state["answer"]
                ans_hash = hashlib.sha1(ans_text.encode("utf-8")).hexdigest()[:12]
                fout.write(
                    f"--- q{idx} pred={pred} label={label} hash={ans_hash}\n"
                )
                fout.write(ans_text)
                fout.write("\n--- end q{}\n\n".format(idx))
        print(f"Per-question answers dumped to {answers_path}")

    if args.dump_logprobs:
        # Per-step token + top-K logprobs dump for diff'ing two runs (e.g.
        # CP=1 vs CP=2). Each line is one decode step:
        #   q<idx> step=<i> picked=<token_id> "<text>" lp=<logprob>
        #     top: [(token_id, "text", logprob), ...]
        # Diff'ing two such files (e.g. tmp_logprobs_srt_cp1.txt vs
        # tmp_logprobs_srt_cp2.txt) lands you on the first divergent token
        # and immediately shows the top-K next-token distribution at that
        # step on both sides, which makes attention-retrieval bugs (where
        # one side's top-1 swaps to a different surface form like
        # "60" vs "80") visible in seconds.
        logprobs_path = f"tmp_logprobs_{args.backend}{suffix}.txt"
        with open(logprobs_path, "w") as fout:
            for idx, state in zip(target_indices, states):
                meta = state.get_meta_info("answer") or {}
                token_lp = meta.get("output_token_logprobs") or []
                top_lp = meta.get("output_top_logprobs") or []
                fout.write(
                    f"=== q{idx} steps={len(token_lp)} top_k={args.top_logprobs}\n"
                )
                for step, picked in enumerate(token_lp):
                    p_lp, p_id, p_txt = picked
                    p_txt_repr = repr(p_txt)
                    fout.write(
                        f"q{idx} step={step} picked={p_id} {p_txt_repr} "
                        f"lp={p_lp:.6f}\n"
                    )
                    if step < len(top_lp) and top_lp[step]:
                        for t_lp, t_id, t_txt in top_lp[step]:
                            t_txt_repr = repr(t_txt)
                            fout.write(
                                f"    top: {t_id} {t_txt_repr} lp={t_lp:.6f}\n"
                            )
                fout.write(f"=== end q{idx}\n\n")
        print(f"Per-step logprobs dumped to {logprobs_path}")

    dump_bench_raw_result(
        path=args.raw_result_file,
        states=states,
        preds=preds,
        labels=labels,
    )

    with open(args.result_file, "a") as fout:
        value = {
            "task": "gsm8k-platinum" if args.platinum else "gsm8k",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "accuracy": round(acc, 3),
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument(
        "--question-index",
        type=int,
        default=None,
        help="Run only the single question at this 0-based index (after the "
        "first --num-shots few-shot examples). Overrides --num-questions.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Suffix appended to the dumped output / answers file names "
        "(e.g. cp1, cp2). Lets two runs over the same --question-index "
        "land in different files for diff'ing.",
    )
    parser.add_argument(
        "--dump-answers",
        action="store_true",
        help="Also dump a per-question answer text + sha1 hash file "
        "(tmp_answers_<backend><suffix>.txt) for easy diff between runs.",
    )
    parser.add_argument(
        "--dump-logprobs",
        action="store_true",
        help="Also dump per-step picked-token + top-K logprobs to "
        "tmp_logprobs_<backend><suffix>.txt for diff'ing two runs to "
        "locate the first divergent decode step.",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=10,
        help="K for --dump-logprobs (top-K next-token distribution per step).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode by wrapping prompts with chat template",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer (required when --enable-thinking is set)",
    )
    parser.add_argument(
        "--platinum",
        action="store_true",
        help="Use GSM8K Platinum dataset (drop-in replacement with corrected labels)",
    )
    args = add_common_sglang_args_and_parse(parser)
    main(args)
