# Adapted from https://github.com/openai/simple-evals/

"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

import random
import re
from typing import Optional

import pandas

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    ANSWER_PATTERN_MULTICHOICE,
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
    format_multichoice_question,
)

GPQA_ANSWER_PATTERN_MULTICHOICE = (
    r"(?im)(?:^|\n)\s*Answer\s*:\s*(?:[\(\[]\s*)?"
    r"([A-D])(?:\s*[\)\]])?\s*\.?\s*$"
)


def format_gpqa_question(row: dict) -> str:
    return f"""
You are answering a GPQA multiple-choice question.
Choose exactly one option from A, B, C, and D.

Question:
{row["Question"]}

A) {row["A"]}
B) {row["B"]}
C) {row["C"]}
D) {row["D"]}

Your final line must be exactly:
Answer: X

Replace X with one of A, B, C, or D.
Do not put any value, formula, option text, or explanation after the final answer line.
""".strip()


def _normalize_answer_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9]+", " ", text)).strip().lower()


def _extract_answer_spans(response_text: str) -> list[tuple[int, str]]:
    spans: list[tuple[int, str]] = []

    for match in re.finditer(GPQA_ANSWER_PATTERN_MULTICHOICE, response_text):
        if response_text[match.end() :].strip():
            continue
        spans.append((match.start(), match.group(1).upper()))

    answer_phrases = [
        r"\b(?:(?i:(?:final\s+(?:answer|option|choice)|"
        r"correct\s+(?:answer|option|choice))\s*(?:is|:)|answer\s+is))\s*"
        r"(?i:(?:option|choice)?)\s*[\(\[]?\s*([A-D])\b"
        r"(?!\s*(?:[-/,]|or\b|and\b))",
        r"(?i:\\boxed\{\s*)([A-D])(?i:\s*\})",
    ]
    for pattern in answer_phrases:
        for match in re.finditer(pattern, response_text):
            prefix = response_text[max(0, match.start() - 80) : match.start()]
            suffix = response_text[match.end() :].split("\n", 1)[0]
            if re.search(
                r"(?i)\b(?:considered|initially|first|thought|tempting)\b",
                prefix,
            ) or re.search(
                r"(?i)(?:\b(?:but|however|though|although|instead|rather|"
                r"wrong|changed|unsure|not\s+sure)\b|actually\s+no)",
                suffix,
            ):
                continue
            spans.append((match.start(), match.group(1).upper()))

    tail_match = re.search(r"(?:^|</think>|\n)\s*([A-D])\s*$", response_text, re.I)
    if tail_match:
        spans.append((tail_match.start(), tail_match.group(1).upper()))

    tail_choice_match = re.search(
        r"(?:^|\n)\s*([A-D])\s*[\)\].:-]\s*\S[^\n]*\s*$",
        response_text,
        re.I,
    )
    if tail_choice_match:
        prefix = response_text[: tail_choice_match.start()].strip()
        if (
            prefix.endswith("</think>")
            or re.search(r"(?i)(?:final|answer|correct)\s*[:\n]?\s*$", prefix[-80:])
        ):
            spans.append((tail_choice_match.start(), tail_choice_match.group(1).upper()))

    return spans


def _match_choice_text(response_text: str, choices_dict: dict) -> Optional[str]:
    tail_text = "\n".join(response_text.strip().splitlines()[-3:])
    normalized_tail = _normalize_answer_text(tail_text)
    if not normalized_tail:
        return None
    if not re.search(r"(?i)\b(answer|final|correct|therefore|thus|hence)\b", tail_text):
        return None
    if re.search(
        r"(?i)(?:\b(?:but|however|though|although|instead|rather|wrong|"
        r"changed|unsure|not\s+sure|not\s+final)\b|actually\s+no)",
        tail_text,
    ):
        return None

    hits = []
    for answer in "ABCD":
        choice_text = _normalize_answer_text(str(choices_dict[answer]))
        if choice_text and re.search(
            rf"(?:^| ){re.escape(choice_text)}(?: |$)", normalized_tail
        ):
            hits.append((answer, len(choice_text)))
    if not hits:
        return None

    max_len = max(length for _, length in hits)
    longest_hits = [answer for answer, length in hits if length == max_len]
    return longest_hits[0] if len(longest_hits) == 1 else None


def extract_gpqa_answer(
    response_text: str, choices_dict: dict, relaxed: bool = True
) -> Optional[str]:
    if not relaxed:
        match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
        return match.group(1) if match else None

    spans = _extract_answer_spans(response_text)
    if spans:
        return max(spans, key=lambda item: item[0])[1]
    return _match_choice_text(response_text, choices_dict)


class GPQAEval(Eval):
    def __init__(
        self,
        filename: str,
        num_examples: Optional[int],
        num_threads: int,
        n_repeats: int = 1,
        prompt_style: str = "official",
        relaxed_extraction: bool = False,
    ):
        if prompt_style not in ("official", "strict-final-line"):
            raise ValueError(
                "prompt_style must be either 'official' or 'strict-final-line'"
            )
        if "://" in filename:
            df = pandas.read_csv(filename, storage_options={"timeout": 30})
        else:
            df = pandas.read_csv(filename)
        examples = [row.to_dict() for _, row in df.iterrows()]
        rng = random.Random(0)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples"
            examples = rng.sample(examples, num_examples)
        examples = examples * n_repeats
        examples = [
            example | {"permutation": rng.sample(range(4), 4)} for example in examples
        ]
        self.examples = examples
        self.n_repeats = n_repeats
        self.num_threads = num_threads
        self.prompt_style = prompt_style
        self.relaxed_extraction = relaxed_extraction

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            choices = [choices[i] for i in row["permutation"]]
            correct_index = choices.index(row["Correct Answer"])
            correct_answer = "ABCD"[correct_index]
            choices_dict = dict(
                A=choices[0],
                B=choices[1],
                C=choices[2],
                D=choices[3],
                Question=row["Question"],
            )
            prompt_text = (
                format_multichoice_question(choices_dict)
                if self.prompt_style == "official"
                else format_gpqa_question(choices_dict)
            )
            prompt_messages = [
                sampler._pack_message(content=prompt_text, role="user")
            ]
            response_text = sampler(prompt_messages)
            if response_text is None:
                response_text = ""
            extracted_answer = extract_gpqa_answer(
                response_text, choices_dict, relaxed=self.relaxed_extraction
            )
            relaxed_answer = (
                extracted_answer
                if self.relaxed_extraction
                else extract_gpqa_answer(response_text, choices_dict, relaxed=True)
            )
            score = 1.0 if extracted_answer == correct_answer else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={
                    "chars": len(response_text),
                    "empty_response": 0.0 if response_text.strip() else 1.0,
                    "answer_extracted": 1.0 if extracted_answer is not None else 0.0,
                    "strict_final_answer_line": (
                        1.0
                        if re.search(GPQA_ANSWER_PATTERN_MULTICHOICE, response_text)
                        else 0.0
                    ),
                    "relaxed_would_extract": (
                        1.0 if relaxed_answer is not None else 0.0
                    ),
                },
            )

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)
