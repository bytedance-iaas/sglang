import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.run_eval import (
    get_thinking_kwargs,
    resolve_chat_template_kwargs,
    should_fallback_to_reasoning_content,
)
from sglang.test.simple_eval_common import ChatCompletionSampler
from sglang.test.simple_eval_gpqa import extract_gpqa_answer, format_gpqa_question

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSV4GPQAEvalContract(unittest.TestCase):
    def _sampler_for_message(
        self, content, reasoning_content, fallback_to_reasoning_content=True
    ):
        message = SimpleNamespace(content=content, reasoning_content=reasoning_content)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(completion_tokens=1),
        )
        completions = SimpleNamespace(create=lambda **_: response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        sampler = object.__new__(ChatCompletionSampler)
        sampler.client = client
        sampler.model = "dummy"
        sampler.system_message = None
        sampler.temperature = 0.0
        sampler.top_p = 1.0
        sampler.max_tokens = 8
        sampler.reasoning_effort = None
        sampler.extra_body = None
        sampler.fallback_to_reasoning_content = fallback_to_reasoning_content
        sampler._completion_tokens = []
        return sampler

    def test_sampler_falls_back_to_reasoning_content_when_content_empty(self):
        sampler = self._sampler_for_message("", "Answer: B")
        self.assertEqual(sampler([{"role": "user", "content": "q"}]), "Answer: B")

    def test_sampler_falls_back_to_reasoning_content_when_content_whitespace(self):
        sampler = self._sampler_for_message("\n  ", "Answer: B")
        self.assertEqual(sampler([{"role": "user", "content": "q"}]), "Answer: B")

    def test_sampler_prefers_content_over_reasoning_content(self):
        sampler = self._sampler_for_message("Answer: C", "Answer: A")
        self.assertEqual(sampler([{"role": "user", "content": "q"}]), "Answer: C")

    def test_sampler_does_not_merge_reasoning_into_nonempty_content(self):
        sampler = self._sampler_for_message("No final answer.", "Answer: A")
        self.assertEqual(
            sampler([{"role": "user", "content": "q"}]),
            "No final answer.",
        )

    def test_sampler_reasoning_fallback_is_opt_in(self):
        sampler = self._sampler_for_message(
            "", "Answer: B", fallback_to_reasoning_content=False
        )
        self.assertEqual(sampler([{"role": "user", "content": "q"}]), "")

    def test_deepseek_v4_uses_thinking_chat_template_key(self):
        args = SimpleNamespace(thinking_mode="deepseek-v4")
        self.assertEqual(get_thinking_kwargs(args), {"thinking": True})

    def test_qwen3_alias_uses_enable_thinking_key(self):
        args = SimpleNamespace(thinking_mode="qwen3")
        self.assertEqual(get_thinking_kwargs(args), {"enable_thinking": True})

    def test_explicit_chat_template_kwargs_override_thinking_mode_default(self):
        args = SimpleNamespace(
            thinking_mode="deepseek-v4",
            chat_template_kwargs={"thinking": False},
        )
        self.assertEqual(resolve_chat_template_kwargs(args), {"thinking": False})

    def test_string_chat_template_kwargs_override_thinking_mode_default(self):
        args = SimpleNamespace(
            thinking_mode="deepseek-v4",
            chat_template_kwargs='{"thinking": false, "foo": "bar"}',
        )
        self.assertEqual(
            resolve_chat_template_kwargs(args),
            {"thinking": False, "foo": "bar"},
        )

    def test_reasoning_content_fallback_is_gpqa_only(self):
        self.assertTrue(
            should_fallback_to_reasoning_content(SimpleNamespace(eval_name="gpqa"))
        )
        for eval_name in ["mmlu", "math", "longbench_v2", None]:
            with self.subTest(eval_name=eval_name):
                self.assertFalse(
                    should_fallback_to_reasoning_content(
                        SimpleNamespace(eval_name=eval_name)
                    )
                )

    def test_gpqa_extractor_uses_last_answer(self):
        choices = {
            "Question": "Which option is correct?",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
        }
        response = "At first I considered A.\nAnswer: A\nFinal check.\nAnswer: C"
        self.assertEqual(extract_gpqa_answer(response, choices), "C")

    def test_gpqa_extractor_rejects_ambiguous_answer_line(self):
        choices = {
            "Question": "Which option is correct?",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
        }
        for response in ["Answer: A or B", "Answer: A/B", "Answer: A and B"]:
            with self.subTest(response=response):
                self.assertIsNone(extract_gpqa_answer(response, choices))

    def test_gpqa_extractor_accepts_parenthesized_answer_line(self):
        choices = {
            "Question": "Which option is correct?",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
        }
        self.assertEqual(extract_gpqa_answer("Answer: (A)", choices), "A")

    def test_gpqa_extractor_rejects_answer_line_with_nonempty_suffix(self):
        choices = {
            "Question": "Which option is correct?",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
        }
        self.assertIsNone(extract_gpqa_answer("Answer: A\nActually no.", choices))

    def test_gpqa_extractor_accepts_later_final_answer_after_old_answer_line(self):
        choices = {
            "Question": "Which option is correct?",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
        }
        response = "Answer: A\nActually no.\nThe final answer is C."
        self.assertEqual(extract_gpqa_answer(response, choices), "C")

    def test_gpqa_extractor_accepts_option_phrase(self):
        choices = {
            "Question": "Which option is correct?",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
        }
        self.assertEqual(
            extract_gpqa_answer("The correct answer is option D.", choices),
            "D",
        )

    def test_gpqa_extractor_does_not_accept_incidental_option_phrase(self):
        choices = {
            "Question": "Which option is correct?",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
        }
        self.assertIsNone(extract_gpqa_answer("It is not option A.", choices))

    def test_gpqa_extractor_does_not_accept_uncommitted_answer_phrase(self):
        choices = {
            "Question": "Which option is correct?",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
        }
        response = "I considered final answer A, but this is wrong."
        self.assertIsNone(extract_gpqa_answer(response, choices))

    def test_gpqa_extractor_rejects_revised_answer_phrase(self):
        choices = {
            "Question": "Which option is correct?",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
        }
        rejected = [
            "The final answer is A but this is wrong.",
            "The answer is C. Actually no.",
            "The correct answer is option D; however, I am not sure.",
            "I thought the final answer is B.",
            "I thought the final answer is B then changed my mind.",
        ]
        for response in rejected:
            with self.subTest(response=response):
                self.assertIsNone(extract_gpqa_answer(response, choices))

    def test_gpqa_extractor_accepts_explained_answer_phrase(self):
        choices = {
            "Question": "Which option is correct?",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
        }
        self.assertEqual(
            extract_gpqa_answer("The answer is C because gamma is correct.", choices),
            "C",
        )

    def test_gpqa_extractor_rejects_bare_option_line(self):
        choices = {
            "Question": "Which option is correct?",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
        }
        self.assertIsNone(extract_gpqa_answer("A) YES", choices))

    def test_gpqa_extractor_does_not_accept_repeated_option_line(self):
        choices = {
            "Question": "Which option is correct?",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
        }
        response = "The options are:\nA) alpha"
        self.assertIsNone(extract_gpqa_answer(response, choices))

    def test_gpqa_extractor_does_not_accept_bare_choice_text(self):
        choices = {
            "Question": "Which option is correct?",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
        }
        self.assertIsNone(extract_gpqa_answer("gamma", choices))

    def test_gpqa_extractor_accepts_option_line_after_final_context(self):
        choices = {
            "Question": "Which option is correct?",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
        }
        response = "Final answer:\nB. beta"
        self.assertEqual(extract_gpqa_answer(response, choices), "B")

    def test_gpqa_extractor_accepts_dsv4_reasoning_boundary_forms(self):
        choices = {
            "Question": "Which option is correct?",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
        }
        accepted = {
            "</think>C": "C",
            "long reasoning\n</think>\nD": "D",
            "<think>reasoning</think>\nAnswer: B": "B",
            "<think>reasoning</think>\nB) beta": "B",
            "<think>reasoning</think>\nThe final answer is A.": "A",
        }
        for response, expected in accepted.items():
            with self.subTest(response=response):
                self.assertEqual(extract_gpqa_answer(response, choices), expected)

    def test_gpqa_extractor_rejects_nonfinal_dsv4_boundary_forms(self):
        choices = {
            "Question": "Which option is correct?",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
        }
        rejected = [
            "<think>Answer: A</think>",
            "<think>Answer: A</think>\nNo final answer",
            "<think>reasoning</think>\nThe final answer is not C.",
            "I will compare A, B, C, and D.",
            "A) alpha\nB) beta\nC) gamma\nD) delta",
            "The answer could be A or B.",
            "The final answer is C/D.",
            "The final answer is A-like.",
            "The final answer is C, not D.",
            "The final answer is C; I changed it to D.",
        ]
        for response in rejected:
            with self.subTest(response=response):
                self.assertIsNone(extract_gpqa_answer(response, choices))

    def test_gpqa_extractor_matches_unique_choice_text(self):
        choices = {
            "Question": "Which option is correct?",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
        }
        self.assertEqual(extract_gpqa_answer("The answer is gamma.", choices), "C")

    def test_gpqa_extractor_does_not_match_stale_choice_text(self):
        choices = {
            "Question": "Which option is correct?",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
        }
        response = "gamma\n" + "\n".join(["irrelevant reasoning"] * 4)
        self.assertIsNone(extract_gpqa_answer(response, choices))

    def test_gpqa_extractor_rejects_revised_choice_text(self):
        choices = {
            "Question": "Which option is correct?",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
        }
        self.assertIsNone(
            extract_gpqa_answer("The answer is beta, but this is not final.", choices)
        )

    def test_gpqa_extractor_does_not_match_choice_text_substrings(self):
        self.assertIsNone(
            extract_gpqa_answer(
                "The answer is condition.",
                {
                    "Question": "Which option is correct?",
                    "A": "ion",
                    "B": "beta",
                    "C": "gamma",
                    "D": "delta",
                },
            )
        )
        self.assertIsNone(
            extract_gpqa_answer(
                "The answer is analysis.",
                {
                    "Question": "Which option is correct?",
                    "A": "A",
                    "B": "beta",
                    "C": "gamma",
                    "D": "delta",
                },
            )
        )

    def test_gpqa_extractor_prefers_longest_unique_choice_text(self):
        self.assertEqual(
            extract_gpqa_answer(
                "The answer is alpha beta.",
                {
                    "Question": "Which option is correct?",
                    "A": "alpha beta",
                    "B": "beta",
                    "C": "gamma",
                    "D": "delta",
                },
            ),
            "A",
        )

    def test_gpqa_prompt_requires_final_answer_line(self):
        prompt = format_gpqa_question(
            {
                "Question": "Which option is correct?",
                "A": "alpha",
                "B": "beta",
                "C": "gamma",
                "D": "delta",
            }
        )
        self.assertIn("Your final line must be exactly:", prompt)
        self.assertIn("Answer: X", prompt)


if __name__ == "__main__":
    unittest.main()
