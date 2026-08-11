"""Regression coverage for speculative req_to_token row sizing."""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_len_per_decode,
    get_alloc_reserve_per_decode,
    get_req_to_token_extra_context_len,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _args(*, page_size, draft_tokens, algorithm="DSPARK", steps=1, topk=1):
    return SimpleNamespace(
        speculative_algorithm=algorithm,
        effective_speculative_algorithm=algorithm,
        speculative_num_steps=steps,
        speculative_eagle_topk=topk,
        max_speculative_num_draft_tokens=draft_tokens,
        page_size=page_size,
    )


class TestReqToTokenRowHeadroom(unittest.TestCase):
    def test_dspark_page_256_reserves_double_window_and_alignment(self):
        args = _args(page_size=256, draft_tokens=6)
        self.assertEqual(get_alloc_len_per_decode(args), 6)
        self.assertEqual(get_alloc_reserve_per_decode(args), 12)
        self.assertEqual(get_req_to_token_extra_context_len(args), 267)

    def test_page_size_one_still_reserves_double_window(self):
        args = _args(page_size=1, draft_tokens=6)
        self.assertEqual(get_req_to_token_extra_context_len(args), 12)

    def test_steps_and_topk_participate_in_reserve(self):
        args = _args(page_size=1, draft_tokens=6, steps=4, topk=2)
        self.assertEqual(get_alloc_len_per_decode(args), 8)
        self.assertEqual(get_alloc_reserve_per_decode(args), 16)

    def test_non_spec_headroom_is_unchanged(self):
        args = _args(page_size=256, draft_tokens=None, algorithm=None)
        self.assertEqual(get_req_to_token_extra_context_len(args), 4)

    def test_prefill_peer_dspark_uses_speculative_headroom(self):
        args = _args(page_size=256, draft_tokens=6, algorithm=None)
        args.effective_speculative_algorithm = "DSPARK"
        self.assertEqual(get_req_to_token_extra_context_len(args), 267)


if __name__ == "__main__":
    unittest.main()
