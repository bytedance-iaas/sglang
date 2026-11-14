# SPDX-License-Identifier: Apache-2.0
"""CPU checks for target/draft embedding sharing before KV pool sizing."""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _run(compatible: bool) -> list[str]:
    calls = []
    worker = EagleDraftWorker.__new__(EagleDraftWorker)
    worker.draft_worker = MagicMock()
    worker.draft_worker.alloc_memory_pool.side_effect = lambda **_: calls.append(
        "draft_pool"
    )
    with (
        patch.object(
            EagleDraftWorker,
            "_can_share_embed_before_pool_sizing",
            return_value=compatible,
        ),
        patch.object(
            EagleDraftWorker, "init_token_map", lambda self: calls.append("token_map")
        ),
        patch.object(
            EagleDraftWorker, "init_lm_head", lambda self: calls.append("lm_head")
        ),
        patch(
            "sglang.srt.speculative.eagle_worker_v2.get_spec",
            return_value=SimpleNamespace(speculative_use_rejection_sampling=False),
        ),
    ):
        worker.target_embed_and_head_shared = False
        worker._maybe_share_embed_before_pool_sizing()
        calls.append("target_pool")
        worker.alloc_memory_pool()
    return calls


class TestEagleDraftEmbedShare(CustomTestCase):
    def test_compatible_layout_is_shared_before_pool_sizing(self):
        self.assertEqual(
            _run(True), ["token_map", "lm_head", "target_pool", "draft_pool"]
        )

    def test_incompatible_layout_keeps_original_order(self):
        self.assertEqual(
            _run(False), ["target_pool", "draft_pool", "token_map", "lm_head"]
        )

    def test_init_ends_with_the_early_share_attempt(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(EagleDraftWorker.__init__)))
        last = tree.body[0].body[-1]
        self.assertEqual(
            ast.unparse(last), "self._maybe_share_embed_before_pool_sizing()"
        )

    def test_layout_signature_includes_vocab_partition(self):
        weight = torch.empty((4, 8), dtype=torch.bfloat16)
        common = dict(
            quant_method=None,
            num_embeddings=8,
            enable_tp=True,
            tp_size=2,
            use_attn_tp_group=True,
        )
        left = SimpleNamespace(
            **common,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=0,
                org_vocab_end_index=4,
                added_vocab_start_index=8,
                added_vocab_end_index=8,
                num_org_vocab_padding=0,
            ),
        )
        right = SimpleNamespace(
            **common,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=4,
                org_vocab_end_index=8,
                added_vocab_start_index=8,
                added_vocab_end_index=8,
                num_org_vocab_padding=0,
            ),
        )
        self.assertNotEqual(
            EagleDraftWorker._weight_layout_signature(left, weight),
            EagleDraftWorker._weight_layout_signature(right, weight),
        )


if __name__ == "__main__":
    unittest.main()
