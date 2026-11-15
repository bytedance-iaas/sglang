import unittest

import torch

from sglang.srt.layers.attention.dsa.pivot_candidate import (
    PivotFallbackReason,
    PivotQueryRow,
    PivotShareIdentity,
)
from sglang.srt.layers.attention.dsa.pivot_reference import (
    PivotReferenceConfig,
    pivot_refine_reference,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPivotReference(unittest.TestCase):
    def setUp(self):
        self.identity = PivotShareIdentity(
            request_generation=1,
            producer_layer_id=3,
            index_share_id="share-0",
            cache_version=7,
            key_representation_version="fp32-reference-v1",
        )
        self.keys = {
            self.identity: torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [-1.0, 1.0],
                    [1.0, -1.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                    [3.0, 3.0],
                ]
            )
        }

    def rows(self, positions=(5, 6), *, rank=0, group="group-0"):
        return [
            PivotQueryRow(self.identity, group, rank, position)
            for position in positions
        ]

    def test_mean_proxy_first_cutoff_window_and_refine(self):
        queries = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ]
        )
        gates = torch.ones((2, 2))
        result = pivot_refine_reference(
            queries=queries,
            gates=gates,
            keys=self.keys,
            rows=self.rows(),
            config=PivotReferenceConfig(
                group_size=2,
                topk=2,
                candidate_budget=4,
                local_window=1,
                scoring_heads=2,
            ),
        )

        self.assertTrue(all(plan.accepted for plan in result.plans))
        # The proxy scans only through the first row's position (ID 5). The
        # staggered local-window union adds IDs 5 and 6, but never future ID 7.
        self.assertNotIn(7, result.plans[1].candidates)
        self.assertIn(5, result.plans[0].candidates)
        self.assertIn(6, result.plans[1].candidates)
        self.assertEqual(result.selected_topk[0], result.exact_topk[0])
        self.assertEqual(result.recall.evaluated_rows, 2)
        self.assertGreater(result.proxy_score_items, 0)
        self.assertGreater(result.refine_score_items, 0)
        self.assertLess(result.ideal_score_work_ratio, 1.5)

    def test_short_group_guardrail_falls_back_to_dense(self):
        result = pivot_refine_reference(
            queries=torch.ones((2, 2, 2)),
            gates=torch.ones((2, 2)),
            keys=self.keys,
            rows=self.rows(positions=(2, 3)),
            config=PivotReferenceConfig(
                group_size=2,
                topk=2,
                candidate_budget=4,
                scoring_heads=2,
            ),
        )
        self.assertTrue(all(not plan.accepted for plan in result.plans))
        self.assertTrue(
            all(
                plan.fallback_reason == PivotFallbackReason.SHORT_CONTEXT
                for plan in result.plans
            )
        )
        self.assertEqual(result.selected_topk, result.exact_topk)
        self.assertEqual(result.proxy_score_items, 0)
        self.assertEqual(result.refine_score_items, 0)

    def test_nonconsecutive_group_fails_closed(self):
        queries = torch.ones((2, 2, 2))
        gates = torch.ones((2, 2))
        result = pivot_refine_reference(
            queries=queries,
            gates=gates,
            keys=self.keys,
            rows=self.rows(positions=(5, 7)),
            config=PivotReferenceConfig(
                group_size=2,
                topk=2,
                candidate_budget=4,
                scoring_heads=2,
            ),
        )
        self.assertTrue(any(not plan.accepted for plan in result.plans))
        self.assertEqual(result.selected_topk, result.exact_topk)

    def test_cross_rank_rows_form_independent_single_query_groups(self):
        queries = torch.ones((2, 2, 2))
        gates = torch.ones((2, 2))
        rows = [self.rows()[0], self.rows(rank=1)[1]]
        result = pivot_refine_reference(
            queries=queries,
            gates=gates,
            keys=self.keys,
            rows=rows,
            config=PivotReferenceConfig(
                group_size=2,
                topk=2,
                candidate_budget=4,
                scoring_heads=2,
            ),
        )
        self.assertTrue(all(plan.accepted for plan in result.plans))
        self.assertEqual(result.selected_topk, result.exact_topk)
        self.assertEqual(result.proxy_score_items, 13)

    def test_candidate_recall_exposes_proxy_omission(self):
        # Mean pooling cancels the first head. Query 0's full scorer still
        # prefers remote ID 5, demonstrating that Refine cannot recover an ID
        # omitted by the shared candidate set.
        queries = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[-1.0, 0.0], [0.0, 1.0]],
            ]
        )
        gates = torch.ones((2, 2))
        result = pivot_refine_reference(
            queries=queries,
            gates=gates,
            keys=self.keys,
            rows=self.rows(),
            config=PivotReferenceConfig(
                group_size=2,
                topk=2,
                candidate_budget=3,
                scoring_heads=2,
            ),
        )
        self.assertGreater(result.recall.severe_omissions, 0)
        self.assertNotEqual(result.selected_topk[0], result.exact_topk[0])


if __name__ == "__main__":
    unittest.main()
