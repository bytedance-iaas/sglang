import unittest

from sglang.srt.layers.attention.dsa.dosa_cp import build_dosa_cp_layout
from sglang.srt.layers.attention.dsa.pivot_candidate import (
    PivotFallbackReason,
    PivotPlannerConfig,
    PivotPlannerContext,
    PivotProposalKey,
    PivotQueryRow,
    PivotShareIdentity,
    audit_pivot_recall,
    build_pivot_rows_from_cp_layout,
    plan_pivot_candidates,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPivotCandidateContract(unittest.TestCase):
    def setUp(self):
        self.config = PivotPlannerConfig(
            topk=4,
            candidate_budget=8,
            scoring_heads=4,
            min_context_tokens=8,
        )
        self.context = PivotPlannerContext(
            enabled=True,
            is_prefill=True,
            is_main_model=True,
            is_ordinary_extend=True,
        )
        self.identity = PivotShareIdentity(11, 3, "share-0", 7, "fp8-index-k-v1")

    def row(self, *, group="qg-0", rank=0, position=15, identity=None):
        return PivotQueryRow(
            identity=identity or self.identity,
            query_group=group,
            cp_rank=rank,
            absolute_position=position,
        )

    def test_cp8_rows_share_only_by_original_query_group(self):
        rows = [
            self.row(rank=rank, group="same", position=15 + rank) for rank in range(8)
        ]
        proposals = {
            PivotProposalKey(self.identity, "same", rank): tuple(range(8))
            for rank in range(8)
        }
        plans = plan_pivot_candidates(
            rows=rows,
            pivot_proposals=proposals,
            row_refinements=None,
            config=self.config,
            context=self.context,
        )
        self.assertTrue(all(plan.accepted for plan in plans))
        self.assertEqual([plan.valid_count for plan in plans], [8] * 8)

        unrelated = self.row(rank=0, group="other", position=23)
        plan = plan_pivot_candidates(
            rows=[unrelated],
            pivot_proposals=proposals,
            row_refinements=None,
            config=self.config,
            context=self.context,
        )[0]
        self.assertEqual(plan.fallback_reason, PivotFallbackReason.MISSING_PROPOSAL)

    def test_cp_layout_bridge_never_crosses_segment_or_request(self):
        layout = build_dosa_cp_layout(
            cp_rank=0,
            cp_size=2,
            request_generations=[11, 22],
            rids=["req-a", "req-b"],
            seq_lens=[14, 11],
            extend_seq_lens=[10, 7],
            query_layout_version="zigzag-2cp-v1",
            cache_versions=[3, 4],
        )
        rows = build_pivot_rows_from_cp_layout(
            layout,
            group_size=2,
            producer_layer_id=7,
            index_share_id="share-7",
            key_representation_version="fp8-index-k-v1",
        )
        self.assertEqual(len(rows), layout.local_row_count)
        for previous, current in zip(rows, rows[1:]):
            if previous.query_group == current.query_group:
                self.assertEqual(previous.cp_rank, current.cp_rank)
                self.assertEqual(previous.identity, current.identity)
                self.assertEqual(
                    current.absolute_position, previous.absolute_position + 1
                )

        rank1 = self.row(rank=1, group="rank-local", position=23)
        plan = plan_pivot_candidates(
            rows=[rank1],
            pivot_proposals={
                PivotProposalKey(self.identity, "rank-local", 0): tuple(range(8))
            },
            row_refinements=None,
            config=self.config,
            context=self.context,
        )[0]
        self.assertEqual(plan.fallback_reason, PivotFallbackReason.MISSING_PROPOSAL)

    def test_causal_filter_and_refine_are_per_row(self):
        row = self.row(position=9)
        key = PivotProposalKey(self.identity, "qg-0", 0)
        plans = plan_pivot_candidates(
            rows=[row],
            pivot_proposals={key: (0, 1, 8, 9, 10, -1)},
            row_refinements={0: (2, 3, 9, 11)},
            config=self.config,
            context=self.context,
        )
        self.assertTrue(plans[0].accepted)
        self.assertEqual(plans[0].valid_count, 6)
        self.assertEqual(plans[0].candidates, (0, 1, 8, 9, 2, 3, -1, -1))

    def test_cache_generation_and_index_share_identity_do_not_alias(self):
        stale = PivotShareIdentity(11, 3, "share-0", 6, "fp8-index-k-v1")
        row = self.row(identity=self.identity)
        plan = plan_pivot_candidates(
            rows=[row],
            pivot_proposals={
                PivotProposalKey(stale, "qg-0", 0): tuple(range(8)),
            },
            row_refinements=None,
            config=self.config,
            context=self.context,
        )[0]
        self.assertEqual(plan.fallback_reason, PivotFallbackReason.MISSING_PROPOSAL)

    def test_underfill_overflow_and_invalid_values_fail_closed(self):
        row = self.row()
        key = PivotProposalKey(self.identity, "qg-0", 0)
        cases = [
            ((0, 1, 2), PivotFallbackReason.CANDIDATE_UNDERFILL),
            (tuple(range(9)), PivotFallbackReason.CANDIDATE_OVERFLOW),
            ((0, 1, 2, -2), PivotFallbackReason.INVALID_CANDIDATE),
        ]
        for proposal, reason in cases:
            with self.subTest(reason=reason):
                plan = plan_pivot_candidates(
                    rows=[row],
                    pivot_proposals={key: proposal},
                    row_refinements=None,
                    config=self.config,
                    context=self.context,
                )[0]
                self.assertEqual(plan.fallback_reason, reason)

    def test_glm_scoring_head_contract_accepts_32_heads(self):
        config = PivotPlannerConfig(
            topk=2,
            candidate_budget=4,
            scoring_heads=32,
            min_context_tokens=1,
        )
        row = self.row(position=7)
        key = PivotProposalKey(self.identity, "qg-0", 0)
        plan = plan_pivot_candidates(
            rows=[row],
            pivot_proposals={key: tuple(range(4))},
            row_refinements=None,
            config=config,
            context=self.context,
        )[0]
        self.assertTrue(plan.accepted)
        self.assertEqual(plan.scoring_heads, 32)
        self.assertEqual(plan.candidates, (0, 1, 2, 3))

    def test_unsupported_routes_and_short_context_fall_back(self):
        row = self.row(position=3)
        key = PivotProposalKey(self.identity, "qg-0", 0)
        proposals = {key: tuple(range(8))}
        plan = plan_pivot_candidates(
            rows=[row],
            pivot_proposals=proposals,
            row_refinements=None,
            config=self.config,
            context=self.context,
        )[0]
        self.assertEqual(plan.fallback_reason, PivotFallbackReason.SHORT_CONTEXT)

        for context in [
            PivotPlannerContext(False, True, True, True),
            PivotPlannerContext(True, False, True, True),
            PivotPlannerContext(True, True, False, True),
            PivotPlannerContext(True, True, True, False),
        ]:
            with self.subTest(context=context):
                plan = plan_pivot_candidates(
                    rows=[self.row()],
                    pivot_proposals=proposals,
                    row_refinements=None,
                    config=self.config,
                    context=context,
                )[0]
                expected = (
                    PivotFallbackReason.FEATURE_DISABLED
                    if not context.enabled
                    else PivotFallbackReason.UNSUPPORTED_PATH
                )
                self.assertEqual(plan.fallback_reason, expected)

    def test_recall_audit_reports_every_true_topk_omission(self):
        row = self.row()
        key = PivotProposalKey(self.identity, "qg-0", 0)
        plan = plan_pivot_candidates(
            rows=[row],
            pivot_proposals={key: tuple(range(8))},
            row_refinements=None,
            config=self.config,
            context=self.context,
        )[0]
        audit = audit_pivot_recall(
            [plan],
            {0: (0, 1, 2, 99)},
        )
        self.assertEqual(audit.accepted_rows, 1)
        self.assertEqual(audit.evaluated_rows, 1)
        self.assertEqual(audit.severe_omissions, 1)
        self.assertEqual(audit.recall, 0.75)


if __name__ == "__main__":
    unittest.main()
