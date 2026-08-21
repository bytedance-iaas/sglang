"""CPU-only tests for the DSV4 memory-pool configurator."""

import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSV4DSparkDraftPoolBudget(CustomTestCase):
    @staticmethod
    def _make_kvc(*, pp_rank, pp_size=4, is_dspark=True, mode="prefill"):
        return SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_dspark=lambda: is_dspark),
            server_args=SimpleNamespace(disaggregation_mode=mode),
            ps=SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size),
        )

    def test_pd_prefill_budgets_draft_pool_only_on_final_pp_stage(self):
        should_budget = [
            DSV4PoolConfigurator._should_budget_draft_pool(
                self._make_kvc(pp_rank=rank)
            )
            for rank in range(4)
        ]

        self.assertEqual(should_budget, [False, False, False, True])

    def test_other_speculative_layouts_keep_existing_budgeting(self):
        cases = (
            self._make_kvc(pp_rank=0, is_dspark=False),
            self._make_kvc(pp_rank=0, mode="decode"),
            self._make_kvc(pp_rank=0, pp_size=1),
        )

        for kvc in cases:
            with self.subTest(kvc=kvc):
                self.assertTrue(
                    DSV4PoolConfigurator._should_budget_draft_pool(kvc)
                )


if __name__ == "__main__":
    unittest.main()
