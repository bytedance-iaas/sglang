import unittest

import torch

from sglang.srt.layers.attention.dsa.dosa_real_reference import (
    DosaRealConfig,
    dosa_real_reference,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDosaRealReference(unittest.TestCase):
    def test_randomized_bound_and_recovery_property(self):
        generator = torch.Generator().manual_seed(9261013990)
        for heads in (4, 8, 16, 32):
            for trial in range(8):
                with self.subTest(heads=heads, trial=trial):
                    dim = (4, 16, 128)[trial % 3]
                    keys_n = 80 + trial
                    queries = torch.randn((heads, dim), generator=generator)
                    weights = torch.randn((heads,), generator=generator)
                    weights[weights == 0] = 0.125
                    keys = torch.randn((keys_n, dim), generator=generator)
                    result = dosa_real_reference(
                        queries=queries,
                        weights=weights,
                        keys=keys,
                        config=DosaRealConfig(
                            topk=8,
                            retained_heads=max(1, heads // 2),
                            seed_size=24,
                            candidate_capacity=None if trial % 3 else trial,
                        ),
                        visible_start=trial % 5,
                        visible_end=keys_n - (trial % 7),
                    )
                    self.assertGreaterEqual(
                        result.minimum_bound_margin,
                        -1e-9,
                    )
                    self.assertTrue(result.exact_match)

    def test_random_certificate_dominates_and_selects_exact_topk(self):
        generator = torch.Generator().manual_seed(20261115)
        for heads, retained in ((8, 4), (12, 6)):
            with self.subTest(heads=heads):
                queries = torch.randn((heads, 16), generator=generator)
                weights = torch.randn((heads,), generator=generator)
                weights[weights == 0] = 0.25
                keys = torch.randn((96, 16), generator=generator)
                result = dosa_real_reference(
                    queries=queries,
                    weights=weights,
                    keys=keys,
                    config=DosaRealConfig(
                        topk=8,
                        retained_heads=retained,
                        seed_size=24,
                    ),
                    visible_start=3,
                    visible_end=91,
                )
                self.assertEqual(result.arithmetic_contract, "real_arithmetic_only")
                self.assertGreaterEqual(result.minimum_bound_margin, -1e-9)
                self.assertTrue(result.exact_match)

    def test_equal_bound_and_threshold_survive_pruning(self):
        queries = torch.eye(2)
        weights = torch.tensor([1.0, -1.0])
        keys = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        result = dosa_real_reference(
            queries=queries,
            weights=weights,
            keys=keys,
            config=DosaRealConfig(topk=1, retained_heads=1, seed_size=1),
            seed_ids=[0],
        )
        self.assertIn(1, result.candidate_ids)
        self.assertTrue(result.exact_match)
        self.assertEqual(result.selected, (0,))

    def test_candidate_overflow_uses_full_domain_recovery(self):
        queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        weights = torch.ones(2)
        keys = torch.ones((32, 2))
        result = dosa_real_reference(
            queries=queries,
            weights=weights,
            keys=keys,
            config=DosaRealConfig(
                topk=4,
                retained_heads=1,
                seed_size=4,
                candidate_capacity=2,
            ),
        )
        self.assertTrue(result.recovered_full_domain)
        self.assertEqual(result.candidate_ids, tuple(range(32)))
        self.assertTrue(result.exact_match)

    def test_invalid_seed_and_nonfinite_inputs_fail_closed(self):
        queries = torch.eye(2)
        weights = torch.ones(2)
        keys = torch.eye(2)
        with self.assertRaisesRegex(ValueError, "distinct"):
            dosa_real_reference(
                queries=queries,
                weights=weights,
                keys=keys,
                config=DosaRealConfig(topk=2, retained_heads=1, seed_size=2),
                seed_ids=[0, 0],
            )
        bad_keys = keys.clone()
        bad_keys[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            dosa_real_reference(
                queries=queries,
                weights=weights,
                keys=bad_keys,
                config=DosaRealConfig(topk=1, retained_heads=1, seed_size=1),
            )


if __name__ == "__main__":
    unittest.main()
