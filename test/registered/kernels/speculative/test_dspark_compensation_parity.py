import unittest

import torch

from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    schedule_verify_lens_topk,
    schedule_verify_lens_topk_triton,
)
from sglang.srt.speculative.dspark_components.dspark_planner import DSparkScheduleConfig
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, suite="base-b-test-1-gpu-small")


class TestCompensationParity(CustomTestCase):
    def test_weighted_topk_matches_reference(self):
        rng = torch.Generator().manual_seed(123)
        for bs in [1, 2, 7, 16, 32]:
            for gamma in [1, 2, 6]:
                for variant in ["random", "ties", "zero", "threshold"]:
                    confidence = (
                        torch.rand(bs, gamma, generator=rng)
                        if variant == "random"
                        else torch.full(
                            (bs, gamma),
                            {"ties": 0.5, "zero": 0.0, "threshold": 0.1}[variant],
                        )
                    )
                    for weights in [
                        None,
                        torch.ones(bs),
                        1 + torch.rand(bs, generator=rng) * 0.5,
                    ]:
                        for budget in sorted(
                            {0, 1, bs, bs * gamma // 2, bs * gamma, bs * (gamma + 1)}
                        ):
                            with self.subTest(
                                bs=bs, gamma=gamma, variant=variant, budget=budget
                            ):
                                cfg = DSparkScheduleConfig(
                                    gamma=gamma,
                                    survival_eps=(
                                        0.1 if variant == "threshold" else 1e-6
                                    ),
                                )
                                expected = schedule_verify_lens_topk(
                                    confidence=confidence,
                                    budget=budget,
                                    cfg=cfg,
                                    request_weights=weights,
                                )
                                actual = schedule_verify_lens_topk_triton(
                                    confidence=confidence.cuda(),
                                    budget=budget,
                                    cfg=cfg,
                                    request_weights=(
                                        None if weights is None else weights.cuda()
                                    ),
                                ).cpu()
                                self.assertTrue(
                                    torch.equal(actual, expected), (actual, expected)
                                )


if __name__ == "__main__":
    unittest.main()
