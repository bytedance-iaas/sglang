import types
import unittest

import torch

from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    schedule_verify_lens_topk,
)
from sglang.srt.speculative.dspark_components.dspark_compensation import (
    CappedRequestCompensation,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
    DSparkVerifyPlanner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestCompensation(CustomTestCase):
    def test_pool_binding_preserves_credit_and_rejects_resize(self):
        state = CappedRequestCompensation(pool_size=0, device="cpu", max_boost=0.25)
        self.assertEqual(state.credit.numel(), 0)
        state.initialize_pool(4)
        state.credit[2] = 3
        state.initialize_pool(4)
        self.assertEqual(state.credit.tolist(), [0, 0, 3, 0])
        with self.assertRaises(ValueError):
            state.initialize_pool(5)
        with self.assertRaises(ValueError):
            state.initialize_pool(0)

    def test_first_verify_binds_late_created_request_pool(self):
        planner = DSparkVerifyPlanner.__new__(DSparkVerifyPlanner)
        planner.model_runner = types.SimpleNamespace(req_to_token_pool=None)
        planner._compensation = CappedRequestCompensation(
            pool_size=0, device="cpu", max_boost=0.5
        )
        # This is the scheduler ordering used by the full model: the pool is
        # attached after constructing the worker, before serving requests.
        planner.model_runner.req_to_token_pool = types.SimpleNamespace(
            req_to_token=torch.empty((4, 8), dtype=torch.int32)
        )
        slots = torch.tensor([2])
        reqs = [object()]
        planner.prepare_compensation(
            reqs=reqs, req_pool_indices=slots, req_pool_indices_cpu=slots
        )
        self.assertEqual(planner.compensation_weights.tolist(), [1.0])
        planner._compensation.credit[2] = 4
        planner.prepare_compensation(
            reqs=reqs, req_pool_indices=slots, req_pool_indices_cpu=slots
        )
        self.assertEqual(planner.compensation_weights.tolist(), [1.25])

    def test_cap_hit_rejection_and_slot_reuse(self):
        state = CappedRequestCompensation(pool_size=4, device="cpu", max_boost=0.5)
        reqs = [
            types.SimpleNamespace(kv=types.SimpleNamespace(req_pool_idx=i))
            for i in [1, 3]
        ]
        slots = torch.tensor([1, 3])
        _, weights = state.prepare(
            reqs=reqs, req_pool_indices=slots, req_pool_indices_cpu=slots
        )
        self.assertEqual(weights.tolist(), [1, 1])
        for _ in range(12):
            state.update(
                req_pool_indices=slots,
                verify_lens=torch.tensor([2, 7]),
                commit_lens=torch.tensor([2, 7]),
                full_width=7,
            )
        credit, weights = state.prepare(
            reqs=reqs[::-1],
            req_pool_indices=slots.flip(0),
            req_pool_indices_cpu=slots.flip(0),
        )
        self.assertEqual(credit.tolist(), [0, 8])
        self.assertEqual(weights.tolist(), [1, 1.5])
        state.update(
            req_pool_indices=slots,
            verify_lens=torch.tensor([3, 7]),
            commit_lens=torch.tensor([1, 7]),
            full_width=7,
        )
        self.assertEqual(state.credit[1].item(), 4)
        # New object at the same pool slot must not inherit credit.
        reqs[0] = types.SimpleNamespace(kv=types.SimpleNamespace(req_pool_idx=1))
        credit, _ = state.prepare(
            reqs=reqs, req_pool_indices=slots, req_pool_indices_cpu=slots
        )
        self.assertEqual(credit.tolist(), [0, 0])
        self.assertEqual(set(state._owners), {1, 3})
        state.prepare(
            reqs=[],
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            req_pool_indices_cpu=torch.empty(0, dtype=torch.int64),
        )
        self.assertFalse(state._owners)

    def test_fixed_budget_reallocates_and_preserves_eligibility(self):
        cfg = DSparkScheduleConfig(gamma=2, survival_eps=0.1)
        confidence = torch.tensor([[0.9, 0.9], [0.6, 0.6]])
        base = schedule_verify_lens_topk(confidence=confidence, budget=2, cfg=cfg)
        fair = schedule_verify_lens_topk(
            confidence=confidence,
            budget=2,
            cfg=cfg,
            request_weights=torch.tensor([1.0, 1.5]),
        )
        self.assertEqual(base.tolist(), [3, 1])
        self.assertEqual(fair.tolist(), [2, 2])
        confidence = torch.tensor([[0.9, 0.05], [0.05, 0.9]])
        fair = schedule_verify_lens_topk(
            confidence=confidence,
            budget=4,
            cfg=cfg,
            request_weights=torch.tensor([2.0, 2.0]),
        )
        self.assertEqual(fair.tolist(), [2, 1])

    def test_unity_weights_and_budget_conservation(self):
        rng = torch.Generator().manual_seed(42)
        for bs in [1, 3, 32]:
            c = torch.rand(bs, 6, generator=rng)
            for budget in [0, 1, bs, 3 * bs, 6 * bs, 7 * bs]:
                cfg = DSparkScheduleConfig(gamma=6)
                base = schedule_verify_lens_topk(confidence=c, budget=budget, cfg=cfg)
                unity = schedule_verify_lens_topk(
                    confidence=c, budget=budget, cfg=cfg, request_weights=torch.ones(bs)
                )
                self.assertTrue(torch.equal(base, unity))
                fair = schedule_verify_lens_topk(
                    confidence=c,
                    budget=budget,
                    cfg=cfg,
                    request_weights=1 + torch.rand(bs, generator=rng),
                )
                self.assertEqual(int(fair.sum()), int(base.sum()))
                self.assertTrue(bool(((fair >= 1) & (fair <= 7)).all()))

    def test_invalid_boost(self):
        for boost in [0, -1, 1.1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                CappedRequestCompensation(pool_size=4, device="cpu", max_boost=boost)


if __name__ == "__main__":
    unittest.main()
