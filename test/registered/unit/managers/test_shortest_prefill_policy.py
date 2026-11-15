import unittest
from array import array
from unittest.mock import patch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import CacheAwarePolicy, SchedulePolicy
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_req(rid: str, uncached: int, *, cached: int = 0, arrived: int = 0):
    req = Req(
        rid,
        "",
        array("q", range(uncached + cached)),
        SamplingParams(max_new_tokens=1),
    )
    req.full_untruncated_fill_ids = list(req.origin_input_ids)
    req.num_matched_prefix_tokens = cached
    req.prefix_indices = list(range(cached))
    req.arrival_processed_tokens = arrived
    req.time_stats.wait_queue_entry_time = float(arrived)
    return req


class TestShortestPrefillPolicy(CustomTestCase):
    def create_policy(self, aging: float = 0.0):
        return SchedulePolicy(
            policy="shortest-prefill-first",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
            shortest_prefill_aging_factor=aging,
        )

    def test_orders_by_residual_work_and_arrival(self):
        policy = self.create_policy()
        cached = make_req("cached", 16, cached=4096, arrived=3)
        older = make_req("older", 32, arrived=1)
        newer = make_req("newer", 32, arrived=2)
        long = make_req("long", 1024, arrived=0)
        queue = [long, newer, older, cached]
        with patch.object(policy, "_compute_prefix_matches", return_value=set()):
            policy.calc_priority(queue, processed_tokens=10)
        self.assertEqual(
            [req.rid for req in queue], ["cached", "older", "newer", "long"]
        )

    def test_token_aging_promotes_waiting_long_request(self):
        policy = self.create_policy(aging=0.5)
        old_long = make_req("old-long", 1000, arrived=0)
        new_short = make_req("new-short", 10, arrived=3000)
        queue = [old_long, new_short]
        policy._sort_by_shortest_prefill(queue, set(), processed_tokens=1000)
        self.assertEqual([req.rid for req in queue], ["new-short", "old-long"])
        policy._sort_by_shortest_prefill(queue, set(), processed_tokens=3000)
        self.assertEqual([req.rid for req in queue], ["old-long", "new-short"])

    def test_chunk_limit_reserves_pages_and_bounds_bypass(self):
        policy = self.create_policy(aging=0.5)
        continuation = make_req("continuation", 16384, arrived=0)
        waiting = [make_req("a", 512), make_req("b", 1024)]
        self.assertEqual(
            policy.shortest_prefill_chunk_limit(
                continuation,
                waiting,
                4096,
                256,
                processed_tokens=0,
            ),
            2560,
        )

        # Once the continuation's aged rank wins, a fresh short request no
        # longer reserves room; the continuation receives the full chunk.
        fresh = make_req("fresh", 10, arrived=40000)
        self.assertIsNone(
            policy.shortest_prefill_chunk_limit(
                continuation,
                [fresh],
                4096,
                256,
                processed_tokens=40000,
            )
        )

    def test_multi_round_timeline_recomputes_hits_and_bounds_aging(self):
        policy = self.create_policy(aging=0.5)
        page_size = 64
        budget = 256
        processed_tokens = 0
        continuation = make_req("continuation", 1024, arrived=0)
        waiting = [
            make_req("short-0", 64, arrived=0),
            make_req("host-hit", 320, arrived=0),
        ]
        timeline = []

        def run_round(round_id):
            nonlocal processed_tokens
            policy._sort_by_shortest_prefill(
                waiting, set(), processed_tokens=processed_tokens
            )
            chunk_limit = policy.shortest_prefill_chunk_limit(
                continuation,
                waiting,
                budget,
                page_size,
                processed_tokens=processed_tokens,
            )
            chunk_tokens = budget if chunk_limit is None else chunk_limit
            chunk_tokens = min(
                chunk_tokens,
                len(continuation.full_untruncated_fill_ids)
                - len(continuation.prefix_indices),
            )
            continuation.prefix_indices = list(
                range(len(continuation.prefix_indices) + chunk_tokens)
            )

            remaining = budget - chunk_tokens
            admitted = []
            for req in list(waiting):
                work = policy._shortest_prefill_work(req)
                charge = -(-work // page_size) * page_size
                if charge > remaining:
                    break
                remaining -= charge
                admitted.append(req.rid)
                waiting.remove(req)

            processed_tokens += chunk_tokens + (budget - chunk_tokens - remaining)
            timeline.append(
                (
                    round_id,
                    chunk_tokens,
                    tuple(admitted),
                    len(continuation.prefix_indices),
                )
            )

        run_round(1)
        self.assertEqual(timeline[-1], (1, 192, ("short-0",), 192))

        # Model an asynchronous HiCache load-back before the next round. The
        # policy must rank the refreshed residual work, not the stale 320-token
        # miss observed in round 1.
        waiting[0].num_matched_prefix_tokens = 256
        waiting[0].prefix_indices = list(range(256))
        waiting.append(make_req("short-1", 64, arrived=processed_tokens))
        run_round(2)
        self.assertEqual(timeline[-1], (2, 128, ("host-hit", "short-1"), 320))

        for round_id in (3, 4):
            waiting.append(
                make_req(f"short-{round_id - 1}", 64, arrived=processed_tokens)
            )
            run_round(round_id)
            self.assertGreaterEqual(timeline[-1][1], page_size)
            self.assertEqual(timeline[-1][2], (f"short-{round_id - 1}",))

        waiting.append(make_req("short-4", 64, arrived=processed_tokens))
        run_round(5)
        self.assertEqual(timeline[-1], (5, 256, (), 960))

        # Aging eventually makes the original continuation outrank a newly
        # arrived short request. Removing the cancelled continuation leaves no
        # policy-side reservation: the waiting request can run immediately.
        cancelled = continuation
        continuation = None
        self.assertEqual(cancelled.rid, "continuation")
        policy._sort_by_shortest_prefill(
            waiting, set(), processed_tokens=processed_tokens
        )
        self.assertEqual([req.rid for req in waiting], ["short-4"])
        self.assertLessEqual(policy._shortest_prefill_work(waiting[0]), budget)

    def test_invalid_policy_configuration_is_rejected(self):
        for schedule_policy, factor in [
            ("shortest-prefill-first", -0.5),
            ("shortest-prefill-first", float("nan")),
            ("shortest-prefill-first", float("inf")),
            ("fcfs", 0.5),
        ]:
            with self.subTest(schedule_policy=schedule_policy, factor=factor):
                with self.assertRaises(ValueError):
                    SchedulePolicy(
                        policy=schedule_policy,
                        tree_cache=RadixCache.create_simulated(),
                        enable_hierarchical_cache=True,
                        enable_priority_scheduling=False,
                        schedule_low_priority_values_first=False,
                        shortest_prefill_aging_factor=factor,
                    )

    def test_requires_prefix_cache(self):
        with self.assertRaises(ValueError):
            SchedulePolicy(
                policy=CacheAwarePolicy.SHORTEST_PREFILL_FIRST.value,
                tree_cache=RadixCache.create_simulated(disable=True),
                enable_hierarchical_cache=False,
                enable_priority_scheduling=False,
                schedule_low_priority_values_first=False,
            )


if __name__ == "__main__":
    unittest.main()
