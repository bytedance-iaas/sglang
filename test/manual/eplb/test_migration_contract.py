"""CPU migration invariants for configurable EP and model dimensions."""

import random
import unittest

from sglang.srt.eplb.capacity_planner import metrics
from sglang.srt.eplb.moe_shape import MoeShape
from sglang.srt.eplb.multi_swap_cpu import plan


class MigrationContract(unittest.TestCase):
    def test_ep_size_and_per_rank_budget(self):
        for world in (8, 16, 32):
            for logical, hidden, topk in (
                (256, 4096, 6),
                (384, 5120, 6),
                (288, 4096, 8),
            ):
                shape = MoeShape(logical, logical + world, world, hidden, topk)
                mapping = list(range(logical)) + list(range(world))
                for budget in (1, 4):
                    counts = [
                        random.Random(i + logical).randrange(1500) for i in mapping
                    ]
                    result = plan(mapping, counts, budget, shape)
                    t0, b0 = metrics(counts, world=world)
                    t1, b1 = metrics(result["quota"], world=world)
                    self.assertLessEqual(max(b1), max(b0))
                    self.assertLessEqual(max(t1), max(t0))
                    self.assertTrue(all(0 <= n <= budget for n in result["counts"]))
                    for expert in range(logical):
                        self.assertEqual(
                            sum(n for e, n in zip(mapping, counts) if e == expert),
                            sum(
                                n
                                for e, n in zip(result["mapping"], result["quota"])
                                if e == expert
                            ),
                        )
                    for cp in result["copies"]:
                        self.assertEqual(
                            mapping[cp["source"]], result["mapping"][cp["destination"]]
                        )

    def test_invalid_slot_divisibility(self):
        with self.assertRaises(ValueError):
            MoeShape(256, 264, 16, 4096, 6)


if __name__ == "__main__":
    unittest.main()
