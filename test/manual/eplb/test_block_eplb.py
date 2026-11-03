"""Check per-step Block-EPLB acceptance and unchanged native seed semantics."""

import unittest

import numpy as np
import torch

from sglang.srt.eplb.eplb_algorithms import block as block_eplb
from sglang.srt.eplb.eplb_algorithms import deepseek


class PlacementContract(unittest.TestCase):
    def test_each_observed_step_and_replica_count(self):
        for logical in (256, 384, 288):
            physical = logical + 8
            for empty in (False, True):
                counts = torch.randint(
                    0,
                    1000,
                    (3, 2, logical),
                    generator=torch.Generator().manual_seed(logical),
                )
                if empty:
                    counts.zero_()
                seed, _, seed_replicas = deepseek.rebalance_experts(
                    weight=counts.sum(0),
                    num_replicas=physical,
                    num_groups=1,
                    num_nodes=1,
                    num_gpus=8,
                    enable_hierarchical=False,
                )
                mapping, inverse, replicas = block_eplb.rebalance_experts(
                    counts, physical, physical // 8, 1, 1, 64, False
                )
                self.assertTrue(torch.equal(replicas, seed_replicas))
                for layer in range(2):
                    samples = counts[:, layer].numpy()
                    before = block_eplb.tails(
                        block_eplb.physical_loads(samples, seed[layer].numpy()), 8, 64
                    )
                    after = block_eplb.tails(
                        block_eplb.physical_loads(samples, mapping[layer].numpy()),
                        8,
                        64,
                    )
                    self.assertTrue(np.all(after <= before), (logical, empty, layer))
                    self.assertEqual(mapping[layer].numel(), physical)
                    for expert in range(logical):
                        ids = torch.where(mapping[layer] == expert)[0]
                        self.assertEqual(len(ids), int(replicas[layer, expert]))
                        self.assertTrue(
                            torch.equal(
                                inverse[layer, expert, : len(ids)].sort().values, ids
                            )
                        )

    def test_tradeoff_on_one_step_is_rejected(self):
        self.assertFalse(
            block_eplb.acceptable(
                np.array([[100, 1000], [100, 1000]]), np.array([[90, 900], [101, 1000]])
            )
        )


if __name__ == "__main__":
    unittest.main()
