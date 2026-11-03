"""A later physical-slot remap must not relabel earlier gate statistics."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_distribution as recorder


class RecordedMapping(unittest.TestCase):
    def test_capture_logical_counts_before_remap(self):
        metadata = SimpleNamespace(
            num_layers=1,
            num_physical_experts=4,
            num_local_physical_experts=2,
            num_logical_experts=3,
            physical_to_logical_map=torch.tensor([[0, 1, 2, 0]]),
        )
        cfg = SimpleNamespace(moe=SimpleNamespace(enable_expert_migration=True))
        with (
            patch.object(recorder, "get_exec", return_value=cfg),
            patch.object(recorder, "get_device", return_value="cpu"),
        ):
            gatherer = recorder._SelectExpertsSinglePassGatherer(metadata, 0)
            gatherer.on_select_experts(0, torch.tensor([[0, 1, 3, -1]]))
            metadata.physical_to_logical_map.copy_(torch.tensor([[1, 0, 2, 0]]))
            sample = gatherer.collect()
            self.assertEqual(sample["global_logical_count"].tolist(), [[2, 1, 0]])
            gatherer.reset()
            self.assertEqual(gatherer.collect()["global_logical_count"].sum(), 0)


if __name__ == "__main__":
    unittest.main()
