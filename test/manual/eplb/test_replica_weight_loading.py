"""Weight loading must include replicas hidden by rank-local dispatch preference."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import List

import torch


def load_method():
    root = Path(__file__).resolve().parents[3]
    path = root / "python/sglang/srt/eplb/expert_location.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ExpertLocationMetadata"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "logical_to_all_physical"
    )
    namespace = {"List": List, "torch": torch}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[method.name]


class ReplicaWeightLoadingTest(unittest.TestCase):
    def test_load_all_physical_slots_even_when_dispatch_prefers_one(self):
        # Rank 0 owns physical slots 0..2. Expert 0 occupies both slots 0 and 1.
        # Local preference may choose slot 0, but a remote rank can choose 1.
        metadata = SimpleNamespace(
            physical_to_logical_map_cpu=torch.tensor([[0, 0, 1, 2, 3, 3]]),
            logical_to_all_physical_map_cpu=torch.tensor(
                [
                    [
                        [0, -1, -1, -1, -1, -1],
                        [2, -1, -1, -1, -1, -1],
                        [3, -1, -1, -1, -1, -1],
                        [4, 5, -1, -1, -1, -1],
                    ]
                ]
            ),
            num_logical_experts=4,
        )
        method = load_method()
        self.assertEqual(method(metadata, 0, 0), [0, 1])
        self.assertEqual(method(metadata, 0, 1), [2])
        self.assertEqual(method(metadata, 0, 3), [4, 5])
        self.assertEqual(method(metadata, 0, 0, True), [0, 4])
        self.assertEqual(method(metadata, 1, 0), [0])


if __name__ == "__main__":
    unittest.main()
