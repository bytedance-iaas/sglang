"""CPU coverage for CUDA graph executable dedup lifecycle."""

import unittest
from unittest.mock import Mock, call, patch

import sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin as dedup_module
from sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin import (
    DedupedCudaGraphRegistry,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _CapturedGraph:
    def __init__(self, raw_graph):
        self._raw_graph = raw_graph

    def raw_cuda_graph(self):
        return self._raw_graph


class TestDedupedCudaGraphRegistry(CustomTestCase):
    def test_compat_exec_is_lazy_until_a_duplicate_graph_arrives(self):
        registry = DedupedCudaGraphRegistry()
        registry.instantiate = Mock(side_effect=[1001, 1002])
        registry.log_new_signature = Mock()

        with (
            patch.object(dedup_module, "graph_signature", return_value=("same",)),
            patch.object(
                dedup_module, "dedup_update", return_value=(True, "")
            ) as update,
        ):
            first = registry.register(_CapturedGraph(101))

            self.assertEqual(registry.instantiate.call_args_list, [call(101)])
            self.assertIsNone(first.group.compat_exec)

            second = registry.register(_CapturedGraph(102))

        self.assertIs(first.group, second.group)
        self.assertEqual(
            registry.instantiate.call_args_list,
            [call(101), call(101)],
        )
        self.assertEqual(first.group.compat_exec, 1002)
        update.assert_called_once_with(1002, 102)
        registry.log_new_signature.assert_called_once_with(("same",))

    def test_unique_signatures_never_allocate_compat_execs(self):
        registry = DedupedCudaGraphRegistry()
        registry.instantiate = Mock(side_effect=[1001, 1002])
        registry.log_new_signature = Mock()

        with patch.object(
            dedup_module,
            "graph_signature",
            side_effect=[("first",), ("second",)],
        ):
            first = registry.register(_CapturedGraph(101))
            second = registry.register(_CapturedGraph(102))

        self.assertEqual(
            registry.instantiate.call_args_list,
            [call(101), call(102)],
        )
        self.assertIsNone(first.group.compat_exec)
        self.assertIsNone(second.group.compat_exec)
        self.assertEqual(registry.log_new_signature.call_count, 2)


if __name__ == "__main__":
    unittest.main()
