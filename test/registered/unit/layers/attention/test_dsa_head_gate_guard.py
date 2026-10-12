import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.utils import is_cuda, is_hip
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

HELPERS = ("logits_head_gate_graph", "scale_head_gate_graph")


class TestDsaHeadGateGuard(CustomTestCase):
    """The head-gate helpers are defined in one module and imported in another.

    Both sides carry their own platform condition, so the two can drift: widen
    the import without widening the definition and every DSA model fails to
    import on the platform in between. That is a startup failure far from its
    cause, and no CUDA runner can see it.
    """

    def test_definition_and_import_agree(self):
        from sglang.srt.layers.attention.dsa import dsa_indexer, head_gate

        for name in HELPERS:
            with self.subTest(helper=name):
                self.assertEqual(
                    name in vars(head_gate),
                    name in vars(dsa_indexer),
                    f"{name} is defined on one side of the guard but not the other",
                )

    def test_helpers_exist_on_cuda_and_hip(self):
        from sglang.srt.layers.attention.dsa import head_gate

        expected = is_cuda() or is_hip()
        for name in HELPERS:
            with self.subTest(helper=name):
                self.assertEqual(name in vars(head_gate), expected)

    def test_kpool_cuda_head_gate_uses_bf16_operands_and_fp32_output(self):
        from sglang.srt.layers.attention.dsa import dsa_indexer_kpool

        x = torch.empty((3, 16), dtype=torch.bfloat16)
        weight = torch.empty((4, 16), dtype=torch.bfloat16)
        output = torch.empty((3, 4), dtype=torch.float32)
        indexer = SimpleNamespace(weights_proj=SimpleNamespace(weight=weight))

        with (
            patch.object(dsa_indexer_kpool, "is_cuda", return_value=True),
            patch.object(torch, "mm", return_value=output) as mm,
        ):
            actual = dsa_indexer_kpool.IndexerKPool._weights_proj_bf16_in_fp32_out(
                indexer, x
            )

        self.assertIs(actual, output)
        mm.assert_called_once()
        args, kwargs = mm.call_args
        self.assertIs(args[0], x)
        self.assertTrue(torch.equal(args[1], weight.t()))
        self.assertEqual(kwargs, {"out_dtype": torch.float32})

    def test_kpool_non_cuda_head_gate_keeps_fp32_result(self):
        from sglang.srt.layers.attention.dsa import dsa_indexer_kpool

        x = torch.empty((3, 16), dtype=torch.bfloat16)
        output = torch.empty((3, 4), dtype=torch.bfloat16)
        projection = Mock(return_value=(output, None))
        indexer = SimpleNamespace(weights_proj=projection)

        with patch.object(dsa_indexer_kpool, "is_cuda", return_value=False):
            actual = dsa_indexer_kpool.IndexerKPool._weights_proj_bf16_in_fp32_out(
                indexer, x
            )

        projection.assert_called_once_with(x)
        self.assertEqual(actual.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
