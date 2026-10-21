"""CP bridge plumbing only; real convolution/state equivalence needs GPU tests."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear import kda_backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestKDAPrefillCPBridge(unittest.TestCase):
    def setUp(self):
        self._head_exchange_env = (
            envs.SGLANG_EXPERIMENTAL_KDA_PREFILL_CP_HEAD_EXCHANGE.override(False)
        )
        self._head_exchange_env.__enter__()
        self.addCleanup(self._head_exchange_env.__exit__, None, None, None)

    def test_non_cp_retains_original_arguments_and_result(self):
        args = [object() for _ in range(5)]
        result = object()
        calls = []

        def run(*received, **kwargs):
            calls.append((received, kwargs))
            return result

        backend = SimpleNamespace(_forward_extend_full_sequence=run)
        with patch.object(kda_backend, "is_cp_active", return_value=False):
            actual = kda_backend.KDAAttnBackend.forward_extend(
                backend, *args, test_marker="retained"
            )
        self.assertIs(actual, result)
        self.assertEqual(calls, [(tuple(args), {"test_marker": "retained"})])

    def test_cp_restores_order_and_preserves_dtype_and_metadata(self):
        # Uneven global length models interleave padding on the second rank.
        layer, batch = object(), SimpleNamespace(query_start_loc=object())
        qkv = torch.arange(35).reshape(7, 5).to(torch.bfloat16)
        a = torch.arange(21).reshape(1, 7, 3).float()
        b = torch.arange(14).reshape(1, 7, 2).to(torch.bfloat16)
        expected = torch.arange(42).reshape(1, 7, 2, 3).to(torch.bfloat16)
        core_calls = []

        def run(*received, **kwargs):
            core_calls.append((received, kwargs))
            return expected

        for rank in (0, 1):
            gathered = iter((qkv, a.movedim(1, 0), b.movedim(1, 0)))
            gather_calls = []

            def gather(local, received_batch):
                self.assertIs(received_batch, batch)
                full = next(gathered)
                torch.testing.assert_close(local, full[rank::2], rtol=0, atol=0)
                self.assertEqual(local.dtype, full.dtype)
                gather_calls.append(local)
                return full

            def shard(full, received_batch):
                self.assertIs(received_batch, batch)
                torch.testing.assert_close(full, expected.movedim(1, 0))
                return full[rank::2]

            with (
                patch.object(kda_backend, "is_cp_active", return_value=True),
                patch.object(kda_backend, "cp_materialize_global_token_order", gather),
                patch.object(kda_backend, "cp_shard_hidden_states", shard),
            ):
                result = kda_backend.KDAAttnBackend.forward_extend(
                    SimpleNamespace(_forward_extend_full_sequence=run),
                    layer,
                    batch,
                    qkv[rank::2],
                    a[:, rank::2],
                    b[:, rank::2],
                    test_marker="preserve",
                )
            torch.testing.assert_close(result, expected[:, rank::2], rtol=0, atol=0)
            self.assertEqual(len(gather_calls), 3)
            received, kwargs = core_calls[-1]
            self.assertIs(received[0], layer)
            self.assertIs(received[1], batch)
            for actual, original in zip(received[2:], (qkv, a, b), strict=True):
                torch.testing.assert_close(actual, original, rtol=0, atol=0)
                self.assertEqual(actual.dtype, original.dtype)
            self.assertEqual(kwargs, {"test_marker": "preserve"})

    def test_cp_rejects_tuple_qkv_before_collective(self):
        with (
            patch.object(kda_backend, "is_cp_active", return_value=True),
            patch.object(kda_backend, "cp_materialize_global_token_order") as gather,
        ):
            with self.assertRaisesRegex(ValueError, "packed QKV"):
                kda_backend.KDAAttnBackend.forward_extend(
                    SimpleNamespace(),
                    object(),
                    object(),
                    (),
                    torch.empty(1),
                    torch.empty(1),
                )
            gather.assert_not_called()

    def test_cp_rejects_gate_batch_before_collective(self):
        with (
            patch.object(kda_backend, "is_cp_active", return_value=True),
            patch.object(kda_backend, "cp_materialize_global_token_order") as gather,
        ):
            with self.assertRaisesRegex(ValueError, "packed batch axis"):
                kda_backend.KDAAttnBackend.forward_extend(
                    SimpleNamespace(),
                    object(),
                    object(),
                    torch.empty(3, 4),
                    torch.empty(2, 3, 4),
                    torch.empty(1, 3, 4),
                )
            gather.assert_not_called()

    def test_selected_head_exchange_bypasses_three_gather_bridge(self):
        result = object()
        plan = object()
        backend = SimpleNamespace()
        args = [object() for _ in range(5)]
        with (
            patch.object(kda_backend, "is_cp_active", return_value=True),
            patch.object(
                kda_backend.KDAAttnBackend,
                "_kda_prefill_cp_head_plan",
                return_value=plan,
            ) as select,
            patch.object(
                kda_backend.KDAAttnBackend,
                "_forward_extend_cp_head_exchange",
                return_value=result,
            ) as exchange,
            patch.object(kda_backend, "cp_materialize_global_token_order") as gather,
        ):
            actual = kda_backend.KDAAttnBackend.forward_extend(
                backend, *args, test_marker="not_forwarded"
            )
        self.assertIs(actual, result)
        select.assert_called_once_with(backend, *args)
        exchange.assert_called_once_with(backend, *args, plan)
        gather.assert_not_called()


if __name__ == "__main__":
    unittest.main()
