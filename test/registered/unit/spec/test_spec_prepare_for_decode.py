"""Unit tests for common speculative decode preparation."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.speculative.spec_utils import spec_prepare_for_decode  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestSpecPrepareForDecode(CustomTestCase):
    def _run_prepare(self, *, is_dflash_family: bool):
        events = []
        req = SimpleNamespace(decode_batch_idx=4)
        batch = SimpleNamespace(
            reqs=[req],
            spec_algorithm=SimpleNamespace(is_dflash_family=lambda: is_dflash_family),
            spec_info=SimpleNamespace(),
        )
        batch.maybe_evict_swa = MagicMock(
            side_effect=lambda: events.append(("evict", req.decode_batch_idx))
        )
        batch.spec_info.prepare_for_decode = MagicMock(
            side_effect=lambda _: events.append(("dflash", req.decode_batch_idx))
        )
        eagle_prepare = MagicMock(
            side_effect=lambda _: events.append(("eagle", req.decode_batch_idx))
        )
        server_args = SimpleNamespace(enable_mamba_extra_buffer_lazy=lambda: False)

        with (
            patch(
                "sglang.srt.speculative.spec_utils.get_server_args",
                return_value=server_args,
            ),
            patch(
                "sglang.srt.speculative.eagle_utils.eagle_prepare_for_decode",
                eagle_prepare,
            ),
        ):
            spec_prepare_for_decode(batch)

        return batch, eagle_prepare, events, req

    def test_dflash_gets_common_swa_bookkeeping_before_prepare(self):
        batch, eagle_prepare, events, req = self._run_prepare(is_dflash_family=True)

        self.assertEqual(req.decode_batch_idx, 5)
        self.assertEqual(events, [("evict", 4), ("dflash", 5)])
        batch.spec_info.prepare_for_decode.assert_called_once_with(batch)
        eagle_prepare.assert_not_called()

    def test_eagle_keeps_single_common_bookkeeping_tick(self):
        batch, eagle_prepare, events, req = self._run_prepare(is_dflash_family=False)

        self.assertEqual(req.decode_batch_idx, 5)
        self.assertEqual(events, [("evict", 4), ("eagle", 5)])
        batch.spec_info.prepare_for_decode.assert_not_called()
        eagle_prepare.assert_called_once_with(batch)


if __name__ == "__main__":
    unittest.main()
