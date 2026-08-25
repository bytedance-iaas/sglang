"""Unit tests for SWA reclamation and decode retraction fail-safes."""

import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import (  # noqa: E402
    FINISH_ABORT,
    NewTokenRatioTracker,
    ScheduleBatch,
    release_req,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestScheduleBatchSwaRetraction(CustomTestCase):
    def test_check_decode_mem_forces_swa_reclamation_before_retraction(self):
        batch = ScheduleBatch(reqs=[])
        batch.forward_mode = MagicMock()
        batch.forward_mode.is_decode.return_value = True
        batch.tree_cache = MagicMock()
        batch.tree_cache.supports_swa.return_value = True
        batch.token_to_kv_pool_allocator = MagicMock()
        batch.token_to_kv_pool_allocator.available_size.side_effect = [3, 8]
        batch.new_tokens_required_next_decode = MagicMock(return_value=8)
        batch.maybe_evict_swa = MagicMock()

        with patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache") as evict:
            self.assertTrue(batch.check_decode_mem())

        batch.maybe_evict_swa.assert_called_once_with(force=True)
        self.assertEqual(evict.call_count, 2)

    def test_force_swa_reclamation_ignores_periodic_interval(self):
        req = SimpleNamespace(decode_batch_idx=1, seqlen=32)
        batch = ScheduleBatch(reqs=[req])
        batch.forward_mode = MagicMock()
        batch.forward_mode.is_decode.return_value = True
        batch.tree_cache = MagicMock(sliding_window_size=16)
        batch.tree_cache.supports_swa.return_value = True
        batch.forward_iter = 1
        batch._evict_swa = MagicMock()

        with (
            patch(
                "sglang.srt.managers.schedule_batch.envs.SGLANG_SWA_EVICTION_INTERVAL.get",
                return_value=8,
            ),
            patch(
                "sglang.srt.managers.schedule_batch.envs."
                "SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW.get",
                return_value=False,
            ),
            patch("sglang.srt.managers.schedule_batch.get_server_args"),
            patch("sglang.srt.managers.schedule_batch.maybe_evict_dsv4_state"),
        ):
            batch.maybe_evict_swa()
            batch._evict_swa.assert_not_called()

            batch.maybe_evict_swa(force=True)

        batch._evict_swa.assert_called_once_with(req, req.seqlen - 1)

    def test_unsupported_backup_aborts_only_retracted_request(self):
        kept_req = MagicMock(rid="kept")
        aborted_req = MagicMock(rid="aborted")
        batch = ScheduleBatch(reqs=[kept_req, aborted_req])
        batch._get_decode_retraction_order = MagicMock(return_value=[0, 1])
        batch.check_decode_mem = MagicMock(return_value=True)
        batch.release_req = MagicMock(return_value=False)
        batch.filter_batch = MagicMock()
        server_args = MagicMock()

        with patch.object(
            NewTokenRatioTracker,
            "estimate_new_token_ratio_after_retract",
            return_value=0.25,
        ):
            retracted, ratio, reqs_to_abort = batch.retract_decode(server_args)

        self.assertEqual(retracted, [])
        self.assertEqual(ratio, 0.25)
        self.assertEqual(reqs_to_abort, [aborted_req])
        self.assertIsInstance(aborted_req.to_finish, FINISH_ABORT)
        self.assertEqual(
            aborted_req.to_finish.status_code, HTTPStatus.INTERNAL_SERVER_ERROR
        )
        batch.release_req.assert_called_once_with(
            1,
            1,
            server_args,
            abort_on_unsupported_backup=True,
        )
        batch.filter_batch.assert_called_once_with(keep_indices=[0])

    def test_last_request_oom_skips_unusable_backup(self):
        req = MagicMock(rid="last")
        batch = ScheduleBatch(reqs=[req])
        batch._get_decode_retraction_order = MagicMock(return_value=[0])
        batch.check_decode_mem = MagicMock(return_value=False)
        batch.release_req = MagicMock(return_value=True)
        batch.filter_batch = MagicMock()
        server_args = MagicMock()

        with patch.object(
            NewTokenRatioTracker,
            "estimate_new_token_ratio_after_retract",
            return_value=0.0,
        ):
            retracted, _, reqs_to_abort = batch.retract_decode(server_args)

        self.assertEqual(retracted, [])
        self.assertEqual(reqs_to_abort, [req])
        batch.release_req.assert_called_once_with(0, 0, server_args, offload_kv=False)

    def test_release_req_handles_unsupported_backup_when_requested(self):
        req = MagicMock(rid="unsupported")
        req.finished.return_value = False
        req.offload_kv_cache.side_effect = NotImplementedError
        server_args = SimpleNamespace(disaggregation_mode="decode")

        with (
            patch("sglang.srt.managers.schedule_batch.release_kv_cache") as release,
            patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache"),
        ):
            backup_succeeded = release_req(
                req=req,
                remaing_req_count=1,
                server_args=server_args,
                req_to_token_pool=MagicMock(),
                token_to_kv_pool_allocator=MagicMock(),
                tree_cache=MagicMock(),
                hisparse_coordinator=None,
                abort_on_unsupported_backup=True,
            )

        self.assertFalse(backup_succeeded)
        self.assertIsNone(req.kv_cache_cpu)
        release.assert_called_once()
        req.reset_for_retract.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
