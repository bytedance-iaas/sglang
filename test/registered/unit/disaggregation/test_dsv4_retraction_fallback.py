import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers.schedule_batch import ScheduleBatch


def _make_req(rid):
    return SimpleNamespace(
        rid=rid,
        seqlen=4096,
        origin_input_ids=[1, 2],
        output_ids=[3, 4],
        to_finish=None,
        offload_kv_cache=MagicMock(),
        reset_for_retract=MagicMock(),
    )


class TestDSV4RetractionFallback(unittest.TestCase):
    def test_release_can_skip_decode_cpu_snapshot(self):
        req = _make_req("req-0")
        batch = SimpleNamespace(
            reqs=[req],
            hisparse_coordinator=None,
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=MagicMock(),
            tree_cache=MagicMock(),
        )
        args = SimpleNamespace(disaggregation_mode="decode")

        with (
            patch("sglang.srt.managers.schedule_batch.release_kv_cache") as release,
            patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache"),
        ):
            ScheduleBatch.release_req(
                batch,
                0,
                0,
                args,
                offload_kv_cache=False,
            )

        req.offload_kv_cache.assert_not_called()
        release.assert_called_once_with(req, batch.tree_cache, is_insert=False)
        req.reset_for_retract.assert_called_once_with()

    def test_release_keeps_decode_cpu_snapshot_by_default(self):
        req = _make_req("req-0")
        batch = SimpleNamespace(
            reqs=[req],
            hisparse_coordinator=None,
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=MagicMock(),
            tree_cache=MagicMock(),
        )
        args = SimpleNamespace(disaggregation_mode="decode")

        with (
            patch("sglang.srt.managers.schedule_batch.release_kv_cache"),
            patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache"),
        ):
            ScheduleBatch.release_req(batch, 0, 0, args)

        req.offload_kv_cache.assert_called_once_with(
            batch.req_to_token_pool,
            batch.token_to_kv_pool_allocator,
        )

    def test_unsupported_dsv4_retraction_aborts_only_selected_request(self):
        keep_req = _make_req("keep")
        abort_req = _make_req("abort")
        kv_pool = SimpleNamespace(supports_decode_retraction_cpu_snapshot=False)
        allocator = MagicMock()
        allocator.get_kvcache.return_value = kv_pool
        allocator.available_size.return_value = 32
        allocator.full_available_size.return_value = 1024
        allocator.swa_available_size.return_value = 32
        released = []

        def release_req(idx, remaining, _server_args, *, offload_kv_cache=True):
            released.append((idx, remaining, offload_kv_cache))

        batch = SimpleNamespace(
            reqs=[keep_req, abort_req],
            token_to_kv_pool_allocator=allocator,
            release_req=release_req,
            check_decode_mem=MagicMock(side_effect=[True, True]),
            filter_batch=MagicMock(),
            _mark_unsupported_decode_retraction_abort=lambda req, pool: (
                ScheduleBatch._mark_unsupported_decode_retraction_abort(
                    batch, req, pool
                )
            ),
        )
        args = SimpleNamespace(
            disaggregation_mode="decode",
            speculative_algorithm="DSPARK",
        )

        with patch(
            "sglang.srt.managers.schedule_batch.NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
            return_value=0.5,
        ):
            retracted, ratio, aborted = ScheduleBatch.retract_decode(batch, args)

        self.assertEqual(retracted, [])
        self.assertEqual(aborted, [abort_req])
        self.assertEqual(ratio, 0.5)
        self.assertEqual(released, [(1, 1, False)])
        self.assertIsNotNone(abort_req.to_finish)
        keep_req.offload_kv_cache.assert_not_called()
        abort_req.offload_kv_cache.assert_not_called()
        batch.filter_batch.assert_called_once_with(keep_indices=[0])

    def test_snapshot_capable_pool_keeps_existing_retraction_behavior(self):
        keep_req = _make_req("keep")
        retract_req = _make_req("retract")
        allocator = MagicMock()
        allocator.get_kvcache.return_value = SimpleNamespace()
        released = []

        def release_req(idx, remaining, _server_args, *, offload_kv_cache=True):
            released.append((idx, remaining, offload_kv_cache))

        batch = SimpleNamespace(
            reqs=[keep_req, retract_req],
            token_to_kv_pool_allocator=allocator,
            release_req=release_req,
            check_decode_mem=MagicMock(side_effect=[True, True]),
            filter_batch=MagicMock(),
        )
        args = SimpleNamespace(
            disaggregation_mode="decode",
            speculative_algorithm=None,
        )

        with patch(
            "sglang.srt.managers.schedule_batch.NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
            return_value=0.5,
        ):
            retracted, _, aborted = ScheduleBatch.retract_decode(batch, args)

        self.assertEqual(retracted, [retract_req])
        self.assertEqual(aborted, [])
        self.assertEqual(released, [(1, 1, True)])

    def test_last_dsv4_request_oom_skips_cpu_snapshot(self):
        abort_req = _make_req("last")
        allocator = MagicMock()
        allocator.get_kvcache.return_value = SimpleNamespace(
            supports_decode_retraction_cpu_snapshot=False
        )
        released = []

        def release_req(idx, remaining, _server_args, *, offload_kv_cache=True):
            released.append((idx, remaining, offload_kv_cache))

        batch = SimpleNamespace(
            reqs=[abort_req],
            token_to_kv_pool_allocator=allocator,
            release_req=release_req,
            check_decode_mem=MagicMock(return_value=False),
            filter_batch=MagicMock(),
        )
        args = SimpleNamespace(
            disaggregation_mode="decode",
            speculative_algorithm="DSPARK",
        )

        with patch(
            "sglang.srt.managers.schedule_batch.NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
            return_value=0.5,
        ):
            retracted, _, aborted = ScheduleBatch.retract_decode(batch, args)

        self.assertEqual(retracted, [])
        self.assertEqual(aborted, [abort_req])
        self.assertEqual(released, [(0, 0, False)])
        abort_req.offload_kv_cache.assert_not_called()
        batch.filter_batch.assert_called_once_with(keep_indices=[])


if __name__ == "__main__":
    unittest.main()
