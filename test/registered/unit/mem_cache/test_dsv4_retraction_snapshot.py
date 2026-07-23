"""DSV4 decode-retraction snapshot and failure-isolation tests."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch
from sglang.srt.mem_cache.deepseek_v4_compress_state import KVAndScore
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4IndexerPool,
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
)


def _make_page_pool(*, layers: int, pages: int, page_size: int, row_bytes: int):
    pool = DeepSeekV4SingleKVPool.__new__(DeepSeekV4SingleKVPool)
    pool.page_size = page_size
    pool.layer_num = layers
    pool.cpu_offloading_chunk_size = 17
    pool.kv_buffer = [
        torch.zeros((pages, row_bytes), dtype=torch.uint8) for _ in range(layers)
    ]
    return pool


def _make_indexer_pool(*, layers: int, pages: int, page_size: int, row_bytes: int):
    pool = DeepSeekV4IndexerPool.__new__(DeepSeekV4IndexerPool)
    pool.page_size = page_size
    pool.layer_num = layers
    pool.cpu_offloading_chunk_size = 19
    pool.index_k_with_scale_buffer = [
        torch.zeros((pages, row_bytes), dtype=torch.uint8) for _ in range(layers)
    ]
    return pool


def _make_state_pool(*, rows: int, dim: int, ring: int, swa_page_size: int):
    pool = SimpleNamespace(
        ring_size=ring,
        swa_page_size=swa_page_size,
        kv_score_buffer=KVAndScore(torch.zeros((rows, dim), dtype=torch.float32)),
    )

    def translate(swa_loc: torch.Tensor) -> torch.Tensor:
        pages = swa_loc // swa_page_size
        state_loc = pages * ring + (swa_loc % ring)
        return torch.where(swa_loc < 0, -1, state_loc)

    pool.translate_from_swa_loc_to_state_loc = translate
    return pool


def _fill_page_pool(pool, base: int):
    for layer_id, buf in enumerate(pool.kv_buffer):
        values = torch.arange(buf.numel(), dtype=torch.int64).reshape(buf.shape)
        buf.copy_(((values + base + layer_id * 37) % 251).to(torch.uint8))


def _fill_indexer_pool(pool, base: int):
    for layer_id, buf in enumerate(pool.index_k_with_scale_buffer):
        values = torch.arange(buf.numel(), dtype=torch.int64).reshape(buf.shape)
        buf.copy_(((values + base + layer_id * 41) % 251).to(torch.uint8))


def _page_snapshot(pool, indices):
    pages = torch.unique_consecutive(indices // pool.page_size)
    return [buf[pages].clone() for buf in pool.kv_buffer]


class TestDSV4RetractionSnapshot(unittest.TestCase):
    def test_round_trip_to_new_full_swa_and_req_mapping(self):
        old_full = torch.arange(1, 257, dtype=torch.int64)
        new_full = torch.arange(513, 769, dtype=torch.int64)
        c4_old = old_full[(old_full + 1) % 4 == 0] // 4
        c4_new = new_full[(new_full + 1) % 4 == 0] // 4
        c128_old = old_full[(old_full + 1) % 128 == 0] // 128
        c128_new = new_full[(new_full + 1) % 128 == 0] // 128

        # Retraction allocates a new req_pool_idx and new physical pages.  The
        # request-row identity is represented here by replacing the entire
        # full->SWA mapping before restore.
        old_swa = torch.arange(17, 81, dtype=torch.int64)
        # The resumed request may retain fewer SWA rows than the save-time
        # request.  This exercises the page-aware filtering added by the
        # latest #27559 head instead of only testing the all-rows fast path.
        new_swa = torch.arange(225, 281, dtype=torch.int64)
        old_mapping = torch.zeros(900, dtype=torch.int64)
        new_mapping = torch.zeros(900, dtype=torch.int64)
        old_mapping[old_full[-64:]] = old_swa
        new_mapping[new_full[-56:]] = new_swa

        token_pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        token_pool.swa_kv_pool = _make_page_pool(
            layers=2, pages=128, page_size=8, row_bytes=13
        )
        token_pool.c4_kv_pool = _make_page_pool(
            layers=3, pages=256, page_size=4, row_bytes=11
        )
        token_pool.c128_kv_pool = _make_page_pool(
            layers=2, pages=32, page_size=1, row_bytes=7
        )
        token_pool.c4_indexer_kv_pool = _make_indexer_pool(
            layers=3, pages=256, page_size=4, row_bytes=17
        )
        token_pool.compress_state_pools = [
            _make_state_pool(rows=512, dim=10, ring=8, swa_page_size=8),
            _make_state_pool(rows=2048, dim=12, ring=32, swa_page_size=8),
        ]
        token_pool.indexer_compress_state_pools = [
            _make_state_pool(rows=512, dim=14, ring=8, swa_page_size=8),
            None,
        ]
        token_pool.full_to_swa_index_mapping = old_mapping

        _fill_page_pool(token_pool.swa_kv_pool, 3)
        _fill_page_pool(token_pool.c4_kv_pool, 11)
        _fill_page_pool(token_pool.c128_kv_pool, 19)
        _fill_indexer_pool(token_pool.c4_indexer_kv_pool, 29)
        for i, pool in enumerate(
            token_pool.compress_state_pools
            + token_pool.indexer_compress_state_pools
        ):
            if pool is not None:
                values = torch.arange(
                    pool.kv_score_buffer.kv_score.numel(), dtype=torch.float32
                ).reshape(pool.kv_score_buffer.kv_score.shape)
                pool.kv_score_buffer.kv_score.copy_(values + 1000 * (i + 1))

        expected_swa = _page_snapshot(token_pool.swa_kv_pool, old_swa[-56:])
        expected_c4 = _page_snapshot(token_pool.c4_kv_pool, c4_old)
        expected_c128 = _page_snapshot(token_pool.c128_kv_pool, c128_old)
        old_indexer_pages = torch.unique_consecutive(
            c4_old // token_pool.c4_indexer_kv_pool.page_size
        )
        expected_indexer = [
            buf[old_indexer_pages].clone()
            for buf in token_pool.c4_indexer_kv_pool.index_k_with_scale_buffer
        ]
        expected_attention_state = [
            pool.kv_score_buffer.kv_score[
                pool.translate_from_swa_loc_to_state_loc(old_swa[-56:])
            ].clone()
            for pool in token_pool.compress_state_pools
        ]
        expected_indexer_state = token_pool.indexer_compress_state_pools[
            0
        ].kv_score_buffer.kv_score[
            token_pool.indexer_compress_state_pools[
                0
            ].translate_from_swa_loc_to_state_loc(old_swa[-56:])
        ].clone()

        cpu_copy = token_pool.get_cpu_copy(old_full)
        self.assertEqual(cpu_copy["format_version"], 1)
        self.assertEqual(int(cpu_copy["swa_mask"].sum()), len(old_swa))

        for pool in (
            token_pool.swa_kv_pool,
            token_pool.c4_kv_pool,
            token_pool.c128_kv_pool,
        ):
            for buf in pool.kv_buffer:
                buf.zero_()
        for buf in token_pool.c4_indexer_kv_pool.index_k_with_scale_buffer:
            buf.zero_()
        for pool in (
            token_pool.compress_state_pools
            + token_pool.indexer_compress_state_pools
        ):
            if pool is not None:
                pool.kv_score_buffer.kv_score.fill_(-999)

        token_pool.full_to_swa_index_mapping = new_mapping
        token_pool.load_cpu_copy(cpu_copy, new_full)

        for pool, indices, expected in (
            (token_pool.swa_kv_pool, new_swa, expected_swa),
            (token_pool.c4_kv_pool, c4_new, expected_c4),
            (token_pool.c128_kv_pool, c128_new, expected_c128),
        ):
            pages = torch.unique_consecutive(indices // pool.page_size)
            for actual, wanted in zip(pool.kv_buffer, expected):
                self.assertTrue(torch.equal(actual[pages], wanted))

        new_indexer_pages = torch.unique_consecutive(
            c4_new // token_pool.c4_indexer_kv_pool.page_size
        )
        for actual, wanted in zip(
            token_pool.c4_indexer_kv_pool.index_k_with_scale_buffer,
            expected_indexer,
        ):
            self.assertTrue(torch.equal(actual[new_indexer_pages], wanted))

        for pool, wanted in zip(
            token_pool.compress_state_pools, expected_attention_state
        ):
            new_locs = pool.translate_from_swa_loc_to_state_loc(new_swa)
            self.assertTrue(
                torch.equal(pool.kv_score_buffer.kv_score[new_locs], wanted)
            )
        indexer_pool = token_pool.indexer_compress_state_pools[0]
        new_locs = indexer_pool.translate_from_swa_loc_to_state_loc(new_swa)
        self.assertTrue(
            torch.equal(
                indexer_pool.kv_score_buffer.kv_score[new_locs],
                expected_indexer_state,
            )
        )

    def test_snapshot_failure_isolated_to_retracted_request(self):
        req = SimpleNamespace(rid="bad-snapshot", to_finish=None)
        req.offload_kv_cache = Mock(side_effect=RuntimeError("snapshot failed"))
        req.reset_for_retract = Mock()
        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch.reqs = [req]
        batch.hisparse_coordinator = None
        batch.req_to_token_pool = Mock()
        batch.token_to_kv_pool_allocator = Mock()
        batch.tree_cache = Mock()
        server_args = SimpleNamespace(disaggregation_mode="decode")

        with (
            patch("sglang.srt.managers.schedule_batch.release_kv_cache") as release,
            patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache"),
        ):
            ok = batch.release_req(
                0, 0, server_args, isolate_snapshot_failure=True
            )

        self.assertFalse(ok)
        release.assert_called_once_with(req, batch.tree_cache, is_insert=False)
        req.reset_for_retract.assert_called_once_with()
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertEqual(req.to_finish.status_code, 500)

    def test_restore_failure_isolated_to_retracted_request(self):
        req = SimpleNamespace(
            rid="bad-restore",
            is_retracted=True,
            return_logprob=False,
            req_pool_idx=1,
            load_kv_cache=Mock(side_effect=RuntimeError("restore failed")),
            kv_cache_cpu={"format_version": 1},
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.retracted_queue = [req]
        queue.req_to_token_pool = Mock()
        queue.req_to_token_pool.available_size.return_value = 1
        queue.token_to_kv_pool_allocator = Mock()
        queue.tree_cache = Mock()
        queue.scheduler = SimpleNamespace(
            output_streamer=SimpleNamespace(stream_output=Mock())
        )
        queue._uses_swa_tail_prealloc = Mock(return_value=False)
        queue._allocatable_token_budgets = Mock(return_value=1024)
        queue._prealloc_required_tokens = Mock(return_value=(16, 16))
        queue._pre_alloc = Mock()

        with patch("sglang.srt.disaggregation.decode.release_kv_cache") as release:
            resumed = queue.resume_retracted_reqs()

        self.assertEqual(resumed, [])
        self.assertEqual(queue.retracted_queue, [])
        release.assert_called_once_with(req, queue.tree_cache, is_insert=False)
        queue.scheduler.output_streamer.stream_output.assert_called_once()
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.status_code, 500)


if __name__ == "__main__":
    unittest.main()
