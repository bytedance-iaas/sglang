"""Unit tests for HiSparse hierarchical sparse KV cache system.

Tests cover:
- CUDA kernel correctness (swap_in_selected_pages vs naive_load_topk oracle)
- Memory allocator lifecycle (alloc / free / available_size)
- Request lifecycle (staging path, direct-to-host path)
- Batch multi-request correctness
"""

import gc
import os
import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseResidencyState
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

# ---------------------------------------------------------------------------
# Test configuration (small-scale for fast CI runs)
# ---------------------------------------------------------------------------
SIZE = 2048  # device buffer pool size (tokens)
PAGE_SIZE = 64  # page size (must be 64 for CUDA, 1 for ROCm)
TOP_K = 256  # top-k selection count
DEVICE_BUFFER_SIZE = 512  # device buffer per request
HOST_TO_DEVICE_RATIO = 2
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
KV_CACHE_DIM = 576  # MLA dim (DeepSeek-style)
LAYER_NUM = 2
MAX_NUM_REQS = 8
MAX_CONTEXT_LEN = 2048


def _make_req(rid="test-req-0", origin_input_ids=None, output_ids=None):
    """Create a minimal mock Req object with the fields HiSparseCoordinator uses."""
    if origin_input_ids is None:
        origin_input_ids = list(range(64))
    if output_ids is None:
        output_ids = []
    req = SimpleNamespace(
        rid=rid,
        origin_input_ids=origin_input_ids,
        output_ids=output_ids,
        fill_ids=origin_input_ids + output_ids,
        seqlen=len(origin_input_ids) + len(output_ids),
        req_pool_idx=None,
        kv_allocated_len=0,
        kv_committed_len=0,
        sampling_params=SimpleNamespace(max_new_tokens=1024),
        finished_reason=None,
        hisparse_staging=False,
        hisparse_resident=False,
        staging=False,
        inflight_middle_chunks=0,
    )
    req.finished = lambda: req.finished_reason is not None
    req.set_extend_range = lambda start, end: setattr(
        req, "extend_range", Range(start, end)
    )
    return req


class TestHiSparseUnit(unittest.TestCase):
    """Test class that builds a minimal HiSparse component stack."""

    # ==================================================================
    # Fixture
    # ==================================================================

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for HiSparse tests.")
        if is_npu() or is_xpu():
            raise unittest.SkipTest("HiSparse tests only support CUDA/ROCm.")
        if not (is_cuda() or is_hip()):
            raise unittest.SkipTest("CUDA/ROCm not available.")

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29599")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
        cls.tp_group = torch.distributed.group.WORLD

        from sglang.srt.mem_cache.pool_host.common import (
            ALLOC_MEMORY_FUNCS,
            alloc_with_pin_memory,
        )

        cls._original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory

        if is_hip():
            from sglang.srt.layers.attention.dsa.utils import (
                aiter_can_use_preshuffle_paged_mqa,
            )

            global_page_size = 64 if aiter_can_use_preshuffle_paged_mqa() else 1
        else:
            global_page_size = PAGE_SIZE

        from sglang.srt.mem_cache.allocator.hisparse import (
            HiSparseTokenToKVPoolAllocator,
        )
        from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool

        cls.device_pool = HiSparseDSATokenToKVPool(
            size=SIZE,
            page_size=global_page_size,
            kv_lora_rank=KV_LORA_RANK,
            dtype=torch.bfloat16,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            layer_num=LAYER_NUM,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=KV_CACHE_DIM,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )
        cls.allocator = HiSparseTokenToKVPoolAllocator(
            size=SIZE,
            page_size=global_page_size,
            dtype=torch.bfloat16,
            device="cuda",
            kvcache=cls.device_pool,
            need_sort=False,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )

        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        cls.req_to_token_pool = ReqToTokenPool(
            size=MAX_NUM_REQS,
            max_context_len=MAX_CONTEXT_LEN,
            device="cuda",
            enable_memory_saver=False,
        )

        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        cls.page_size = global_page_size
        cls.coordinator = HiSparseCoordinator(
            req_to_token_pool=cls.req_to_token_pool,
            token_to_kv_pool_allocator=cls.allocator,
            top_k=TOP_K,
            device_buffer_size=DEVICE_BUFFER_SIZE,
            device="cuda",
            tp_group=cls.tp_group,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
            dynamic_residency=True,
            dynamic_residency_max_tokens=MAX_CONTEXT_LEN,
            dynamic_residency_max_requests=MAX_NUM_REQS,
            dynamic_residency_min_remaining_tokens=0,
            dynamic_residency_promote_watermark=0.01,
            dynamic_residency_demote_watermark=0.0,
            dynamic_residency_cooldown_steps=0,
        )

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS

        ALLOC_MEMORY_FUNCS["cuda"] = cls._original_alloc
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def setUp(self):
        """Reset shared allocator / coordinator state so tests are isolated.

        Without this, a mid-test assertion failure skips cleanup and leaks
        resources, causing unrelated failures in later tests.
        """
        self.allocator.clear()
        self.req_to_token_pool.clear()
        self.coordinator.mem_pool_host.clear()
        # Reset per-request coordinator bookkeeping
        self.coordinator.req_to_device_buffer.zero_()
        self.coordinator.req_device_buffer_size.zero_()
        self.coordinator.req_to_host_pool.fill_(-1)
        self.coordinator.req_to_host_pool_allocated_len.zero_()
        self.coordinator.req_device_buffer_tokens.fill_(-1)
        self.coordinator.req_device_buffer_token_locs.fill_(-1)
        self.coordinator.lru_slots[:] = self.coordinator._lru_init.view(1, 1, -1)
        self.coordinator.ack_staging_queue.clear()
        self.coordinator._has_pending_backup = False
        self.coordinator._pending_draft_extend_backup = None
        self.coordinator.active_hisparse_reqs.clear()
        self.coordinator._residency_states.clear()
        self.coordinator._last_residency_transition_step.clear()
        self.coordinator._ever_resident_requests.clear()
        self.coordinator._pressure_demoted_requests.clear()
        self.coordinator._repromotion_suppression_reported.clear()
        self.coordinator._decode_step = 0
        self.coordinator._promotion_count = 0
        self.coordinator._demotion_count = 0
        self.coordinator._promotion_failure_count = 0
        self.coordinator._promotion_migration_acts.clear()
        self.coordinator._promotion_migrated_bytes = 0
        self.coordinator._promotion_migration_seconds = 0.0
        self.coordinator._demotion_reclaimed_bytes = 0
        self.coordinator._demotion_transition_seconds = 0.0
        self.coordinator._repromotion_suppressed_count = 0
        for i in range(len(self.coordinator._skip_first_backup)):
            self.coordinator._skip_first_backup[i] = False

    # ==================================================================
    # Low-level helpers
    # ==================================================================

    def _alloc_req_slot(self, req):
        """Allocate a req_pool_idx for the request."""
        indices = self.req_to_token_pool.alloc([req])
        self.assertIsNotNone(indices, "Failed to allocate req pool slot")
        return req.req_pool_idx

    def _free_req_slot(self, req):
        """Free the req_pool_idx."""
        if req.req_pool_idx is not None:
            self.req_to_token_pool.free(req)

    def _alloc_kv(self, req, fill_len, *, logical_only=False):
        """Allocate KV indices, write req_to_token_pool, update req fields.
        If logical_only=True, uses alloc_logical_only (PD-separated path).
        Returns kv_loc tensor."""
        device = self.allocator.device
        alloc_fn = (
            self.allocator.alloc_logical_only
            if logical_only
            else self.allocator.alloc_extend
        )
        kv_loc = alloc_fn(
            prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
            extend_num_tokens=fill_len,
        )
        self.assertIsNotNone(kv_loc, "KV alloc failed")
        self.req_to_token_pool.write((req.req_pool_idx, slice(0, len(kv_loc))), kv_loc)
        req.kv_allocated_len = fill_len
        req.kv_committed_len = fill_len
        req.full_untruncated_fill_ids = array("q", range(fill_len))
        req.extend_range = Range(0, fill_len)
        return kv_loc

    # ==================================================================
    # Mid-level helpers
    # ==================================================================

    @staticmethod
    def _kv_pattern(layer_id, token_id):
        """Deterministic KV value for (layer, token) — used by write & verify."""
        v = (layer_id * 10000 + token_id + 1) * 0.001
        return float(torch.tensor(v, dtype=torch.bfloat16))

    def _write_device_patterns(self, kv_loc, fill_len):
        """Write distinguishable patterns into device KV buffer for all layers.

        kv_loc contains *logical* indices; we must translate them to hisparse
        device indices before indexing kv_buffer (which is sized for the
        hisparse pool, not the larger logical space).
        """
        hisparse_locs = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        for lid in range(LAYER_NUM):
            for i in range(fill_len):
                self.device_pool.kv_buffer[lid][hisparse_locs[i]] = self._kv_pattern(
                    lid, i
                )

    def _populate_host_pool(self, req, fill_len):
        """Allocate host slots, write known patterns, register in coordinator.
        Returns host_indices (cuda tensor)."""
        host_pool = self.coordinator.mem_pool_host
        host_indices = host_pool.alloc(fill_len)
        self.assertIsNotNone(host_indices, "Host alloc failed")
        host_indices = host_indices.to(device="cuda")
        self.coordinator.req_to_host_pool[req.req_pool_idx, :fill_len] = host_indices
        self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx] = fill_len
        for lid in range(LAYER_NUM):
            for i in range(fill_len):
                host_pool.kv_buffer[lid][host_indices[i]] = self._kv_pattern(lid, i)
        return host_indices

    def _build_topk_tokens(self, fill_len, *, include_newest=False):
        """Build a 1-D [TOP_K] int32 cuda tensor of token positions.

        If include_newest=True, fill_len-1 is guaranteed as the last valid slot.
        Pads with -1 when fill_len (or fill_len-1) < TOP_K.

        For long-sequence tests (fill_len > DEVICE_BUFFER_SIZE) where the
        "newest token" reserved slot is not populated (it requires an actual
        decode step + map_last_loc_to_buffer), callers should pass
        ``fill_len - 1`` as the effective pool size so position fill_len-1 is
        never randomly selected.
        """
        n = min(fill_len, TOP_K)
        if include_newest and n > 1:
            tokens = torch.randperm(fill_len - 1, device="cuda")[: n - 1].to(
                torch.int32
            )
            tokens = torch.cat(
                [tokens, torch.tensor([fill_len - 1], dtype=torch.int32, device="cuda")]
            )
        else:
            tokens = torch.randperm(fill_len, device="cuda")[:n].to(torch.int32)
        if n < TOP_K:
            pad = torch.full((TOP_K - n,), -1, dtype=torch.int32, device="cuda")
            tokens = torch.cat([tokens, pad])
        return tokens

    def _make_batch_tensors(self, reqs, fill_lens):
        """Build (req_pool_indices [int64], seq_lens [int32]) on cuda."""
        rpi = torch.tensor(
            [r.req_pool_idx for r in reqs], dtype=torch.int64, device="cuda"
        )
        sls = torch.tensor(fill_lens, dtype=torch.int32, device="cuda")
        return rpi, sls

    def _assert_kv_correct(self, locs_row, tokens_row, layer_id, count, msg=""):
        """Assert device KV data at *locs_row[:count]* matches the written
        pattern for the corresponding *tokens_row[:count]* positions."""
        for i in range(count):
            tok = int(tokens_row[i].item())
            if tok < 0:
                continue
            expected = self._kv_pattern(layer_id, tok)
            actual = self.device_pool.kv_buffer[layer_id][locs_row[i].long()]
            self.assertTrue(
                torch.allclose(
                    actual.float(),
                    torch.full_like(actual.float(), expected),
                    atol=1e-2,
                ),
                f"{msg}layer {layer_id}, token {tok}: KV data mismatch",
            )

    def _assert_matches_naive(self, rpi, sls, batch, kernel_locs, layer_id, msg=""):
        """Assert kernel swap_in KV data matches naive_load_topk KV data."""
        naive_locs = self.coordinator.naive_load_topk(rpi, sls, batch, layer_id)
        for b in range(batch.shape[0]):
            for i in range(TOP_K):
                if batch[b, i] < 0:
                    continue
                naive_data = self.device_pool.kv_buffer[layer_id][
                    naive_locs[b, i].long()
                ]
                kernel_data = self.device_pool.kv_buffer[layer_id][
                    kernel_locs[b, i].long()
                ]
                self.assertTrue(
                    torch.allclose(naive_data.float(), kernel_data.float(), atol=1e-2),
                    f"{msg}layer {layer_id}, b{b} idx {i}: naive != kernel",
                )

    def _swap_in_selected_pages(
        self,
        rpi: torch.Tensor,
        sls: torch.Tensor,
        batch: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Wrapper that sets num_real_reqs before calling swap_in_selected_pages.

        In production, model_runner sets num_real_reqs before each forward
        pass.  Tests must replicate that to get correct kernel behaviour.
        """
        self.coordinator.num_real_reqs[0] = rpi.shape[0]
        return self.coordinator.swap_in_selected_pages(rpi, sls, batch, layer_id)

    def _cleanup_req(self, req, kv_loc, *, logical_only=False):
        """request_finished -> free KV -> free req slot."""
        was_resident = self.coordinator._is_resident(req.req_pool_idx)
        self.coordinator.request_finished(req)
        if logical_only and not was_resident:
            self.allocator.logical_attn_allocator.free(kv_loc)
        else:
            self.allocator.free(kv_loc)
        self._free_req_slot(req)

    def _admit_direct_to_buffer(self, req):
        """Force direct admission to exercise the HBM-full fallback path."""
        allocator = self.allocator.hisparse_attn_allocator
        buffer_size = self.coordinator._device_buffer_alloc_size(req.kv_allocated_len)
        host_len = self.coordinator.host_token_len(req.kv_allocated_len)
        promotion_size = (
            (host_len + self.page_size - 1) // self.page_size
        ) * self.page_size
        self.assertGreater(promotion_size, buffer_size)

        held = allocator.alloc(allocator.available_size() - buffer_size)
        self.assertIsNotNone(held)
        try:
            self.coordinator.admit_request_direct(req)
            self.assertFalse(self.coordinator._is_resident(req.req_pool_idx))
        finally:
            allocator.free(held)

    def _get_initial_sizes(self):
        """Snapshot allocator available sizes."""
        return (
            self.allocator.logical_attn_allocator.available_size(),
            self.allocator.hisparse_attn_allocator.available_size(),
            self.coordinator.mem_pool_host.available_size(),
        )

    def _assert_sizes_restored(self, initial_sizes, msg=""):
        """Assert allocator sizes match the snapshot."""
        logical, hisparse, host = self._get_initial_sizes()
        self.assertEqual(logical, initial_sizes[0], f"Logical leak {msg}")
        self.assertEqual(hisparse, initial_sizes[1], f"HiSparse leak {msg}")
        self.assertEqual(host, initial_sizes[2], f"Host leak {msg}")

    # ==================================================================
    # Test: Kernel correctness — short sequence (fast path)
    # ==================================================================
    def test_kernel_correctness_short_seq(self):
        """Short seq (len <= device_buffer_size): kernel fast path returns
        device buffer locs, matching naive_load_topk."""
        initial = self._get_initial_sizes()
        req = _make_req("short-seq", list(range(self.page_size)))
        self._alloc_req_slot(req)

        fill_len = self.page_size
        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)
        self.coordinator.alloc_device_buffer(req)

        tokens = self._build_topk_tokens(fill_len)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        for lid in range(LAYER_NUM):
            naive_locs = self.coordinator.naive_load_topk(rpi, sls, batch, lid)
            kernel_locs = self._swap_in_selected_pages(rpi, sls, batch, lid)
            valid = batch[0] >= 0
            self.assertTrue(
                torch.equal(naive_locs[0][valid].cpu(), kernel_locs[0][valid].cpu()),
                f"Layer {lid}: kernel locs != naive oracle",
            )

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "short_seq")

    # ==================================================================
    # Test: Kernel correctness — long sequence (cache miss + host DMA)
    # ==================================================================
    def test_kernel_correctness_long_seq(self):
        """Long seq (len > device_buffer_size): kernel loads from host,
        matching naive_load_topk for data correctness."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size * 2
        req = _make_req("long-seq", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self._admit_direct_to_buffer(req)

        # Pass fill_len-1 so position fill_len-1 ("newest token") is never
        # randomly selected — its reserved device-buffer slot is only valid
        # after map_last_loc_to_buffer in a real decode step.
        tokens = self._build_topk_tokens(fill_len - 1)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        for lid in range(LAYER_NUM):
            naive_locs = self.coordinator.naive_load_topk(rpi, sls, batch, lid)
            kernel_locs = self._swap_in_selected_pages(rpi, sls, batch, lid)
            self.assertTrue(torch.all(naive_locs[0, :TOP_K] >= 0))
            self.assertTrue(torch.all(kernel_locs[0, :TOP_K] >= 0))
            # Verify both return correct KV data independently
            self._assert_kv_correct(naive_locs[0], tokens, lid, TOP_K, msg="Naive: ")
            self._assert_kv_correct(kernel_locs[0], tokens, lid, TOP_K, msg="Kernel: ")

        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "long_seq")

    # ==================================================================
    # Test: Kernel LRU replacement across multiple decode steps
    # ==================================================================
    def test_kernel_lru_replacement(self):
        """Multi-step swap-in: second call hits cached tokens, only
        evicts/loads new misses."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size * 2
        req = _make_req("lru-test", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self._admit_direct_to_buffer(req)

        rpi, sls = self._make_batch_tensors([req], [fill_len])

        # Step 1: load the first TOP_K positions from host (no newest token —
        # the reserved slot is only valid after map_last_loc_to_buffer which is
        # called during an actual decode step, not modelled here).
        tokens_s1 = torch.arange(TOP_K, dtype=torch.int32, device="cuda")
        locs1 = self._swap_in_selected_pages(
            rpi, sls, tokens_s1.unsqueeze(0), layer_id=0
        )
        self.assertTrue(torch.all(locs1[0, :TOP_K] >= 0))

        # Step 2: half overlap (hit) + half new (miss).
        # Choose new tokens from a range safely below fill_len.
        half = TOP_K // 2
        new_start = TOP_K  # first position not in step-1
        tokens_s2 = torch.cat(
            [
                tokens_s1[:half],  # hits
                torch.arange(
                    new_start, new_start + half, dtype=torch.int32, device="cuda"
                ),  # misses
            ]
        )
        locs2 = self._swap_in_selected_pages(
            rpi, sls, tokens_s2.unsqueeze(0), layer_id=0
        )
        self.assertTrue(torch.all(locs2[0, :TOP_K] >= 0))

        # Verify repeated (hit) tokens still have correct KV data
        self._assert_kv_correct(
            locs2[0], tokens_s2, layer_id=0, count=half, msg="LRU hit: "
        )
        # Also verify new (miss) tokens loaded correctly
        self._assert_kv_correct(
            locs2[0, half:],
            tokens_s2[half:],
            layer_id=0,
            count=half,
            msg="LRU miss: ",
        )

        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "lru_replacement")

    # ==================================================================
    # Test: Allocator alloc/free lifecycle
    # ==================================================================
    def test_allocator_alloc_free_cycle(self):
        """alloc_extend / alloc_device_buffer / free restores available_size."""
        initial = self._get_initial_sizes()
        device = self.allocator.device
        fill_len = self.page_size * 2

        kv_loc = self.allocator.alloc_extend(
            prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
            extend_num_tokens=fill_len,
        )
        self.assertIsNotNone(kv_loc)
        self.assertEqual(len(kv_loc), fill_len)

        mapping = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping > 0), "Mapping should be non-zero")
        self.assertLess(self.allocator.available_size(), initial[0])

        need_size = min(
            ((fill_len + self.page_size - 1) // self.page_size) * self.page_size,
            DEVICE_BUFFER_SIZE,
        )
        buf_idx = self.allocator.alloc_device_buffer(kv_loc, need_size)
        self.assertIsNotNone(buf_idx)
        mapping_after = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping_after == 0), "Mapping should be cleared")

        self.allocator.free_hisparse_indices(buf_idx)
        self.allocator.logical_attn_allocator.free(kv_loc)
        self._assert_sizes_restored(initial, "alloc_free_cycle")

    def test_allocator_page_size_one_alloc_free_cycle(self):
        """alloc() maps logical to hisparse indices for ROCm page_size=1."""
        if self.page_size != 1:
            self.skipTest("page_size=1 alloc path is ROCm-specific")

        initial = self._get_initial_sizes()
        need_size = 16

        kv_loc = self.allocator.alloc(need_size)
        self.assertIsNotNone(kv_loc)
        self.assertEqual(len(kv_loc), need_size)

        mapping = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping > 0), "Mapping should be non-zero")
        self.assertLess(self.allocator.available_size(), initial[0])

        self.allocator.free(kv_loc)
        mapping_after = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping_after == 0), "Mapping should be cleared")
        self._assert_sizes_restored(initial, "page_size_one_alloc_free_cycle")

    def test_decode_remap_frees_stale_page_size_one_mapping(self):
        """map_last_loc_to_buffer frees the temporary alloc() hisparse slot."""
        if self.page_size != 1:
            self.skipTest("page_size=1 decode remap path is ROCm-specific")

        initial = self._get_initial_sizes()
        device = self.allocator.device
        fill_len = 2
        req = _make_req("decode-remap", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self.coordinator.alloc_device_buffer(req)
        self.coordinator._skip_first_backup[req.req_pool_idx] = True

        out_loc = self.allocator.alloc(1)
        self.assertIsNotNone(out_loc)
        stale_loc = self.allocator.full_to_hisparse_device_index_mapping[
            out_loc
        ].clone()
        self.assertTrue(torch.all(stale_loc > 0), "Temporary mapping should exist")

        seq_len = fill_len + 1
        self.req_to_token_pool.write((req.req_pool_idx, fill_len), out_loc)
        req.kv_allocated_len = seq_len
        req.kv_committed_len = seq_len

        self.coordinator.map_last_loc_to_buffer(
            seq_lens=torch.tensor([seq_len], dtype=torch.int64, device=device),
            out_cache_loc=out_loc,
            req_pool_indices=torch.tensor(
                [req.req_pool_idx], dtype=torch.int64, device=device
            ),
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([req.req_pool_idx], dtype=torch.int64),
        )

        remapped_loc = self.allocator.full_to_hisparse_device_index_mapping[out_loc]
        self.assertTrue(torch.all(remapped_loc > 0), "Remapped loc should exist")
        self.assertFalse(
            torch.equal(stale_loc, remapped_loc),
            "Decode loc should move from temporary mapping to device buffer",
        )
        self.assertEqual(
            self.allocator.hisparse_attn_allocator.available_size(),
            initial[1] - seq_len,
        )

        self.coordinator.request_finished(req)
        self.allocator.logical_attn_allocator.free(torch.cat([kv_loc, out_loc]))
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "decode_remap")

    # ==================================================================
    # Test: Staging (PD Colocate) path
    # ==================================================================
    def test_request_lifecycle_staging_path(self):
        """prefill -> staging DMA -> collect_ready -> swap-in -> finish."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size
        req = _make_req("staging-req", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)

        self.coordinator.admit_request_into_staging(req)
        self.assertTrue(req.hisparse_staging)

        torch.cuda.synchronize()
        ready = self.coordinator.collect_ready_reqs()
        self.assertEqual(len(ready), 1)
        self.assertFalse(req.hisparse_staging)
        self.assertTrue(self.coordinator._is_resident(req.req_pool_idx))
        self.assertEqual(self.coordinator._promotion_count, 0)
        self.assertTrue(self.coordinator._skip_first_backup[req.req_pool_idx])

        tokens = self._build_topk_tokens(fill_len)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        locs = self._swap_in_selected_pages(rpi, sls, batch, layer_id=0)
        valid_n = min(fill_len, TOP_K)
        self.assertTrue(torch.all(locs[0, :valid_n] >= 0))
        self._assert_kv_correct(
            locs[0], tokens, layer_id=0, count=valid_n, msg="Staging: "
        )

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "staging_path")

    # ==================================================================
    # Test: Single-node staging host page allocation
    # ==================================================================
    def test_single_node_staging_allocates_paged_host_slots(self):
        """Single-node staging should allocate host slots at page granularity."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size * 2 + 1
        rounded_len = (fill_len + self.page_size - 1) // self.page_size * self.page_size
        req = _make_req("single-node-staging-pages", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)

        self.coordinator.admit_request_into_staging(req)
        torch.cuda.synchronize()
        ready = self.coordinator.collect_ready_reqs()
        self.assertEqual(ready, [req])

        host_row = self.coordinator.req_to_host_pool[req.req_pool_idx, :rounded_len]
        self.assertTrue(torch.all(host_row >= 0))
        self.assertEqual(torch.unique(host_row).numel(), rounded_len)
        self.assertEqual(
            int(self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx]),
            rounded_len,
        )

        available_size = self.coordinator.mem_pool_host.available_size()
        next_host_index = self.coordinator.mem_pool_host.alloc_paged_token_slots(
            self.coordinator.req_to_host_pool,
            self.coordinator.req_to_host_pool_allocated_len,
            req.req_pool_idx,
            fill_len,
            1,
        )
        # With page_size>1 the rounded-up staging allocation provides headroom,
        # so no new pages are needed.  With page_size=1 there is no headroom and
        # exactly one new page is allocated for the next token.
        expected_new_pages = 0 if fill_len < rounded_len else 1
        self.assertEqual(
            self.coordinator.mem_pool_host.available_size(),
            available_size - expected_new_pages,
        )
        self.assertTrue(torch.all(next_host_index >= 0))

        expected_total = rounded_len + expected_new_pages * self.page_size
        allocated_host_indices = self.coordinator.mem_pool_host.allocated_host_indices(
            self.coordinator.req_to_host_pool,
            req.req_pool_idx,
            int(self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx]),
        )
        self.assertEqual(allocated_host_indices.numel(), expected_total)

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "single_node_staging_pages")

    # ==================================================================
    # Test: Direct-to-host (PD separated) path
    # ==================================================================
    def test_request_lifecycle_direct_path(self):
        """alloc_logical_only -> host write -> resident promotion -> finish."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size
        req = _make_req("direct-req", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        self.assertFalse(req.staging)
        self.assertTrue(self.coordinator._is_resident(req.req_pool_idx))
        self.assertEqual(self.coordinator._promotion_count, 0)
        self.assertTrue(self.coordinator._skip_first_backup[req.req_pool_idx])
        self.assertEqual(self.coordinator.req_device_buffer_size[req.req_pool_idx], 0)

        tokens = self._build_topk_tokens(fill_len - 1)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        for layer_id in range(LAYER_NUM):
            locs = self._swap_in_selected_pages(rpi, sls, batch, layer_id)
            self.assertTrue(torch.all(locs[0, :TOP_K] >= 0))
            self._assert_kv_correct(
                locs[0], tokens, layer_id=layer_id, count=TOP_K, msg="Direct: "
            )

        self._cleanup_req(req, kv_loc, logical_only=True)

    def test_admission_once_suppresses_repromotion_after_demotion(self):
        """A pressure-demoted request remains host-backed until it finishes."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + 2 * self.page_size
        req = _make_req("admission-once", list(range(fill_len)))
        self._alloc_req_slot(req)
        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)

        previous_mode = self.coordinator.dynamic_residency_mode
        self.coordinator.dynamic_residency_mode = "admission_once"
        try:
            self.coordinator.admit_request_direct(req)
            self.assertTrue(self.coordinator._is_resident(req.req_pool_idx))

            self.coordinator._demote_resident_request(req)
            self.assertFalse(self.coordinator._is_resident(req.req_pool_idx))
            self.assertFalse(self.coordinator._try_promote_from_host(req))
            self.assertFalse(self.coordinator._try_promote_from_host(req))

            stats = self.coordinator.get_token_stats()
            self.assertEqual(stats.promotions, 0)
            self.assertEqual(stats.demotions, 1)
            self.assertEqual(stats.repromotion_suppressed, 1)
        finally:
            self.coordinator.dynamic_residency_mode = previous_mode
            self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "admission_once")

    def test_admission_window_only_selects_resident_at_direct_admission(self):
        """A worker lease survives finish and runtime rebalance cannot acquire it."""
        initial = self._get_initial_sizes()
        previous_mode = self.coordinator.dynamic_residency_mode
        previous_seconds = (
            self.coordinator.dynamic_residency_admission_window_seconds
        )
        previous_deadline = self.coordinator._admission_window_next_allowed_at
        previous_owner = self.coordinator._admission_window_owner_req_idx
        self.coordinator.dynamic_residency_mode = "admission_window"
        self.coordinator.dynamic_residency_admission_window_seconds = 1800
        self.coordinator._admission_window_next_allowed_at = 0.0
        self.coordinator._admission_window_owner_req_idx = None
        try:
            first_len = DEVICE_BUFFER_SIZE + 2 * self.page_size
            first = _make_req("admission-window-first", list(range(first_len)))
            self._alloc_req_slot(first)
            first_kv = self._alloc_kv(first, first_len, logical_only=True)
            self._populate_host_pool(first, first_len)
            self.coordinator.admit_request_direct(first)
            self.assertTrue(self.coordinator._is_resident(first.req_pool_idx))
            self.assertEqual(
                self.coordinator._admission_window_owner_req_idx,
                first.req_pool_idx,
            )
            self._cleanup_req(first, first_kv, logical_only=True)

            second_len = DEVICE_BUFFER_SIZE + 2 * self.page_size
            second = _make_req("admission-window-second", list(range(second_len)))
            self._alloc_req_slot(second)
            second_kv = self._alloc_kv(second, second_len, logical_only=True)
            self._populate_host_pool(second, second_len)
            self.coordinator.admit_request_direct(second)
            self.assertEqual(
                self.coordinator._state(second.req_pool_idx),
                HiSparseResidencyState.DEVICE_BUFFERED,
            )

            # Even after the wall-clock lease is made available, a request
            # already executing decode cannot acquire residency.  Only a new
            # direct-admission boundary may do so.
            self.coordinator._admission_window_next_allowed_at = 0.0
            self.assertFalse(self.coordinator._try_promote_from_host(second))
            self._cleanup_req(second, second_kv, logical_only=True)

            third_len = DEVICE_BUFFER_SIZE + 2 * self.page_size
            third = _make_req("admission-window-third", list(range(third_len)))
            self._alloc_req_slot(third)
            third_kv = self._alloc_kv(third, third_len, logical_only=True)
            self._populate_host_pool(third, third_len)
            self.coordinator.admit_request_direct(third)
            self.assertTrue(self.coordinator._is_resident(third.req_pool_idx))
            self._cleanup_req(third, third_kv, logical_only=True)
        finally:
            self.coordinator.dynamic_residency_mode = previous_mode
            self.coordinator.dynamic_residency_admission_window_seconds = (
                previous_seconds
            )
            self.coordinator._admission_window_next_allowed_at = previous_deadline
            self.coordinator._admission_window_owner_req_idx = previous_owner
        self._assert_sizes_restored(initial, "admission_window")

    def test_runtime_promotion_detaches_speculative_tail_from_old_buffer(self):
        """Promotion transfers ownership without leaving a double-free tail.

        A target-verify batch may over-allocate logical slots beyond the
        sequence length used by the next residency rebalance.  Those slots can
        still point at the host-backed device buffer when promotion publishes
        the new resident prefix.  Releasing the old buffer must detach every
        such request-local mapping before generic request cleanup runs.
        """
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + 2 * self.page_size
        committed_len = fill_len - 3
        req = _make_req("promotion-spec-tail", list(range(fill_len)))
        self._alloc_req_slot(req)
        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self._admit_direct_to_buffer(req)

        req_idx = req.req_pool_idx
        old_buffer_locs = self.coordinator.req_to_device_buffer[
            req_idx, : int(self.coordinator.req_device_buffer_size[req_idx])
        ].clone()
        # The last three already-allocated logical slots model target-verify
        # over-allocation beyond the committed sequence span.
        extra_locs = kv_loc[-3:]
        req.kv_allocated_len = fill_len
        req.kv_committed_len = committed_len

        mapping = self.allocator.full_to_hisparse_device_index_mapping
        extra_compressed = self.device_pool.translate_loc_from_full_to_compressed(
            extra_locs
        )
        stale_page_locs = old_buffer_locs[-3:]
        mapping[extra_compressed] = stale_page_locs

        self.assertTrue(
            self.coordinator._try_promote_from_host(req, logical_len=committed_len)
        )
        self.assertTrue(self.coordinator._is_resident(req_idx))
        self.assertTrue(torch.all(mapping[extra_compressed] == 0))

        self.coordinator.request_finished(req)
        self.allocator.free(kv_loc)
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "promotion_spec_tail")

    def test_forced_host_backed_skips_direct_residency(self):
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size
        req = _make_req("forced-host-backed", list(range(fill_len)))
        self._alloc_req_slot(req)
        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)

        previous_mode = self.coordinator.dynamic_residency_mode
        self.coordinator.dynamic_residency_mode = "forced_host_backed"
        try:
            self.coordinator.admit_request_direct(req)
            self.assertEqual(
                self.coordinator._state(req.req_pool_idx),
                HiSparseResidencyState.DEVICE_BUFFERED,
            )
        finally:
            self.coordinator.dynamic_residency_mode = previous_mode
            self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "forced_host_backed")

    def test_dynamic_target_draft_share_residency_transitions(self):
        """Target owns allocations; draft mirrors resident and swap slots."""
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
        from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool

        initial = self._get_initial_sizes()
        # Keep the resident allocation larger than the fixed host-backed
        # device buffer so demotion has a measurable net capacity benefit.
        fill_len = DEVICE_BUFFER_SIZE + 2 * self.page_size
        req = _make_req("dynamic-target-draft", list(range(fill_len)))
        self._alloc_req_slot(req)
        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        host_indices = self._populate_host_pool(req, fill_len)

        draft_pool = HiSparseDSATokenToKVPool(
            size=SIZE,
            page_size=self.page_size,
            kv_lora_rank=KV_LORA_RANK,
            dtype=torch.bfloat16,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            layer_num=1,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=KV_CACHE_DIM,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )
        draft_pool.register_mapping(
            self.allocator.full_to_hisparse_device_index_mapping
        )
        draft = HiSparseCoordinator(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            top_k=TOP_K,
            device_buffer_size=DEVICE_BUFFER_SIZE,
            device="cuda",
            tp_group=self.tp_group,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
            max_num_steps=3,
            mem_pool_device_override=draft_pool,
            dynamic_residency=True,
            dynamic_residency_max_tokens=MAX_CONTEXT_LEN,
            dynamic_residency_max_requests=MAX_NUM_REQS,
            dynamic_residency_min_remaining_tokens=0,
            dynamic_residency_promote_watermark=0.01,
            dynamic_residency_demote_watermark=0.0,
            dynamic_residency_cooldown_steps=0,
        )
        self.coordinator.register_device_slot_mirror(draft)
        draft.mirror_host_slots_from(self.coordinator, req.req_pool_idx)
        for i in range(fill_len):
            draft.mem_pool_host.kv_buffer[0][host_indices[i]] = self._kv_pattern(0, i)

        try:
            self.coordinator.admit_request_direct(req)
            draft.admit_request_direct(req, device_slot_owner=self.coordinator)
            self.assertTrue(self.coordinator._is_resident(req.req_pool_idx))
            self.assertTrue(draft._is_resident(req.req_pool_idx))

            self.coordinator._demote_resident_request(req)
            self.assertFalse(self.coordinator._is_resident(req.req_pool_idx))
            self.assertFalse(draft._is_resident(req.req_pool_idx))
            self.assertTrue(
                torch.equal(
                    self.coordinator.req_to_device_buffer[req.req_pool_idx],
                    draft.req_to_device_buffer[req.req_pool_idx],
                )
            )

            self.assertTrue(self.coordinator._try_promote_from_host(req))
            self.assertTrue(self.coordinator._is_resident(req.req_pool_idx))
            self.assertTrue(draft._is_resident(req.req_pool_idx))
            self.assertEqual(self.coordinator._promotion_count, 1)
            self.assertEqual(draft._promotion_count, 1)
            self.assertEqual(
                int(self.coordinator.req_device_buffer_size[req.req_pool_idx]), 0
            )
            self.assertEqual(int(draft.req_device_buffer_size[req.req_pool_idx]), 0)

            torch.cuda.synchronize()
            migration_stats = self.coordinator.get_token_stats()
            expected_promotion_bytes = fill_len * (
                self.device_pool.bytes_per_token * self.device_pool.layer_num
                + draft_pool.bytes_per_token * draft_pool.layer_num
            )
            self.assertEqual(
                migration_stats.promotion_migrated_bytes,
                expected_promotion_bytes,
            )
            self.assertGreaterEqual(migration_stats.promotion_migration_seconds, 0.0)
            self.assertGreater(migration_stats.demotion_reclaimed_bytes, 0)
            self.assertGreaterEqual(migration_stats.demotion_transition_seconds, 0.0)

            mapped = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
            self.assertTrue(torch.all(mapped > 0))
            self.assertTrue(
                torch.allclose(
                    draft_pool.kv_buffer[0][mapped].float(),
                    self.device_pool.kv_buffer[0][mapped].float(),
                    atol=1e-2,
                )
            )

            req_idx = req.req_pool_idx
            req_pool_indices = torch.tensor([req_idx], dtype=torch.int64, device="cuda")
            tokens_per_req_cpu = torch.tensor([3], dtype=torch.int64)
            start_positions_cpu = torch.tensor([fill_len], dtype=torch.int64)
            available_before_spec = (
                self.allocator.hisparse_attn_allocator.available_size()
            )
            target_slots = self.coordinator.get_draft_device_slots_variable(
                req_pool_indices,
                tokens_per_req_cpu,
                start_positions_cpu,
            )
            draft.mirror_device_slots_from(self.coordinator, req_idx)
            draft_slots = draft.get_draft_device_slots_variable(
                req_pool_indices,
                tokens_per_req_cpu,
                start_positions_cpu,
            )
            self.assertTrue(torch.equal(target_slots, draft_slots))
            self.assertEqual(
                available_before_spec
                - self.allocator.hisparse_attn_allocator.available_size(),
                self.page_size,
                "Resident MTP must allocate one speculative page, not a hot buffer",
            )
            self.assertTrue(
                torch.all(
                    self.coordinator.req_device_buffer_token_locs[
                        :, req_idx, :DEVICE_BUFFER_SIZE
                    ]
                    == -1
                ),
                "Resident hot-buffer metadata must stay invalid",
            )

            verify_locs = self.allocator.alloc_extend_with_device_mapping(
                prefix_lens=torch.tensor([fill_len], dtype=torch.int64, device="cuda"),
                prefix_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                seq_lens=torch.tensor([fill_len + 3], dtype=torch.int64, device="cuda"),
                seq_lens_cpu=torch.tensor([fill_len + 3], dtype=torch.int64),
                last_loc=kv_loc[-1:],
                extend_num_tokens=3,
                device_slots=target_slots,
            )
            self.req_to_token_pool.write(
                (req_idx, slice(fill_len, fill_len + 3)), verify_locs
            )
            req.kv_allocated_len = fill_len + 3
            req.kv_committed_len = fill_len + 3
            for lid in range(LAYER_NUM):
                for pos in range(3):
                    self.device_pool.kv_buffer[lid][target_slots[pos]] = (
                        self._kv_pattern(lid, fill_len + pos)
                    )
            for pos in range(3):
                draft_pool.kv_buffer[0][draft_slots[pos]] = self._kv_pattern(
                    0, fill_len + pos
                )

            self.coordinator.finalize_accepted_tokens_spec_v2(
                req_pool_indices=req_pool_indices,
                seq_lens=torch.tensor([fill_len], dtype=torch.int64, device="cuda"),
                verify_cache_locs=verify_locs,
                accept_index=torch.tensor(
                    [[0, 1, -1]], dtype=torch.int64, device="cuda"
                ),
                mirror=draft,
            )
            torch.cuda.synchronize()
            permanent = self.allocator.full_to_hisparse_device_index_mapping[
                verify_locs
            ]
            self.assertTrue(torch.all(permanent[:2] > 0))
            self.assertTrue(torch.all(permanent[2:] == 0))
            self.assertTrue(torch.all(permanent[:2] != target_slots[:2]))
            for pos in range(2):
                for lid in range(LAYER_NUM):
                    self.assertTrue(
                        torch.allclose(
                            self.device_pool.kv_buffer[lid][permanent[pos]].float(),
                            torch.full_like(
                                self.device_pool.kv_buffer[lid][permanent[pos]].float(),
                                self._kv_pattern(lid, fill_len + pos),
                            ),
                            atol=1e-2,
                        )
                    )
                self.assertTrue(
                    torch.allclose(
                        draft_pool.kv_buffer[0][permanent[pos]].float(),
                        torch.full_like(
                            draft_pool.kv_buffer[0][permanent[pos]].float(),
                            self._kv_pattern(0, fill_len + pos),
                        ),
                        atol=1e-2,
                    )
                )
            self.assertEqual(
                int(self.coordinator.req_device_buffer_size[req_idx]),
                self.coordinator.padded_buffer_size,
                "Resident speculative page must remain graph-stable",
            )
        finally:
            self.coordinator.request_finished(req)
            draft.request_finished(req)
            allocated_locs = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : req.kv_allocated_len
            ].clone()
            self.allocator.free(allocated_locs)
            self._free_req_slot(req)
            self.coordinator._device_slot_mirrors.remove(draft)
            # The fixture replaces cudaHostRegister with torch pin_memory.
            # HiSparseCoordinator.destroy() intentionally calls
            # cudaHostUnregister for production buffers and is therefore not
            # valid for this test-only allocator override; normal tensor GC is
            # the matching cleanup here.
            del draft
            del draft_pool
            gc.collect()
            torch.cuda.synchronize()
            self.allocator.set_demote_until_hisparse_available(
                self.coordinator.demote_until_hisparse_available
            )
            self.allocator.set_schedulable_hisparse_available(
                self.coordinator.schedulable_hisparse_available
            )

        self._assert_sizes_restored(initial, "dynamic_target_draft")
        self._assert_sizes_restored(initial, "direct_path")

    def test_dynamic_mirror_accepts_allocator_terminal_page(self):
        """The last padding-shifted allocator page is backed by both KV pools."""
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
        from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool

        draft_pool = HiSparseDSATokenToKVPool(
            size=SIZE,
            page_size=self.page_size,
            kv_lora_rank=KV_LORA_RANK,
            dtype=torch.bfloat16,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            layer_num=1,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=KV_CACHE_DIM,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )
        draft_pool.register_mapping(
            self.allocator.full_to_hisparse_device_index_mapping
        )
        draft = HiSparseCoordinator(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            top_k=TOP_K,
            device_buffer_size=DEVICE_BUFFER_SIZE,
            device="cuda",
            tp_group=self.tp_group,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
            max_num_steps=3,
            mem_pool_device_override=draft_pool,
            dynamic_residency=True,
            dynamic_residency_max_tokens=MAX_CONTEXT_LEN,
            dynamic_residency_max_requests=MAX_NUM_REQS,
            dynamic_residency_min_remaining_tokens=0,
            dynamic_residency_promote_watermark=0.01,
            dynamic_residency_demote_watermark=0.0,
            dynamic_residency_cooldown_steps=0,
        )

        addressable_slots = SIZE + self.page_size
        terminal_slot = addressable_slots - 1
        terminal = torch.tensor([terminal_slot], dtype=torch.int64, device="cuda")
        draft._validate_mirror_device_locs(terminal)

        req_idx = 0
        self.coordinator.req_device_buffer_size[req_idx] = 1
        self.coordinator.req_to_device_buffer[req_idx, 0] = terminal_slot
        draft.mirror_device_slots_from(self.coordinator, req_idx)
        self.assertEqual(
            int(draft.req_to_device_buffer[req_idx, 0]), terminal_slot
        )

        with self.assertRaisesRegex(
            RuntimeError, f"slot={addressable_slots}, capacity={addressable_slots}"
        ):
            draft._validate_mirror_device_locs(
                torch.tensor([addressable_slots], dtype=torch.int64, device="cuda")
            )

        del draft
        del draft_pool
        gc.collect()

    def test_dynamic_demote_detaches_live_resident_spec_page(self):
        """Demotion must not return a graph-stable MTP page twice.

        Production can demote after target verify has bound over-allocated
        logical slots to the resident speculative page.  The transition must
        detach those mappings before alloc_device_buffer consumes the remaining
        resident mapping and releases its physical pages.
        """
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + 2 * self.page_size
        req = _make_req("dynamic-demote-live-spec", list(range(fill_len)))
        self._alloc_req_slot(req)
        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)
        self.assertTrue(self.coordinator._is_resident(req.req_pool_idx))

        req_pool_indices = torch.tensor(
            [req.req_pool_idx], dtype=torch.int64, device="cuda"
        )
        draft_slots = self.coordinator.get_draft_device_slots(
            req_pool_indices,
            3,
            torch.tensor([fill_len], dtype=torch.int64),
        )
        verify_locs = self.allocator.alloc_extend_with_device_mapping(
            prefix_lens=torch.tensor([fill_len], dtype=torch.int64, device="cuda"),
            prefix_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
            seq_lens=torch.tensor(
                [fill_len + 3], dtype=torch.int64, device="cuda"
            ),
            seq_lens_cpu=torch.tensor([fill_len + 3], dtype=torch.int64),
            last_loc=kv_loc[-1:],
            extend_num_tokens=3,
            device_slots=draft_slots,
        )
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(fill_len, fill_len + 3)), verify_locs
        )
        req.kv_allocated_len = fill_len + 3
        self.assertTrue(
            torch.all(
                self.allocator.full_to_hisparse_device_index_mapping[verify_locs]
                > 0
            )
        )

        self.coordinator._demote_resident_request(req)
        mapping = self.allocator.full_to_hisparse_device_index_mapping
        self.assertTrue(torch.all(mapping[verify_locs] == 0))
        physical = self.allocator.hisparse_attn_allocator
        self.assertLessEqual(physical.available_size(), physical.size)
        self.assertEqual(
            len(torch.unique(physical.free_pages)),
            len(physical.free_pages),
            "Demotion returned the resident speculative page more than once",
        )

        self.coordinator.request_finished(req)
        self.allocator.free(
            self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : req.kv_allocated_len
            ].clone()
        )
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "dynamic_demote_live_spec")

    def test_dynamic_repeated_mtp_residency_cycles_restore_physical_pages(self):
        """Repeated MTP commits and residency swaps must preserve page ownership.

        The production GLM-5.2 path can alternate between full residency and the
        fixed HiSparse device buffer several times while speculative verification
        keeps committing new tokens.  Exercise that combined lifecycle and leave
        the request buffered at finish, matching the long-generation failure mode.
        """
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
        from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool

        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + 2 * self.page_size
        req = _make_req("dynamic-repeated-mtp", list(range(fill_len)))
        self._alloc_req_slot(req)
        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        host_indices = self._populate_host_pool(req, fill_len)

        draft_pool = HiSparseDSATokenToKVPool(
            size=SIZE,
            page_size=self.page_size,
            kv_lora_rank=KV_LORA_RANK,
            dtype=torch.bfloat16,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            layer_num=1,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=KV_CACHE_DIM,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )
        draft_pool.register_mapping(
            self.allocator.full_to_hisparse_device_index_mapping
        )
        draft = HiSparseCoordinator(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            top_k=TOP_K,
            device_buffer_size=DEVICE_BUFFER_SIZE,
            device="cuda",
            tp_group=self.tp_group,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
            max_num_steps=3,
            mem_pool_device_override=draft_pool,
            dynamic_residency=True,
            dynamic_residency_max_tokens=MAX_CONTEXT_LEN,
            dynamic_residency_max_requests=MAX_NUM_REQS,
            dynamic_residency_min_remaining_tokens=0,
            dynamic_residency_promote_watermark=0.01,
            dynamic_residency_demote_watermark=0.0,
            dynamic_residency_cooldown_steps=0,
        )
        self.coordinator.register_device_slot_mirror(draft)
        draft.mirror_host_slots_from(self.coordinator, req.req_pool_idx)
        for i in range(fill_len):
            draft.mem_pool_host.kv_buffer[0][host_indices[i]] = self._kv_pattern(0, i)

        self.coordinator.admit_request_direct(req)
        draft.admit_request_direct(req, device_slot_owner=self.coordinator)
        self.assertTrue(self.coordinator._is_resident(req.req_pool_idx))

        physical = self.allocator.hisparse_attn_allocator

        def assert_physical_pages_unique(stage):
            self.assertLessEqual(
                physical.available_size(), physical.size, f"overflow at {stage}"
            )
            self.assertEqual(
                torch.unique(physical.free_pages).numel(),
                physical.free_pages.numel(),
                f"duplicate physical page at {stage}",
            )

        all_locs = [kv_loc]
        req_pool_indices = torch.tensor(
            [req.req_pool_idx], dtype=torch.int64, device="cuda"
        )

        def commit_three_tokens():
            start = req.kv_allocated_len
            device_slots = self.coordinator.get_draft_device_slots_variable(
                req_pool_indices,
                torch.tensor([3], dtype=torch.int64),
                torch.tensor([start], dtype=torch.int64),
            )
            last_loc = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, start - 1 : start
            ]
            verify_locs = self.allocator.alloc_extend_with_device_mapping(
                prefix_lens=torch.tensor([start], dtype=torch.int64, device="cuda"),
                prefix_lens_cpu=torch.tensor([start], dtype=torch.int64),
                seq_lens=torch.tensor([start + 3], dtype=torch.int64, device="cuda"),
                seq_lens_cpu=torch.tensor([start + 3], dtype=torch.int64),
                last_loc=last_loc,
                extend_num_tokens=3,
                device_slots=device_slots,
            )
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(start, start + 3)), verify_locs
            )
            req.kv_allocated_len = start + 3
            req.kv_committed_len = start + 3
            all_locs.append(verify_locs)

            self.coordinator.finalize_accepted_tokens_spec_v2(
                req_pool_indices=req_pool_indices,
                seq_lens=torch.tensor([start], dtype=torch.int64, device="cuda"),
                verify_cache_locs=verify_locs,
                accept_index=torch.tensor(
                    [[0, 1, 2]], dtype=torch.int32, device="cuda"
                ),
                mirror=draft,
            )
            return start

        for cycle in range(5):
            commit_three_tokens()
            assert_physical_pages_unique(f"cycle-{cycle}-commit")

            self.coordinator._demote_resident_request(req)
            self.assertFalse(self.coordinator._is_resident(req.req_pool_idx))
            assert_physical_pages_unique(f"cycle-{cycle}-demote")

            # Production remains host-backed for multiple target/draft steps
            # before the high watermark allows promotion again.
            for buffered_step in range(12):
                self.coordinator.finish_pending_draft_extend_backup()
                draft.finish_pending_draft_extend_backup()
                commit_three_tokens()
                assert_physical_pages_unique(
                    f"cycle-{cycle}-buffered-{buffered_step}"
                )
            self.coordinator.finish_pending_draft_extend_backup()
            draft.finish_pending_draft_extend_backup()

            if cycle < 4:
                self.assertTrue(self.coordinator._try_promote_from_host(req))
                self.assertTrue(self.coordinator._is_resident(req.req_pool_idx))
                assert_physical_pages_unique(f"cycle-{cycle}-promote")

        try:
            self.coordinator.request_finished(req)
            draft.request_finished(req)
            assert_physical_pages_unique("request-finished")
            self.allocator.free(torch.cat(all_locs))
            assert_physical_pages_unique("allocator-free")
            self._free_req_slot(req)
            self._assert_sizes_restored(initial, "dynamic_repeated_mtp_cycles")
        finally:
            self.coordinator._device_slot_mirrors.remove(draft)
            del draft
            del draft_pool
            gc.collect()
            torch.cuda.synchronize()
            self.allocator.set_demote_until_hisparse_available(
                self.coordinator.demote_until_hisparse_available
            )
            self.allocator.set_schedulable_hisparse_available(
                self.coordinator.schedulable_hisparse_available
            )

    def test_debug_lifecycle_rejects_physical_page_double_free(self):
        """The opt-in lifecycle probe fails before corrupting the free list."""
        physical = self.allocator.hisparse_attn_allocator
        initial = physical.available_size()
        page = physical.alloc(self.page_size)
        self.assertIsNotNone(page)
        old_debug = self.allocator.debug_validate_lifecycle
        self.allocator.debug_validate_lifecycle = True
        try:
            self.allocator.free_hisparse_indices(page)
            with self.assertRaisesRegex(
                RuntimeError, "HiSparse physical page double-free detected"
            ):
                self.allocator.free_hisparse_indices(page)
        finally:
            self.allocator.debug_validate_lifecycle = old_debug
        self.assertEqual(physical.available_size(), initial)
        self.assertEqual(
            torch.unique(physical.free_pages).numel(), physical.free_pages.numel()
        )

    def test_device_buffer_slot_cut_preserves_shared_physical_page(self):
        """A mapping hole must not free a page retained by the device buffer.

        Resident MTP cleanup can detach speculative token mappings before
        demotion.  Filtering those zero mappings shifts a slot-count cut away
        from a physical-page boundary.  The retained and surplus slot slices
        can then touch the same page, which remains owned by the buffer until
        the buffer itself is released.
        """
        physical = self.allocator.hisparse_attn_allocator
        initial = physical.available_size()
        resident = physical.alloc(3 * self.page_size)
        self.assertIsNotNone(resident)

        logical = torch.arange(
            1,
            resident.numel() + 1,
            dtype=torch.int64,
            device=self.allocator.device,
        )
        mapping = self.allocator.full_to_hisparse_device_index_mapping
        mapping[logical] = resident
        mapping[logical[0]] = 0

        buffer = self.allocator.alloc_device_buffer(logical, self.page_size)
        buffer_pages = torch.unique(buffer // self.page_size)
        self.assertFalse(
            torch.any(torch.isin(buffer_pages, physical.free_pages)),
            "A retained device-buffer page was returned by the surplus free",
        )

        self.allocator.free_hisparse_indices(buffer)
        self.assertEqual(physical.available_size(), initial)
        self.assertEqual(
            torch.unique(physical.free_pages).numel(), physical.free_pages.numel()
        )

    def test_device_buffer_discards_mapping_to_already_released_page(self):
        """A later demotion must not re-free a stale speculative mapping.

        A previous residency transition can release a page while a logical
        location outside that transition's request span still points to it.
        Once the over-allocated location enters a later demotion span, the
        allocator must detach and ignore the stale slot before splitting the
        retained device buffer from the surplus pages.
        """
        physical = self.allocator.hisparse_attn_allocator
        for stale_page_idx, placement in ((0, "retained"), (2, "surplus")):
            with self.subTest(placement=placement):
                initial = physical.available_size()
                resident = physical.alloc(3 * self.page_size)
                self.assertIsNotNone(resident)

                logical = torch.arange(
                    1,
                    resident.numel() + 1,
                    dtype=torch.int64,
                    device=self.allocator.device,
                )
                mapping = self.allocator.full_to_hisparse_device_index_mapping
                mapping[logical] = resident

                stale_start = stale_page_idx * self.page_size
                stale_page = resident[stale_start : stale_start + self.page_size]
                physical.free(stale_page)
                self.assertTrue(
                    torch.all(
                        torch.isin(
                            stale_page // self.page_size, physical.free_pages
                        )
                    )
                )

                old_debug = self.allocator.debug_validate_lifecycle
                self.allocator.debug_validate_lifecycle = True
                try:
                    buffer = self.allocator.alloc_device_buffer(
                        logical, self.page_size
                    )
                    buffer_pages = torch.unique(buffer // self.page_size)
                    self.assertFalse(
                        torch.any(torch.isin(buffer_pages, physical.free_pages))
                    )
                    self.allocator.free_hisparse_indices(buffer)
                finally:
                    self.allocator.debug_validate_lifecycle = old_debug

                self.assertEqual(physical.available_size(), initial)
                self.assertEqual(
                    torch.unique(physical.free_pages).numel(),
                    physical.free_pages.numel(),
                )

    def test_dynamic_spec_v2_mixed_resident_and_buffered_finalize(self):
        """One MTP batch commits resident and buffered rows independently."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size
        resident = _make_req("mtp-mixed-resident", list(range(fill_len)))
        buffered = _make_req("mtp-mixed-buffered", list(range(fill_len)))
        reqs = [resident, buffered]
        kv_locs = []

        for req in reqs:
            self._alloc_req_slot(req)
            kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
            self._populate_host_pool(req, fill_len)
            self.coordinator.admit_request_direct(req)
            kv_locs.append(kv_loc)
        self.coordinator._demote_resident_request(buffered)

        req_pool_indices = torch.tensor(
            [req.req_pool_idx for req in reqs], dtype=torch.int64, device="cuda"
        )
        tokens_per_req_cpu = torch.tensor([3, 3], dtype=torch.int64)
        start_positions_cpu = torch.tensor([fill_len, fill_len], dtype=torch.int64)
        device_slots = self.coordinator.get_draft_device_slots_variable(
            req_pool_indices, tokens_per_req_cpu, start_positions_cpu
        )
        verify_locs = self.allocator.alloc_extend_with_device_mapping(
            prefix_lens=torch.tensor(
                [fill_len, fill_len], dtype=torch.int64, device="cuda"
            ),
            prefix_lens_cpu=torch.tensor([fill_len, fill_len], dtype=torch.int64),
            seq_lens=torch.tensor(
                [fill_len + 3, fill_len + 3],
                dtype=torch.int64,
                device="cuda",
            ),
            seq_lens_cpu=torch.tensor([fill_len + 3, fill_len + 3], dtype=torch.int64),
            last_loc=torch.stack([kv_locs[0][-1], kv_locs[1][-1]]),
            extend_num_tokens=6,
            device_slots=device_slots,
        )
        for row, req in enumerate(reqs):
            row_locs = verify_locs[row * 3 : (row + 1) * 3]
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(fill_len, fill_len + 3)), row_locs
            )
            req.kv_allocated_len = fill_len + 3
            req.kv_committed_len = fill_len + 3
        for lid in range(LAYER_NUM):
            for pos, slot in enumerate(device_slots):
                self.device_pool.kv_buffer[lid][slot] = self._kv_pattern(
                    lid, fill_len + pos % 3
                )

        resident_sources = device_slots[:2].clone()
        self.coordinator.finalize_accepted_tokens_spec_v2(
            req_pool_indices=req_pool_indices,
            seq_lens=torch.tensor(
                [fill_len, fill_len], dtype=torch.int64, device="cuda"
            ),
            verify_cache_locs=verify_locs,
            accept_index=torch.tensor(
                [[0, 1, -1], [3, -1, -1]],
                # Runtime Spec V2 accept indices are int32 graph buffers.
                # Keep the mixed resident/buffered regression aligned with
                # that contract so row-local remapping cannot widen to int64.
                dtype=torch.int32,
                device="cuda",
            ),
        )
        torch.cuda.synchronize()

        mapping = self.allocator.full_to_hisparse_device_index_mapping[verify_locs]
        self.assertTrue(torch.all(mapping[:2] > 0))
        self.assertTrue(torch.all(mapping[:2] != resident_sources))
        self.assertEqual(
            int(mapping[3]),
            int(
                self.coordinator.req_to_device_buffer[
                    buffered.req_pool_idx, DEVICE_BUFFER_SIZE
                ]
            ),
        )
        self.assertTrue(torch.all(mapping[2:3] == 0))
        self.assertTrue(torch.all(mapping[4:] == 0))

        allocated_locs = []
        for req in reqs:
            allocated_locs.append(
                self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, : req.kv_allocated_len
                ].clone()
            )
            self.coordinator.request_finished(req)
            self._free_req_slot(req)
        self.allocator.free(torch.cat(allocated_locs))
        self._assert_sizes_restored(initial, "dynamic_spec_v2_mixed")

    def test_dynamic_resident_spec_v2_reuses_partially_filled_permanent_page(self):
        """Repeated accepts consume one resident page, not one page per verify.

        Spec V2 reuses a graph-stable side page for each verification step.  The
        accepted tokens must be copied into the resident request's current tail
        page until that page is full.  Grouping by transient logical page would
        instead allocate a fresh physical page on every iteration.
        """
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size
        req = _make_req("mtp-resident-page-reuse", list(range(fill_len)))
        self._alloc_req_slot(req)
        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        req_pool_indices = torch.tensor(
            [req.req_pool_idx], dtype=torch.int64, device="cuda"
        )
        self.coordinator.get_draft_device_slots(
            req_pool_indices,
            3,
            torch.tensor([fill_len], dtype=torch.int64),
        )
        available_before_commits = (
            self.allocator.hisparse_attn_allocator.available_size()
        )
        all_logical_locs = [kv_loc]

        for step in range(10):
            start = fill_len + step * 2
            device_slots = self.coordinator.get_draft_device_slots(
                req_pool_indices,
                3,
                torch.tensor([start], dtype=torch.int64),
            )
            verify_locs = self.allocator.alloc_extend_with_device_mapping(
                prefix_lens=torch.tensor([start], dtype=torch.int64, device="cuda"),
                prefix_lens_cpu=torch.tensor([start], dtype=torch.int64),
                seq_lens=torch.tensor([start + 3], dtype=torch.int64, device="cuda"),
                seq_lens_cpu=torch.tensor([start + 3], dtype=torch.int64),
                last_loc=self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, start - 1 : start
                ],
                extend_num_tokens=3,
                device_slots=device_slots,
            )
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(start, start + 3)), verify_locs
            )
            all_logical_locs.append(verify_locs)
            for lid in range(LAYER_NUM):
                for pos, slot in enumerate(device_slots):
                    self.device_pool.kv_buffer[lid][slot] = self._kv_pattern(
                        lid, start + pos
                    )

            self.coordinator.finalize_accepted_tokens_spec_v2(
                req_pool_indices=req_pool_indices,
                seq_lens=torch.tensor([start], dtype=torch.int64, device="cuda"),
                verify_cache_locs=verify_locs,
                accept_index=torch.tensor(
                    [[0, 1, -1]], dtype=torch.int32, device="cuda"
                ),
            )
            self.req_to_token_pool.req_to_token[req.req_pool_idx, start + 2] = 0
            req.kv_allocated_len = start + 2
            req.kv_committed_len = start + 2

        available_after_commits = (
            self.allocator.hisparse_attn_allocator.available_size()
        )
        self.assertEqual(
            available_before_commits - available_after_commits,
            self.page_size,
            "Twenty sequential accepted tokens should share one permanent page",
        )

        self.coordinator.request_finished(req)
        self._free_req_slot(req)
        self.allocator.free(torch.cat(all_logical_locs))
        self._assert_sizes_restored(initial, "dynamic_spec_v2_page_reuse")

    def test_dynamic_resident_spec_v2_backup_failure_rolls_back(self):
        """A failed resident commit preserves side mappings and frees new pages."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size
        req = _make_req("mtp-resident-rollback", list(range(fill_len)))
        self._alloc_req_slot(req)
        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        req_pool_indices = torch.tensor(
            [req.req_pool_idx], dtype=torch.int64, device="cuda"
        )
        device_slots = self.coordinator.get_draft_device_slots(
            req_pool_indices,
            3,
            torch.tensor([fill_len], dtype=torch.int64),
        )
        verify_locs = self.allocator.alloc_extend_with_device_mapping(
            prefix_lens=torch.tensor([fill_len], dtype=torch.int64, device="cuda"),
            prefix_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
            seq_lens=torch.tensor([fill_len + 3], dtype=torch.int64, device="cuda"),
            seq_lens_cpu=torch.tensor([fill_len + 3], dtype=torch.int64),
            last_loc=kv_loc[-1:],
            extend_num_tokens=3,
            device_slots=device_slots,
        )
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(fill_len, fill_len + 3)), verify_locs
        )
        req.kv_allocated_len = fill_len + 3
        req.kv_committed_len = fill_len + 3
        mapping_before = self.allocator.full_to_hisparse_device_index_mapping[
            verify_locs
        ].clone()
        available_before = self.allocator.hisparse_attn_allocator.available_size()
        original_backup = self.coordinator._backup_device_locs_to_host
        self.coordinator._backup_device_locs_to_host = lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(RuntimeError("injected backup failure"))
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "injected backup failure"):
                self.coordinator.finalize_accepted_tokens_spec_v2(
                    req_pool_indices=req_pool_indices,
                    seq_lens=torch.tensor([fill_len], dtype=torch.int64, device="cuda"),
                    verify_cache_locs=verify_locs,
                    accept_index=torch.tensor(
                        [[0, 1, -1]], dtype=torch.int64, device="cuda"
                    ),
                )
        finally:
            self.coordinator._backup_device_locs_to_host = original_backup

        self.assertTrue(
            torch.equal(
                self.allocator.full_to_hisparse_device_index_mapping[verify_locs],
                mapping_before,
            )
        )
        self.assertEqual(
            self.allocator.hisparse_attn_allocator.available_size(), available_before
        )

        self.allocator.clear_device_mapping(verify_locs)
        self.allocator.logical_attn_allocator.free(verify_locs)
        req.kv_allocated_len = fill_len
        self.coordinator.request_finished(req)
        self.allocator.free(kv_loc)
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "dynamic_spec_v2_rollback")

    # ==================================================================
    # Test: PD decode prealloc host page allocation
    # ==================================================================
    def test_pd_decode_prealloc_hisparse_host_slots(self):
        """PD decode prealloc should allocate RDMA targets through the host pool."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size * 2 + 1
        req = _make_req("pd-decode-prealloc", list(range(fill_len)))

        from sglang.srt.disaggregation.decode import DecodePreallocQueue

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.req_to_token_pool = self.req_to_token_pool
        queue.token_to_kv_pool_allocator = self.allocator
        queue.token_to_kv_pool = self.allocator.get_kvcache()
        queue.tree_cache = SimpleNamespace(
            evictable_size=lambda: 0,
            protected_size=lambda: 0,
        )
        queue.scheduler = SimpleNamespace(
            enable_hisparse=True,
            hisparse_coordinator=self.coordinator,
            draft_hisparse_coordinator=None,
            server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=False),
        )

        host_indices = queue._pre_alloc(req)
        self.assertEqual(host_indices.numel(), fill_len)
        self.assertTrue(torch.all(host_indices >= 0))
        self.assertTrue(
            torch.equal(
                host_indices,
                self.coordinator.req_to_host_pool[req.req_pool_idx, :fill_len],
            )
        )
        self.assertEqual(req.kv_allocated_len, fill_len)
        self.assertEqual(req.kv_committed_len, fill_len)
        self.assertEqual(req.extend_range.length, fill_len)

        reserved_len = fill_len + req.sampling_params.max_new_tokens
        rounded_len = (
            (reserved_len + self.page_size - 1) // self.page_size * self.page_size
        )
        self.assertEqual(
            int(self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx]),
            rounded_len,
        )
        allocated_host_indices = self.coordinator.mem_pool_host.allocated_host_indices(
            self.coordinator.req_to_host_pool,
            req.req_pool_idx,
            int(self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx]),
        )
        self.assertEqual(allocated_host_indices.numel(), rounded_len)

        kv_loc = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.kv_allocated_len
        ].clone()
        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "pd_decode_prealloc_hisparse")

    # ==================================================================
    # Test: Batch multiple requests
    # ==================================================================
    def test_batch_multiple_requests(self):
        """Mix of short & long requests in batch: kernel correct + no leaks."""
        initial = self._get_initial_sizes()

        configs = [
            ("batch-short-0", self.page_size),
            ("batch-short-1", self.page_size),
            ("batch-long-0", DEVICE_BUFFER_SIZE + self.page_size),
            ("batch-long-1", DEVICE_BUFFER_SIZE + self.page_size * 2),
        ]

        reqs, kv_locs = [], []
        for rid, fl in configs:
            req = _make_req(rid, list(range(fl)))
            self._alloc_req_slot(req)
            is_long = fl > DEVICE_BUFFER_SIZE
            kv_loc = self._alloc_kv(req, fl, logical_only=is_long)
            if is_long:
                self._populate_host_pool(req, fl)
                self.coordinator.admit_request_direct(req)
            else:
                self._write_device_patterns(kv_loc, fl)
                self.coordinator.alloc_device_buffer(req)
            reqs.append(req)
            kv_locs.append(kv_loc)

        rpi, sls = self._make_batch_tensors(reqs, [c[1] for c in configs])
        top_k_batch = torch.stack(
            [
                # For long sequences pass fl-1 to exclude the "newest token" position
                # whose reserved device-buffer slot is not populated in unit tests.
                self._build_topk_tokens(fl - 1 if fl > DEVICE_BUFFER_SIZE else fl)
                for _, fl in configs
            ]
        )

        for lid in range(LAYER_NUM):
            locs = self._swap_in_selected_pages(rpi, sls, top_k_batch, lid)
            for i, (rid, fl) in enumerate(configs):
                vn = min(fl, TOP_K)
                self.assertTrue(
                    torch.all(locs[i, :vn] >= 0),
                    f"Req {rid}, layer {lid}: negative locs",
                )
                self._assert_kv_correct(
                    locs[i], top_k_batch[i], lid, vn, msg=f"{rid}: "
                )

        for i, req in enumerate(reqs):
            is_long = configs[i][1] > DEVICE_BUFFER_SIZE
            self._cleanup_req(req, kv_locs[i], logical_only=is_long)

        self._assert_sizes_restored(initial, "batch_multiple")

    # ==================================================================
    # Test: HiSparse MTP draft slots
    # ==================================================================
    def test_draft_slots_use_extra_page_after_newest_slot(self):
        """Uniform draft slots start after the newest-token slot."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE
        draft_num = 3
        req = _make_req("draft-slots-uniform", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self.coordinator.alloc_device_buffer(req)

        req_idx = req.req_pool_idx
        extra_start = DEVICE_BUFFER_SIZE + 1
        req_pool_indices = torch.tensor([req_idx], dtype=torch.int64, device="cuda")
        start_positions_cpu = torch.tensor([DEVICE_BUFFER_SIZE], dtype=torch.int64)

        hot_tokens_before = self.coordinator.req_device_buffer_tokens[
            :, req_idx, :DEVICE_BUFFER_SIZE
        ].clone()
        newest_token_sentinel = torch.full(
            (LAYER_NUM,), 777, dtype=torch.int32, device="cuda"
        )
        self.coordinator.req_device_buffer_tokens[:, req_idx, DEVICE_BUFFER_SIZE] = (
            newest_token_sentinel
        )

        device_slots = self.coordinator.get_draft_device_slots(
            req_pool_indices,
            draft_num,
            start_positions_cpu,
        )

        expected_slots = self.coordinator.req_to_device_buffer[
            req_idx, extra_start : extra_start + draft_num
        ]
        expected_token_positions = torch.arange(
            DEVICE_BUFFER_SIZE,
            DEVICE_BUFFER_SIZE + draft_num,
            dtype=torch.int32,
            device="cuda",
        )

        self.assertTrue(torch.equal(device_slots, expected_slots))
        self.assertTrue(
            torch.equal(
                self.coordinator.req_device_buffer_tokens[
                    :, req_idx, :DEVICE_BUFFER_SIZE
                ],
                hot_tokens_before,
            ),
            "Draft slots must not rewrite hot-buffer token metadata",
        )
        self.assertTrue(
            torch.equal(
                self.coordinator.req_device_buffer_tokens[
                    :, req_idx, DEVICE_BUFFER_SIZE
                ],
                newest_token_sentinel,
            ),
            "Draft slots must not overwrite the newest-token slot",
        )
        self.assertTrue(
            torch.equal(
                self.coordinator.req_device_buffer_tokens[
                    :, req_idx, extra_start : extra_start + draft_num
                ],
                expected_token_positions.unsqueeze(0).expand(LAYER_NUM, -1),
            )
        )

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "draft_slots_uniform")

    def test_multistep_swap_resolves_graph_stable_extra_page_slots(self):
        """Target-verify must not treat draft extra-page slots as host misses."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size
        req = _make_req("multistep-extra-page", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        req_idx = req.req_pool_idx
        num_steps = 3
        extra_start = DEVICE_BUFFER_SIZE + 1
        req_pool_indices = torch.tensor([req_idx], dtype=torch.int64, device="cuda")
        expected_slots = self.coordinator.get_draft_device_slots(
            req_pool_indices,
            num_steps,
            torch.tensor([fill_len], dtype=torch.int64),
        )
        verify_locs = self.allocator.alloc_extend_with_device_mapping(
            prefix_lens=torch.tensor([fill_len], dtype=torch.int64, device="cuda"),
            prefix_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
            seq_lens=torch.tensor(
                [fill_len + num_steps], dtype=torch.int64, device="cuda"
            ),
            seq_lens_cpu=torch.tensor([fill_len + num_steps], dtype=torch.int64),
            last_loc=kv_loc[-1:],
            extend_num_tokens=num_steps,
            device_slots=expected_slots,
        )
        self.req_to_token_pool.write(
            (req_idx, slice(fill_len, fill_len + num_steps)), verify_locs
        )
        draft_positions = torch.arange(
            fill_len,
            fill_len + num_steps,
            dtype=torch.int32,
            device="cuda",
        )

        topk = torch.full((1, num_steps, TOP_K), -1, dtype=torch.int32, device="cuda")
        for step in range(num_steps):
            topk[0, step, : step + 1] = draft_positions[: step + 1]
        seq_lens = torch.arange(
            fill_len + 1,
            fill_len + num_steps + 1,
            dtype=torch.int32,
            device="cuda",
        )
        self.coordinator.num_real_reqs[0] = 1

        # Warm the JIT module before capture; the measured path below must be
        # graph-capturable and replay the same stable extra-page mappings.
        self.coordinator.swap_in_selected_pages(
            req_pool_indices=req_pool_indices,
            compressed_seq_lens=seq_lens,
            top_k_result=topk,
            layer_id=0,
            token_position_space="full",
            num_steps=num_steps,
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            locs = self.coordinator.swap_in_selected_pages(
                req_pool_indices=req_pool_indices,
                compressed_seq_lens=seq_lens,
                top_k_result=topk,
                layer_id=0,
                token_position_space="full",
                num_steps=num_steps,
            )
        graph.replay()
        torch.cuda.synchronize()

        expected = self.coordinator.req_device_buffer_token_locs[
            0, req_idx, extra_start : extra_start + num_steps
        ]
        self.assertTrue(torch.equal(expected, expected_slots.to(torch.int32)))
        for step in range(num_steps):
            self.assertTrue(
                torch.equal(locs[0, step, : step + 1], expected[: step + 1])
            )
            self.assertTrue(torch.all(locs[0, step, step + 1 :] == -1))
            self.assertTrue(
                torch.all(
                    self.coordinator.req_to_host_pool[
                        req_idx, fill_len : fill_len + step + 1
                    ]
                    == -1
                )
            )

        self.allocator.clear_device_mapping(verify_locs)
        self.allocator.logical_attn_allocator.free(verify_locs)
        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "multistep_extra_page")

    def test_multistep_short_seq_prefers_extra_page_draft_slots(self):
        """Short-sequence fast path must not alias draft positions to hot slots."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size
        req = _make_req("multistep-short-extra-page", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self.coordinator.alloc_device_buffer(req)
        req_idx = req.req_pool_idx
        num_steps = 3
        extra_start = DEVICE_BUFFER_SIZE + 1
        draft_positions = torch.arange(
            fill_len,
            fill_len + num_steps,
            dtype=torch.int32,
            device="cuda",
        )
        self.coordinator.req_device_buffer_tokens[
            :, req_idx, extra_start : extra_start + num_steps
        ] = draft_positions

        topk = torch.full((1, num_steps, TOP_K), -1, dtype=torch.int32, device="cuda")
        for step in range(num_steps):
            topk[0, step, 0] = draft_positions[step]
        seq_lens = torch.arange(
            fill_len + 1,
            fill_len + num_steps + 1,
            dtype=torch.int32,
            device="cuda",
        )
        req_pool_indices = torch.tensor([req_idx], dtype=torch.int64, device="cuda")
        self.coordinator.num_real_reqs[0] = 1
        locs = self.coordinator.swap_in_selected_pages(
            req_pool_indices=req_pool_indices,
            compressed_seq_lens=seq_lens,
            top_k_result=topk,
            layer_id=0,
            token_position_space="full",
            num_steps=num_steps,
        )
        torch.cuda.synchronize()

        expected = self.coordinator.req_device_buffer_token_locs[
            0, req_idx, extra_start : extra_start + num_steps
        ]
        self.assertTrue(torch.equal(locs[0, :, 0], expected))
        self.assertTrue(torch.all(locs[0, :, 1:] == -1))

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "multistep_short_extra_page")

    def test_draft_slots_variable_respect_per_request_counts(self):
        """Variable draft slots only populate each request's actual token count."""
        initial = self._get_initial_sizes()
        reqs = [
            _make_req("draft-slots-variable-0"),
            _make_req("draft-slots-variable-1"),
        ]
        for req in reqs:
            self._alloc_req_slot(req)

        req_pool_indices = torch.tensor(
            [req.req_pool_idx for req in reqs], dtype=torch.int64, device="cuda"
        )
        tokens_per_req_cpu = torch.tensor([1, 3], dtype=torch.int64)
        start_positions_cpu = torch.tensor(
            [DEVICE_BUFFER_SIZE, DEVICE_BUFFER_SIZE], dtype=torch.int64
        )
        extra_start = DEVICE_BUFFER_SIZE + 1

        device_slots = self.coordinator.get_draft_device_slots_variable(
            req_pool_indices,
            tokens_per_req_cpu,
            start_positions_cpu,
        )

        expected_slots = torch.cat(
            [
                self.coordinator.req_to_device_buffer[
                    reqs[0].req_pool_idx, extra_start : extra_start + 1
                ],
                self.coordinator.req_to_device_buffer[
                    reqs[1].req_pool_idx, extra_start : extra_start + 3
                ],
            ]
        )
        self.assertTrue(torch.equal(device_slots, expected_slots))

        req0_expected = torch.tensor(
            [DEVICE_BUFFER_SIZE, -1, -1], dtype=torch.int32, device="cuda"
        )
        req1_expected = torch.arange(
            DEVICE_BUFFER_SIZE,
            DEVICE_BUFFER_SIZE + 3,
            dtype=torch.int32,
            device="cuda",
        )
        self.assertTrue(
            torch.equal(
                self.coordinator.req_device_buffer_tokens[
                    :, reqs[0].req_pool_idx, extra_start : extra_start + 3
                ],
                req0_expected.unsqueeze(0).expand(LAYER_NUM, -1),
            )
        )
        self.assertTrue(
            torch.equal(
                self.coordinator.req_device_buffer_tokens[
                    :, reqs[1].req_pool_idx, extra_start : extra_start + 3
                ],
                req1_expected.unsqueeze(0).expand(LAYER_NUM, -1),
            )
        )

        for req in reqs:
            self.coordinator.request_finished(req)
            self._free_req_slot(req)
        self._assert_sizes_restored(initial, "draft_slots_variable")

    def test_finalize_accepted_tokens_remaps_newest_and_clears_rejected(self):
        """Accepted MTP tokens move the last accepted KV to newest slot."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE
        draft_num = 4
        req = _make_req("finalize-accepted", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self.coordinator.alloc_device_buffer(req)

        req_idx = req.req_pool_idx
        req_pool_indices = torch.tensor([req_idx], dtype=torch.int64, device="cuda")
        start_positions_cpu = torch.tensor([fill_len], dtype=torch.int64)
        draft_device_slots = self.coordinator.get_draft_device_slots(
            req_pool_indices,
            draft_num,
            start_positions_cpu,
        )

        device = self.allocator.device
        prefix_lens = torch.tensor([fill_len], dtype=torch.int64, device=device)
        prefix_lens_cpu = torch.tensor([fill_len], dtype=torch.int64)
        seq_lens = torch.tensor(
            [fill_len + draft_num], dtype=torch.int64, device=device
        )
        seq_lens_cpu = torch.tensor([fill_len + draft_num], dtype=torch.int64)
        last_loc = kv_loc[-1:].to(device=device)
        draft_cache_locs = self.allocator.alloc_extend_with_device_mapping(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            draft_num,
            draft_device_slots,
        )
        self.req_to_token_pool.write(
            (req_idx, slice(fill_len, fill_len + draft_num)), draft_cache_locs
        )
        req.kv_allocated_len = fill_len + draft_num
        req.kv_committed_len = fill_len + draft_num

        for lid in range(LAYER_NUM):
            for i in range(draft_num):
                self.device_pool.kv_buffer[lid][draft_device_slots[i]] = (
                    self._kv_pattern(lid, fill_len + i)
                )

        accepted_cache_locs = draft_cache_locs[:2]
        accepted_token_positions = torch.tensor(
            [fill_len, fill_len + 1], dtype=torch.int64, device="cuda"
        )
        self.coordinator.finalize_accepted_tokens(
            req_pool_indices=req_pool_indices,
            accepted_cache_locs=accepted_cache_locs,
            draft_cache_locs=draft_cache_locs,
            num_correct_drafts=torch.tensor([1], dtype=torch.int64, device="cuda"),
            num_correct_drafts_cpu=torch.tensor([1], dtype=torch.int64),
            accepted_token_positions=accepted_token_positions,
        )

        mapping = self.allocator.full_to_hisparse_device_index_mapping
        newest_slot = self.coordinator.req_to_device_buffer[req_idx, DEVICE_BUFFER_SIZE]

        self.assertEqual(
            int(mapping[accepted_cache_locs[0]].item()),
            int(draft_device_slots[0].item()),
        )
        self.assertEqual(int(mapping[accepted_cache_locs[-1]].item()), int(newest_slot))
        self.assertTrue(torch.all(mapping[draft_cache_locs[2:]] == 0))
        self.assertTrue(
            torch.all(
                self.coordinator.req_device_buffer_tokens[
                    :, req_idx, DEVICE_BUFFER_SIZE
                ]
                == fill_len + 1
            )
        )
        self.assertTrue(
            torch.all(
                self.coordinator.req_to_host_pool[req_idx, accepted_token_positions]
                >= 0
            )
        )
        for lid in range(LAYER_NUM):
            expected = self._kv_pattern(lid, fill_len + 1)
            actual = self.device_pool.kv_buffer[lid][newest_slot.long()]
            self.assertTrue(
                torch.allclose(
                    actual.float(),
                    torch.full_like(actual.float(), expected),
                    atol=1e-2,
                ),
                f"Layer {lid}: newest slot was not updated from last accepted token",
            )

        self.coordinator.finish_pending_draft_extend_backup()
        self.assertEqual(int(mapping[accepted_cache_locs[0]].item()), 0)
        self.assertEqual(int(mapping[accepted_cache_locs[-1]].item()), int(newest_slot))

        self.coordinator.request_finished(req)
        self.allocator.free(torch.cat([kv_loc, draft_cache_locs]))
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "finalize_accepted_tokens")

    def test_finalize_accepted_tokens_extends_mixed_host_boundary_without_duplicates(
        self,
    ):
        """A final verify window preserves preallocated slots at a host boundary."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size
        seq_len = fill_len - 2
        draft_num = 3
        req = _make_req("finalize-mixed-host-boundary", list(range(seq_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, seq_len)
        self.coordinator.alloc_device_buffer(req)
        preallocated_host_locs = self._populate_host_pool(req, fill_len)

        req_idx = req.req_pool_idx
        req_pool_indices = torch.tensor([req_idx], dtype=torch.int64, device="cuda")
        draft_device_slots = self.coordinator.get_draft_device_slots(
            req_pool_indices,
            draft_num,
            torch.tensor([seq_len], dtype=torch.int64),
        )
        draft_cache_locs = self.allocator.alloc_extend_with_device_mapping(
            prefix_lens=torch.tensor([seq_len], dtype=torch.int64, device="cuda"),
            prefix_lens_cpu=torch.tensor([seq_len], dtype=torch.int64),
            seq_lens=torch.tensor(
                [seq_len + draft_num], dtype=torch.int64, device="cuda"
            ),
            seq_lens_cpu=torch.tensor([seq_len + draft_num], dtype=torch.int64),
            last_loc=kv_loc[-1:],
            extend_num_tokens=draft_num,
            device_slots=draft_device_slots,
        )
        self.req_to_token_pool.write(
            (req_idx, slice(seq_len, seq_len + draft_num)), draft_cache_locs
        )
        req.kv_allocated_len = seq_len + draft_num
        req.kv_committed_len = seq_len + draft_num
        for lid in range(LAYER_NUM):
            for pos in range(draft_num):
                self.device_pool.kv_buffer[lid][draft_device_slots[pos]] = (
                    self._kv_pattern(lid, seq_len + pos)
                )

        accepted_positions = torch.tensor(
            [fill_len - 2, fill_len - 1, fill_len],
            dtype=torch.int64,
            device="cuda",
        )
        self.coordinator.finalize_accepted_tokens(
            req_pool_indices=req_pool_indices,
            accepted_cache_locs=draft_cache_locs,
            draft_cache_locs=draft_cache_locs,
            num_correct_drafts=torch.tensor([2], dtype=torch.int64, device="cuda"),
            num_correct_drafts_cpu=torch.tensor([2], dtype=torch.int64),
            accepted_token_positions=accepted_positions,
        )

        host_row = self.coordinator.req_to_host_pool[
            req_idx,
            : int(self.coordinator.req_to_host_pool_allocated_len[req_idx]),
        ]
        torch.testing.assert_close(
            host_row[fill_len - 2 : fill_len],
            preallocated_host_locs[fill_len - 2 : fill_len],
        )
        self.assertGreaterEqual(len(host_row), fill_len + 1)
        self.assertGreaterEqual(int(host_row[fill_len]), 0)
        self.assertEqual(torch.unique(host_row).numel(), host_row.numel())

        self.coordinator.finish_pending_draft_extend_backup()
        self.coordinator.request_finished(req)
        self.allocator.free(torch.cat([kv_loc, draft_cache_locs]))
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "finalize_mixed_host_boundary")

    def test_finalize_accepted_tokens_keeps_short_last_token_in_hot_slot(self):
        """Short-context accepted tokens use the hot slot read by the fast path."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size
        draft_num = 4
        req = _make_req("finalize-accepted-short", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self.coordinator.alloc_device_buffer(req)

        req_idx = req.req_pool_idx
        req_pool_indices = torch.tensor([req_idx], dtype=torch.int64, device="cuda")
        start_positions_cpu = torch.tensor([fill_len], dtype=torch.int64)
        draft_device_slots = self.coordinator.get_draft_device_slots(
            req_pool_indices,
            draft_num,
            start_positions_cpu,
        )

        device = self.allocator.device
        prefix_lens = torch.tensor([fill_len], dtype=torch.int64, device=device)
        prefix_lens_cpu = torch.tensor([fill_len], dtype=torch.int64)
        seq_lens = torch.tensor(
            [fill_len + draft_num], dtype=torch.int64, device=device
        )
        seq_lens_cpu = torch.tensor([fill_len + draft_num], dtype=torch.int64)
        last_loc = kv_loc[-1:].to(device=device)
        draft_cache_locs = self.allocator.alloc_extend_with_device_mapping(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            draft_num,
            draft_device_slots,
        )
        self.req_to_token_pool.write(
            (req_idx, slice(fill_len, fill_len + draft_num)), draft_cache_locs
        )
        req.kv_allocated_len = fill_len + draft_num
        req.kv_committed_len = fill_len + draft_num

        accepted_cache_locs = draft_cache_locs[:2]
        accepted_token_positions = torch.tensor(
            [fill_len, fill_len + 1], dtype=torch.int64, device="cuda"
        )
        self.coordinator.finalize_accepted_tokens(
            req_pool_indices=req_pool_indices,
            accepted_cache_locs=accepted_cache_locs,
            draft_cache_locs=draft_cache_locs,
            num_correct_drafts=torch.tensor([1], dtype=torch.int64, device="cuda"),
            num_correct_drafts_cpu=torch.tensor([1], dtype=torch.int64),
            accepted_token_positions=accepted_token_positions,
        )

        mapping = self.allocator.full_to_hisparse_device_index_mapping
        last_hot_slot = self.coordinator.req_to_device_buffer[req_idx, fill_len + 1]
        extra_newest_slot = self.coordinator.req_to_device_buffer[
            req_idx, DEVICE_BUFFER_SIZE
        ]

        self.assertEqual(
            int(mapping[accepted_cache_locs[-1]].item()), int(last_hot_slot.item())
        )
        self.assertNotEqual(int(last_hot_slot.item()), int(extra_newest_slot.item()))
        self.assertTrue(
            torch.all(
                self.coordinator.req_device_buffer_tokens[:, req_idx, fill_len + 1]
                == fill_len + 1
            )
        )
        self.assertTrue(
            torch.all(
                self.coordinator.req_device_buffer_token_locs[:, req_idx, fill_len + 1]
                == last_hot_slot.to(torch.int32)
            )
        )
        self.assertIsNone(self.coordinator._pending_draft_extend_backup)

        self.coordinator.request_finished(req)
        self.allocator.free(torch.cat([kv_loc, draft_cache_locs]))
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "finalize_accepted_tokens_short")

    def test_prepare_verify_slots_reuses_graph_stable_extra_page(self):
        """Replay updates logical mappings without reallocating physical slots."""
        initial = self._get_initial_sizes()
        req = _make_req("graph-stable-verify", list(range(DEVICE_BUFFER_SIZE)))
        self._alloc_req_slot(req)
        kv_loc = self._alloc_kv(req, DEVICE_BUFFER_SIZE)
        self.coordinator.alloc_device_buffer(req)

        req_idx = req.req_pool_idx
        req_pool_indices = torch.tensor([req_idx], dtype=torch.int64, device="cuda")
        start_positions_cpu = torch.tensor([DEVICE_BUFFER_SIZE], dtype=torch.int64)
        prefix_lens = torch.tensor(
            [DEVICE_BUFFER_SIZE], dtype=torch.int64, device="cuda"
        )
        prefix_lens_cpu = torch.tensor([DEVICE_BUFFER_SIZE], dtype=torch.int64)
        seq_lens = torch.tensor(
            [DEVICE_BUFFER_SIZE + 3], dtype=torch.int64, device="cuda"
        )
        seq_lens_cpu = torch.tensor([DEVICE_BUFFER_SIZE + 3], dtype=torch.int64)
        logical_a = self.allocator.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            kv_loc[-1:],
            3,
        )
        self.assertIsNotNone(logical_a)

        self.coordinator.prepare_verify_slots_spec_v2(
            req_pool_indices, logical_a, 3, start_positions_cpu
        )
        mapping = self.allocator.full_to_hisparse_device_index_mapping
        physical_a = mapping[logical_a].clone()
        physical_available = self.allocator.hisparse_attn_allocator.available_size()

        self.allocator.clear_device_mapping(logical_a)
        self.allocator.logical_attn_allocator.free(logical_a)
        logical_b = self.allocator.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            kv_loc[-1:],
            3,
        )
        self.assertIsNotNone(logical_b)

        self.coordinator.prepare_verify_slots_spec_v2(
            req_pool_indices, logical_b, 3, start_positions_cpu
        )
        physical_b = mapping[logical_b].clone()
        self.assertTrue(torch.equal(physical_a, physical_b))
        self.assertEqual(
            physical_available,
            self.allocator.hisparse_attn_allocator.available_size(),
            "Graph replay must not allocate a second physical draft page",
        )

        self.allocator.clear_device_mapping(logical_b)
        self.allocator.logical_attn_allocator.free(logical_b)
        self.coordinator.request_finished(req)
        self.allocator.free(kv_loc)
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "graph_stable_verify_slots")

    def test_dynamic_mixed_resident_and_swap_requests(self):
        """Dynamic mixed batch: resident rows use mapping, swap rows use LRU."""
        initial = self._get_initial_sizes()

        resident_len = DEVICE_BUFFER_SIZE + self.page_size * 3
        swap_len = DEVICE_BUFFER_SIZE + self.page_size * 4
        resident = _make_req("dynamic-resident", list(range(resident_len)))
        swap = _make_req("dynamic-swap", list(range(swap_len)))

        self._alloc_req_slot(resident)
        resident_kv = self._alloc_kv(resident, resident_len)
        self._write_device_patterns(resident_kv, resident_len)
        self.coordinator.active_hisparse_reqs[resident.req_pool_idx] = resident
        self.coordinator._set_residency_state(
            resident.req_pool_idx,
            HiSparseResidencyState.RESIDENT,
            count_transition=False,
        )

        self._alloc_req_slot(swap)
        swap_kv = self._alloc_kv(swap, swap_len, logical_only=True)
        self._populate_host_pool(swap, swap_len)
        self.coordinator.admit_request_direct(swap)

        reqs = [resident, swap]
        fill_lens = [resident_len, swap_len]
        rpi, sls = self._make_batch_tensors(reqs, fill_lens)
        top_k_batch = torch.stack(
            [
                self._build_topk_tokens(resident_len - 1),
                self._build_topk_tokens(swap_len - 1),
            ]
        )

        self.coordinator.num_real_reqs[0] = len(reqs)

        for lid in range(LAYER_NUM):
            locs = self.coordinator.swap_in_selected_pages(rpi, sls, top_k_batch, lid)

            resident_tokens = top_k_batch[0]
            valid = resident_tokens >= 0
            logical_locs = self.req_to_token_pool.req_to_token[
                resident.req_pool_idx, resident_tokens[valid].long()
            ]
            expected_locs = (
                self.allocator.full_to_hisparse_device_index_mapping[
                    logical_locs.long()
                ]
                .to(torch.int32)
                .cpu()
            )
            self.assertTrue(
                torch.equal(locs[0][valid].cpu(), expected_locs),
                f"Layer {lid}: dynamic resident locs != mapping",
            )
            self._assert_kv_correct(
                locs[0],
                resident_tokens,
                lid,
                int(valid.sum().item()),
                msg="Dynamic resident: ",
            )
            self._assert_kv_correct(
                locs[1],
                top_k_batch[1],
                lid,
                TOP_K,
                msg="Dynamic swap: ",
            )

        self._cleanup_req(resident, resident_kv)
        self._cleanup_req(swap, swap_kv, logical_only=True)
        self._assert_sizes_restored(initial, "dynamic_mixed")


if __name__ == "__main__":
    unittest.main()
