"""The donated mamba checkpoint depth must land on the tree page.

DCP widens the tree page past the mamba chunk grid. A checkpoint picked on the
finer grid names a depth no radix node can carry, so it gets attached to the
preceding node and a later request resumes from a state that already covers
tokens past that node.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    HybridLinearKVPool,
    HybridReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _compressed_index_tree_params,
)
from sglang.srt.runtime_context import get_context, mamba_track_grid, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CHUNK = 64


def _track_seqlen(*, tree_page: int, prefix_len: int, extend_len: int) -> int:
    """Run one extend through the tracker and report the donated depth."""
    server_args = ServerArgs(model_path="dummy", page_size=CHUNK)
    # The property would otherwise load the HF config for the dummy model.
    server_args._mamba_cache_chunk_size = CHUNK
    set_global_server_args_for_scheduler(server_args)

    sampling_params = SamplingParams(max_new_tokens=1)
    sampling_params.normalize(None)
    req = Req(
        rid="req",
        origin_input_text="",
        origin_input_ids=array("q", [1] * (prefix_len + extend_len)),
        sampling_params=sampling_params,
        vocab_size=128,
    )
    req.prefix_indices = torch.arange(prefix_len, dtype=torch.int64)
    req.set_extend_range(prefix_len, prefix_len + extend_len)
    req.kv.mamba_ping_pong_track_buffer = torch.tensor([0, 1], dtype=torch.int64)
    req.kv.mamba_next_track_idx = 0
    req.mamba_branching_seqlen = None

    batch = ScheduleBatch(reqs=[req])
    batch.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(mamba_chunk_size=CHUNK)
    )
    batch.tree_cache = SimpleNamespace(page_size=tree_page)
    batch.req_to_token_pool = MagicMock()
    batch.req_to_token_pool.get_mamba_ping_pong_other_idx.return_value = 1

    batch._mamba_radix_cache_v2_req_prepare_for_extend(req)
    return req.kv.mamba_last_track_seqlen


class TestMambaCheckpointDepth(unittest.TestCase):
    def test_widened_tree_page_moves_the_donated_depth_onto_it(self):
        # 4066 tokens past a 16384 prefix: the chunk grid would stop at 20416,
        # which a 256-token page cannot name.
        depth = _track_seqlen(tree_page=256, prefix_len=16384, extend_len=4066)
        self.assertEqual(depth % 256, 0)
        self.assertEqual(depth, 20224)

    def test_unwidened_tree_page_keeps_the_chunk_grid(self):
        depth = _track_seqlen(tree_page=CHUNK, prefix_len=16384, extend_len=4066)
        self.assertEqual(depth, 20416)


class TestMambaTrackGrid(unittest.TestCase):
    def _grid(self, *, interval: int, tree_page: int, chunk: int = CHUNK) -> int:
        with get_context().override_server_args(
            mamba_track_interval=interval, _mamba_cache_chunk_size=chunk
        ):
            return mamba_track_grid(tree_page)

    def test_widened_tree_page_rounds_the_interval_up(self):
        self.assertEqual(self._grid(interval=256, tree_page=512), 512)

    def test_interval_already_on_the_tree_page_is_untouched(self):
        self.assertEqual(self._grid(interval=256, tree_page=256), 256)
        self.assertEqual(self._grid(interval=256, tree_page=128), 256)
        self.assertEqual(self._grid(interval=256, tree_page=64), 256)

    def test_grid_stays_on_the_chunk_size(self):
        self.assertEqual(self._grid(interval=192, tree_page=64, chunk=128) % 128, 0)


class TestCompressedIndexCheckpoint(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def _fixture(self):
        args = ServerArgs(
            model_path="dummy", page_size=64, mamba_radix_cache_strategy="extra_buffer"
        )
        args._mamba_cache_chunk_size = 64
        set_global_server_args_for_scheduler(args)
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=16,
            n_groups=1,
            num_heads=1,
            head_dim=16,
            state_size=2,
            conv_kernel=4,
        )
        req_pool = HybridReqToTokenPool(
            size=4,
            mamba_size=32,
            mamba_spec_state_size=4,
            max_context_len=2048,
            device="cpu",
            enable_memory_saver=False,
            cache_params=Mamba2CacheParams(shape=shape, layers=[0]),
            mamba_layer_ids=[0],
            enable_mamba_extra_buffer=True,
        )
        dsa = object.__new__(DSATokenToKVPool)
        dsa.page_size = 64
        dsa.index_kpool = 4
        dsa.kpool_use_compress = True
        kv_pool = object.__new__(HybridLinearKVPool)
        kv_pool.full_kv_pool = dsa
        allocator = TokenToKVPoolAllocator(
            size=4096,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=64,
            tree_components=(ComponentType.FULL, ComponentType.MAMBA),
            enable_mamba_extra_buffer=True,
        )
        return UnifiedRadixCache(params), allocator, req_pool, params

    def _request(self, cache, allocator, pool, length=1000):
        sampling = SamplingParams(max_new_tokens=1)
        sampling.normalize(None)
        req = Req(
            rid="compressed-index",
            origin_input_text="",
            origin_input_ids=array("q", [1] * length),
            sampling_params=sampling,
            vocab_size=128,
        )
        pool.alloc([req])
        req.full_untruncated_fill_ids = array("q", req.origin_input_ids)
        pool.write((req.kv.req_pool_idx, slice(0, length)), allocator.alloc(length))
        req.kv.kv_allocated_len = length
        req.last_node = cache.root_node_handle()
        req.prefix_indices = torch.empty(0, dtype=torch.int64)
        return req

    def _chunk(self, cache, pool, req, end, publish=True):
        prefix = len(req.prefix_indices)
        req.set_extend_range(prefix, end)
        req.kv.kv_committed_len = end
        batch = ScheduleBatch(reqs=[req])
        batch.model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(mamba_chunk_size=64)
        )
        batch.tree_cache = cache
        batch.req_to_token_pool = pool
        entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(req)
        forward = SimpleNamespace(
            extend_seq_lens=torch.tensor([end - prefix]),
            extend_prefix_lens=torch.tensor([prefix]),
            mamba_track_seqlens=torch.tensor([entry.track_seqlen]),
            mamba_track_mask=torch.tensor([entry.track_mask]),
            mamba_track_indices=torch.tensor([entry.track_index]),
        )
        indices = MambaAttnBackendBase._init_track_ssm_indices(
            SimpleNamespace(device="cpu", mamba_chunk_size=64),
            req.kv.mamba_pool_idx.unsqueeze(0),
            forward,
        )
        h_src, h_dst = indices[1:3]
        final_src, final_dst = indices[4:6]
        states = pool.mamba_pool.mamba_cache.temporal
        states[:, req.kv.mamba_pool_idx].fill_(end)
        for src, dst in zip(h_src.tolist(), h_dst.tolist()):
            states[:, dst].fill_(prefix + src * 64)
        for src, dst in zip(final_src.tolist(), final_dst.tolist()):
            states[:, dst].copy_(states[:, src])
        for conv in pool.mamba_pool.mamba_cache.conv:
            positions = MambaAttnBackendBase._init_track_conv_indices(
                SimpleNamespace(device="cpu", conv_states_shape=conv.shape[2:]),
                torch.tensor([0, end - prefix]),
                forward,
            )
            if entry.track_mask:
                conv[:, entry.track_index] = (prefix + positions[0] + 1).to(conv.dtype)
        checkpoint = req.kv.mamba_last_track_seqlen
        if publish:
            cache.cache_unfinished_req(req, chunked=True)
        return checkpoint, entry

    def _check_state(self, cache, pool, tokens, expected):
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
        self.assertEqual(len(result.device_indices), expected)
        if expected:
            slot = cache.tree_core.get_component_device_value(
                result.best_match_node, ComponentType.MAMBA
            )
            self.assertTrue(
                torch.all(
                    pool.mamba_pool.mamba_cache.temporal[:, slot] == expected
                ).item()
            )
            for conv in pool.mamba_pool.mamba_cache.conv:
                window = (
                    torch.arange(conv.shape[-1]) + expected - conv.shape[-1] + 1
                ).to(conv.dtype)
                self.assertTrue(torch.all(conv[:, slot] == window).item())

    def test_tree_grid_preserves_physical_pool_and_transfer_pages(self):
        cache, allocator, _, params = self._fixture()
        self.assertEqual(cache.page_size, 256)
        self.assertEqual(params.page_size, 64)
        self.assertEqual(cache._transfer_page_size, 64)
        self.assertEqual(allocator.get_kvcache().full_kv_pool.page_size, 64)
        params.page_size = 96
        self.assertEqual(_compressed_index_tree_params(params).page_size, 768)
        params.disable = True
        self.assertIs(_compressed_index_tree_params(params), params)
        params.disable = False
        allocator.get_kvcache().full_kv_pool.kpool_use_compress = False
        self.assertIs(_compressed_index_tree_params(params), params)
        with self.assertRaisesRegex(ValueError, "index-buffer restore"):
            cache.init_hicache(None, params)
        with self.assertRaisesRegex(ValueError, "external cache linker"):
            cache.init_cache_linker(None)

    def test_successive_chunks_keep_state_at_absolute_tree_boundary(self):
        for ends in ((320, 640, 960, 1000), tuple(range(64, 961, 64))):
            with self.subTest(ends=ends):
                cache, allocator, pool, _ = self._fixture()
                req = self._request(cache, allocator, pool)
                for end in ends:
                    self._chunk(cache, pool, req, end)
                    self.assertEqual(len(req.prefix_indices), end)
                    self._check_state(
                        cache, pool, req.origin_input_ids[:end], end // 256 * 256
                    )

    def test_branch_at_616_cannot_share_the_512_to_767_index_page(self):
        cache, allocator, pool, _ = self._fixture()
        req = self._request(cache, allocator, pool)
        self._chunk(cache, pool, req, 768)
        tokens = array("q", req.origin_input_ids)
        tokens[616:] = array("q", [2] * (len(tokens) - 616))
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
        self.assertEqual(result.full_kv_hit_length, 512)
        self.assertEqual(result.mamba_branching_seqlen, 512)
        self.assertEqual(len(result.device_indices), 0)
        for branch, expected in ((576, 768), (512, 512)):
            with self.subTest(branch=branch):
                cache, allocator, pool, _ = self._fixture()
                req = self._request(cache, allocator, pool)
                req.mamba_branching_seqlen = branch
                checkpoint, _ = self._chunk(cache, pool, req, 960)
                self.assertEqual(checkpoint, expected)
                self._check_state(cache, pool, req.origin_input_ids[:960], expected)

    def test_missing_relative_kernel_snapshot_is_not_published(self):
        cache, allocator, pool, _ = self._fixture()
        req = self._request(cache, allocator, pool)
        self._chunk(cache, pool, req, 324)
        checkpoint, entry = self._chunk(cache, pool, req, 644)
        self.assertFalse(entry.track_mask)
        self.assertIsNone(checkpoint)
        self._check_state(cache, pool, req.origin_input_ids[:644], 256)

    def test_divergent_request_copies_an_independent_checkpoint(self):
        cache, allocator, pool, _ = self._fixture()
        req = self._request(cache, allocator, pool)
        self._chunk(cache, pool, req, 320)
        self._chunk(cache, pool, req, 640)
        other = self._request(cache, allocator, pool, length=700)
        other.origin_input_ids[512:] = array("q", [2] * 188)
        match = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(other.origin_input_ids), req=other, cow_mamba=True
            )
        )
        self.assertEqual(len(match.device_indices), 512)
        batch = ScheduleBatch(reqs=[other])
        batch._collect_deferred_mamba_cow_and_clear([other])
        pool.mamba_pool.copy_from(
            batch.mamba_cow_src_indices, batch.mamba_cow_dst_indices
        )
        state = pool.mamba_pool.mamba_cache.temporal[:, other.kv.mamba_pool_idx]
        self.assertTrue(torch.all(state == 512).item())
        state.fill_(-1)
        self._check_state(cache, pool, other.origin_input_ids, 512)

    def test_finished_request_keeps_only_exact_cached_checkpoints(self):
        cache, allocator, pool, _ = self._fixture()
        available = pool.mamba_allocator.available_size()
        req = self._request(cache, allocator, pool)
        self._chunk(cache, pool, req, 320)
        self._chunk(cache, pool, req, 640, publish=False)
        cache.cache_finished_req(req, kv_len_to_handle=640)
        self._check_state(cache, pool, req.origin_input_ids[:640], 512)
        self.assertEqual(pool.mamba_allocator.available_size(), available - 2)

    def test_component_key_limit_precedes_checkpoint_donation(self):
        for mamba_first in (False, True):
            for extra_buffer in (False, True):
                with self.subTest(mamba_first=mamba_first, extra_buffer=extra_buffer):
                    full = MagicMock(component_type=ComponentType.FULL)
                    full.prepare_for_caching_req.return_value = 32
                    mamba = MagicMock(component_type=ComponentType.MAMBA)
                    mamba.prepare_for_caching_req.return_value = 0
                    cache = SimpleNamespace(
                        _components_tuple=(mamba, full)
                        if mamba_first
                        else (full, mamba),
                        enable_mamba_extra_buffer=extra_buffer,
                    )
                    req, params = object(), object()
                    length = UnifiedRadixCache._prepare_for_caching_req(
                        cache, req, params, 64, is_finished=False
                    )
                    self.assertEqual(length, 0)
                    if extra_buffer:
                        mamba.prepare_for_caching_req.assert_called_once_with(
                            req=req,
                            insert_params=params,
                            token_ids_len=32,
                            is_finished=False,
                        )
                    else:
                        mamba.prepare_for_caching_req.assert_not_called()

    def test_invalid_checkpoint_keeps_or_frees_request_owned_slots(self):
        for finished in (False, True):
            for checkpoint in (0, 576, 768, None):
                with self.subTest(finished=finished, checkpoint=checkpoint):
                    cache, allocator, pool, _ = self._fixture()
                    available = pool.mamba_allocator.available_size()
                    req = self._request(cache, allocator, pool)
                    req.set_extend_range(0, 640)
                    req.kv.kv_committed_len = 640
                    req.kv.mamba_last_track_seqlen = checkpoint
                    before = pool.mamba_allocator.available_size()
                    if finished:
                        cache.cache_finished_req(req, kv_len_to_handle=640)
                    else:
                        cache.cache_unfinished_req(req, chunked=True)
                    self.assertEqual(
                        pool.mamba_allocator.available_size(),
                        available if finished else before,
                    )
                    self._check_state(cache, pool, req.origin_input_ids[:640], 0)


if __name__ == "__main__":
    unittest.main()
