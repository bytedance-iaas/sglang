"""An unpublished prefill must keep releasing SWA past its branch point."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _drive_unpublished(backend, chunk_ends, *, skip_insert):
    pool = ReqToTokenPool(
        size=4, max_context_len=128, device="cpu", enable_memory_saver=False
    )
    # The prompt is longer than the entire SWA pool.
    swa_size = 40
    kv = SWAKVPool(
        size=256,
        size_swa=swa_size,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=8,
        swa_attention_layer_ids=[0],
        full_attention_layer_ids=[1],
        device="cpu",
    )
    allocator = SWATokenToKVPoolAllocator(
        size=256,
        size_swa=swa_size,
        page_size=1,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv,
        need_sort=False,
        req_to_token_pool=pool,
    )
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            sliding_window_size=4,
            tree_core_backend=backend,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
        )
    )
    # Do not count a fallback to Python as Rust coverage.
    if cache._tree_core_backend != backend:
        raise unittest.SkipTest(f"{backend} tree core is unavailable on this platform")

    full_total = allocator.full_available_size()
    swa_total = allocator.swa_available_size()

    def insert(tokens):
        indices = allocator.alloc(len(tokens))
        cache.insert(InsertParams(key=RadixKey(array("q", tokens)), value=indices))

    prefix = list(range(1, 5))
    old_suffix = list(range(100, 116))
    insert(prefix)
    insert(prefix + old_suffix)
    base = cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", prefix)))
    ).last_device_node
    held = cache.lock(base)
    cache.evict(EvictParams(swa_num_tokens=len(old_suffix)))
    cache.unlock(held)

    tokens = prefix + old_suffix[:8] + list(range(200, 208))
    tokens += list(range(300, 300 + chunk_ends[-1] - len(tokens)))
    req = Req("branched", None, array("q", tokens), SamplingParams(max_new_tokens=1))
    pool.alloc([req])
    req.init_next_round_input(tree_cache=cache)
    assert req.swa_branching_seqlen == 12
    pool.write(
        (req.kv.req_pool_idx, slice(0, len(prefix))),
        cache.path_device_indices(req.last_node),
    )
    req.extend_end = len(prefix)
    req.lock = cache.lock(req.last_node)
    req.pending_bootstrap = True
    req.skip_radix_cache_insert = skip_insert
    scheduler = SchedulerDisaggregationPrefillMixin()
    scheduler.tree_cache = cache
    steps = []

    for end in chunk_ends:
        need = end - req.extend_end
        available = allocator.swa_available_size()
        assert available + cache.swa_evictable_size() >= need, (
            f"prefill stalled at {req.extend_end}, protected={req.kv.cache_protected_len}"
        )
        if available < need:
            cache.evict(EvictParams(swa_num_tokens=need - available))
        indices = allocator.alloc(need)
        assert indices is not None
        pool.write((req.kv.req_pool_idx, slice(req.extend_end, end)), indices)
        req.kv.kv_committed_len = req.kv.kv_allocated_len = end
        req.extend_end = end
        scheduler.checkpoint_disagg_prefill(req)
        cache.sanity_check()
        steps.append(
            SimpleNamespace(
                protected=req.kv.cache_protected_len,
                swa_evicted=req.kv.get_evicted_seqlen(ComponentType.SWA),
                branch_hit=cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", tokens[:12])))
                ).device_prefix_len,
            )
        )

    release_kv_cache(req, cache, checkpoint=False)
    assert allocator.full_available_size() + cache.full_evictable_size() == full_total
    assert allocator.swa_available_size() + cache.swa_evictable_size() == swa_total
    cache.sanity_check()
    return steps


class TestPrefillSendProtection(CustomTestCase):
    def _check_unpublished_branch(self, *, skip_insert):
        ends = tuple(range(20, 125, 8))
        for backend in ("python", "rust"):
            with (
                self.subTest(backend=backend),
                get_context().override_server_args(page_size=1),
                envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.override(True),
                patch(
                    "sglang.srt.disaggregation.prefill._uses_write_through_cache",
                    return_value=True,
                ),
            ):
                steps = _drive_unpublished(backend, ends, skip_insert=skip_insert)
                self.assertEqual([s.swa_evicted for s in steps], [e - 5 for e in ends])
                boundary = 4 if skip_insert else 12
                self.assertEqual(
                    {(s.protected, s.branch_hit) for s in steps},
                    {(boundary, boundary)},
                )

    def test_unpublished_branch_keeps_releasing_swa(self):
        self._check_unpublished_branch(skip_insert=False)

    def test_unpublished_branch_without_insert_keeps_releasing_swa(self):
        self._check_unpublished_branch(skip_insert=True)


if __name__ == "__main__":
    unittest.main()
