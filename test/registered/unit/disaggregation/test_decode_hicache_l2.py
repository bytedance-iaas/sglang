"""PD admission and ownership while UnifiedRadixCache restores a host prefix."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import DecodeRequest
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)
from sglang.srt.managers.cache_controller import HiCacheAck, PrefetchOperation
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import match_kv_cache
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import PrefetchSubmission
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Event:
    def __init__(self, done=True):
        self.done = done

    def query(self):
        return self.done

    def synchronize(self):
        assert self.done


class _CPUHostPool(TokenToKVPoolAllocator):
    """Host slots with the anchor-entry view the tree's backup path reads: no
    shared allocation domain, so the tree evicts host space itself."""

    anchor_entry = SimpleNamespace(
        host_pool=SimpleNamespace(shared_allocation_domain=None), host_evict_fn=None
    )


class _CPUController:
    """Replace DMA submission/completion; tree allocation and locks stay real."""

    def __init__(self, allocator):
        self.allocator = allocator
        self.write_policy = "write_through"
        self.mem_pool_host = _CPUHostPool(
            size=1024, dtype=torch.float32, device="cpu", kvcache=None, need_sort=False
        )
        self.layer_done_counter = SimpleNamespace(
            producer_index=-1,
            num_counters=3,
            events=[SimpleNamespace(finish_event=_Event()) for _ in range(3)],
        )
        self.load_queue = []
        self.ack_load_queue = []
        self.ack_write_queue = []
        self.prefetch_tokens_occupied = 0
        self.decline_prefetch = False

    def get_prefetch_submission(self, rid):
        return None

    def prefetch_rate_limited(self):
        return False

    def submit_prefetch(
        self,
        handle,
        key,
        last_hash,
        prefix_keys,
        matched_prefix_tokens,
        pool_transfers,
        assume_stored=False,
    ):
        # Only the external storage submission is replaced; the actual cache
        # computes the admission span, pins its anchor, and registers ownership.
        if self.decline_prefetch:
            return PrefetchSubmission(decision=False)
        operation = PrefetchOperation(handle, key, last_hash, prefix_keys)
        operation.assume_stored = assume_stored
        return PrefetchSubmission(operation=operation)

    def append_host_mem_release(self, extra_pools):
        assert not extra_pools

    def write(self, device_indices, node_id, extra_pools, flush=True):
        self.ack_write_queue.append(HiCacheAck(_Event(), _Event(), [node_id]))
        return self.mem_pool_host.alloc(len(device_indices))

    def load(self, host_indices, node_id, extra_pools):
        indices = self.allocator.alloc(len(host_indices))
        if indices is not None:
            self.load_queue.append(node_id)
        return indices

    def start_loading(self):
        if not self.load_queue:
            return -1
        counter = self.layer_done_counter
        counter.producer_index = (counter.producer_index + 1) % counter.num_counters
        event = counter.events[counter.producer_index].finish_event
        assert event.done
        event.done = False
        self.ack_load_queue.append(HiCacheAck(_Event(), event, self.load_queue[:]))
        self.load_queue.clear()
        return counter.producer_index


def _match_state(req):
    return (
        req.prefix_len,
        req.kv.cache_protected_len,
        req.host_hit_length,
        req.last_node,
    )


class _Queue(DecodeHiCachePreallocMixin, DecodeHiCacheTransferMixin):
    def __init__(self, cache):
        self.tree_cache = cache
        self.scheduler = SimpleNamespace(enable_decode_hicache=True)


class TestDecodeHiCacheL2(CustomTestCase):
    def setUp(self):
        super().setUp()
        context = get_context().override_server_args(disaggregation_mode="decode")
        context.install()
        self.addCleanup(context.restore)
        self.allocator = TokenToKVPoolAllocator(
            size=1024, dtype=torch.float32, device="cpu", kvcache=None, need_sort=False
        )
        self.pool = ReqToTokenPool(
            size=8, max_context_len=1024, device="cpu", enable_memory_saver=False
        )
        self.cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                # _backup builds actions by hand, which the Rust core does not expose.
                tree_core_backend="python",
                req_to_token_pool=self.pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=128,
                tree_components=(ComponentType.FULL,),
            )
        )
        self.controller = _CPUController(self.allocator)
        self.cache.cache_controller = self.controller
        self.cache.components[
            ComponentType.FULL
        ]._full_kv_pool_host = self.controller.mem_pool_host
        self.cache.tree_core.set_hicache_enabled()
        self.cache.load_back_threshold = 10
        self.queue = _Queue(self.cache)
        storage_probe = patch.object(
            self.cache,
            "query_storage_hit_length",
            side_effect=AssertionError("L2-only cache queried L3"),
        )
        storage_probe.start()
        self.addCleanup(storage_probe.stop)
        self.tokens = array("q", range(384))

    def _insert(self, length):
        key = RadixKey(self.tokens[:length])
        slots = self.allocator.alloc(length)
        result = self.cache.insert(InsertParams(key=key, value=slots))
        self.allocator.free(slots[: result.prefix_len])
        return self.cache.match_prefix(MatchPrefixParams(key=key)).last_device_node

    def _backup(self, node):
        action = self.cache.tree_core.build_backup_kv_action(node)
        self.cache._execute_and_commit_kv_backup(action)
        self.cache.writing_check()

    def _request(self, index, length=384):
        req = Req(
            rid=str(index),
            origin_input_text="",
            origin_input_ids=self.tokens[:length],
            sampling_params=SamplingParams(max_new_tokens=8),
        )
        req.kv.req_pool_idx = index
        result = match_kv_cache(req, self.cache, req.origin_input_ids)
        req.lock = self.cache.lock(result.last_device_node)
        pm = self.queue._build_decode_prefix_match(req, result)
        self.pool.write((index, slice(0, pm.l1_prefix_len)), pm.prefix_indices)
        return DecodeRequest(
            req=req,
            kv_receiver=SimpleNamespace(poll=lambda: KVPoll.Success),
            prefix_match=pm,
        )

    def _host_prefix(self):
        l1 = self._insert(128)
        holder = self.cache.lock(l1)
        l2 = self._insert(256)
        self._backup(l2)
        self.cache.evict(EvictParams(num_tokens=128))
        self.addCleanup(self.cache.unlock, holder)
        return l1

    def _complete(self):
        for ack in self.controller.ack_load_queue:
            ack.finish_event.done = True
        self.cache.loading_check()

    def test_l2_miss_and_host_hit_do_not_query_storage(self):
        dr = self._request(0)
        self.assertEqual(dr.prefix_match.decode_prefix_len, 0)
        self.cache.unlock(dr.req.lock)
        self._host_prefix()
        dr = self._request(1)
        self.assertEqual(dr.prefix_match.l1_prefix_len, 128)
        self.assertEqual(dr.prefix_match.l2_host_hit_length, 128)
        self.assertEqual(dr.prefix_match.l3_storage_hit_length, 0)
        self.cache.unlock(dr.req.lock)

    def test_l3_prefetch_keeps_the_actual_l1_l2_prefix_context(self):
        self._host_prefix()
        requests = [self._request(0), self._request(1)]
        anchor = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(self.tokens))
        ).last_host_node
        self.cache.enable_storage = True
        self.cache.prefetch_threshold = 128
        self.cache.hicache_storage_pass_prefix_keys = False
        for dr, decline in zip(requests, (False, True)):
            with self.subTest(declined=decline):
                pm = dr.prefix_match
                self.assertEqual((pm.l1_prefix_len, pm.l2_host_hit_length), (128, 128))
                pm.l3_storage_hit_length, pm.last_host_node = 128, anchor
                self.controller.decline_prefetch = decline
                self.queue._start_hicache_prefetch(dr.req, pm)
                self.assertEqual(pm.prefetch_registered, not decline)
                self.assertEqual(pm.decode_prefix_len, 256 if decline else 384)
                if not decline:
                    info = self.cache.ongoing_prefetch.pop(dr.req.cache_request_handle)
                    self.assertEqual(info.operation.storage_start, 256)
                    self.assertEqual(len(info.prefetch_key), 128)
                    self.assertTrue(info.operation.assume_stored)
                    self.cache.dec_host_lock_ref(anchor, info.anchor_lock_params)
                self.cache.unlock(dr.req.lock)
        self.assertEqual(self.cache._prefetch_outcome_stats["issued_assumed_stored"], 1)
        self.cache.sanity_check()

    def test_restore_failure_is_visible_to_transfer_polling(self):
        dr = self._request(0)
        poller = HiCacheRestoreGatedKVReceiver(dr)
        for status, expected in (
            (HiCacheRestoreResult.PENDING, KVPoll.Transferring),
            (HiCacheRestoreResult.FAILED, KVPoll.Failed),
            (HiCacheRestoreResult.READY, KVPoll.Success),
        ):
            dr.hicache_restore_status = status
            self.assertEqual(poller.poll(), expected)
        self.cache.unlock(dr.req.lock)

    def test_concurrent_shared_prefix_and_l1_follower_wait_for_dma(self):
        self._host_prefix()
        requests = [self._request(0), self._request(1)]
        # The rematch must not overwrite the admission-time match on the request.
        admitted = [_match_state(dr.req) for dr in requests]
        self.queue._process_hicache_local_restores(requests)
        self.assertEqual(len(self.controller.ack_load_queue), 1)
        follower = self._request(2)
        self.assertFalse(follower.prefix_match.needs_local_restore)
        requests.append(follower)
        admitted.append(_match_state(follower.req))
        self.queue._process_hicache_local_restores(requests)
        for dr in requests:
            self.assertEqual(
                HiCacheRestoreGatedKVReceiver(dr).poll(), KVPoll.Transferring
            )
            self.assertEqual(dr.req.last_node, dr.prefix_match.last_device_node)
        self._complete()
        self.queue._process_hicache_local_restores(requests)
        self.assertEqual([_match_state(dr.req) for dr in requests], admitted)
        for dr in requests:
            self.assertEqual(HiCacheRestoreGatedKVReceiver(dr).poll(), KVPoll.Success)
            self.queue._commit_hicache_local_restore_to_req(dr)
            self.assertEqual(dr.req.prefix_len, 256)
            torch.testing.assert_close(
                self.pool.req_to_token[dr.req.kv.req_pool_idx, :256].long(),
                self.cache.prefix_device_indices(dr.req).long(),
            )
            self.cache.unlock(dr.req.lock)
        self.cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
