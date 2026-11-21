"""Protect the input boundary in the device radix and page allocator."""

import unittest
from array import array

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDecodeInputOnlyCache(unittest.TestCase):
    page_size = 64
    capacity = 512
    output_length = 128

    def retention_case(self, prompt_length, *, enabled, role, checkpoint, insert):
        with (
            get_context().override_server_args(
                model_path="dummy",
                device="cpu",
                page_size=self.page_size,
                disaggregation_mode=role,
            ),
            envs.SGLANG_PD_DECODE_CACHE_INPUT_ONLY.override(enabled),
            envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"),
        ):
            requests = ReqToTokenPool(
                2, self.capacity, "cpu", enable_memory_saver=False
            )
            pool = MHATokenToKVPool(
                size=self.capacity,
                page_size=self.page_size,
                dtype=torch.float32,
                head_num=1,
                head_dim=4,
                layer_num=1,
                device="cpu",
                enable_memory_saver=False,
            )
            allocator = PagedTokenToKVPoolAllocator(
                size=self.capacity,
                page_size=self.page_size,
                dtype=torch.float32,
                device="cpu",
                kvcache=pool,
                need_sort=False,
            )
            cache = UnifiedRadixCache(
                CacheInitParams(
                    disable=False,
                    req_to_token_pool=requests,
                    token_to_kv_pool_allocator=allocator,
                    page_size=self.page_size,
                    is_eagle=True,
                    tree_components=(ComponentType.FULL,),
                )
            )
            length = prompt_length + self.output_length
            tokens = array("q", range(length))
            req = Req(
                rid="input-only",
                origin_input_text="",
                origin_input_ids=tokens[:prompt_length],
                sampling_params=SamplingParams(temperature=0, max_new_tokens=128),
            )
            self.assertIsNotNone(requests.alloc([req]))
            req.output_ids = tokens[prompt_length:]
            req.full_untruncated_fill_ids = tokens
            req.set_extend_range(0, length)
            req.kv.kv_committed_len = req.kv.kv_allocated_len = length
            req.last_node = cache.root_node_handle()
            allocated = (length + self.page_size - 1) // self.page_size * self.page_size
            indices = allocator.alloc(allocated)
            requests.write((req.kv.req_pool_idx, slice(0, length)), indices[:length])

            boundary = prompt_length if enabled and role == "decode" else length
            retained = (
                (boundary - 1) // self.page_size * self.page_size if insert else 0
            )
            if checkpoint:
                cache.cache_unfinished_req(req)
                self.assertEqual(req.kv.cache_protected_len, retained)
                torch.testing.assert_close(req.prefix_indices, indices[:length])
            cache.cache_finished_req(req, is_insert=insert, kv_len_to_handle=length)
            match = cache.match_prefix(
                MatchPrefixParams(key=RadixKey(tokens, is_bigram=True))
            )
            self.assertEqual(len(match.device_indices), retained)
            self.assertEqual(cache.protected_size(), 0)
            self.assertEqual(allocator.available_size(), self.capacity - retained)
            self.assertEqual(
                cache.evict(EvictParams(num_tokens=self.capacity)).num_tokens_evicted,
                retained,
            )
            self.assertEqual(allocator.available_size(), self.capacity)
            self.assertEqual(
                allocator.alloc(self.capacity).unique().numel(), self.capacity
            )

    def test_checkpoint_and_finish_keep_only_aligned_input(self):
        for prompt_length in (63, 64, 65, 127, 128, 129):
            with self.subTest(prompt_length=prompt_length):
                self.retention_case(
                    prompt_length,
                    enabled=True,
                    role="decode",
                    checkpoint=True,
                    insert=True,
                )

    def test_finish_without_insertion_returns_all_slots(self):
        self.retention_case(
            129, enabled=True, role="decode", checkpoint=False, insert=False
        )

    def test_default_and_prefill_keep_existing_retention(self):
        for role, enabled in (("decode", False), ("prefill", True)):
            with self.subTest(role=role, enabled=enabled):
                self.retention_case(
                    129, enabled=enabled, role=role, checkpoint=False, insert=True
                )


if __name__ == "__main__":
    unittest.main()
