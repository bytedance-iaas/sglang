"""Focused CPU tests for the DSV4 EAGLE3 HiCache backport."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.kv_cache_builder import maybe_register_hicache_draft
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSV4Eagle3HiCache(unittest.TestCase):
    def test_external_draft_falls_back_from_dsv4_backend(self):
        fallback = mock.Mock(return_value="fa3")
        holder = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
            ),
            server_args=SimpleNamespace(_get_default_attn_backend=fallback),
            use_mla_backend=False,
        )

        result = ModelRunner._fallback_dsv4_backend_for_draft(holder, "dsv4")

        self.assertEqual(result, "fa3")
        fallback.assert_called_once_with(
            use_mla_backend=False,
            model_config=holder.model_config,
        )

    def test_dsv4_draft_keeps_dsv4_backend(self):
        fallback = mock.Mock(return_value="fa3")
        holder = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["DeepseekV4ForCausalLMNextN"]
                )
            ),
            server_args=SimpleNamespace(_get_default_attn_backend=fallback),
            use_mla_backend=True,
        )

        result = ModelRunner._fallback_dsv4_backend_for_draft(holder, "dsv4")

        self.assertEqual(result, "dsv4")
        fallback.assert_not_called()

    def test_eagle3_capture_layers_follow_upstream_offset(self):
        holder = SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=True),
            capture_aux_hidden_states=False,
            config=SimpleNamespace(num_hidden_layers=80),
            model=SimpleNamespace(layers_to_capture=[]),
        )

        DeepseekV4ForCausalLM.set_eagle3_layers_to_capture(holder, [1, 39, 76])

        self.assertTrue(holder.capture_aux_hidden_states)
        self.assertEqual(holder.model.layers_to_capture, [2, 40, 77])

    def test_build_full_draft_sidecar_uses_target_kv_indices(self):
        draft_pool = mock.Mock(spec=MHATokenToKVPool)
        draft_pool.layer_num = 1
        draft_pool.size = 1024
        fake_host_pool = SimpleNamespace(layer_num=1)
        tree_cache = SimpleNamespace(
            cache_controller=SimpleNamespace(
                page_size=256,
                mem_pool_host=SimpleNamespace(size=2048),
            )
        )
        server_args = SimpleNamespace(
            hicache_mem_layout="layer_first",
            hicache_storage_backend=None,
        )

        with mock.patch.object(
            hybrid_pool_assembler,
            "_build_draft_host_pool",
            return_value=fake_host_pool,
        ):
            specs, entries = hybrid_pool_assembler.build_hicache_draft_sidecars(
                draft_device_pools=(draft_pool,),
                tree_cache=tree_cache,
                server_args=server_args,
            )

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].pool_name, PoolName.DRAFT)
        self.assertEqual(specs[0].indices_from_pool, PoolName.KV)
        self.assertEqual(entries[0].name, PoolName.DRAFT)
        self.assertIs(entries[0].device_pool, draft_pool)
        self.assertEqual(entries[0].layer_mapper(0), 0)
        self.assertIsNone(entries[0].layer_mapper(1))

    def test_unified_cache_registers_sidecar_instead_of_legacy_piggyback(self):
        draft_pool = object()
        draft_runner = SimpleNamespace(
            token_to_kv_pool=draft_pool,
            model_config=object(),
        )
        draft_worker = SimpleNamespace(model_runner=draft_runner)
        spec_algorithm = SimpleNamespace(
            is_ngram=mock.Mock(return_value=False),
            is_eagle3=mock.Mock(return_value=True),
            supports_spec_v2=mock.Mock(return_value=False),
        )
        tree_cache = object.__new__(UnifiedRadixCache)
        tree_cache.register_hicache_draft_pools = mock.Mock()
        tree_cache.cache_controller = SimpleNamespace(set_draft_kv_pool=mock.Mock())
        specs = [object()]
        entries = [object()]

        with mock.patch(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
            "build_hicache_draft_sidecars",
            return_value=(specs, entries),
        ):
            maybe_register_hicache_draft(
                tree_cache=tree_cache,
                draft_worker=draft_worker,
                spec_algorithm=spec_algorithm,
                server_args=SimpleNamespace(hicache_storage_backend=None),
                enable_hierarchical_cache=True,
                enable_overlap=False,
                page_size=256,
            )

        tree_cache.register_hicache_draft_pools.assert_called_once_with(
            specs, entries
        )
        tree_cache.cache_controller.set_draft_kv_pool.assert_not_called()


if __name__ == "__main__":
    unittest.main()
