import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _with_mtp_layer_mapping,
)
from sglang.srt.mem_cache.kv_cache_builder import (
    HiCacheDraftMode,
    get_draft_kv_pool,
    prepare_dspark_hicache_draft_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _DSparkAlgorithm:
    def is_dspark(self):
        return True

    def is_ngram(self):
        return False


def _pool(mapping, *, page_size=128, item_bytes=64):
    swa_pool = SimpleNamespace(
        bytes_per_page_padded=item_bytes,
        kv_buffer=[object()],
    )
    return SimpleNamespace(
        swa_page_size=page_size,
        swa_kv_pool=swa_pool,
        full_to_swa_index_mapping=mapping,
    )


def _runner(pool, *, pp_rank, pp_size):
    return SimpleNamespace(
        pp_rank=pp_rank,
        pp_size=pp_size,
        token_to_kv_pool=pool,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["DeepseekV4ForCausalLMDSpark"]
            )
        ),
    )


class TestDeepSeekV4DSparkHiCachePlan(CustomTestCase):
    def test_only_final_pp_stage_packs_dspark_swa(self):
        mapping = object()
        target_pool = _pool(mapping)
        draft_pool = _pool(mapping)
        args = SimpleNamespace(
            enable_hierarchical_cache=True,
            disaggregation_mode="prefill",
        )

        cases = ((0, HiCacheDraftMode.NONE), (3, HiCacheDraftMode.PACKED))
        for pp_rank, expected in cases:
            target_runner = _runner(target_pool, pp_rank=pp_rank, pp_size=4)
            draft_runner = _runner(draft_pool, pp_rank=pp_rank, pp_size=4)
            draft_worker = SimpleNamespace(
                is_lifecycle_only_pp_prefill_rank=False,
                draft_model_runner=draft_runner,
            )

            plan = prepare_dspark_hicache_draft_plan(
                target_worker=SimpleNamespace(model_runner=target_runner),
                draft_worker=draft_worker,
                spec_algorithm=_DSparkAlgorithm(),
                server_args=args,
            )

            self.assertEqual(plan.mode, expected)
            self.assertEqual(plan.device_pools, (draft_pool,) if pp_rank == 3 else ())

    def test_lifecycle_only_rank_has_no_draft_pool(self):
        worker = SimpleNamespace(
            is_lifecycle_only_pp_prefill_rank=True,
            primary_draft_kv_pool=None,
            draft_model_runner=SimpleNamespace(model_config=object()),
        )
        self.assertEqual(
            get_draft_kv_pool(
                draft_worker=worker,
                spec_algorithm=_DSparkAlgorithm(),
                server_args=SimpleNamespace(),
                enable_overlap=False,
            ),
            (None, None),
        )

    def test_packed_plan_requires_shared_mapping(self):
        target_runner = _runner(_pool(object()), pp_rank=3, pp_size=4)
        draft_runner = _runner(_pool(object()), pp_rank=3, pp_size=4)

        with self.assertRaisesRegex(ValueError, "Full-to-SWA"):
            prepare_dspark_hicache_draft_plan(
                target_worker=SimpleNamespace(model_runner=target_runner),
                draft_worker=SimpleNamespace(
                    is_lifecycle_only_pp_prefill_rank=False,
                    draft_model_runner=draft_runner,
                ),
                spec_algorithm=_DSparkAlgorithm(),
                server_args=SimpleNamespace(
                    enable_hierarchical_cache=True,
                    disaggregation_mode="prefill",
                ),
            )

    def test_packed_layers_follow_target_layers(self):
        self.assertEqual(
            _with_mtp_layer_mapping(
                {0: 0, 1: 1},
                target_layer_num=2,
                draft_layer_num=3,
            ),
            {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        )


if __name__ == "__main__":
    unittest.main()
