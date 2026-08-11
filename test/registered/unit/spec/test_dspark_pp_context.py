import inspect
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.layernorm import RMSNorm  # noqa: E402
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod  # noqa: E402
from sglang.srt.managers.tp_worker import TpModelWorker  # noqa: E402
from sglang.srt.mem_cache.kv_cache_builder import get_draft_kv_pool  # noqa: E402
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (  # noqa: E402
    DeepSeekV4TokenToKVPool,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode  # noqa: E402
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig  # noqa: E402
from sglang.srt.models.deepseek_v4 import (  # noqa: E402
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
)
from sglang.srt.models.deepseek_v4_dspark import (  # noqa: E402
    DSparkAttention,
    DSparkV4Stage,
    DeepseekV4ForCausalLMDSpark,
    _BlockFp8LinearSlice,
)
from sglang.srt.speculative.dspark_components.dspark_config import (  # noqa: E402
    resolve_single_owner_pp_rank,
    use_lifecycle_only_draft_model,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (  # noqa: E402
    DSparkWorkerV2,
    _is_context_only_pp_prefill_rank,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _TupleLinear(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        return self.linear(hidden_states), None


class _AttentionWithoutDecodeTpHook(torch.nn.Module):
    """The fork's CP1 attention contract has no CP-decode context hook."""

    def forward(self, positions, hidden_states, forward_batch):
        del positions, forward_batch
        return hidden_states + 1


def _make_deepseek_v4_dspark_projection_model(
    *, hidden_size: int, num_target_features: int
) -> DeepseekV4ForCausalLMDSpark:
    model = DeepseekV4ForCausalLMDSpark.__new__(DeepseekV4ForCausalLMDSpark)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=hidden_size)
    model.num_target_features = num_target_features
    stage = torch.nn.Module()
    stage.main_proj = _TupleLinear(hidden_size * num_target_features, hidden_size)
    stage.main_norm = RMSNorm(hidden_size, eps=1e-6)
    model.stages = torch.nn.ModuleList([stage])
    model.markov_head = torch.nn.Identity()
    model.confidence_head = torch.nn.Identity()
    model.embed_tokens = None
    model.lm_head = None
    model.is_lifecycle_only = False
    model._partial_feature_indices = ()
    model._partial_main_proj = None
    return model


class TestDSparkPPContext(CustomTestCase):
    def test_target_worker_accepts_explicit_hidden_capture_mode(self):
        parameter = inspect.signature(
            TpModelWorker.forward_batch_generation
        ).parameters["capture_hidden_mode"]

        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(parameter.default)

    def test_target_worker_forwards_explicit_hidden_capture_mode(self):
        worker = TpModelWorker.__new__(TpModelWorker)
        forward_batch = object()
        model_output = SimpleNamespace(
            logits_output="pp-proxy",
            can_run_graph=False,
            expert_distribution_metrics=None,
        )
        worker._model_runner = SimpleNamespace(
            forward=Mock(return_value=model_output)
        )
        worker.pp_group = SimpleNamespace(is_last_rank=False)
        worker.set_hicache_consumer = Mock()
        worker.is_dllm = Mock(return_value=False)
        batch = SimpleNamespace(hicache_consumer_index=3)

        with patch(
            "sglang.srt.managers.tp_worker.ForwardBatch.init_new",
            return_value=forward_batch,
        ) as init_new:
            result = worker.forward_batch_generation(
                batch,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )

        init_new.assert_called_once_with(
            batch,
            worker.model_runner,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        worker.model_runner.forward.assert_called_once_with(
            forward_batch,
            pp_proxy_tensors=None,
            skip_attn_backend_init=False,
        )
        self.assertEqual(result.pp_hidden_states_proxy_tensors, "pp-proxy")

    def test_lifecycle_only_prefill_requests_no_hidden_capture(self):
        batch_output = SimpleNamespace(logits_output=None, new_seq_lens=None)
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.server_args = SimpleNamespace(enable_dp_attention=False)
        worker._target_worker = SimpleNamespace(
            forward_batch_generation=Mock(return_value=batch_output)
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_idle=lambda: False,
                is_extend=lambda: True,
            ),
            is_extend_in_batch=False,
            seq_lens="seq-lens",
        )

        result = worker._forward_lifecycle_only_prefill(
            batch=batch,
            on_publish=None,
            pp_proxy_tensors="pp-proxy",
        )

        worker.target_worker.forward_batch_generation.assert_called_once_with(
            batch,
            pp_proxy_tensors="pp-proxy",
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self.assertIs(result, batch_output)
        self.assertEqual(result.new_seq_lens, "seq-lens")

    def test_active_prefill_idle_participation_requests_full_hidden_capture(self):
        idle_result = object()
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.server_args = SimpleNamespace(enable_dp_attention=True)
        worker._target_worker = SimpleNamespace(forward_batch_generation=Mock())
        worker._decode_idle_result = Mock(return_value=idle_result)
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: True),
        )

        result = worker._forward_prefill(
            batch,
            on_publish=None,
            pp_proxy_tensors="pp-proxy",
        )

        worker.target_worker.forward_batch_generation.assert_called_once_with(
            batch,
            pp_proxy_tensors="pp-proxy",
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        worker._decode_idle_result.assert_called_once_with(on_publish=None)
        self.assertIs(result, idle_result)

    def test_decoder_constructor_uses_overridable_attention_factory(self):
        custom_attention = torch.nn.Identity()

        class _DecoderWithCustomAttention(DeepseekV4DecoderLayer):
            def _build_self_attn(self, **kwargs):
                self.attention_factory_kwargs = kwargs
                return custom_attention

        config = SimpleNamespace(
            hidden_size=8,
            rms_norm_eps=1e-6,
            hc_mult=2,
            hc_sinkhorn_iters=1,
            hc_eps=1e-6,
        )
        with (
            patch(
                "sglang.srt.models.deepseek_v4.deepseek_v2.DeepseekV2MoE",
                return_value=torch.nn.Identity(),
            ),
            patch(
                "sglang.srt.models.deepseek_v4.is_nsa_enable_prefill_cp",
                return_value=False,
            ),
        ):
            layer = _DecoderWithCustomAttention(config=config, layer_id=3)

        self.assertIs(layer.self_attn, custom_attention)
        self.assertEqual(layer.attention_factory_kwargs["layer_id"], 3)

    def test_dspark_stage_cp1_forward_does_not_require_cp_decode_tp_hook(self):
        stage = DSparkV4Stage.__new__(DSparkV4Stage)
        torch.nn.Module.__init__(stage)
        stage.self_attn = _AttentionWithoutDecodeTpHook()
        stage.input_layernorm = torch.nn.Identity()
        stage.post_attention_layernorm = torch.nn.Identity()
        stage.hc_attn_fn = None
        stage.hc_attn_scale = None
        stage.hc_attn_base = None
        stage.hc_ffn_fn = None
        stage.hc_ffn_scale = None
        stage.hc_ffn_base = None
        stage._hc_pre_block = Mock(side_effect=lambda x, *_: (x, None, None))
        stage._hc_post_block = Mock(side_effect=lambda x, *_: x)
        stage._run_ffn = Mock(side_effect=lambda x, _: x)

        hidden_states = torch.zeros(2, 3)
        actual = stage.forward(
            positions=torch.arange(2),
            hidden_states=hidden_states,
            forward_batch=object(),
        )

        torch.testing.assert_close(actual, hidden_states + 1)

    def test_dspark_stage_reuses_target_moe_dp_sync_helper(self):
        self.assertIs(
            DSparkV4Stage._run_moe_ffn_dp_sync,
            DeepseekV4DecoderLayer._run_moe_ffn_dp_sync,
        )

    def test_swa_fused_write_accepts_target_and_dspark_locations(self):
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool._should_cache_swa = False
        pool.translate_loc_from_full_to_swa = Mock(return_value="mapped-swa-loc")
        pool._swa_local_layer_id = Mock(return_value=0)
        pool.swa_kv_pool = SimpleNamespace(kv_buffer=["buffer"], page_size=256)
        common = dict(
            layer_id=0,
            kv="kv",
            kv_weight="weight",
            eps=1e-6,
            freqs_cis="freqs",
            positions="positions",
        )

        with patch(
            "sglang.srt.mem_cache.deepseek_v4_memory_pool."
            "fused_k_norm_rope_flashmla"
        ) as fused_write:
            pool.set_swa_key_buffer_radix_fused_norm_rope(
                raw_loc="full-loc", **common
            )
            self.assertEqual(fused_write.call_args.kwargs["out_loc"], "mapped-swa-loc")

            pool.set_swa_key_buffer_radix_fused_norm_rope(
                swa_loc="direct-swa-loc", **common
            )
            self.assertEqual(fused_write.call_args.kwargs["out_loc"], "direct-swa-loc")

        pool.translate_loc_from_full_to_swa.assert_called_once_with("full-loc")

        with self.assertRaisesRegex(ValueError, "Exactly one"):
            pool.set_swa_key_buffer_radix_fused_norm_rope(**common)

    def test_dspark_attention_uses_fork_full_to_swa_translation_contract(self):
        attention = DSparkAttention.__new__(DSparkAttention)
        torch.nn.Module.__init__(attention)
        attention.layer_id = 7
        attention.kv_norm = SimpleNamespace(
            weight=SimpleNamespace(data="kv-weight")
        )
        attention.eps = 1e-6
        attention.freqs_cis = "freqs"
        pool = Mock()
        forward_batch = SimpleNamespace(out_cache_loc="full-loc")

        attention._store_block_kv(
            kv="kv",
            positions="positions",
            forward_batch=forward_batch,
            pool=pool,
        )

        pool.set_swa_key_buffer_radix_fused_norm_rope.assert_called_once_with(
            layer_id=7,
            raw_loc="full-loc",
            kv="kv",
            kv_weight="kv-weight",
            eps=1e-6,
            freqs_cis="freqs",
            positions="positions",
        )

    def test_dspark_pd_state_layer_ids_follow_pp_local_buffer_order(self):
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool._stage_start = 10
        pool._stage_end = 14
        pool.compression_ratios = [0] * 10 + [0, 4, 128, 4]

        self.assertEqual(
            pool.get_dspark_pd_state_layer_ids(),
            [10, 11, 12, 13, 11, 13, 11, 13],
        )
        self.assertEqual(pool.get_c128_state_layer_ids(), [12])

    def test_full_projection_fast_path_requires_final_pp_owner(self):
        """Only final-rank ownership can bypass the ctx_acc handoff."""
        with patch.dict(
            os.environ,
            {"SGLANG_PP_LAYER_PARTITION": "6,5,6,5,6,5,5,5"},
        ):
            self.assertEqual(
                resolve_single_owner_pp_rank(
                    target_layer_ids=[40, 41, 42],
                    num_hidden_layers=43,
                    pp_size=8,
                ),
                7,
            )
            self.assertIsNone(
                resolve_single_owner_pp_rank(
                    target_layer_ids=[35, 40],
                    num_hidden_layers=43,
                    pp_size=8,
                )
            )

    def test_lifecycle_only_draft_model_is_limited_to_non_owner_ranks(self):
        common = dict(
            disaggregation_mode="prefill",
            pp_size=8,
            target_layer_ids=[40, 41, 42],
            num_hidden_layers=43,
        )
        for pp_rank in range(7):
            self.assertTrue(use_lifecycle_only_draft_model(pp_rank=pp_rank, **common))
        self.assertFalse(use_lifecycle_only_draft_model(pp_rank=7, **common))
        self.assertFalse(
            use_lifecycle_only_draft_model(
                pp_rank=0,
                **{**common, "target_layer_ids": [35, 40]},
            )
        )

    def test_lifecycle_only_model_does_not_consume_checkpoint_weights(self):
        model = DeepseekV4ForCausalLMDSpark.__new__(DeepseekV4ForCausalLMDSpark)
        torch.nn.Module.__init__(model)
        model.is_lifecycle_only = True

        def weights():
            raise AssertionError("lifecycle-only model consumed checkpoint weights")
            yield

        model.load_weights(weights())

    def test_all_separate_shared_expert_weights_and_scales_must_load(self):
        model = DeepseekV4ForCausalLMDSpark.__new__(
            DeepseekV4ForCausalLMDSpark
        )
        torch.nn.Module.__init__(model)
        model.num_stages = 1
        params = {
            "stages.0.mlp.shared_experts.gate_up_proj.weight": object(),
            "stages.0.mlp.shared_experts.gate_up_proj.weight_scale_inv": object(),
        }

        with self.assertRaisesRegex(ValueError, "missing.*weights or scales"):
            model._assert_shared_experts_loaded(
                params_dict=params,
                loaded_params={
                    "stages.0.mlp.shared_experts.gate_up_proj.weight"
                },
            )

        model._assert_shared_experts_loaded(
            params_dict=params,
            loaded_params=set(params),
        )

    def test_block_fp8_projection_slice_selects_matching_weight_and_scale_blocks(self):
        feature_width = 128
        output_size = 2
        quant_method = Fp8LinearMethod(
            Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                weight_block_size=[128, 128],
            )
        )
        weight = torch.arange(
            output_size * feature_width * 3, dtype=torch.float32
        ).reshape(output_size, feature_width * 3)
        weight_scale = torch.tensor([[11.0, 22.0, 33.0]])
        source = SimpleNamespace(
            quant_method=quant_method,
            weight=torch.nn.Parameter(weight, requires_grad=False),
            weight_scale_inv=torch.nn.Parameter(weight_scale, requires_grad=False),
        )
        source.weight_scale_inv.format_ue8m0 = False

        projection_slice = _BlockFp8LinearSlice(
            source=source,
            feature_indices=[0, 2],
            feature_width=feature_width,
        )

        expected_weight = torch.cat(
            [weight[:, :feature_width], weight[:, 2 * feature_width :]], dim=1
        )
        self.assertTrue(torch.equal(projection_slice.weight, expected_weight))
        self.assertTrue(
            torch.equal(
                projection_slice.weight_scale_inv,
                torch.tensor([[11.0, 33.0]]),
            )
        )

    def test_partial_projection_uses_prepared_quantized_slice(self):
        model = _make_deepseek_v4_dspark_projection_model(
            hidden_size=4, num_target_features=3
        )
        expected = torch.randn(2, 4)
        projection_slice = Mock(return_value=expected)
        model._partial_feature_indices = (1,)
        model._partial_main_proj = projection_slice
        local_hidden = torch.randn(2, 4)

        actual = model.project_target_hidden_partial(local_hidden, [1])

        projection_slice.assert_called_once_with(local_hidden)
        self.assertIs(actual, expected)

    def test_partial_projection_sum_matches_full_projection(self):
        """PP partial projections must preserve the full pre-norm FC result."""
        torch.manual_seed(0)
        hidden_size = 4
        model = _make_deepseek_v4_dspark_projection_model(
            hidden_size=hidden_size, num_target_features=3
        )

        feature_hidden = [
            torch.randn(5, hidden_size, dtype=torch.float32) for _ in range(3)
        ]
        full_hidden = torch.cat(feature_hidden, dim=-1)
        full_projected = model.project_target_hidden(full_hidden)

        stage_0 = model.project_target_hidden_partial(
            torch.cat([feature_hidden[0], feature_hidden[2]], dim=-1),
            [0, 2],
        )
        stage_1 = model.project_target_hidden_partial(feature_hidden[1], [1])
        pp_projected = model.stages[0].main_norm(stage_0 + stage_1)

        torch.testing.assert_close(pp_projected, full_projected)

    def test_deepseek_v4_partial_projection_survives_context_only_pruning(self):
        """Cross-rank contributions must retain full projection math after pruning."""
        torch.manual_seed(1)
        hidden_size = 4
        model = _make_deepseek_v4_dspark_projection_model(
            hidden_size=hidden_size, num_target_features=3
        )
        features = [torch.randn(5, hidden_size, dtype=torch.float32) for _ in range(3)]
        full_projected = model.project_target_hidden(torch.cat(features, dim=-1))

        stage_0 = model.project_target_hidden_partial(
            torch.cat([features[0], features[2]], dim=-1),
            [0, 2],
        )
        model.prune_to_ctx_projection()
        stage_1 = model.project_target_hidden_partial(features[1], [1])
        pp_projected = model.stages[0].main_norm(stage_0 + stage_1)
        write_context_hidden_kv = Mock()
        model._write_context_hidden_kv = write_context_hidden_kv
        model.write_projected_context_kv(
            projected_context=stage_0 + stage_1,
            swa_loc=torch.arange(5),
            positions=torch.arange(5),
            pool=object(),
        )

        self.assertEqual(list(model.stages[0]._modules), ["main_proj", "main_norm"])
        torch.testing.assert_close(pp_projected, full_projected)
        torch.testing.assert_close(
            write_context_hidden_kv.call_args.kwargs["main_x"],
            full_projected,
        )

    def test_deepseek_v4_capture_is_local_to_each_pp_rank(self):
        """A target feature on a non-last rank must not disappear from ctx_acc."""
        model = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
        torch.nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(is_last_rank=False)
        model.model = SimpleNamespace(
            start_layer=10,
            end_layer=20,
            dspark_layers_to_capture=None,
        )
        model.capture_aux_hidden_states = False

        model.set_dspark_layers_to_capture([5, 12, 18, 25])

        self.assertTrue(model.capture_aux_hidden_states)
        self.assertEqual(model.model.dspark_layers_to_capture, [12, 18])

    def test_non_last_pp_prefill_does_not_require_target_lm_head(self):
        """PP0-PP(N-2) must initialize without the last rank's lm_head."""
        self.assertTrue(
            _is_context_only_pp_prefill_rank(
                disaggregation_mode="prefill",
                pp_rank=0,
                pp_size=8,
            )
        )
        self.assertFalse(
            _is_context_only_pp_prefill_rank(
                disaggregation_mode="prefill",
                pp_rank=7,
                pp_size=8,
            )
        )

    def test_context_only_rank_does_not_require_draft_attention_backend(self):
        """Projection-only ranks must initialize overlap state without draft attention."""
        target_backend = object()
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._is_context_only_pp_prefill_rank = True
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=target_backend)
        )
        worker.draft_model_runner = SimpleNamespace()

        self.assertEqual(worker.spec_v2_attn_backends, (target_backend,))

    def test_pp_context_uses_fork_dspark_capture_config(self):
        """The fork stores DSpark capture ids directly on ModelRunner."""
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.ps = SimpleNamespace(pp_rank=1, pp_size=4, tp_rank=0)
        worker.model_runner = SimpleNamespace(dspark_target_layer_ids=[2, 5, 9])
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(start_layer=4, end_layer=8)
            )
        )
        worker.draft_model = SimpleNamespace(
            project_target_hidden_partial=Mock(),
            prepare_target_hidden_partial=Mock(),
        )
        worker._draft_is_moe = True
        worker._use_full_projection_prefill = False
        worker._pp_context_feature_indices = None

        worker._init_pp_context_feature_indices()

        self.assertEqual(worker._pp_context_feature_indices, [1])
        worker.draft_model.prepare_target_hidden_partial.assert_called_once_with([1])

    def test_lifecycle_only_rank_does_not_allocate_draft_pool(self):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._draft_worker = Mock()
        worker._is_lifecycle_only_pp_prefill_rank = True

        worker.alloc_memory_pool(memory_pool_config=Mock())

        worker._draft_worker.alloc_memory_pool.assert_not_called()

    def test_lifecycle_only_rank_does_not_publish_draft_pool(self):
        worker = SimpleNamespace(is_lifecycle_only_pp_prefill_rank=True)
        spec_algorithm = SimpleNamespace(
            is_ngram=lambda: False,
            is_dspark=lambda: True,
        )

        self.assertEqual(
            get_draft_kv_pool(
                draft_worker=worker,
                spec_algorithm=spec_algorithm,
                server_args=SimpleNamespace(enable_multi_layer_eagle=False),
                enable_overlap=False,
            ),
            (None, None),
        )

    def test_non_last_pp_prefill_uses_minimal_draft_kv_pool(self):
        """A context-only PP rank must not reserve the full draft KV capacity."""
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._draft_worker = Mock()
        worker._is_pd_prefill = True
        worker._draft_is_moe = True
        worker._is_context_only_pp_prefill_rank = True
        worker._is_lifecycle_only_pp_prefill_rank = False
        worker.ps = SimpleNamespace(pp_rank=0, pp_size=2)
        worker.page_size = 64
        full_config = MemoryPoolConfig(
            max_total_num_tokens=4096,
            max_running_requests=32,
        )

        worker.alloc_memory_pool(memory_pool_config=full_config)

        passed_config = worker._draft_worker.alloc_memory_pool.call_args.kwargs[
            "memory_pool_config"
        ]
        self.assertEqual(passed_config.max_total_num_tokens, 64)
        self.assertEqual(passed_config.max_running_requests, 32)
        self.assertEqual(full_config.max_total_num_tokens, 4096)

    def test_last_pp_prefill_keeps_full_draft_kv_pool(self):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._draft_worker = Mock()
        worker._is_pd_prefill = True
        worker._draft_is_moe = True
        worker._is_context_only_pp_prefill_rank = False
        worker._is_lifecycle_only_pp_prefill_rank = False
        worker.ps = SimpleNamespace(pp_rank=1, pp_size=2)
        worker.page_size = 64
        full_config = MemoryPoolConfig(
            max_total_num_tokens=4096,
            max_running_requests=32,
        )

        worker.alloc_memory_pool(memory_pool_config=full_config)

        passed_config = worker._draft_worker.alloc_memory_pool.call_args.kwargs[
            "memory_pool_config"
        ]
        self.assertIs(passed_config, full_config)


if __name__ == "__main__":
    unittest.main()
