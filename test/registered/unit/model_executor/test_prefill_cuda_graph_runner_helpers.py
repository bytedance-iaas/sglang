"""Unit tests for prefill CUDA graph wrapper helpers."""

import unittest
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    build_prefill_registry,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
    _build_layer_model_forward_kwargs,
    _resolve_transformer_layer_model,
)
from sglang.srt.model_executor.runner_utils.buffers import PrefillInputBuffers
from sglang.srt.model_loader.utils import resolve_language_model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _LayerModel:
    def __init__(self):
        self.layers = [object()]

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return input_embeds


def _make_pp_buffers_and_registry():
    base = torch.zeros(3, dtype=torch.int64)
    buffers = SimpleNamespace(
        **{name: base.clone() for name in ("input_ids", "positions", "out_cache_loc")},
        pp_proxy_tensors={
            key: torch.zeros((3, 2)) for key in ("hidden_states", "residual")
        },
    )
    registry = build_prefill_registry(
        device=base.device,
        max_bs=1,
        max_num_token=len(base),
        cache_loc_dtype=torch.int64,
        share_pool=False,
        source=buffers,
    )
    return buffers, registry


class TestPrefillCudaGraphRunnerHelpers(CustomTestCase):
    def test_pp_proxy_stable_buffers_accept_full_and_hidden_only_contracts(self):
        buffers, registry = _make_pp_buffers_and_registry()
        full_proxy = PPProxyTensors(
            {
                "hidden_states": torch.full((3, 2), 2.0),
                "residual": torch.full((3, 2), 3.0),
            }
        )
        values = torch.arange(3)
        fill = partial(
            registry.fill_from,
            SimpleNamespace(input_ids=values, positions=values, out_cache_loc=values),
            raw_bs=1,
            padded_bs=1,
            raw_num_tokens=3,
            padded_num_tokens=3,
        )
        fill(pp_proxy_tensors=full_proxy)

        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.buffers = buffers
        runner.capture_forward_mode = ForwardMode.TARGET_VERIFY
        runner.require_attn_tp_gather = True
        runner.model_runner = SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=False),
            ps=SimpleNamespace(pp_rank=1, attn_tp_size=2, attn_cp_size=1),
            get_pp_proxy_input_token_scatter_factor=lambda: 2,
        )
        capture_proxy = runner._capture_pp_proxy_tensors(3)
        self.assertEqual(tuple(capture_proxy["hidden_states"].shape), (2, 2))
        self.assertEqual(tuple(capture_proxy["residual"].shape), (2, 2))
        torch.testing.assert_close(
            capture_proxy["hidden_states"], full_proxy["hidden_states"][:2]
        )
        torch.testing.assert_close(
            capture_proxy["residual"], full_proxy["residual"][:2]
        )
        self.assertEqual(
            capture_proxy["hidden_states"].data_ptr(),
            buffers.pp_proxy_tensors["hidden_states"].data_ptr(),
        )

        hidden_only_proxy = PPProxyTensors({"hidden_states": torch.full((3, 2), 4.0)})
        fill(pp_proxy_tensors=hidden_only_proxy)
        torch.testing.assert_close(
            buffers.pp_proxy_tensors["hidden_states"][:3],
            hidden_only_proxy["hidden_states"],
        )

    def test_layer_model_kwargs_bind_optional_inputs_by_signature(self):
        def proxy_before_embeds(a, b, c, pp_proxy_tensors=None, inputs_embeds=None):
            pass

        cases = (
            (_LayerModel(), {"input_embeds": "embeds"}),
            (
                SimpleNamespace(forward=proxy_before_embeds),
                {"inputs_embeds": "embeds", "pp_proxy_tensors": "proxy"},
            ),
        )
        forward_batch = SimpleNamespace(input_embeds="embeds")
        for layer_model, expected in cases:
            with self.subTest(signature=layer_model.forward.__name__):
                kwargs = _build_layer_model_forward_kwargs(
                    layer_model, forward_batch, "proxy"
                )
                self.assertEqual(kwargs, expected)
                layer_model.forward(None, None, forward_batch, **kwargs)

    def test_finalize_pp_proxy_trims_padded_token_rows(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.raw_num_tokens = 3
        runner.capture_forward_mode = ForwardMode.EXTEND
        runner.require_attn_tp_gather = True
        runner.model_runner = SimpleNamespace(
            ps=SimpleNamespace(pp_rank=0, attn_tp_size=2, attn_cp_size=1),
            get_pp_proxy_output_token_scatter_factor=lambda: 2,
        )
        output = PPProxyTensors({"hidden_states": torch.arange(10).reshape(5, 2)})
        trimmed = runner._finalize_execute_output(
            output, forward_mode=runner.capture_forward_mode
        )
        self.assertIsInstance(trimmed, PPProxyTensors)
        self.assertEqual(tuple(trimmed["hidden_states"].shape), (2, 2))
        torch.testing.assert_close(
            trimmed["hidden_states"][-1], output["hidden_states"][1]
        )

    def test_resolve_layer_model_from_language_model_wrapper(self):
        layer_model = _LayerModel()
        model = SimpleNamespace(language_model=SimpleNamespace(model=layer_model))

        self.assertIs(_resolve_transformer_layer_model(model), layer_model)

    def test_resolve_layer_model_from_nested_model_wrapper(self):
        layer_model = _LayerModel()
        model = SimpleNamespace(model=SimpleNamespace(model=layer_model))

        self.assertIs(_resolve_transformer_layer_model(model), layer_model)

    def test_resolve_layer_model_rejects_wrapper_without_layers(self):
        model = SimpleNamespace()
        model.model = model

        with self.assertRaisesRegex(RuntimeError, "without layers"):
            _resolve_transformer_layer_model(model)

    def test_resolve_language_model_accepts_asr_style_wrapper(self):
        language_model = object()
        self.assertIs(
            resolve_language_model(SimpleNamespace(language_model=language_model)),
            language_model,
        )

    def test_resolve_language_model_accepts_omni_style_wrapper(self):
        language_model = object()
        omni_model = type("Qwen3OmniMoeForConditionalGeneration", (), {})()
        omni_model.thinker = SimpleNamespace(model=language_model)
        self.assertIs(resolve_language_model(omni_model), language_model)

    def test_resolve_language_model_rejects_non_language_wrapper(self):
        with self.assertRaises(AttributeError):
            resolve_language_model(SimpleNamespace())

    def test_prefill_buffers_allocate_pipeline_proxy_token_rows(self):
        buffers = PrefillInputBuffers.create(
            device=torch.device("cpu"),
            max_bs=4,
            max_num_tokens=16,
            cache_loc_dtype=torch.int64,
            is_multimodal=False,
            hidden_size=8,
            dtype=torch.bfloat16,
            enable_mamba_track=False,
            pp_size=2,
            is_first_pp_rank=False,
            pp_proxy_residual_num_blocks=3,
        )

        self.assertEqual(
            {
                key: tuple(value.shape)
                for key, value in buffers.pp_proxy_tensors.items()
            },
            {"hidden_states": (16, 8), "residual": (16, 3, 8)},
        )

    def test_pipeline_proxy_output_is_supported(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.raw_num_tokens = 3
        runner.capture_forward_mode = ForwardMode.EXTEND
        runner.require_attn_tp_gather = True
        runner.model_runner = SimpleNamespace(
            ps=SimpleNamespace(pp_rank=0, attn_tp_size=2, attn_cp_size=1),
            get_pp_proxy_output_token_scatter_factor=lambda: 2,
        )
        output = PPProxyTensors({"hidden_states": torch.zeros((8, 8))})

        finalized = runner._finalize_execute_output(
            output, forward_mode=runner.capture_forward_mode
        )
        self.assertEqual(finalized["hidden_states"].shape, (2, 8))

    def test_pp_proxy_widths_follow_tp_and_topk_contracts(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.raw_num_tokens = 5
        runner.capture_forward_mode = ForwardMode.TARGET_VERIFY
        runner.require_attn_tp_gather = True
        runner.model_runner = SimpleNamespace(
            ps=SimpleNamespace(pp_rank=1, attn_tp_size=2, attn_cp_size=1),
            get_pp_proxy_output_token_scatter_factor=lambda: 2,
        )
        output = PPProxyTensors(
            {
                "hidden_states": torch.zeros((8, 4)),
                "residual": torch.zeros((8, 4)),
                "topk_indices": torch.zeros((8,), dtype=torch.int64),
            }
        )

        finalized = runner._finalize_execute_output(
            output, forward_mode=runner.capture_forward_mode
        )

        self.assertEqual(tuple(finalized["hidden_states"].shape), (3, 4))
        self.assertEqual(tuple(finalized["residual"].shape), (3, 4))
        self.assertEqual(tuple(finalized["topk_indices"].shape), (5,))

    def test_pp_proxy_widths_follow_boundary_not_global_gather_gate(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.raw_num_tokens = 5
        runner.capture_forward_mode = ForwardMode.TARGET_VERIFY
        # This intentionally disagrees with the boundary capability.  CUDA
        # graph geometry must follow the model's actual PP input ownership.
        runner.require_attn_tp_gather = True
        runner.model_runner = SimpleNamespace(
            ps=SimpleNamespace(pp_rank=1, attn_tp_size=2, attn_cp_size=1),
            get_pp_proxy_output_token_scatter_factor=lambda: 1,
        )
        output = PPProxyTensors({"hidden_states": torch.zeros((8, 4))})

        finalized = runner._finalize_execute_output(
            output, forward_mode=runner.capture_forward_mode
        )

        self.assertEqual(tuple(finalized["hidden_states"].shape), (5, 4))

    def test_pp_proxy_widths_use_context_parallel_for_extend_and_mixed(self):
        for forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            with self.subTest(forward_mode=forward_mode):
                runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
                runner.raw_num_tokens = 5
                runner.capture_forward_mode = forward_mode
                runner.require_attn_tp_gather = True
                runner.model_runner = SimpleNamespace(
                    ps=SimpleNamespace(pp_rank=1, attn_tp_size=2, attn_cp_size=4),
                    get_pp_proxy_output_token_scatter_factor=lambda: 2,
                )
                output = PPProxyTensors(
                    {
                        "hidden_states": torch.zeros((8, 4)),
                        "residual": torch.zeros((8, 4)),
                        "topk_indices": torch.zeros((8,), dtype=torch.int64),
                    }
                )

                finalized = runner._finalize_execute_output(
                    output, forward_mode=forward_mode
                )

                self.assertEqual(tuple(finalized["hidden_states"].shape), (2, 4))
                self.assertEqual(tuple(finalized["residual"].shape), (2, 4))
                self.assertEqual(tuple(finalized["topk_indices"].shape), (2,))

    def test_draft_extend_with_cp_falls_back_from_prefill_graph(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.model_runner = SimpleNamespace(ps=SimpleNamespace(attn_cp_size=4))

        self.assertFalse(
            runner.can_replay_locally(
                batch_size=1,
                num_tokens=4,
                input_embeds=None,
                replace_embeds=None,
                prefix_lens=None,
                is_target_verify=False,
                is_draft_extend_v2=True,
                capture_hidden_mode=None,
                return_logprob=False,
            )
        )

    def test_bcg_eager_tail_uses_live_multimodal_embeddings(self):
        live_embeds = object()
        live_batch = SimpleNamespace(mm_input_embeds=live_embeds)
        static_batch = SimpleNamespace(
            input_ids=None,
            positions=None,
            mm_input_embeds=None,
        )

        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner._input_embeds_arg_idx = None
        runner.buffer_registry = SimpleNamespace(has_slot=lambda _name: False)
        runner.backend = SimpleNamespace(replay=lambda *_args, **_kwargs: None)
        runner.layer_model = SimpleNamespace(forward=lambda *_args, **_kwargs: None)
        runner.model_runner = SimpleNamespace(
            model=SimpleNamespace(
                forward=lambda _ids, _positions, batch, **_kwargs: batch.mm_input_embeds
            )
        )
        runner._prefill_forward_context = lambda *_args, **_kwargs: nullcontext()

        output = runner._execute_body_capture(
            live_batch,
            static_batch,
            static_num_tokens=1,
            raw_num_tokens=1,
            shape_key=object(),
        )

        self.assertIs(output, live_embeds)


if __name__ == "__main__":
    unittest.main()
