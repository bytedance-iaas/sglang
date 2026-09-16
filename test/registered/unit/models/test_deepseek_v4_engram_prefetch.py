"""Engram rows-only prefetch policy and image-token masking, without a GPU."""

import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _EngramStub:
    def __init__(self, layer_hash_index, *, shared=True, pinned=True):
        self.layer_hash_index = layer_hash_index
        self.embed = Mock(side_effect=lambda ids: (ids.float() + 1).unsqueeze(-1))
        self.embed._shared = shared
        self.embed.host_table = SimpleNamespace(registered=pinned)
        self.project_from_embeddings = Mock(side_effect=lambda emb: emb * 2)
        self.apply_gate = Mock(side_effect=lambda hidden, kv: hidden + kv)
        self.sync_calls = 0

    def __call__(self, hidden, ids, *args, **kwargs):
        self.sync_calls += 1
        return self.apply_gate(hidden, self.project_from_embeddings(self.embed(ids)))


class TestDeepseekV4EngramPrefetch(CustomTestCase):
    def _make_model(self, vision_n_layers=2, *, enabled=True, shared=True, pinned=True):
        config = SimpleNamespace(
            model_type="deepseek_v41",
            vision_n_layers=vision_n_layers,
            image_token_id=7,
            hidden_size=2,
            vocab_size=16,
            num_hidden_layers=15,
            rms_norm_eps=1e-6,
            hc_eps=1e-6,
            hc_mult=2,
            hc_pre_from_prev_sublayer=True,
        )
        layers = [
            SimpleNamespace(
                engram=None,
                forward_hc_pre_from_prev=lambda **kwargs: (
                    kwargs["hidden_states"],
                    None,
                ),
            )
            for _ in range(config.num_hidden_layers)
        ]
        layers[1].engram = _EngramStub(0)
        layers[14].engram = _EngramStub(1, shared=shared, pinned=pinned)
        hasher = Mock(
            side_effect=lambda ids, batch: torch.stack((ids % 3, ids % 5), dim=1)[
                :, :, None
            ]
        )
        pp_group = SimpleNamespace(
            world_size=1, rank_in_group=0, is_first_rank=True, is_last_rank=True
        )
        stream, ready = Mock(), Mock()
        with ExitStack() as stack:
            for name, value in (
                ("SGLANG_ENABLE_DSV41_ENGRAM_EMBED_PREFETCH", enabled),
                ("SGLANG_ENABLE_DSV41_ENGRAM_KV_PREFETCH", False),
                ("SGLANG_OPT_USE_MULTI_STREAM_OVERLAP", False),
            ):
                stack.enter_context(getattr(envs, name).override(value))
            stack.enter_context(
                patch.multiple(
                    deepseek_v4,
                    _is_cuda=True,
                    _is_hip=False,
                    _is_npu=False,
                    get_pp_group=Mock(return_value=pp_group),
                    is_dp_attention_enabled=Mock(return_value=True),
                    get_platform=Mock(
                        return_value=SimpleNamespace(is_blackwell=False, is_sm90=True)
                    ),
                    VocabParallelEmbedding=Mock(return_value=nn.Identity()),
                    RMSNorm=Mock(return_value=nn.Identity()),
                    make_layers=Mock(return_value=(layers, 0, len(layers))),
                    build_engram_layout=Mock(return_value=object()),
                    is_cross_layer_mhc_fusion_enabled=Mock(return_value=False),
                    _is_fused_mhc_post_pre_enabled_xpu=Mock(return_value=False),
                    get_exec=Mock(
                        return_value=SimpleNamespace(
                            features=SimpleNamespace(
                                enable_decoder_swa_bounded_replay=False
                            )
                        )
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    deepseek_v4.EngramHasher, "from_config", return_value=hasher
                )
            )
            stack.enter_context(
                patch.object(torch, "get_device_module", return_value=Mock())
            )
            stack.enter_context(patch.object(torch.cuda, "Stream", return_value=stream))
            stack.enter_context(patch.object(torch.cuda, "Event", return_value=ready))
            return deepseek_v4.DeepseekV4Model(config)

    def _forward(self, model, ids, *, mode=ForwardMode.DECODE, capture=False):
        hidden = torch.arange(ids.numel() * 4, dtype=torch.float32).reshape(-1, 2, 2)
        main_stream = Mock()
        with (
            patch.object(deepseek_v4, "is_cp_active", return_value=False),
            patch.object(deepseek_v4, "is_in_breakable_cuda_graph", return_value=False),
            patch.object(deepseek_v4, "get_is_capture_mode", return_value=capture),
            patch.object(deepseek_v4, "check_cuda_graph_backend", return_value=True),
            patch.object(torch.cuda, "current_stream", return_value=main_stream),
            patch.object(
                torch.cuda, "stream", side_effect=lambda stream: nullcontext()
            ),
            patch.object(torch.Tensor, "record_stream"),
        ):
            output, _, _ = model._forward_layers_hc_pre_from_prev(
                positions=torch.arange(ids.numel()),
                hidden_states=hidden,
                forward_batch=SimpleNamespace(forward_mode=mode),
                input_ids=ids,
                input_ids_global=ids,
                capture_dspark=False,
                dspark_aux_hidden_states=[],
            )
        return hidden, output, main_stream

    def test_vision_config_does_not_disable_embedding_prefetch(self):
        for vision_n_layers in (0, 2):
            with self.subTest(vision_n_layers=vision_n_layers):
                model = self._make_model(vision_n_layers)
                self.assertIsNotNone(model.engram_embed_prefetch_stream)
                self.assertIsNotNone(model.engram_embed_prefetch_ready)
                self.assertIsNone(model.engram_prefetch_stream)

    def test_opt_in_and_shared_pinned_guards_remain(self):
        for kwargs in (
            {"enabled": False},
            {"shared": False},
            {"pinned": False},
        ):
            with self.subTest(**kwargs):
                model = self._make_model(**kwargs)
                self.assertIsNone(model.engram_embed_prefetch_stream)
                self.assertIsNone(model.engram_embed_prefetch_ready)

    def test_decode_matches_sync_and_preserves_image_token_mask(self):
        for token_ids in ([1], [1, 7, 2]):
            with self.subTest(token_ids=token_ids):
                ids = torch.tensor(token_ids)
                model = self._make_model()
                hidden, actual, main_stream = self._forward(model, ids)
                _, expected, _ = self._forward(self._make_model(enabled=False), ids)
                torch.testing.assert_close(actual, expected)
                image = ids == model.config.image_token_id
                torch.testing.assert_close(actual[image], hidden[image])
                increment = 2 * ((ids % 3 + 1) + (ids % 5 + 1))
                torch.testing.assert_close(
                    actual[~image], hidden[~image] + increment[~image, None, None]
                )
                # Only L14 uses the prefetched embedding; L1 still looks up inline.
                self.assertEqual(model.layers[1].engram.sync_calls, 1)
                self.assertEqual(model.layers[14].engram.sync_calls, 0)
                model.layers[14].engram.embed.assert_called_once()
                model.engram_embed_prefetch_stream.wait_stream.assert_called_once_with(
                    main_stream
                )
                model.engram_embed_prefetch_ready.record.assert_called_once_with(
                    model.engram_embed_prefetch_stream
                )
                main_stream.wait_event.assert_called_once_with(
                    model.engram_embed_prefetch_ready
                )

    def test_non_decode_capture_and_empty_batches_keep_sync_path(self):
        for mode, capture, token_ids in (
            (ForwardMode.EXTEND, False, [1, 7, 2]),
            (ForwardMode.TARGET_VERIFY, False, [1, 7, 2]),
            (ForwardMode.DECODE, True, [1, 7, 2]),
            (ForwardMode.DECODE, False, []),
        ):
            with self.subTest(mode=mode, capture=capture, token_ids=token_ids):
                ids = torch.tensor(token_ids, dtype=torch.int64)
                model = self._make_model()
                self.assertIsNotNone(model.engram_embed_prefetch_ready)
                _, actual, main_stream = self._forward(
                    model, ids, mode=mode, capture=capture
                )
                _, expected, _ = self._forward(
                    self._make_model(enabled=False), ids, mode=mode, capture=capture
                )
                torch.testing.assert_close(actual, expected)
                self.assertEqual(model.layers[14].engram.sync_calls, 1)
                model.engram_embed_prefetch_ready.record.assert_not_called()
                main_stream.wait_event.assert_not_called()


if __name__ == "__main__":
    unittest.main()
