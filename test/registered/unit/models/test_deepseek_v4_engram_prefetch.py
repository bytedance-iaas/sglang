"""Engram rows-only prefetch policy and image-token masking, without a GPU."""

import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch
from torch import nn

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.srt.environ import envs
from sglang.srt.layers.engram import EngramEmbedding, EngramHasher
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
        self.embed.dim = 1
        self.embed._empty = lambda ids: torch.empty(*ids.shape, 1)
        self.embed._gather_shared_into = Mock(
            side_effect=lambda ids, out: out.copy_((ids.float() + 1).unsqueeze(-1))
        )
        self.embed.prefetch = Mock(
            side_effect=lambda *args: EngramEmbedding.prefetch(self.embed, *args)
        )
        self.project_from_embeddings = Mock(side_effect=lambda emb: emb * 2)
        self.apply_gate = Mock(side_effect=lambda hidden, kv: hidden + kv)
        self.sync_calls = 0

    def __call__(self, hidden, ids, *args, **kwargs):
        self.sync_calls += 1
        return self.apply_gate(hidden, self.project_from_embeddings(self.embed(ids)))


class TestDeepseekV4EngramPrefetch(CustomTestCase):
    def _make_model(
        self,
        vision_n_layers=2,
        *,
        enabled=True,
        shared=True,
        pinned=True,
        layer_options=None,
        pp_world_size=1,
        layer_range=None,
    ):
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
        for index, layer_id in enumerate((1, 14)):
            options = (layer_options or {}).get(
                layer_id, dict(shared=shared, pinned=pinned)
            )
            layers[layer_id].engram = (
                _EngramStub(index, **options) if options is not None else None
            )
        hasher = Mock(
            side_effect=lambda ids, batch: torch.stack((ids % 3, ids % 5), dim=1)[
                :, :, None
            ]
        )
        start_layer, end_layer = layer_range or (0, len(layers))
        pp_group = SimpleNamespace(
            world_size=pp_world_size,
            rank_in_group=0,
            is_first_rank=start_layer == 0,
            is_last_rank=end_layer == len(layers),
        )
        stream = Mock()
        with ExitStack() as stack:
            for name, value in (
                ("SGLANG_ENABLE_DSV41_ENGRAM_EMBED_PREFETCH", enabled),
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
                    make_layers=Mock(return_value=(layers, start_layer, end_layer)),
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
                    deepseek_v4.EngramLayout, "from_config", return_value=object()
                )
            )
            hasher_factory = stack.enter_context(
                patch.object(
                    deepseek_v4.EngramHasher, "from_config", return_value=hasher
                )
            )
            stack.enter_context(
                patch.object(torch, "get_device_module", return_value=Mock())
            )
            stack.enter_context(patch.object(torch.cuda, "Stream", return_value=stream))
            stack.enter_context(patch.object(torch.cuda, "Event", side_effect=Mock))
            model = deepseek_v4.DeepseekV4Model(config)
            hasher_factory.assert_called_once_with(
                config,
                model.engram_layout,
                image_token_id=config.image_token_id if vision_n_layers > 0 else None,
            )
            return model

    def _forward(
        self,
        model,
        ids,
        *,
        mode=ForwardMode.DECODE,
        capture=False,
        piecewise=False,
        breakable=False,
        compiling=False,
        tail=None,
        cp=False,
        dp_size=1,
        dp_layout=None,
    ):
        hidden = torch.arange(ids.numel() * 4, dtype=torch.float32).reshape(-1, 2, 2)
        batch = SimpleNamespace(
            forward_mode=mode,
            is_extend_in_batch=dp_layout is not None,
            input_ids=ids,
            attn_cp_metadata=SimpleNamespace(total_seq_lens=ids.numel()),
        )
        backend = Mock(tail_forward_metadata=SimpleNamespace(late_layer_tail=tail))
        model._check_late_layer_tail_readers = Mock()
        if cp:
            hidden = hidden[1::2]
        main_stream = Mock()
        with (
            patch.object(deepseek_v4, "is_cp_active", return_value=cp),
            patch.object(
                deepseek_v4,
                "get_parallel",
                return_value=SimpleNamespace(
                    attn_dp_size=dp_size,
                    attn_cp_rank=1,
                    attn_cp_size=2,
                ),
            ),
            patch.object(
                deepseek_v4.LateLayerDPLayout, "prepare", return_value=dp_layout
            ),
            patch.object(
                deepseek_v4,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_none=lambda: True),
            ),
            patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
            patch.object(
                deepseek_v4, "is_in_breakable_cuda_graph", return_value=breakable
            ),
            patch.object(
                deepseek_v4, "is_in_tc_piecewise_cuda_graph", return_value=piecewise
            ),
            patch.object(deepseek_v4, "get_is_capture_mode", return_value=capture),
            patch.object(torch.compiler, "is_compiling", return_value=compiling),
            patch.object(
                deepseek_v4,
                "bcg_deepseek_v4_engram_hash_ids",
                side_effect=lambda hasher, ids: hasher(ids, batch),
            ),
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
                forward_batch=batch,
                input_ids=ids[1::2] if cp else ids,
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
                self.assertEqual(list(model.engram_embed_prefetch_events), [1, 14])
                self.assertIsNot(
                    model.engram_embed_prefetch_events[1],
                    model.engram_embed_prefetch_events[14],
                )

    def test_opt_in_and_shared_pinned_guards_remain(self):
        for kwargs in (
            {"enabled": False},
            {"shared": False},
            {"pinned": False},
        ):
            with self.subTest(**kwargs):
                model = self._make_model(**kwargs)
                self.assertIsNone(model.engram_embed_prefetch_stream)
                self.assertEqual(model.engram_embed_prefetch_events, {})

    def test_layer_guards_are_independent(self):
        for layer_id in (1, 14):
            for options in (None, {"shared": False}, {"pinned": False}):
                with self.subTest(layer=layer_id, options=options):
                    model = self._make_model(layer_options={layer_id: options})
                    self.assertEqual(
                        list(model.engram_embed_prefetch_events),
                        [14 if layer_id == 1 else 1],
                    )
                    self.assertIsNotNone(model.engram_embed_prefetch_stream)

    def test_pipeline_stages_prefetch_only_local_engram_layers(self):
        for layer_range, expected in (
            ((0, 14), [1]),
            ((14, 15), [14]),
            ((2, 14), []),
        ):
            with self.subTest(layer_range=layer_range):
                model = self._make_model(
                    pp_world_size=2,
                    layer_range=layer_range,
                )
                self.assertEqual(list(model.engram_embed_prefetch_events), expected)
                self.assertEqual(
                    model.engram_embed_prefetch_stream is not None,
                    bool(expected),
                )

    def test_decode_matches_sync_and_preserves_image_token_mask(self):
        for token_ids in ([1], [1, 7, 2]):
            with self.subTest(token_ids=token_ids):
                ids = torch.tensor(token_ids)
                model = self._make_model()
                launches = Mock()
                for layer_id in (1, 14):
                    launches.attach_mock(
                        model.layers[layer_id].engram.embed.prefetch, f"l{layer_id}"
                    )
                hidden, actual, main_stream = self._forward(model, ids)
                _, expected, _ = self._forward(self._make_model(enabled=False), ids)
                torch.testing.assert_close(actual, expected)
                image = ids == model.config.image_token_id
                torch.testing.assert_close(actual[image], hidden[image])
                increment = 2 * ((ids % 3 + 1) + (ids % 5 + 1))
                torch.testing.assert_close(
                    actual[~image], hidden[~image] + increment[~image, None, None]
                )
                self.assertEqual(
                    [entry[0] for entry in launches.mock_calls], ["l1", "l14"]
                )
                self.assertEqual(
                    model.engram_embed_prefetch_stream.wait_stream.call_args_list,
                    [call(main_stream), call(main_stream)],
                )
                self.assertEqual(
                    main_stream.wait_event.call_args_list,
                    [call(model.engram_embed_prefetch_events[i]) for i in (1, 14)],
                )
                for layer_id in (1, 14):
                    self.assertEqual(model.layers[layer_id].engram.sync_calls, 0)
                    model.layers[layer_id].engram.embed.prefetch.assert_called_once()
                    model.engram_embed_prefetch_events[
                        layer_id
                    ].record.assert_called_once_with(model.engram_embed_prefetch_stream)

    def test_decode_capture_and_eager_extend_match_sync(self):
        for mode, capture in (
            (ForwardMode.DECODE, True),
            (ForwardMode.EXTEND, False),
            (ForwardMode.MIXED, False),
        ):
            with self.subTest(mode=mode, capture=capture):
                ids = torch.tensor([1, 7, 2, 3, 4])
                model = self._make_model()
                _, actual, _ = self._forward(model, ids, mode=mode, capture=capture)
                _, expected, _ = self._forward(
                    self._make_model(enabled=False), ids, mode=mode, capture=capture
                )
                torch.testing.assert_close(actual, expected)
                for layer_id in (1, 14):
                    model.layers[layer_id].engram.embed.prefetch.assert_called_once()
                    self.assertEqual(model.layers[layer_id].engram.sync_calls, 0)

    def test_target_verify_eager_and_capture_keep_all_local_rows(self):
        # gamma=5: each rank-local request verifies anchor + five drafts.
        for capture in (False, True):
            for requests in (1, 3, 8):
                with self.subTest(capture=capture, requests=requests):
                    ids = (torch.arange(requests * 6) + requests) % 16
                    model = self._make_model()
                    sync = self._make_model(enabled=False)
                    # Bounded prefill must not trim a verify window.
                    model.late_layer_start = sync.late_layer_start = 10
                    kwargs = dict(mode=ForwardMode.TARGET_VERIFY, capture=capture)
                    _, actual, stream = self._forward(model, ids, **kwargs)
                    _, expected, _ = self._forward(sync, ids, **kwargs)
                    torch.testing.assert_close(actual, expected)
                    model.engram_hasher.assert_called_once()
                    model._check_late_layer_tail_readers.assert_not_called()
                    self.assertEqual(
                        stream.wait_event.call_args_list,
                        [call(model.engram_embed_prefetch_events[i]) for i in (1, 14)],
                    )
                    for layer_id, divisor in ((1, 3), (14, 5)):
                        engram = model.layers[layer_id].engram
                        self.assertEqual(engram.sync_calls, 0)
                        engram.embed.prefetch.assert_called_once()
                        torch.testing.assert_close(
                            engram.embed.prefetch.call_args.args[0],
                            (ids % divisor)[:, None],
                        )

    def test_prefill_graph_unsupported_and_empty_batches_keep_sync_path(self):
        for mode, capture, token_ids in (
            (ForwardMode.EXTEND, True, [1, 7, 2]),
            (ForwardMode.DRAFT_EXTEND_V2, False, [1, 7, 2]),
            (ForwardMode.SPLIT_PREFILL, False, [1, 7, 2]),
            (ForwardMode.IDLE, False, [1, 7, 2]),
            (ForwardMode.DECODE, False, []),
            (ForwardMode.TARGET_VERIFY, True, []),
        ):
            with self.subTest(mode=mode, capture=capture, token_ids=token_ids):
                ids = torch.tensor(token_ids, dtype=torch.int64)
                model = self._make_model()
                self.assertEqual(list(model.engram_embed_prefetch_events), [1, 14])
                _, actual, main_stream = self._forward(
                    model, ids, mode=mode, capture=capture
                )
                _, expected, _ = self._forward(
                    self._make_model(enabled=False), ids, mode=mode, capture=capture
                )
                torch.testing.assert_close(actual, expected)
                for layer_id in (1, 14):
                    self.assertEqual(model.layers[layer_id].engram.sync_calls, 1)
                    model.engram_embed_prefetch_events[
                        layer_id
                    ].record.assert_not_called()
                main_stream.wait_event.assert_not_called()

    def test_piecewise_prefill_keeps_sync_path(self):
        model = self._make_model()
        self._forward(
            model, torch.tensor([1, 2]), mode=ForwardMode.EXTEND, piecewise=True
        )
        for layer_id in (1, 14):
            model.layers[layer_id].engram.embed.prefetch.assert_not_called()

    def test_verify_partial_graphs_and_compile_keep_sync_path(self):
        ids = torch.arange(12)
        for kwargs in ({"piecewise": True}, {"breakable": True}, {"compiling": True}):
            with self.subTest(**kwargs):
                model = self._make_model()
                _, actual, stream = self._forward(
                    model, ids, mode=ForwardMode.TARGET_VERIFY, **kwargs
                )
                _, expected, _ = self._forward(
                    self._make_model(enabled=False),
                    ids,
                    mode=ForwardMode.TARGET_VERIFY,
                    **kwargs,
                )
                torch.testing.assert_close(actual, expected)
                stream.wait_event.assert_not_called()
                for layer_id in (1, 14):
                    self.assertEqual(model.layers[layer_id].engram.sync_calls, 1)

    def test_verify_budget_counts_padded_token_rows_not_requests(self):
        # Two graph request slots, six tokens each, even if only one is live.
        ids = torch.arange(12)
        per_layer_bytes = ids.numel() * 2
        for budget, selected in (
            (0, []),
            (per_layer_bytes - 1, []),
            (per_layer_bytes, [14]),
            (2 * per_layer_bytes, [1, 14]),
        ):
            with self.subTest(budget=budget):
                model = self._make_model()
                model.engram_embed_prefetch_max_bytes = budget
                _, actual, stream = self._forward(
                    model, ids, mode=ForwardMode.TARGET_VERIFY, capture=True
                )
                _, expected, _ = self._forward(
                    self._make_model(enabled=False), ids, mode=ForwardMode.TARGET_VERIFY
                )
                torch.testing.assert_close(actual, expected)
                self.assertEqual(
                    stream.wait_event.call_args_list,
                    [call(model.engram_embed_prefetch_events[i]) for i in selected],
                )

    def test_total_buffer_budget_reserves_l14_before_l1(self):
        ids = torch.tensor([1, 7, 2, 3, 4, 5])
        per_layer_bytes = ids.numel() * 2
        _, expected, _ = self._forward(self._make_model(enabled=False), ids)
        for budget, selected in (
            (0, []),
            (per_layer_bytes - 1, []),
            (per_layer_bytes, [14]),
            (2 * per_layer_bytes - 1, [14]),
            (2 * per_layer_bytes, [1, 14]),
        ):
            with self.subTest(budget=budget):
                model = self._make_model()
                model.engram_embed_prefetch_max_bytes = budget
                _, actual, stream = self._forward(model, ids)
                torch.testing.assert_close(actual, expected)
                self.assertEqual(
                    stream.wait_event.call_args_list,
                    [call(model.engram_embed_prefetch_events[i]) for i in selected],
                )
                for i in (1, 14):
                    self.assertEqual(
                        model.layers[i].engram.sync_calls, int(i not in selected)
                    )

    def test_buffer_budget_and_tail_row_selection(self):
        ids = torch.tensor([1, 7, 2, 3, 4, 5])
        for rows in (torch.tensor([3, 4, 5]), torch.tensor([1, 4])):
            tail = SimpleNamespace(
                rows=lambda x: None if x is None else x[rows], positions=rows
            )
            for budget in (0, rows.numel() * 2 - 1, rows.numel() * 2):
                with self.subTest(rows=rows, budget=budget):
                    model, sync = self._make_model(), self._make_model(enabled=False)
                    model.late_layer_start = sync.late_layer_start = 10
                    model.engram_embed_prefetch_max_bytes = budget
                    _, actual, _ = self._forward(
                        model, ids, mode=ForwardMode.EXTEND, tail=tail
                    )
                    _, expected, _ = self._forward(
                        sync, ids, mode=ForwardMode.EXTEND, tail=tail
                    )
                    torch.testing.assert_close(actual, expected)
                    prefetch = model.layers[14].engram.embed.prefetch
                    if budget == rows.numel() * 2:
                        prefetch.assert_called_once()
                        torch.testing.assert_close(
                            prefetch.call_args.args[0], (ids[rows] % 5)[:, None]
                        )
                    else:
                        prefetch.assert_not_called()

    def test_each_layer_selects_only_its_own_bounded_tail(self):
        ids = torch.tensor([1, 7, 2, 3, 4, 5])
        for start in (1, 10):
            for rows in (torch.tensor([3, 4, 5]), torch.tensor([1, 4])):
                with self.subTest(start=start, rows=rows):
                    tail = SimpleNamespace(
                        rows=lambda x: None if x is None else x[rows], positions=rows
                    )
                    model, sync = self._make_model(), self._make_model(enabled=False)
                    model.late_layer_start = sync.late_layer_start = start
                    _, actual, _ = self._forward(
                        model, ids, mode=ForwardMode.EXTEND, tail=tail
                    )
                    _, expected, _ = self._forward(
                        sync, ids, mode=ForwardMode.EXTEND, tail=tail
                    )
                    torch.testing.assert_close(actual, expected)
                    for layer_id, divisor in ((1, 3), (14, 5)):
                        selected_ids = ids[rows] if start <= layer_id else ids
                        torch.testing.assert_close(
                            model.layers[layer_id].engram.embed.prefetch.call_args.args[
                                0
                            ],
                            (selected_ids % divisor)[:, None],
                        )

    def test_cp_extend_uses_local_hash_rows(self):
        ids = torch.tensor([1, 7, 2, 3, 4, 5])
        model = self._make_model()
        _, actual, _ = self._forward(model, ids, mode=ForwardMode.EXTEND, cp=True)
        _, expected, _ = self._forward(
            self._make_model(enabled=False), ids, mode=ForwardMode.EXTEND, cp=True
        )
        torch.testing.assert_close(actual, expected)
        for layer_id, divisor in ((1, 3), (14, 5)):
            torch.testing.assert_close(
                model.layers[layer_id].engram.embed.prefetch.call_args.args[0],
                (ids[1::2] % divisor)[:, None],
            )

    def test_dp_prefetch_keeps_rank_local_rows(self):
        for rows in (0, 1, 3):
            with self.subTest(local_rows=rows):
                ids = torch.arange(rows, dtype=torch.int64)
                model, sync = self._make_model(), self._make_model(enabled=False)
                _, actual, _ = self._forward(
                    model, ids, mode=ForwardMode.EXTEND, dp_size=8
                )
                _, expected, _ = self._forward(
                    sync, ids, mode=ForwardMode.EXTEND, dp_size=8
                )
                torch.testing.assert_close(actual, expected)
                for layer_id in (1, 14):
                    if rows:
                        self.assertEqual(
                            model.layers[layer_id]
                            .engram.embed.prefetch.call_args.args[0]
                            .shape[0],
                            rows,
                        )
                    else:
                        model.layers[layer_id].engram.embed.prefetch.assert_not_called()

    def test_late_dp_padding_keeps_prefetch_rank_local(self):
        for mode in (ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.IDLE):
            for rows in (0, 1, 3):
                with self.subTest(mode=mode, local_rows=rows):
                    ids = torch.arange(rows, dtype=torch.int64)
                    layout = SimpleNamespace(
                        local_rows=rows,
                        pad=lambda x: torch.cat(
                            [x, x.new_zeros(7 - rows, *x.shape[1:])]
                        ),
                        activate=lambda batch: nullcontext(),
                        gather_input_ids=Mock(side_effect=lambda ids, batch, **kw: ids),
                    )
                    model, sync = self._make_model(), self._make_model(enabled=False)
                    model.late_layer_start = sync.late_layer_start = 10
                    kwargs = dict(mode=mode, dp_size=8, dp_layout=layout)
                    _, actual, _ = self._forward(model, ids, **kwargs)
                    _, expected, _ = self._forward(sync, ids, **kwargs)
                    torch.testing.assert_close(actual, expected)
                    self.assertEqual(layout.gather_input_ids.call_count, 2)
                    # The synchronous fallback retains the target branch's
                    # communication padding, while shared-table prefetch only
                    # gathers local rows and needs no Engram collective.
                    self.assertEqual(
                        sync.layers[14].engram.embed.call_args.args[0].shape[0], 7
                    )
                    for layer_id in (1, 14):
                        prefetch = model.layers[layer_id].engram.embed.prefetch
                        if rows and mode != ForwardMode.IDLE:
                            self.assertEqual(prefetch.call_args.args[0].shape[0], rows)
                            self.assertEqual(
                                model.layers[layer_id].engram.sync_calls, 0
                            )
                        else:
                            prefetch.assert_not_called()

    def test_padded_idle_hashes_do_not_touch_history(self):
        hasher = EngramHasher.__new__(EngramHasher)
        nn.Module.__init__(hasher)
        hasher.max_ngram_size = 4
        hasher.primes = torch.zeros(2, 3, 8, dtype=torch.int64)
        hasher.offsets = torch.zeros(2, 24, dtype=torch.int64)
        hasher.history = torch.arange(15).reshape(5, 3)
        before = hasher.history.clone()
        for tokens in (0, 1, 8):
            hashes = hasher(
                torch.zeros(tokens, dtype=torch.int64),
                SimpleNamespace(forward_mode=ForwardMode.IDLE),
            )
            self.assertEqual(hashes.shape, (tokens, 2, 24))
            self.assertEqual(hashes.count_nonzero().item(), 0)
            torch.testing.assert_close(hasher.history, before)


if __name__ == "__main__":
    unittest.main()
