"""Static DSpark PD validation and heterogeneous attention-TP rank mapping."""

import copy
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.arg_groups.deepseek_v4_hook import (
    _dsv41_dspark_pd_parallelism_supported,
    validate_deepseek_v41_features,
)
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    CommonKVSender,
)
from sglang.srt.disaggregation.utils import get_dsv41_spec_layout
from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    request_scoped_state_transfer_indices,
)
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_layout():
    args = SimpleNamespace(
        mla_compression_ratios=[0, 2, 1],
        kv_layer_ids=[1, 2],
        kv_item_lens=[512, 1024],
        state_types=[StateType.SWA, StateType.DSV4_REQUEST_STATE, StateType.SWA],
        state_item_lens=[[512], [32768], [512]],
    )
    with get_context().override_server_args(
        speculative_algorithm="DSPARK", speculative_num_draft_tokens=6
    ):
        return get_dsv41_spec_layout(args)


class TestDSV41DSparkPD(CustomTestCase):
    def test_dp_attention_allowed_with_static_mooncake_pd(self):
        from sglang.srt.arg_groups import deepseek_v4_hook as hook
        from sglang.srt.model_executor.cuda_graph_config import Backend
        from sglang.srt.speculative.ragged_verify import RaggedVerifyMode

        cfg = SimpleNamespace(
            enable_encoder_swa_bounded_replay=False,
            enable_decoder_swa_bounded_replay=True,
            enable_dp_attention=True,
            enable_prefill_cp=False,
            enable_prefill_context_parallel=False,
            dp_size=8,
            speculative_algorithm="DSPARK",
            enable_hisparse=False,
            dsv4_attn_backend="auto",
            enable_two_batch_overlap=False,
            pp_size=1,
            attn_cp_size=1,
            dcp_size=1,
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.DISABLED, max_seq_len=4096)
            ),
        )
        model = SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41"))
        with (
            patch.object(hook, "resolving_view", return_value=cfg),
            patch.object(hook, "model_config_of", return_value=model),
            patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
                return_value=False,
            ),
            patch(
                "sglang.srt.speculative.ragged_verify.read_ragged_verify_mode",
                return_value=RaggedVerifyMode.STATIC,
            ) as verify_mode,
        ):
            for role in ("prefill", "decode"):
                for dp_size in (1, 8):
                    with self.subTest(role=role, dp_size=dp_size):
                        cfg.disaggregation_mode = role
                        cfg.dp_size = dp_size
                        if dp_size == 1:
                            hook.validate_deepseek_v41_features(object())
                        else:
                            with self.assertRaisesRegex(
                                ValueError, "decoder-swa-bounded-replay"
                            ):
                                hook.validate_deepseek_v41_features(object())

            for name, value in (
                ("disaggregation_transfer_backend", "nixl"),
                ("attn_cp_size", 2),
                ("dcp_size", 2),
            ):
                with self.subTest(unsupported=name), patch.object(cfg, name, value):
                    with self.assertRaisesRegex(ValueError, "DSpark PD requires"):
                        hook.validate_deepseek_v41_features(object())
            verify_mode.return_value = RaggedVerifyMode.COMPACT
            with self.assertRaisesRegex(ValueError, "DSpark PD requires"):
                hook.validate_deepseek_v41_features(object())

    def test_bootstrap_validates_before_caching(self):
        layout = make_layout()
        cases = [("matching", layout, layout, 4, True), ("legacy", None, None, 2, True)]
        for key, value in (
            ("num_draft_tokens", 5),
            ("kv_layer_ids", [2, 1]),
            ("kv_item_lens", [256, 1024]),
            ("state_types", ["swa", "c128_state"]),
            ("state_item_lens", [[512], [8192], [512]]),
        ):
            different = copy.deepcopy(layout)
            different[key] = value
            cases.append((key, layout, different, 4, False))
        cases += [
            ("prefill_only", None, layout, 4, False),
            ("decode_only_or_old_prefill", layout, None, 4, False),
            ("smaller_prefill_tp", layout, layout, 2, True),
            ("larger_prefill_tp", layout, layout, 8, True),
        ]
        for name, local, peer, tp_size, supported in cases:
            with self.subTest(name=name):
                manager = object.__new__(CommonKVManager)
                manager.prefill_info_table = {}
                manager.kv_args = SimpleNamespace(page_size=256)
                manager.kv_cache_dtype_str = "fp8_e4m3"
                manager.dsv41_spec_layout = local
                manager.attn_tp_size = 4
                manager.attn_cp_size = 1
                manager.dcp_size = 1
                manager._resolve_rank_mapping = Mock()
                response = Mock(status_code=200)
                response.json.return_value = dict(
                    attn_tp_size=tp_size,
                    attn_cp_size=1,
                    dp_size=1,
                    pp_size=1,
                    page_size=256,
                    kv_cache_dtype="fp8_e4m3",
                    follow_bootstrap_room=True,
                    dsv41_spec_layout=peer,
                )
                with patch(
                    "sglang.srt.disaggregation.common.conn.requests.get",
                    return_value=response,
                ) as fetch:
                    if supported:
                        self.assertTrue(
                            manager.try_ensure_parallel_info("prefill:8998")
                        )
                        self.assertTrue(
                            manager.try_ensure_parallel_info("prefill:8998")
                        )
                        fetch.assert_called_once()
                        manager._resolve_rank_mapping.assert_called_once()
                    else:
                        with self.assertRaisesRegex(
                            RuntimeError, "DeepSeek-V4.1 DSpark PD"
                        ):
                            manager.try_ensure_parallel_info("prefill:8998")
                        self.assertFalse(manager.prefill_info_table)
                        manager._resolve_rank_mapping.assert_not_called()

    def test_mla_heterogeneous_tp_bootstrap_rank_mapping(self):
        layout = make_layout()
        for prefill_tp, decode_tp in ((8, 1), (8, 4), (1, 8), (4, 8)):
            for rank in range(decode_tp):
                with self.subTest(
                    prefill_tp=prefill_tp, decode_tp=decode_tp, rank=rank
                ):
                    manager = object.__new__(CommonKVManager)
                    manager.prefill_info_table = {}
                    manager.kv_args = SimpleNamespace(page_size=256, engine_rank=rank)
                    manager.kv_cache_dtype_str = "fp8_e4m3"
                    manager.dsv41_spec_layout = layout
                    manager.attn_tp_size = decode_tp
                    manager.attn_cp_size = manager.dcp_size = manager.pp_size = 1
                    manager.attn_cp_rank = manager.pp_rank = 0
                    manager.is_mla_backend = True
                    manager.is_hybrid_mla_backend = False
                    response = Mock(status_code=200)
                    response.json.return_value = dict(
                        attn_tp_size=prefill_tp,
                        attn_cp_size=1,
                        dp_size=1,
                        pp_size=1,
                        page_size=256,
                        kv_cache_dtype="fp8_e4m3",
                        follow_bootstrap_room=True,
                        dsv41_spec_layout=layout,
                    )
                    with patch(
                        "sglang.srt.disaggregation.common.conn.requests.get",
                        return_value=response,
                    ):
                        self.assertTrue(
                            manager.try_ensure_parallel_info("prefill:8998")
                        )
                    info = manager.prefill_info_table["prefill:8998"]
                    self.assertEqual(
                        info.target_tp_rank, rank * prefill_tp // decode_tp
                    )
                    self.assertEqual(info.required_prefill_response_num, 1)
                    self.assertEqual(
                        info.required_dst_info_num, max(1, decode_tp // prefill_tp)
                    )
                    self.assertEqual(
                        len(info.target_tp_ranks), max(1, prefill_tp // decode_tp)
                    )

    def test_prefill_cp_parallelism_is_supported(self):
        base = dict(
            disaggregation_mode="prefill",
            dp_size=1,
            dcp_size=1,
            enable_dp_attention=False,
            enable_prefill_cp=False,
            enable_prefill_context_parallel=False,
            attn_cp_size=1,
        )
        cases = (
            (
                "prefill_cp",
                dict(enable_prefill_cp=True, enable_dp_attention=True, attn_cp_size=8),
                True,
            ),
            ("prefill_tp", {}, True),
            ("prefill_dp_attention", dict(enable_dp_attention=True), True),
            ("prefill_noncanonical_cp", dict(attn_cp_size=8), False),
            ("decode_tp", dict(disaggregation_mode="decode"), True),
            (
                "decode_cp",
                dict(
                    disaggregation_mode="decode",
                    enable_prefill_cp=True,
                    enable_dp_attention=True,
                    attn_cp_size=8,
                ),
                False,
            ),
            ("decode_dcp", dict(disaggregation_mode="decode", dcp_size=2), False),
        )
        for name, overrides, supported in cases:
            with self.subTest(name=name):
                values = base | overrides
                self.assertEqual(
                    _dsv41_dspark_pd_parallelism_supported(SimpleNamespace(**values)),
                    supported,
                )

    def test_prefill_cp_allows_decoder_bounded_replay(self):
        cfg = SimpleNamespace(
            enable_encoder_swa_bounded_replay=False,
            enable_decoder_swa_bounded_replay=True,
            enable_hisparse=False,
            enable_unified_memory=False,
            enable_two_batch_overlap=False,
            dsv4_attn_backend="triton",
            enable_dp_attention=True,
            enable_prefill_cp=True,
            enable_prefill_context_parallel=False,
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="mooncake",
            speculative_algorithm="DSPARK",
            dp_size=1,
            attn_cp_size=8,
            dcp_size=1,
            pp_size=1,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.DISABLED, max_seq_len=None)
            ),
        )
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41")
        )
        with (
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook.resolving_view",
                return_value=cfg,
            ),
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook.model_config_of",
                return_value=model_config,
            ),
        ):
            validate_deepseek_v41_features(SimpleNamespace())

    def test_bootstrap_accepts_prefill_cp_fan_in(self):
        manager = object.__new__(CommonKVManager)
        manager.prefill_info_table = {}
        manager.kv_args = SimpleNamespace(page_size=256, engine_rank=3)
        manager.kv_cache_dtype_str = "fp8_e4m3"
        manager.dsv41_spec_layout = make_layout()
        manager.attn_tp_size = 8
        manager.attn_cp_size = 1
        manager.attn_cp_rank = 0
        manager.dcp_size = 1
        manager.pp_size = 1
        manager.pp_rank = 0
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.enable_all_cp_ranks_for_transfer = True
        response = Mock(status_code=200)
        response.json.return_value = dict(
            attn_tp_size=1,
            attn_cp_size=8,
            dp_size=1,
            pp_size=1,
            page_size=256,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=True,
            dsv41_spec_layout=manager.dsv41_spec_layout,
        )
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=response,
        ):
            self.assertTrue(manager.try_ensure_parallel_info("prefill:8998"))

        info = manager.prefill_info_table["prefill:8998"]
        self.assertEqual(info.target_tp_rank, 0)
        self.assertEqual(info.target_tp_ranks, [0])
        self.assertEqual(info.target_cp_ranks, list(range(8)))
        self.assertEqual(info.required_dst_info_num, 8)
        self.assertEqual(info.required_prefill_response_num, 8)

    def test_bootstrap_rejects_mismatched_total_attention_width(self):
        manager = object.__new__(CommonKVManager)
        manager.prefill_info_table = {}
        manager.kv_args = SimpleNamespace(page_size=256)
        manager.kv_cache_dtype_str = "fp8_e4m3"
        manager.dsv41_spec_layout = make_layout()
        manager.attn_tp_size = 8
        manager.attn_cp_size = 1
        manager.dcp_size = 1
        manager._resolve_rank_mapping = Mock()
        response = Mock(status_code=200)
        response.json.return_value = dict(
            attn_tp_size=1,
            attn_cp_size=4,
            dp_size=1,
            pp_size=1,
            page_size=256,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=True,
            dsv41_spec_layout=manager.dsv41_spec_layout,
        )
        with (
            patch(
                "sglang.srt.disaggregation.common.conn.requests.get",
                return_value=response,
            ),
            self.assertRaisesRegex(RuntimeError, "attention parallel width"),
        ):
            manager.try_ensure_parallel_info("prefill:8998")
        manager._resolve_rank_mapping.assert_not_called()

    def test_prefill_cp_partitions_every_kv_page_once(self):
        pages = np.arange(11, dtype=np.int32)
        owned = []
        with get_context().override_server_args(enable_dsa_cache_layer_split=False):
            for cp_rank in range(4):
                manager = SimpleNamespace(
                    enable_all_cp_ranks_for_transfer=True,
                    is_dummy_cp_rank=False,
                    attn_cp_rank=cp_rank,
                    attn_cp_size=4,
                )
                sender = SimpleNamespace()
                sender.kv_mgr = manager
                sender.curr_idx = 0
                sender.num_kv_indices = len(pages)
                local_pages, _, is_last, should_skip = (
                    CommonKVSender._prepare_send_indices(sender, pages)
                )
                self.assertTrue(is_last)
                self.assertFalse(should_skip)
                owned.extend(local_pages.tolist())
        self.assertEqual(sorted(owned), pages.tolist())
        self.assertEqual(len(owned), len(set(owned)))

    def test_prefill_cp_uses_torch_indexer_before_sm100(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = object.__new__(DeepseekV4AttnBackend)
        backend.forward_metadata = SimpleNamespace(late_layer_tail=None)
        backend._use_dense_fp4_prefill_indexer = Mock(return_value=True)
        backend._low_ratio_index_topk_dense = Mock()
        backend._low_ratio_index_topk_torch = Mock()
        layer = SimpleNamespace(compressor=None, indexer=object())
        forward_batch = SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(total_seq_lens=8),
            extend_seq_lens_cpu=[8],
            extend_seq_lens=torch.tensor([8], dtype=torch.int32),
            req_pool_indices=torch.tensor([7], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([8], dtype=torch.int32),
            positions=torch.arange(8),
            forward_mode=ForwardMode.EXTEND,
        )
        x = torch.zeros(4, 8)
        q_lora = torch.zeros(4, 4)
        positions = torch.arange(4)

        with (
            patch(
                "sglang.srt.layers.attention.deepseek_v4_backend.get_parallel",
                return_value=SimpleNamespace(attn_cp_rank=0, attn_cp_size=2),
            ),
            patch(
                "sglang.srt.layers.attention.deepseek_v4_backend._is_sm100_or_newer",
                return_value=False,
            ),
        ):
            backend._forward_low_ratio_sources_cp(
                layer=layer,
                x=x,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                run_compressor=False,
                run_indexer=True,
            )

        backend._low_ratio_index_topk_dense.assert_not_called()
        args = backend._low_ratio_index_topk_torch.call_args.args
        self.assertIs(args[0], layer)
        torch.testing.assert_close(args[1], x)
        torch.testing.assert_close(args[2], q_lora)
        torch.testing.assert_close(args[3], torch.full((4,), 7, dtype=torch.int64))
        torch.testing.assert_close(args[4], positions.to(torch.int64))
        self.assertIs(
            backend._low_ratio_index_topk_torch.call_args.kwargs["request_ids"],
            forward_batch.req_pool_indices,
        )
        self.assertEqual(
            backend._low_ratio_index_topk_torch.call_args.kwargs["q_lens_cpu"], [4]
        )

    def test_prefill_cp_torch_indexer_keeps_empty_request_placeholders(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        backend = object.__new__(DeepseekV4AttnBackend)
        page_indices = torch.empty((3, 1), dtype=torch.int32)
        backend.forward_metadata = SimpleNamespace(
            core_metadata=SimpleNamespace(
                sparse_page_indices=Mock(return_value=page_indices),
                sparse_raw_indices=Mock(return_value=None),
            ),
            candidate_metadata=None,
        )
        backend.req_to_token = torch.arange(24).view(8, 3)
        backend.token_to_kv_pool = SimpleNamespace(
            get_low_ratio_index_k_dequant=Mock(
                side_effect=lambda _layer_id, slots: torch.zeros(len(slots), 1)
            )
        )
        indexer = SimpleNamespace(
            index_topk=1,
            is_candidate_source=True,
            uses_candidates=False,
            candidate_topk_blocks=1,
            candidate_block_size=1,
            queries=Mock(return_value=torch.zeros(3, 1, 1)),
            head_weights=Mock(return_value=torch.zeros(3, 1)),
            scores=Mock(
                side_effect=lambda q, k, _w: torch.zeros(q.shape[0], k.shape[0])
            ),
        )
        layer = SimpleNamespace(
            compress_ratio=1,
            indexer=indexer,
            layer_id=0,
            freqs_cis=torch.zeros(3, 1),
        )

        backend._low_ratio_index_topk_torch(
            layer,
            torch.zeros(3, 1),
            torch.zeros(3, 1),
            torch.full((3,), 7, dtype=torch.int64),
            torch.arange(3),
            request_ids=torch.tensor([6, 7]),
            q_lens_cpu=[0, 3],
        )

        masks = backend.forward_metadata.candidate_metadata.request_masks
        self.assertEqual(len(masks), 2)
        self.assertEqual(tuple(masks[0].shape), (0, 0))
        self.assertEqual(tuple(masks[1].shape), (3, 3))

    def test_c2_handoff_keeps_request_ring_indexing(self):
        self.assertEqual(
            request_scoped_state_transfer_indices(
                req_pool_idx=7,
                seq_len=101,
                ratio=100,
                online=True,
                ring_size=100,
            ).tolist(),
            [7],
        )
        self.assertEqual(
            request_scoped_state_transfer_indices(
                req_pool_idx=7,
                seq_len=100,
                ratio=100,
                online=True,
                ring_size=100,
            ).tolist(),
            [],
        )

    def test_multimodal_request_is_rejected_at_runtime_under_cp(self):
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        model = object.__new__(DeepseekV4ForCausalLM)
        torch.nn.Module.__init__(model)
        model.vision = torch.nn.Identity()
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode=lambda: False,
                is_target_verify=lambda: False,
            ),
            mm_inputs=[object()],
        )
        with (
            patch(
                "sglang.srt.models.deepseek_v4.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=2),
            ),
            self.assertRaisesRegex(ValueError, "multimodal requests"),
        ):
            model.forward(
                torch.tensor([1]),
                torch.tensor([0]),
                forward_batch,
            )

    def test_text_request_can_enter_v41_model_with_cp(self):
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        model = object.__new__(DeepseekV4ForCausalLM)
        torch.nn.Module.__init__(model)
        model.vision = torch.nn.Identity()
        model.config = SimpleNamespace(image_token_id=-1)
        model.dsa_enable_prefill_cp = False
        model.pp_group = SimpleNamespace(is_last_rank=False)
        model.model = Mock()
        model.model.forward.return_value = "hidden"
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode=lambda: False,
                is_decode_or_idle=lambda: False,
                is_target_verify=lambda: False,
            ),
            mm_inputs=[None],
        )
        with (
            patch(
                "sglang.srt.models.deepseek_v4.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=2),
            ),
            patch(
                "sglang.srt.models.deepseek_v4.get_attn_tp_context"
            ) as get_attn_tp_context,
        ):
            get_attn_tp_context.return_value.maybe_input_scattered.return_value.__enter__.return_value = None
            get_attn_tp_context.return_value.maybe_input_scattered.return_value.__exit__.return_value = False
            self.assertEqual(
                model.forward(
                    torch.tensor([1]),
                    torch.tensor([0]),
                    forward_batch,
                ),
                "hidden",
            )

    def test_cp_v2_engram_uses_local_input_ids(self):
        from sglang.srt.models.deepseek_v4 import DeepseekV4Model

        local_input_ids = torch.arange(16)
        local_input_ids[3] = 99
        hidden_states = torch.zeros(16, 1, 4)
        engram = Mock(return_value=torch.ones_like(hidden_states))
        engram.layer_hash_index = 0

        def forward_layer(**kwargs):
            return kwargs["hidden_states"], torch.zeros_like(kwargs["hidden_states"])

        model = object.__new__(DeepseekV4Model)
        torch.nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(world_size=1)
        model.config = SimpleNamespace(
            model_type="deepseek_v41",
            vision_n_layers=1,
            image_token_id=99,
        )
        model.start_layer = 0
        model.end_layer = 1
        model.late_layer_start = None
        model.engram_hasher = Mock(return_value=torch.zeros(128, 1, dtype=torch.int64))
        model.engram_embed_prefetch_stream = None
        model.layers = [
            SimpleNamespace(
                engram=engram,
                forward_hc_pre_from_prev=forward_layer,
            )
        ]
        model.dspark_layers_to_capture = None
        forward_batch = SimpleNamespace(
            input_ids=torch.arange(128),
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            attn_cp_metadata=SimpleNamespace(total_seq_lens=128),
        )

        with (
            patch(
                "sglang.srt.models.deepseek_v4.is_cp_active",
                return_value=True,
            ),
            patch(
                "sglang.srt.models.deepseek_v4.cp_shard_hidden_states",
                return_value=local_input_ids,
            ),
            patch(
                "sglang.srt.models.deepseek_v4.get_parallel",
                return_value=SimpleNamespace(attn_cp_rank=2, attn_cp_size=8),
            ),
            patch(
                "sglang.srt.models.deepseek_v4.check_cuda_graph_backend",
                return_value=True,
            ),
        ):
            output, _, _ = model._forward_layers_hc_pre_from_prev(
                positions=torch.arange(16),
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                input_ids=torch.arange(128),
                input_ids_global=torch.arange(128),
                capture_dspark=False,
                dspark_aux_hidden_states=[],
            )

        torch.testing.assert_close(output[3], torch.zeros_like(output[3]))
        torch.testing.assert_close(output[:3], torch.ones_like(output[:3]))
        torch.testing.assert_close(output[4:], torch.ones_like(output[4:]))

    def test_cp_v2_rebuilds_global_input_ids_for_decoder_replay(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import LateLayerTail
        from sglang.srt.models.deepseek_v4 import DeepseekV4Model

        tail_cp_metadata = object()
        tail = LateLayerTail(
            token_indices=torch.tensor([2, 3]),
            positions=torch.tensor([6, 7]),
            extend_seq_lens=torch.tensor([4], dtype=torch.int32),
            extend_seq_lens_cpu=[4],
            swa_out_cache_loc=torch.tensor([4, 5, 6, 7]),
            cp_metadata=tail_cp_metadata,
            global_token_indices=torch.tensor([4, 5, 6, 7]),
        )
        seen_input_ids = []

        def forward_layer(**kwargs):
            seen_input_ids.append(
                (kwargs["input_ids"].clone(), kwargs["input_ids_global"].clone())
            )
            return kwargs["hidden_states"], torch.ones_like(kwargs["hidden_states"])

        model = object.__new__(DeepseekV4Model)
        torch.nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(world_size=1)
        model.config = SimpleNamespace(
            model_type="deepseek_v41",
            vision_n_layers=1,
            image_token_id=99,
        )
        model.start_layer = 0
        model.end_layer = 2
        model.late_layer_start = 1
        model.engram_hasher = None
        model.engram_embed_prefetch_stream = None
        model.layers = [
            SimpleNamespace(engram=None, forward_hc_pre_from_prev=forward_layer),
            SimpleNamespace(engram=None, forward_hc_pre_from_prev=forward_layer),
        ]
        model.dspark_layers_to_capture = None
        full_cp_metadata = object()
        forward_batch = SimpleNamespace(
            input_ids=torch.arange(8),
            forward_mode=SimpleNamespace(
                is_extend=lambda: True,
                is_extend_without_speculative=lambda: True,
            ),
            attn_cp_metadata=full_cp_metadata,
            capture_hidden_mode=object(),
            return_logprob=False,
        )
        backend = SimpleNamespace(
            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail),
        )

        def enter_late_layer_tail(batch):
            batch.attn_cp_metadata = tail_cp_metadata
            return object()

        backend.enter_late_layer_tail = Mock(side_effect=enter_late_layer_tail)
        backend.exit_late_layer_tail = Mock()
        tail_global_input_ids = torch.tensor([4, 6, 5, 7])

        with (
            patch(
                "sglang.srt.models.deepseek_v4.is_cp_active",
                return_value=True,
            ),
            patch(
                "sglang.srt.models.deepseek_v4.cp_shard_hidden_states",
                return_value=torch.tensor([0, 2, 4, 6]),
            ),
            patch(
                "sglang.srt.models.deepseek_v4.cp_interleave_input_ids",
                return_value=tail_global_input_ids,
            ) as reorder,
            patch(
                "sglang.srt.models.deepseek_v4.get_attn_backend",
                return_value=backend,
            ),
            patch(
                "sglang.srt.models.deepseek_v4.get_parallel",
                return_value=SimpleNamespace(attn_dp_size=1),
            ),
            patch(
                "sglang.srt.models.deepseek_v4.check_cuda_graph_backend",
                return_value=True,
            ),
        ):
            model._forward_layers_hc_pre_from_prev(
                positions=torch.arange(4),
                hidden_states=torch.zeros(4, 1, 4),
                forward_batch=forward_batch,
                input_ids=torch.arange(8),
                input_ids_global=torch.arange(8),
                capture_dspark=False,
                dspark_aux_hidden_states=[],
            )

        torch.testing.assert_close(
            reorder.call_args.args[0], torch.tensor([4, 5, 6, 7])
        )
        self.assertIs(reorder.call_args.args[1].attn_cp_metadata, tail_cp_metadata)
        torch.testing.assert_close(seen_input_ids[1][0], tail_global_input_ids)
        torch.testing.assert_close(seen_input_ids[1][1], tail_global_input_ids)

    def test_cp_v2_preserves_decoder_replay_hidden_indices(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import LateLayerTail
        from sglang.srt.model_executor.runner.eager_runner import EagerRunner
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        token_indices = torch.tensor([2, 3])
        tail_cp_metadata = object()
        tail = LateLayerTail(
            token_indices=torch.tensor([0, 1]),
            positions=torch.tensor([2, 3]),
            extend_seq_lens=torch.tensor([2], dtype=torch.int32),
            extend_seq_lens_cpu=[2],
            swa_out_cache_loc=torch.tensor([2, 3]),
            cp_metadata=tail_cp_metadata,
            global_token_indices=token_indices,
        )
        output = SimpleNamespace(hidden_states_token_indices=None)
        model = object.__new__(DeepseekV4ForCausalLM)
        torch.nn.Module.__init__(model)
        model.capture_aux_hidden_states = True
        model.pp_group = SimpleNamespace(is_last_rank=True)
        model.model = Mock()
        model.model.late_layer_start = 21
        model.model.return_value = (
            (torch.ones(2, 4), torch.ones(2, 4)),
            [torch.ones(2, 4)],
        )
        model.lm_head = None
        model.logits_processor = Mock(return_value=output)
        runner = object.__new__(EagerRunner)
        runner.model_runner = SimpleNamespace(
            model=model,
            attn_backend=SimpleNamespace(
                tail_forward_metadata=SimpleNamespace(late_layer_tail=tail)
            ),
        )
        full_cp_metadata = object()
        forward_batch = SimpleNamespace(
            input_ids=torch.tensor([1, 2, 3, 4]),
            positions=torch.tensor([0, 1]),
            attn_cp_metadata=full_cp_metadata,
        )
        gather_metadata = []

        def gather(value, batch, *_):
            gather_metadata.append(batch.attn_cp_metadata)
            return value

        with (
            patch(
                "sglang.srt.model_executor.runner.eager_runner.cp_shard_model_inputs",
                return_value=nullcontext(
                    (
                        torch.ones(2, 4),
                        forward_batch.positions,
                        forward_batch.input_ids,
                    )
                ),
            ),
            patch(
                "sglang.srt.model_executor.runner.eager_runner.cp_gather_after_forward",
                side_effect=gather,
            ),
            patch(
                "sglang.srt.layers.logits_processor.LogitsMetadata.from_forward_batch",
                return_value=SimpleNamespace(),
            ),
            patch("torch.cuda.current_stream", return_value=Mock()),
        ):
            result = EagerRunner._execute_extend_cp(
                runner,
                forward_batch,
                {"input_embeds": torch.ones(2, 4)},
            )
        self.assertIs(result, output)
        self.assertIs(result.hidden_states_token_indices, token_indices)
        self.assertTrue(all(x is tail_cp_metadata for x in gather_metadata))
        self.assertIs(forward_batch.attn_cp_metadata, full_cp_metadata)
        self.assertTrue(
            torch.equal(model.logits_processor.call_args.args[0], torch.tensor([3, 4]))
        )


if __name__ == "__main__":
    unittest.main()
