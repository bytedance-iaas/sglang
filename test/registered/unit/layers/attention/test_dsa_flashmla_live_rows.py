import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSAFlashMLALiveRows(CustomTestCase):
    def test_eager_scheduler_uses_live_rows_before_dp_padding(self):
        """DP/CP padding must not enlarge FlashMLA's scheduler axis."""
        flashmla_metadata = object()
        backend = SimpleNamespace(
            speculative_num_draft_tokens=4,
            use_mha=False,
            dsa_decode_impl="flashmla_kv",
            dsa_index_topk=16,
            real_page_size=1,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(64, dtype=torch.int32).view(4, 16)
            ),
            set_dsa_prefill_impl=MagicMock(),
            get_topk_transform_method=MagicMock(),
            _draft_decode_seq_len_offset=MagicMock(return_value=1),
            _cal_indexer_k_start_end=MagicMock(return_value=(None, None)),
            get_device_int32_arange=lambda size: torch.arange(
                size, dtype=torch.int32
            ),
            _compute_flashmla_metadata=MagicMock(return_value=flashmla_metadata),
            _transform_table_1_to_real=MagicMock(
                return_value=torch.zeros((1, 16), dtype=torch.int32)
            ),
            _build_topk_v2_plan=MagicMock(return_value=None),
        )
        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens=torch.tensor([7]),
            seq_lens_cpu=torch.tensor([7]),
            req_pool_indices=torch.tensor([0]),
            spec_info=object(),
            forward_mode=ForwardMode.DECODE,
            seq_lens_sum=7,
        )
        physical_dsa_lengths = torch.tensor([8, 0, 0, 0], dtype=torch.int32)

        with (
            patch(
                "sglang.srt.layers.attention.dsa_backend.pad_dsa_cache_seqlens",
                return_value=physical_dsa_lengths,
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.is_cuda",
                return_value=False,
            ),
        ):
            DeepseekSparseAttnBackend.init_forward_metadata(backend, forward_batch)

        flashmla_lengths = backend._compute_flashmla_metadata.call_args.kwargs[
            "cache_seqlens"
        ]
        # The speculative offset is handled by the backend's length transform;
        # the regression contract here is the single live row, not its value.
        self.assertEqual(flashmla_lengths.tolist(), [7])
        self.assertIs(backend.forward_metadata.flashmla_metadata, flashmla_metadata)
        self.assertTrue(
            torch.equal(
                backend.forward_metadata.dsa_cache_seqlens_int32,
                physical_dsa_lengths,
            )
        )

    def test_metadata_owns_logical_extend_lengths(self):
        """Later eager padding must not widen the saved logical row count."""
        backend = SimpleNamespace(
            speculative_num_draft_tokens=4,
            use_mha=False,
            dsa_decode_impl="fa3",
            dsa_index_topk=16,
            real_page_size=1,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(64, dtype=torch.int32).view(4, 16)
            ),
            set_dsa_prefill_impl=MagicMock(),
            get_topk_transform_method=MagicMock(),
            _draft_decode_seq_len_offset=MagicMock(return_value=0),
            _cal_indexer_k_start_end=MagicMock(return_value=(None, None)),
            get_device_int32_arange=lambda size: torch.arange(
                size, dtype=torch.int32
            ),
            _transform_table_1_to_real=lambda table: table,
            _build_topk_v2_plan=MagicMock(return_value=None),
        )
        logical_lengths = [4]
        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens=torch.tensor([13]),
            seq_lens_cpu=torch.tensor([13]),
            req_pool_indices=torch.tensor([0]),
            spec_info=object(),
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            seq_lens_sum=13,
            extend_prefix_lens_cpu=[9],
            extend_prefix_lens=torch.tensor([9]),
            extend_seq_lens_cpu=logical_lengths,
            extend_seq_lens=torch.tensor([4], dtype=torch.int32),
            extend_num_tokens=4,
        )

        with (
            patch(
                "sglang.srt.layers.attention.dsa_backend.is_cuda",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.seqlens_expand_triton",
                return_value=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.pad_dsa_cache_seqlens",
                side_effect=lambda _batch, lengths: lengths,
            ),
        ):
            DeepseekSparseAttnBackend.init_forward_metadata(backend, forward_batch)

        logical_lengths.append(4)
        self.assertEqual(backend.forward_metadata.dsa_extend_seq_lens_list, [4])

    def test_decode_trims_physical_lengths_to_live_rows(self):
        """The kernel must receive one length and index row per live query."""
        captured = {}
        flashmla = ModuleType("sgl_kernel.flash_mla")

        def fake_flash_mla_with_kvcache(**kwargs):
            captured.update(kwargs)
            return torch.ones((1, 1, 2, 2)), None

        flashmla.flash_mla_with_kvcache = fake_flash_mla_with_kvcache
        sgl_kernel = ModuleType("sgl_kernel")
        sgl_kernel.flash_mla = flashmla
        backend = SimpleNamespace(
            flashmla_kv_num_q_heads=2,
            real_page_size=64,
            kv_cache_dim=3,
            dsa_kv_cache_store_fp8=True,
            dsa_index_topk=2,
        )
        metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=torch.tensor([8, 0, 0, 0], dtype=torch.int32),
            flashmla_metadata=SimpleNamespace(
                flashmla_metadata=torch.empty((1,), dtype=torch.int32),
                num_splits=torch.empty((2,), dtype=torch.int32),
            ),
        )

        with patch.dict(
            "sys.modules",
            {
                "sgl_kernel": sgl_kernel,
                "sgl_kernel.flash_mla": flashmla,
            },
        ):
            output = DeepseekSparseAttnBackend._forward_flashmla_kv(
                backend,
                q_all=torch.empty((1, 2, 3)),
                kv_cache=torch.empty((64, 3)),
                v_head_dim=2,
                sm_scale=1.0,
                layer=SimpleNamespace(tp_q_head_num=2, head_dim=3),
                metadata=metadata,
                page_table_1=torch.zeros((1, 2), dtype=torch.int32),
            )

        self.assertEqual(captured["cache_seqlens"].tolist(), [8])
        self.assertEqual(captured["num_splits"].shape[0], 2)
        self.assertEqual(captured["indices"].shape[0], 1)
        self.assertEqual(output.shape, (1, 1, 2, 2))

    def test_draft_extend_trims_eager_padding_and_restores_output(self):
        """MTP draft attention uses logical rows and restores MLP padding."""
        captured = {}
        metadata = SimpleNamespace(
            dsa_extend_seq_lens_list=[4],
            cu_seqlens_q=torch.arange(5, dtype=torch.int32),
            page_table_1=torch.zeros((1, 16), dtype=torch.int32),
            flashmla_metadata=SimpleNamespace(
                flashmla_metadata=torch.empty((1,), dtype=torch.int32),
                num_splits=torch.empty((5,), dtype=torch.int32),
            ),
        )

        def fake_flashmla(**kwargs):
            captured.update(
                q_rows=kwargs["q_all"].shape[0],
                page_table_rows=kwargs["page_table_1"].shape[0],
            )
            return torch.ones((4, 1, 2, 2))

        backend = SimpleNamespace(
            forward_metadata=metadata,
            dsa_decode_impl="flashmla_kv",
            dsa_prefill_impl="fa3",
            use_mha=False,
            use_fused_topk=False,
            hisparse_coordinator=None,
            token_to_kv_pool=SimpleNamespace(
                get_key_buffer=lambda _layer_id: torch.empty((64, 3))
            ),
            get_topk_transform_method=MagicMock(
                return_value=TopkTransformMethod.PAGED
            ),
            _forward_flashmla_kv=fake_flashmla,
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
        )
        layer = SimpleNamespace(
            is_cross_attention=False,
            layer_id=0,
            tp_q_head_num=2,
            v_head_dim=2,
            head_dim=3,
            scaling=1.0,
        )

        with patch(
            "sglang.srt.layers.attention.dsa_backend.transform_index_page_table_prefill",
            return_value=torch.zeros((4, 2), dtype=torch.int32),
        ):
            output = DeepseekSparseAttnBackend.forward_extend(
                backend,
                q=torch.empty((8, 4)),
                k=None,
                v=None,
                layer=layer,
                forward_batch=forward_batch,
                q_rope=torch.empty((8, 2)),
                topk_indices=torch.zeros((8, 16), dtype=torch.int32),
            )

        self.assertEqual(captured, {"q_rows": 4, "page_table_rows": 4})
        self.assertEqual(output.shape, (8, 1, 2, 2))
        self.assertTrue(torch.equal(output[:4], torch.ones_like(output[:4])))
        self.assertTrue(torch.equal(output[4:], torch.zeros_like(output[4:])))

    def test_decode_rejects_each_misaligned_row_axis(self):
        """Reject stale metadata before it reaches the FlashMLA kernel."""
        flashmla = ModuleType("sgl_kernel.flash_mla")
        flashmla.flash_mla_with_kvcache = MagicMock()
        sgl_kernel = ModuleType("sgl_kernel")
        sgl_kernel.flash_mla = flashmla

        cases = (
            (
                torch.tensor([8]),
                torch.empty((3,), dtype=torch.int32),
                2,
                "length_rows=1",
            ),
            (
                torch.tensor([8, 9]),
                torch.empty((4,), dtype=torch.int32),
                2,
                "num_splits=4",
            ),
            (
                torch.tensor([8, 9]),
                torch.empty((3,), dtype=torch.int32),
                1,
                "index_rows=1",
            ),
        )
        with patch.dict(
            "sys.modules",
            {
                "sgl_kernel": sgl_kernel,
                "sgl_kernel.flash_mla": flashmla,
            },
        ):
            for cache_seqlens, num_splits, index_rows, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(
                    RuntimeError, message
                ):
                    DeepseekSparseAttnBackend._forward_flashmla_kv(
                        SimpleNamespace(),
                        q_all=torch.empty((2, 2, 3)),
                        kv_cache=torch.empty((64, 3)),
                        v_head_dim=2,
                        sm_scale=1.0,
                        layer=SimpleNamespace(tp_q_head_num=2, head_dim=3),
                        metadata=SimpleNamespace(
                            dsa_cache_seqlens_int32=cache_seqlens,
                            flashmla_metadata=SimpleNamespace(
                                flashmla_metadata=torch.empty(
                                    (1,), dtype=torch.int32
                                ),
                                num_splits=num_splits,
                            ),
                        ),
                        page_table_1=torch.zeros(
                            (index_rows, 2), dtype=torch.int32
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
