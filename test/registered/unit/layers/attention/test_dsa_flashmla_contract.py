import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    _validate_flashmla_kv_decode_shapes,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFlashMLAKVDecodeShapeContract(unittest.TestCase):
    def _valid_inputs(self, batch_size=4, seq_len_q=1, topk=8):
        return {
            "q": torch.empty((batch_size, seq_len_q, 4, 16)),
            "indices": torch.full((batch_size, seq_len_q, topk), -1, dtype=torch.int32),
            "cache_seqlens": torch.zeros(batch_size, dtype=torch.int32),
            "num_splits": torch.zeros(batch_size + 1, dtype=torch.int32),
            "expected_topk": topk,
        }

    def test_accepts_physical_padding_and_all_minus_one_rows(self):
        inputs = self._valid_inputs()
        inputs["indices"][0].fill_(-1)
        _validate_flashmla_kv_decode_shapes(**inputs)

    def test_rejects_query_axis_mismatches(self):
        cases = {
            "indices_batch": {"indices": torch.empty((3, 1, 8))},
            "indices_query": {"indices": torch.empty((4, 2, 8))},
            "cache_seqlens": {"cache_seqlens": torch.empty(3, dtype=torch.int32)},
            "num_splits": {"num_splits": torch.empty(4, dtype=torch.int32)},
        }
        for name, replacement in cases.items():
            with self.subTest(name=name):
                inputs = self._valid_inputs()
                inputs.update(replacement)
                with self.assertRaisesRegex(ValueError, "query-axis mismatch"):
                    _validate_flashmla_kv_decode_shapes(**inputs)

    def test_rejects_rank_and_topk_mismatches(self):
        cases = {
            "q_rank": ({"q": torch.empty((4, 4, 16))}, "q must have rank 4"),
            "indices_rank": (
                {"indices": torch.empty((4, 8))},
                "indices must have rank 3",
            ),
            "cache_rank": (
                {"cache_seqlens": torch.empty((4, 1), dtype=torch.int32)},
                "cache_seqlens must have rank 1",
            ),
            "num_splits_rank": (
                {"num_splits": torch.empty((5, 1), dtype=torch.int32)},
                "num_splits must have rank 1",
            ),
            "topk": ({"indices": torch.empty((4, 1, 7))}, "top-k mismatch"),
        }
        for name, (replacement, message) in cases.items():
            with self.subTest(name=name):
                inputs = self._valid_inputs()
                inputs.update(replacement)
                with self.assertRaisesRegex(ValueError, message):
                    _validate_flashmla_kv_decode_shapes(**inputs)

    def test_caller_rejects_bad_metadata_before_native_launch(self):
        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.flashmla_kv_num_q_heads = 4
        backend.real_page_size = 64
        backend.kv_cache_dim = 16
        backend.dsa_kv_cache_store_fp8 = True
        backend.dsa_index_topk = 8
        layer = SimpleNamespace(tp_q_head_num=4, head_dim=16, layer_id=7)
        metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=torch.zeros(2, dtype=torch.int32),
            flashmla_metadata=SimpleNamespace(
                flashmla_metadata=torch.empty((1, 1), dtype=torch.int32),
                num_splits=torch.zeros(2, dtype=torch.int32),
            ),
        )

        with (
            patch("sgl_kernel.flash_mla.flash_mla_with_kvcache") as native_flashmla,
            self.assertRaisesRegex(ValueError, "query-axis mismatch"),
        ):
            backend._forward_flashmla_kv(
                q_all=torch.empty((2, 4, 16)),
                kv_cache=torch.empty((64, 16)),
                v_head_dim=16,
                sm_scale=1.0,
                layer=layer,
                metadata=metadata,
                page_table_1=torch.full((2, 8), -1, dtype=torch.int32),
            )
        native_flashmla.assert_not_called()

    def test_caller_launches_for_legal_physical_padding(self):
        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.flashmla_kv_num_q_heads = 4
        backend.real_page_size = 64
        backend.kv_cache_dim = 16
        backend.dsa_kv_cache_store_fp8 = True
        backend.dsa_index_topk = 8
        layer = SimpleNamespace(tp_q_head_num=4, head_dim=16, layer_id=7)
        metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=torch.tensor([17, 0], dtype=torch.int32),
            flashmla_metadata=SimpleNamespace(
                flashmla_metadata=torch.empty((1, 1), dtype=torch.int32),
                num_splits=torch.zeros(3, dtype=torch.int32),
            ),
        )
        native_output = torch.empty((2, 1, 4, 16))

        with (
            patch(
                "sglang.srt.environ.envs.SGLANG_ENABLE_ASYNC_ASSERT.get",
                return_value=False,
            ),
            patch.object(torch, "_assert_async") as async_assert,
            patch(
                "sgl_kernel.flash_mla.flash_mla_with_kvcache",
                return_value=(native_output, torch.empty((2, 4, 1))),
            ) as native_flashmla,
        ):
            output = backend._forward_flashmla_kv(
                q_all=torch.empty((2, 4, 16)),
                kv_cache=torch.empty((64, 16)),
                v_head_dim=16,
                sm_scale=1.0,
                layer=layer,
                metadata=metadata,
                page_table_1=torch.full((2, 8), -1, dtype=torch.int32),
            )

        self.assertIs(output, native_output)
        native_flashmla.assert_called_once()
        call = native_flashmla.call_args.kwargs
        self.assertEqual(call["q"].shape[:2], (2, 1))
        self.assertEqual(call["indices"].shape, (2, 1, 8))
        self.assertEqual(call["cache_seqlens"].shape, (2,))
        self.assertEqual(call["num_splits"].shape, (3,))
        async_assert.assert_not_called()

    def test_enabled_selected_index_probe_precedes_native_launch(self):
        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.flashmla_kv_num_q_heads = 4
        backend.real_page_size = 64
        backend.kv_cache_dim = 16
        backend.dsa_kv_cache_store_fp8 = True
        backend.dsa_index_topk = 8
        layer = SimpleNamespace(tp_q_head_num=4, head_dim=16, layer_id=7)
        metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=torch.tensor([17, 0], dtype=torch.int32),
            flashmla_metadata=SimpleNamespace(
                flashmla_metadata=torch.empty((1, 1), dtype=torch.int32),
                num_splits=torch.zeros(3, dtype=torch.int32),
            ),
        )
        capacity = 128

        for name, selected, expected_conditions in (
            ("legal", [-1, 0, capacity - 1], [True, True]),
            ("below_sentinel", [-2, 0], [False, True]),
            ("at_capacity", [-1, capacity], [True, False]),
        ):
            with self.subTest(name=name):
                page_table = torch.full((2, 8), -1, dtype=torch.int32)
                page_table[0, : len(selected)] = torch.tensor(
                    selected, dtype=torch.int32
                )
                events = []

                def record_assert(condition, message):
                    events.append(("assert", bool(condition), message))

                def record_native(**kwargs):
                    events.append(("native",))
                    return torch.empty((2, 1, 4, 16)), torch.empty((2, 4, 1))

                with (
                    patch(
                        "sglang.srt.environ.envs.SGLANG_ENABLE_ASYNC_ASSERT.get",
                        return_value=True,
                    ),
                    patch.object(
                        torch, "_assert_async", side_effect=record_assert
                    ) as async_assert,
                    patch(
                        "sgl_kernel.flash_mla.flash_mla_with_kvcache",
                        side_effect=record_native,
                    ) as native_flashmla,
                ):
                    backend._forward_flashmla_kv(
                        q_all=torch.empty((2, 4, 16)),
                        kv_cache=torch.empty((capacity, 16)),
                        v_head_dim=16,
                        sm_scale=1.0,
                        layer=layer,
                        metadata=metadata,
                        page_table_1=page_table,
                    )

                self.assertEqual(
                    [event[1] for event in events[:2]], expected_conditions
                )
                self.assertEqual(events[2], ("native",))
                self.assertEqual(async_assert.call_count, 2)
                native_flashmla.assert_called_once()
                self.assertEqual(
                    [item.args[1] for item in async_assert.call_args_list],
                    [
                        "index < -1 (negative / unmasked sentinel?): "
                        "FlashMLA sparse decode selected physical KV slot layer=7",
                        f"index >= {capacity} (out of range): "
                        "FlashMLA sparse decode selected physical KV slot layer=7",
                    ],
                )


if __name__ == "__main__":
    unittest.main()
