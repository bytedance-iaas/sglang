"""Derived-property tests for the PD whole-envelope transfer addressing.

PD disaggregation transfers the unified memory pool as whole envelopes with
``addr = raw_ptr + physical_index * item_len`` (see
``UnifiedMLATokenToKVPool.get_contiguous_buf_infos`` /
``UnifiedMambaPool.get_contiguous_buf_infos`` and mooncake's
``_send_kvcache_generic`` / ``_send_mamba_state``). That contract only holds if
the page-major view builders keep (a) one page's data for ALL layers inside one
contiguous ``page_envelope_bytes`` block, and (b) one mamba slot's conv+temporal
state for all layers inside one contiguous ``entry_bytes`` block. A
"looks equivalent" reordering of the view layout (e.g. layer-major across
pages) would silently corrupt every PD transfer while all kernels keep working,
because kernels read through the strided views, not through raw offsets.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from sglang.srt.mem_cache.layout.page_major import (
    build_mla_views,
    build_page_major_mamba_views,
    mamba_entry_bytes,
    mla_entry_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestMLAEnvelopeTransferAddressing(CustomTestCase):
    def test_page_envelope_matches_per_layer_views(self):
        """Every (page, layer, slot) row written through the MLA views
        must land at raw_ptr + page * page_envelope_bytes + layer-block offset,
        i.e. inside the page's transfer envelope."""
        layer_num, page_size, kv_dim, num_pages = 3, 4, 8, 6
        store_dtype = torch.bfloat16
        row_bytes = kv_dim * store_dtype.itemsize
        page_bytes = page_size * layer_num * row_bytes
        self.assertEqual(
            page_bytes,
            page_size
            * mla_entry_bytes(
                layer_num=layer_num,
                kv_cache_dim=kv_dim,
                itemsize=store_dtype.itemsize,
            ),
        )
        # +1 page envelope of tail pad, as UnifiedKVPool allocates for MLA.
        raw = torch.zeros((num_pages + 1) * page_bytes, dtype=torch.uint8)
        views = build_mla_views(
            raw,
            layer_num=layer_num,
            kv_cache_dim=kv_dim,
            store_dtype=store_dtype,
            page_size=page_size,
            num_pages=num_pages,
            anchor_bytes=0,
        )
        torch.manual_seed(0)
        for page in range(num_pages):
            for layer in range(layer_num):
                for off in range(page_size):
                    kernel_id = page * layer_num * page_size + off
                    val = torch.randn(kv_dim, dtype=store_dtype)
                    views[layer][kernel_id, 0] = val
                    start = (
                        page * page_bytes
                        + layer * page_size * row_bytes
                        + off * row_bytes
                    )
                    got = raw[start : start + row_bytes].view(store_dtype)
                    self.assertTrue(torch.equal(got, val), (page, layer, off))


class TestMambaEnvelopeTransferAddressing(CustomTestCase):
    def test_slot_envelope_is_self_contained(self):
        """A slot's conv+temporal state for all layers must live exactly in
        raw[slot * entry_bytes : (slot+1) * entry_bytes]: no byte outside the
        envelope may change, and the payload byte count must fill it."""
        layer_num, max_slots = 2, 5
        conv_shapes = ((3, 4), (2, 6))
        temporal_shape = (2, 3, 4)
        conv_dtype = torch.bfloat16
        temporal_dtype = torch.float32
        entry = mamba_entry_bytes(
            layer_num=layer_num,
            conv_state_shapes=conv_shapes,
            conv_dtype=conv_dtype,
            temporal_state_shape=temporal_shape,
            temporal_dtype=temporal_dtype,
        )
        raw = torch.zeros(max_slots * entry, dtype=torch.uint8)
        conv_views, temporal_view = build_page_major_mamba_views(
            raw,
            layer_num=layer_num,
            conv_state_shapes=conv_shapes,
            conv_dtype=conv_dtype,
            temporal_state_shape=temporal_shape,
            temporal_dtype=temporal_dtype,
            max_slots=max_slots,
            anchor_bytes=0,
        )
        torch.manual_seed(0)
        for slot in range(max_slots):
            raw.zero_()
            n_payload = 0
            for i, conv_view in enumerate(conv_views):
                val = torch.randn((layer_num,) + conv_shapes[i], dtype=conv_dtype)
                conv_view[:, slot] = val
                n_payload += val.numel() * val.element_size()
            val = torch.randn((layer_num,) + temporal_shape, dtype=temporal_dtype)
            temporal_view[:, slot] = val
            n_payload += val.numel() * val.element_size()

            outside = torch.cat([raw[: slot * entry], raw[(slot + 1) * entry :]])
            self.assertTrue(
                bool(outside.eq(0).all()),
                f"slot {slot} state bled outside its transfer envelope",
            )
            self.assertEqual(n_payload, entry)


class TestMLAPeerLayout(CustomTestCase):
    def _manager(self, page_bytes, *, hybrid=False, unified=False):
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args(enable_unified_memory=unified)
        override.install()
        self.addCleanup(override.restore)
        manager = object.__new__(MooncakeKVManager)
        manager.is_mla_backend = not hybrid
        manager.is_hybrid_mla_backend = hybrid
        manager.attn_tp_size = 4
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[1000],
            kv_item_lens=[page_bytes],
            kv_layer_ids=[3],
            mla_compression_ratios=None,
        )
        manager._send_kvcache_generic = MagicMock(return_value=0)
        return manager

    def _send(self, manager, dst_page_bytes, dst_tp=1):
        return manager.send_kvcache(
            "session",
            np.array([1], dtype=np.int32),
            [10000],
            np.array([7], dtype=np.int32),
            MagicMock(),
            dst_layer_ids=[3],
            dst_kv_item_len=dst_page_bytes,
            dst_attn_tp_size=dst_tp,
        )

    def test_raw_and_group_layouts_match_across_attention_tp(self):
        for dimension in (512, 528):
            for hybrid in (False, True):
                with self.subTest(dimension=dimension, hybrid=hybrid):
                    manager = self._manager(dimension * 64, hybrid=hybrid)
                    self.assertEqual(self._send(manager, dimension * 64), 0)
                    manager._send_kvcache_generic.assert_called_once()

    def test_mismatch_or_missing_layout_rejected_before_transfer(self):
        for src, dst in ((512, 528), (528, 512), (528, None)):
            for hybrid in (False, True):
                with self.subTest(src=src, dst=dst, hybrid=hybrid):
                    manager = self._manager(src * 64, hybrid=hybrid)
                    with self.assertRaisesRegex(
                        RuntimeError, "PD MLA KV layout mismatch"
                    ):
                        self._send(manager, dst * 64 if dst is not None else None)
                    manager._send_kvcache_generic.assert_not_called()

    def test_unified_envelope_rejects_shape_and_tp_mismatches(self):
        for dst_bytes, dst_tp in ((128, 4), (256, 1)):
            with self.subTest(dst_bytes=dst_bytes, dst_tp=dst_tp):
                manager = self._manager(256, unified=True)
                with self.assertRaises(RuntimeError):
                    self._send(manager, dst_bytes, dst_tp)
                manager._send_kvcache_generic.assert_not_called()

    def test_mha_does_not_require_mla_page_layout(self):
        manager = self._manager(256)
        manager.is_mla_backend = False
        self.assertEqual(self._send(manager, 128), 0)


if __name__ == "__main__":
    unittest.main()
