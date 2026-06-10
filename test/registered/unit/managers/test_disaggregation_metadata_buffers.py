"""Stage 1 protocol-extension regression tests.

These tests guard the layout-only extension that reserves prefix-aware slots
in MetadataBuffers (``prefix_aligned_len``, ``prefix_last_topk_p``,
``prefix_last_topk_index``, ``prefix_last_hidden_states``).

Stage 1 explicitly does NOT consume the new fields anywhere; the only contract
is that:

  * the fields exist with the expected shapes / dtypes,
  * they default to zero,
  * ``get_buf_infos`` / ``get_buf`` agree on the field count and ordering, and
  * the bootstrap_room slot stays the last entry (so existing decode-side
    unpacking that relies on its position keeps working).

If any of these break, downstream stages that wire prefill -> decode prefix
anchors will silently misalign, which is exactly the failure mode this test
exists to catch.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.disaggregation.utils import MetadataBuffers


class TestMetadataBuffersStage1Layout(unittest.TestCase):
    HIDDEN_SIZE = 32
    SIZE = 4

    def setUp(self) -> None:
        self.buf = MetadataBuffers(
            size=self.SIZE,
            hidden_size=self.HIDDEN_SIZE,
            hidden_states_dtype=torch.float32,
        )

    def test_prefix_fields_exist_with_expected_shape_and_dtype(self) -> None:
        self.assertEqual(
            self.buf.prefix_aligned_len.shape, torch.Size([self.SIZE, 16])
        )
        self.assertEqual(self.buf.prefix_aligned_len.dtype, torch.int32)

        self.assertEqual(
            self.buf.prefix_last_topk_p.shape, torch.Size([self.SIZE, 16])
        )
        self.assertEqual(self.buf.prefix_last_topk_p.dtype, torch.float32)

        self.assertEqual(
            self.buf.prefix_last_topk_index.shape, torch.Size([self.SIZE, 16])
        )
        self.assertEqual(self.buf.prefix_last_topk_index.dtype, torch.int64)

        self.assertEqual(
            self.buf.prefix_last_hidden_states.shape,
            torch.Size([self.SIZE, self.HIDDEN_SIZE]),
        )
        self.assertEqual(
            self.buf.prefix_last_hidden_states.dtype, torch.float32
        )

    def test_prefix_fields_default_to_zero(self) -> None:
        # Stage 1 must not change runtime behavior, which means the new slots
        # have to be observably zero for every existing consumer.
        self.assertTrue(torch.all(self.buf.prefix_aligned_len == 0))
        self.assertTrue(torch.all(self.buf.prefix_last_topk_p == 0))
        self.assertTrue(torch.all(self.buf.prefix_last_topk_index == 0))
        self.assertTrue(torch.all(self.buf.prefix_last_hidden_states == 0))

    def test_get_buf_infos_lengths_match_and_include_prefix_slots(self) -> None:
        ptrs, data_lens, item_lens = self.buf.get_buf_infos()
        self.assertEqual(len(ptrs), len(data_lens))
        self.assertEqual(len(ptrs), len(item_lens))
        # 9 legacy aux entries + 4 prefix-aware entries + bootstrap_room = 14.
        self.assertEqual(len(ptrs), 14)

        # The bootstrap_room slot must remain the last entry; decode-side
        # corruption checks (and any backend code that accesses the trailing
        # entry) rely on this position.
        self.assertEqual(ptrs[-1], self.buf.bootstrap_room.data_ptr())

    def test_get_buf_tuple_shape_matches_get_buf_infos(self) -> None:
        infos_len = len(self.buf.get_buf_infos()[0])
        tuple_len = len(self.buf.get_buf(0))
        self.assertEqual(infos_len, tuple_len)

        # Spot-check the prefix-aware slots in the unpacked tuple. Indices are
        # locked to the layout in MetadataBuffers; if someone reorders slots
        # this test breaks loudly instead of producing silently swapped data.
        unpacked = self.buf.get_buf(0)
        prefix_aligned_len = unpacked[9]
        prefix_last_topk_p = unpacked[10]
        prefix_last_topk_index = unpacked[11]
        prefix_last_hidden_states = unpacked[12]
        bootstrap_room = unpacked[13]

        self.assertEqual(prefix_aligned_len.shape, torch.Size([16]))
        self.assertEqual(prefix_aligned_len.dtype, torch.int32)
        self.assertEqual(prefix_last_topk_p.shape, torch.Size([16]))
        self.assertEqual(prefix_last_topk_p.dtype, torch.float32)
        self.assertEqual(prefix_last_topk_index.shape, torch.Size([16]))
        self.assertEqual(prefix_last_topk_index.dtype, torch.int64)
        self.assertEqual(
            prefix_last_hidden_states.shape, torch.Size([self.HIDDEN_SIZE])
        )
        self.assertEqual(bootstrap_room.shape, torch.Size([8]))


if __name__ == "__main__":
    unittest.main()
