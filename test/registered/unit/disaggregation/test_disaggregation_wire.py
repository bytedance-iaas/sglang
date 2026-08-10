import unittest

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.utils import (
    pack_int_lists,
    pack_list_of_buffers,
    unpack_int_lists,
    unpack_list_of_buffers,
)
from sglang.srt.disaggregation.utils import (
    get_dsv4_full_indexed_c128_state_indices,
    pack_state_types,
    resolve_state_component_dst_index,
    unpack_state_types,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDisaggregationWire(unittest.TestCase):
    def test_int_lists_roundtrip(self):
        cases = [
            ("Q", [[1, 2, 3], [4]]),
            ("I", [[10, 20], [30, 40, 50]]),
            ("i", [[-1, 2], [3, -4, 5]]),
        ]
        for fmt, sample in cases:
            packed = pack_int_lists(sample, fmt)
            self.assertEqual(unpack_int_lists(packed, fmt), sample, msg=fmt)

    def test_pack_accepts_ndarray(self):
        arrs = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5], dtype=np.int32),
        ]
        packed = pack_int_lists(arrs, "i")
        self.assertEqual(unpack_int_lists(packed, "i"), [[1, 2, 3], [4, 5]])

    def test_empty_outer_list(self):
        self.assertEqual(pack_int_lists([], "Q"), b"")
        self.assertEqual(unpack_int_lists(b"", "Q"), [])

    def test_empty_inner_list(self):
        packed = pack_int_lists([[]], "I")
        self.assertEqual(unpack_int_lists(packed, "I"), [[]])

    def test_list_of_buffers_roundtrip(self):
        bufs = [b"abc", b"", b"de", b"x" * 17]
        self.assertEqual(unpack_list_of_buffers(pack_list_of_buffers(bufs)), bufs)

    def test_state_component_matching_uses_type_occurrence(self):
        src_state_types = [StateType.SWA, StateType.C128_STATE, StateType.SWA]
        dst_state_types = [StateType.SWA, StateType.C128_STATE, StateType.SWA]

        self.assertEqual(
            resolve_state_component_dst_index(src_state_types, dst_state_types, 0),
            0,
        )
        self.assertEqual(
            resolve_state_component_dst_index(src_state_types, dst_state_types, 2),
            2,
        )

    def test_state_types_roundtrip(self):
        state_types = [StateType.SWA, StateType.C128_STATE, StateType.SWA]
        self.assertEqual(unpack_state_types(pack_state_types(state_types)), state_types)

    def test_full_indexed_c128_state_uses_current_chunk_page(self):
        req_to_token = torch.zeros((2, 512), dtype=torch.int32)
        req_to_token[1, 128] = 640

        np.testing.assert_array_equal(
            get_dsv4_full_indexed_c128_state_indices(req_to_token, 1, 129),
            np.array([5], dtype=np.int32),
        )

    def test_full_indexed_c128_state_skips_closed_chunk(self):
        req_to_token = torch.zeros((1, 256), dtype=torch.int32)
        self.assertEqual(
            get_dsv4_full_indexed_c128_state_indices(req_to_token, 0, 256).size,
            0,
        )

    def test_full_indexed_c128_state_rejects_padding_slot_zero(self):
        req_to_token = torch.zeros((1, 256), dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "unallocated Full KV slot"):
            get_dsv4_full_indexed_c128_state_indices(req_to_token, 0, 129)


if __name__ == "__main__":
    unittest.main()
