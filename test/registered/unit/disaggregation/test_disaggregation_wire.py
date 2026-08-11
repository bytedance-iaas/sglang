import unittest
from types import SimpleNamespace
from unittest.mock import patch

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
    setup_state_kv_args,
    unpack_state_types,
)
from sglang.srt.server_args import ServerArgs
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

    def test_dspark_state_matching_requires_wire_metadata(self):
        with self.assertRaisesRegex(RuntimeError, "state_types metadata"):
            resolve_state_component_dst_index(
                [StateType.SWA],
                [],
                0,
                require_metadata=True,
            )

    def test_state_matching_rejects_missing_occurrence(self):
        with self.assertRaisesRegex(RuntimeError, "occurrence 2"):
            resolve_state_component_dst_index(
                [StateType.SWA, StateType.SWA],
                [StateType.SWA, StateType.C128_STATE],
                1,
                require_metadata=True,
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

    def test_dspark_nonfinal_and_final_pp_state_registration(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        mapping = object()
        target = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        target.compression_ratios = [0, 128]
        target.page_size = 256
        target.swa_page_size = 256
        target.swa_window_size = 4096
        target.full_to_swa_index_mapping = mapping
        target.get_state_buf_infos = lambda: ([10, 40], [20, 50], [30, 60])
        target.get_dspark_pd_state_buf_infos = lambda: ([10], [20], [30])
        target.get_dspark_pd_state_layer_ids = lambda: [7]
        target.get_c128_state_buf_infos = lambda: ([40], [50], [60])
        target.get_c128_state_layer_ids = lambda: [8]

        draft = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        draft.compression_ratios = [0]
        draft.page_size = 256
        draft.swa_page_size = 256
        draft.swa_window_size = 4096
        draft.full_to_swa_index_mapping = mapping
        draft.get_state_buf_infos = lambda: ([70], [80], [90])

        prefill_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="prefill",
            disaggregation_peer_speculative_algorithm="DSPARK",
        )
        self.assertIsNone(prefill_args.speculative_algorithm)

        with patch(
            "sglang.srt.server_args.get_global_server_args",
            return_value=prefill_args,
        ):
            nonfinal = SimpleNamespace()
            setup_state_kv_args(nonfinal, target, draft_token_to_kv_pool=None)
            final = SimpleNamespace()
            setup_state_kv_args(final, target, draft_token_to_kv_pool=draft)

        decode_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            speculative_algorithm="DSPARK",
        )
        with patch(
            "sglang.srt.server_args.get_global_server_args",
            return_value=decode_args,
        ):
            decode = SimpleNamespace()
            setup_state_kv_args(decode, target, draft_token_to_kv_pool=draft)

        self.assertEqual(
            nonfinal.state_types,
            [StateType.SWA, StateType.C128_STATE],
        )
        self.assertEqual(nonfinal.state_layer_ids, [[7], [8]])
        self.assertEqual(
            final.state_types,
            [StateType.SWA, StateType.C128_STATE, StateType.SWA],
        )
        self.assertEqual(final.state_layer_ids[-1], [1_000_000])
        self.assertEqual(final.state_types, decode.state_types)
        self.assertEqual(final.state_layer_ids, decode.state_layer_ids)

        legacy_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="prefill",
        )
        with patch(
            "sglang.srt.server_args.get_global_server_args",
            return_value=legacy_args,
        ):
            legacy = SimpleNamespace()
            setup_state_kv_args(legacy, target, draft_token_to_kv_pool=None)

        self.assertEqual(legacy.state_types, [StateType.SWA])
        self.assertEqual(legacy.state_layer_ids, [[]])


if __name__ == "__main__":
    unittest.main()
