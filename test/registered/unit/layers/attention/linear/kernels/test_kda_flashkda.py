"""CPU routing tests for the FlashKDA prefill wrapper."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kernels.kda_flashkda import (
    FlashKDAKernel,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlashKDATrackRouting(CustomTestCase):
    @staticmethod
    def _inputs(seq_len=8192):
        qkv = torch.empty(1, 1, 1, 1, dtype=torch.bfloat16)
        return {
            "q": qkv,
            "k": qkv,
            "v": qkv,
            "g": qkv,
            "beta": torch.empty(1, 1, 1, dtype=torch.bfloat16),
            "ssm_states": torch.empty(1, 1, 1, 1, dtype=torch.float32),
            "cache_indices": torch.tensor([0], dtype=torch.int32),
            "query_start_loc": torch.tensor([0, seq_len], dtype=torch.int32),
            "A_log": torch.empty(1, dtype=torch.float32),
            "dt_bias": torch.empty(1, dtype=torch.float32),
            "lower_bound": -5.0,
            "extend_seq_lens_cpu": [seq_len],
        }

    def test_final_only_tracking_stays_on_flashkda(self):
        kernel = FlashKDAKernel()
        fast_out = object()

        with (
            patch.object(kernel, "_flashkda_extend", return_value=fast_out) as fast,
            patch(
                "sglang.srt.layers.attention.linear.kernels.kda_flashkda._triton_fallback"
            ) as fallback,
        ):
            out, h = kernel.extend(
                **self._inputs(),
                return_intermediate_states=True,
                track_ssm_h_src=torch.empty(0, dtype=torch.int64),
                track_state=None,
                track_chunk_idx=None,
            )

        self.assertIs(out, fast_out)
        self.assertIsNone(h)
        fast.assert_called_once()
        fallback.assert_not_called()

    def test_interior_snapshot_routes_to_triton(self):
        kernel = FlashKDAKernel()
        snapshot = torch.empty(1, 1, 1, 1, dtype=torch.float32)
        chunk_idx = torch.tensor([3], dtype=torch.int32)
        fallback_out = object()

        with (
            patch.object(kernel, "_flashkda_extend") as fast,
            patch(
                "sglang.srt.layers.attention.linear.kernels.kda_flashkda._triton_fallback",
                return_value=fallback_out,
            ) as fallback,
        ):
            out = kernel.extend(
                **self._inputs(),
                return_intermediate_states=True,
                track_ssm_h_src=torch.tensor([3], dtype=torch.int64),
                track_state=snapshot,
                track_chunk_idx=chunk_idx,
            )

        self.assertIs(out, fallback_out)
        fast.assert_not_called()
        fallback.assert_called_once()
        self.assertIs(fallback.call_args.kwargs["track_state"], snapshot)
        self.assertIs(fallback.call_args.kwargs["track_chunk_idx"], chunk_idx)

    def test_only_validated_long_shape_uses_flashkda(self):
        kernel = FlashKDAKernel()

        with (
            patch.object(kernel, "_flashkda_extend", return_value="fast") as fast,
            patch(
                "sglang.srt.layers.attention.linear.kernels.kda_flashkda._triton_fallback",
                return_value="fallback",
            ) as fallback,
        ):
            self.assertEqual(kernel.extend(**self._inputs(8192)), ("fast", None))
            self.assertEqual(kernel.extend(**self._inputs(8193)), "fallback")

        fast.assert_called_once()
        fallback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
