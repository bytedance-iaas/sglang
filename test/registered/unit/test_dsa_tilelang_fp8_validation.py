"""Validate explicit raw and group-scaled CUDA DSA KV contracts."""

import unittest
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import _check_tilelang_dsa_fp8_kv
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDsaTilelangFp8Validation(CustomTestCase):
    def test_cuda_fp8_tilelang_decode_rejected(self):
        with self.assertRaises(ValueError):
            _check_tilelang_dsa_fp8_kv("fp8_e4m3", "flashmla_kv", "tilelang", hip=False)

    def test_cuda_fp8_tilelang_prefill_rejected(self):
        with self.assertRaises(ValueError):
            _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "trtllm", hip=False)

    def test_cuda_fp8_tilelang_pair_allowed_on_hopper(self):
        with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
            _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "tilelang", hip=False)

    def test_cuda_fp8_tilelang_pair_rejected_with_dcp(self):
        with self.assertRaisesRegex(ValueError, "incompatible with --dcp-size > 1"):
            _check_tilelang_dsa_fp8_kv(
                "fp8_e4m3", "tilelang", "tilelang", hip=False, dcp_size=2
            )

    def test_cuda_fp8_tilelang_pair_rejected_before_sm89(self):
        with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
            with self.assertRaisesRegex(ValueError, r"SM89\+"):
                _check_tilelang_dsa_fp8_kv(
                    "fp8_e4m3", "tilelang", "tilelang", hip=False
                )

    def test_cuda_group_scaled_nope_allowed(self):
        for prefill in ("tilelang", "cutedsl_h16"):
            _check_tilelang_dsa_fp8_kv(
                "fp8_e4m3",
                prefill,
                "tilelang",
                hip=False,
                nope_group_scaled=True,
                kv_layout="group528",
            )

    def test_explicit_layout_rejects_incompatible_consumers(self):
        for layout, prefill, decode, nope, dcp in (
            ("group528", "tilelang", "trtllm", True, 1),
            ("group528", "cutedsl_h16", "tilelang", False, 1),
            ("group528", "cutedsl_h16", "tilelang", True, 2),
            ("raw512", "cutedsl_h16", "tilelang", True, 1),
            ("auto", "cutedsl_h16", "tilelang", True, 1),
            ("unknown", "tilelang", "tilelang", True, 1),
        ):
            with self.subTest(
                layout=layout, prefill=prefill, decode=decode, nope=nope, dcp=dcp
            ):
                with self.assertRaises(ValueError):
                    _check_tilelang_dsa_fp8_kv(
                        "fp8_e4m3",
                        prefill,
                        decode,
                        hip=False,
                        nope_group_scaled=nope,
                        kv_layout=layout,
                        dcp_size=dcp,
                    )

    def test_explicit_raw_allowed_on_hopper(self):
        with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
            _check_tilelang_dsa_fp8_kv(
                "fp8_e4m3",
                "tilelang",
                "tilelang",
                hip=False,
                nope_group_scaled=True,
                kv_layout="raw512",
            )

    def test_hip_fp8_tilelang_allowed(self):
        # ROCm has a real fp8 tilelang kernel
        _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "tilelang", hip=True)

    def test_bf16_tilelang_allowed(self):
        # what the CUDA kernel expects
        _check_tilelang_dsa_fp8_kv("bfloat16", "tilelang", "tilelang", hip=False)

    def test_cuda_fp8_non_tilelang_allowed(self):
        # fp8-capable backends must pass
        _check_tilelang_dsa_fp8_kv("fp8_e4m3", "flashmla_kv", "trtllm", hip=False)


if __name__ == "__main__":
    unittest.main()
