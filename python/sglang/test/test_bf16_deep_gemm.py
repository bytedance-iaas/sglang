"""
Unit tests for DeepGEMM BF16 kernel wrappers.

Tests the correctness of:
  - grouped_gemm_nt_bf16bf16bf16_masked
  - grouped_gemm_nt_bf16bf16bf16_contig
  - gemm_nt_bf16bf16bf16

Against torch.matmul as reference.

Requires:
  - CUDA GPU with SM >= 90 (Hopper+)
  - `deep_gemm` package with BF16 kernel support
"""

import itertools
import unittest

import torch

from sglang.test.test_utils import CustomTestCase


def _has_bf16_deep_gemm():
    """Check if deep_gemm with BF16 support is available."""
    try:
        import deep_gemm

        return (
            hasattr(deep_gemm, "bf16_gemm_nt")
            and hasattr(deep_gemm, "m_grouped_bf16_gemm_nt_contiguous")
            and hasattr(deep_gemm, "m_grouped_bf16_gemm_nt_masked")
        )
    except ImportError:
        return False


_is_cuda = torch.cuda.is_available() and torch.version.cuda is not None
_has_bf16_dg = _has_bf16_deep_gemm() if _is_cuda else False


class TestDeepGemmBf16Normal(CustomTestCase):
    """Test bf16_gemm_nt (dense GEMM)."""

    M_SIZES = [1, 7, 64, 128, 512]
    NK_SIZES = [
        (128, 256),
        (1024, 512),
        (2112, 7168),
        (1536, 7168),
    ]
    SEEDS = [0, 42]

    @classmethod
    def setUpClass(cls):
        if not _is_cuda:
            raise unittest.SkipTest("CUDA is not available")
        if not _has_bf16_dg:
            raise unittest.SkipTest(
                "deep_gemm with BF16 support is not available"
            )
        torch.set_default_device("cuda")

    def _test_gemm_nt_bf16(self, M, NK, seed):
        N, K = NK
        torch.manual_seed(seed)

        A = torch.randn((M, K), dtype=torch.bfloat16)
        B = torch.randn((N, K), dtype=torch.bfloat16)

        # Reference: torch.matmul (A @ B^T)
        ref_out = torch.matmul(A.float(), B.float().t()).to(torch.bfloat16)

        # DeepGEMM
        out = torch.empty((M, N), dtype=torch.bfloat16)
        from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (
            gemm_nt_bf16bf16bf16,
        )

        with torch.inference_mode():
            gemm_nt_bf16bf16bf16(A, B, out)

        torch.testing.assert_close(out, ref_out, atol=1e-1, rtol=1e-2)

    def test_gemm_nt_bf16(self):
        for params in itertools.product(self.M_SIZES, self.NK_SIZES, self.SEEDS):
            with self.subTest(M=params[0], NK=params[1], seed=params[2]):
                self._test_gemm_nt_bf16(*params)


class TestDeepGemmBf16GroupedMasked(CustomTestCase):
    """Test m_grouped_bf16_gemm_nt_masked."""

    NUM_GROUPS_LIST = [4, 8, 16]
    M_SIZES = [1, 32, 64, 128]
    NK_SIZES = [
        (128, 256),
        (1536, 7168),
    ]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not _is_cuda:
            raise unittest.SkipTest("CUDA is not available")
        if not _has_bf16_dg:
            raise unittest.SkipTest(
                "deep_gemm with BF16 support is not available"
            )
        torch.set_default_device("cuda")

    def _test_grouped_masked(self, num_groups, M, NK, seed):
        N, K = NK
        torch.manual_seed(seed)

        # Create input with varying masked_m per group
        lhs = torch.randn((num_groups, M, K), dtype=torch.bfloat16)
        rhs = torch.randn((num_groups, N, K), dtype=torch.bfloat16)
        masked_m = torch.randint(
            0, M + 1, (num_groups,), dtype=torch.int32, device="cuda"
        )
        expected_m = M

        out = torch.zeros((num_groups, M, N), dtype=torch.bfloat16)

        from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (
            grouped_gemm_nt_bf16bf16bf16_masked,
        )

        with torch.inference_mode():
            grouped_gemm_nt_bf16bf16bf16_masked(
                lhs, rhs, out, masked_m, expected_m
            )

        # Reference: per-group matmul
        for g in range(num_groups):
            cur_m = masked_m[g].item()
            if cur_m > 0:
                ref = torch.matmul(
                    lhs[g, :cur_m].float(), rhs[g].float().t()
                ).to(torch.bfloat16)
                torch.testing.assert_close(
                    out[g, :cur_m], ref, atol=1e-1, rtol=1e-2
                )

    def test_grouped_masked(self):
        for params in itertools.product(
            self.NUM_GROUPS_LIST, self.M_SIZES, self.NK_SIZES, self.SEEDS
        ):
            with self.subTest(
                num_groups=params[0], M=params[1], NK=params[2], seed=params[3]
            ):
                self._test_grouped_masked(*params)


class TestDeepGemmBf16GroupedContiguous(CustomTestCase):
    """Test m_grouped_bf16_gemm_nt_contiguous."""

    NUM_GROUPS_LIST = [4, 8]
    M_SIZES = [128, 256, 512]  # Must be aligned to contiguous layout
    NK_SIZES = [
        (128, 256),
        (1536, 7168),
    ]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not _is_cuda:
            raise unittest.SkipTest("CUDA is not available")
        if not _has_bf16_dg:
            raise unittest.SkipTest(
                "deep_gemm with BF16 support is not available"
            )
        torch.set_default_device("cuda")

    def _test_grouped_contiguous(self, num_groups, M, NK, seed):
        N, K = NK
        torch.manual_seed(seed)

        # A is (M, K), where tokens are grouped by m_indices
        lhs = torch.randn((M, K), dtype=torch.bfloat16)
        rhs = torch.randn((num_groups, N, K), dtype=torch.bfloat16)

        # Assign tokens to groups uniformly
        m_indices = torch.arange(M, dtype=torch.int32, device="cuda") % num_groups

        out = torch.empty((M, N), dtype=torch.bfloat16)

        from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (
            grouped_gemm_nt_bf16bf16bf16_contig,
        )

        with torch.inference_mode():
            grouped_gemm_nt_bf16bf16bf16_contig(lhs, rhs, out, m_indices)

        # Reference: per-token matmul using the assigned group weight
        for i in range(M):
            group = m_indices[i].item()
            ref = torch.matmul(
                lhs[i : i + 1].float(), rhs[group].float().t()
            ).to(torch.bfloat16)
            torch.testing.assert_close(
                out[i : i + 1], ref, atol=1e-1, rtol=1e-2
            )

    def test_grouped_contiguous(self):
        for params in itertools.product(
            self.NUM_GROUPS_LIST, self.M_SIZES, self.NK_SIZES, self.SEEDS
        ):
            with self.subTest(
                num_groups=params[0], M=params[1], NK=params[2], seed=params[3]
            ):
                self._test_grouped_contiguous(*params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
