"""Correctness coverage for the SM90 FP8 prefill indexer."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
    "requires an SM90 GPU",
)
class TestSm90Fp4IndexerPrefill(CustomTestCase):
    def test_fused_scores_and_topk_match_reference(self):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            store_fp4_index_k_cache,
        )
        from sglang.kernels.ops.attention.dsv4.sm90_fp4_indexer import (
            fp8_index_logits_prefill,
            quantize_bf16_index_queries_fp8,
            unpack_fp4_index_keys_to_fp8,
        )
        from sglang.kernels.ops.attention.dsv4.torch_quant import fake_quant_fp4

        generator = torch.Generator(device="cuda").manual_seed(7)
        rows, heads, head_dim = 9, 64, 128
        width, page_size, pool_size = 193, 64, 256
        slots = torch.randperm(pool_size, device="cuda", generator=generator)[
            :width
        ].to(torch.int64)
        table = torch.zeros(
            pool_size // page_size,
            page_size * 68,
            dtype=torch.uint8,
            device="cuda",
        )
        keys = fake_quant_fp4(
            torch.randn(
                width,
                head_dim,
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
        )
        store_fp4_index_k_cache(
            input=keys,
            cache=table,
            loc=slots.to(torch.int32),
            page_size=page_size,
            rne=True,
        )
        queries = fake_quant_fp4(
            torch.randn(
                rows,
                heads,
                head_dim,
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
        )
        weights = torch.randn(
            rows,
            heads,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        lens = torch.tensor([1, 17, 64, 65, 96, 129, 160, 192, 193], device="cuda")

        fp8_keys = unpack_fp4_index_keys_to_fp8(slots, table, page_size)
        fp8_queries = quantize_bf16_index_queries_fp8(queries)
        torch.testing.assert_close(fp8_keys.bfloat16(), keys, atol=0, rtol=0)
        torch.testing.assert_close(fp8_queries.bfloat16(), queries, atol=0, rtol=0)

        actual = fp8_index_logits_prefill(
            fp8_queries,
            weights,
            fp8_keys,
            lens,
        )
        expected = torch.einsum("bhd,nd->bhn", fp8_queries.float(), fp8_keys.float())
        expected = expected.bfloat16().float().relu()
        expected = (expected * weights.float().unsqueeze(-1)).bfloat16().float()
        expected = expected.sum(dim=1).bfloat16().float()
        columns = torch.arange(width, device="cuda")
        expected.masked_fill_(columns[None, :] >= lens[:, None], -torch.inf)

        self.assertEqual(actual.shape, (rows, 196))
        torch.testing.assert_close(actual[:, :width], expected, atol=0.125, rtol=0.02)
        self.assertTrue(torch.isneginf(actual[:, width:]).all())
        for row, visible in enumerate(lens.tolist()):
            k = min(32, visible)
            self.assertTrue(
                torch.equal(
                    actual[row, :visible].topk(k).indices,
                    expected[row, :visible].topk(k).indices,
                )
            )


if __name__ == "__main__":
    unittest.main()
