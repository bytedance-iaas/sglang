"""Exact CP visibility, independent KV ownership, scratch and stream contracts."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "GPU required")
class TestSm120Prefill(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("SM120 required")
        from flashinfer.dsa_indexer import dsa_indexer_det_topk  # noqa: F401

        from sglang.srt.layers.attention.dsa.sm120_fused_indexer import prefill_topk

        self.prefill_topk = prefill_topk

    def inputs(self, ties=False):
        gen = torch.Generator(device="cuda").manual_seed(17)
        lengths, queries = [3003, 4101], [17, 19]
        pages_per = [(n + 63) // 64 for n in lengths]
        page_count = sum(pages_per) + 2
        keys = (
            torch.randn(page_count, 64, 128, generator=gen, device="cuda") * 0.5
        ).to(torch.float8_e4m3fn)
        if ties:
            keys.zero_()
        scales = torch.rand(page_count, 64, generator=gen, device="cuda") + 0.5
        raw = torch.empty(page_count, 64 * 132, dtype=torch.uint8, device="cuda")
        raw[:, : 64 * 128] = keys.view(torch.uint8).reshape(page_count, -1)
        raw[:, 64 * 128 :] = scales.view(torch.uint8).reshape(page_count, -1)
        permutation = torch.randperm(page_count, generator=gen, device="cuda").to(
            torch.int32
        )
        width = max(pages_per)
        table = torch.zeros(2, width, dtype=torch.int32, device="cuda")
        at = 0
        for batch, count in enumerate(pages_per):
            table[batch, :count] = permutation[at : at + count]
            at += count
        output_table = table + 101  # MLA slots are a separate ownership surface.
        rows = sum(queries)
        q = (torch.randn(rows, 32, 128, generator=gen, device="cuda") * 0.5).to(
            torch.float8_e4m3fn
        )
        w = torch.randn(rows, 32, generator=gen, device="cuda")
        row_batch = torch.repeat_interleave(
            torch.arange(2, device="cuda"), torch.tensor(queries, device="cuda")
        ).to(torch.int32)
        starts, ends, base = [], [], 0
        for length, count in zip(lengths, queries):
            for row in range(count):
                starts.append(base)
                ends.append(base + length - 4 * count + 2 + 4 * row)
            base += length
        return (
            q,
            w,
            raw,
            table,
            output_table,
            row_batch,
            torch.tensor(starts, dtype=torch.int32, device="cuda"),
            torch.tensor(ends, dtype=torch.int32, device="cuda"),
            keys,
            scales,
        )

    def assert_exact(self, values, out):
        q, w, _, table, output_table, batch, starts, ends, keys, scales = values
        for row in range(q.shape[0]):
            b = int(batch[row])
            visible = int(ends[row] - starts[row])
            count = (visible + 63) // 64
            import deep_gemm

            k = keys[table[b, :count].long()].reshape(-1, 128)[:visible].contiguous()
            s = scales[table[b, :count].long()].reshape(-1)[:visible].contiguous()
            scores = deep_gemm.fp8_mqa_logits(
                q[row : row + 1],
                (k, s),
                w[row : row + 1],
                torch.zeros(1, dtype=torch.int32, device="cuda"),
                torch.tensor([visible], dtype=torch.int32, device="cuda"),
                clean_logits=False,
            )[0, :visible]
            wanted = (
                visible - 1 - scores.flip(0).argsort(descending=True, stable=True)[:64]
            )
            expected = output_table[b, wanted // 64].long() * 64 + wanted % 64
            self.assertTrue(
                torch.equal(out[row].long().sort().values, expected.sort().values)
            )

    def call(self, values, budget=1 << 26, check_padding=True):
        q, w, raw, table, target, batch, starts, ends, *_ = values
        out = torch.full((q.shape[0] + 2, 64), -1, dtype=torch.int32, device="cuda")
        result, reason = self.prefill_topk(
            q, w, raw, table, target, batch, starts, ends, out, 64, budget
        )
        self.assertIsNone(reason)
        self.assertIs(result, out)
        if check_padding:
            self.assertTrue(torch.all(out[q.shape[0] :] == -1))
        return out[: q.shape[0]]

    def test_cp_full_tail_and_independent_slot_ownership(self):
        values = self.inputs()
        self.assert_exact(values, self.call(values, budget=16 * 8 * 115840))

    def test_equal_scores_choose_larger_ids(self):
        values = self.inputs(ties=True)
        self.assert_exact(values, self.call(values))

    def test_unsupported_format_and_budget_leave_output(self):
        values = self.inputs()
        q, w, raw, table, target, batch, starts, ends, *_ = values
        out = torch.full((q.shape[0], 64), 71, dtype=torch.int32, device="cuda")
        result, reason = self.prefill_topk(
            q.to(torch.bfloat16),
            w,
            raw,
            table,
            target,
            batch,
            starts,
            ends,
            out,
            64,
            1 << 20,
        )
        self.assertIsNone(result)
        self.assertIsNotNone(reason)
        self.assertTrue(torch.all(out == 71))
        result, reason = self.prefill_topk(
            q, w, raw, table, target, batch, starts, ends, out, 64, 1
        )
        self.assertIsNone(result)
        self.assertIsNotNone(reason)
        self.assertTrue(torch.all(out == 71))

    def test_two_streams_share_paged_workspace_safely(self):
        values = self.inputs()
        other = self.inputs(ties=True)
        first, second = torch.cuda.Stream(), torch.cuda.Stream()
        first.wait_stream(torch.cuda.current_stream())
        second.wait_stream(torch.cuda.current_stream())
        # Queue both calls before any result read, which would synchronize
        # the producer stream and hide shared-workspace races.
        with torch.cuda.stream(first):
            a = self.call(values, check_padding=False)
        with torch.cuda.stream(second):
            b = self.call(other, check_padding=False)
        torch.cuda.synchronize()
        self.assert_exact(values, a)
        self.assert_exact(other, b)

    def test_wide_table_retained_workspace_stays_bounded(self):
        from flashinfer.experimental.sm120_dsa_indexer import _api

        values = list(self.inputs())
        # The shape fixes the scan bound even when these rows see fewer keys.
        # Grow the table to the million-token capacity without changing the
        # visible keys, and exercise shared-buffer growth after short calls.
        wide = torch.zeros(2, 16384, dtype=torch.int32, device="cuda")
        wide[:, : values[3].shape[1]] = values[3]
        values[3] = wide
        self.assert_exact(values, self.call(values))
        buffers = [*_api._LOGITS.values(), *_api._LONG_WS.values(), *_api._RETIRED]
        self.assertLessEqual(
            sum(t.numel() * t.element_size() for t in buffers), 1 << 26
        )


if __name__ == "__main__":
    unittest.main()
