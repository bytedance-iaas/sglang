import unittest

import torch

from sglang.srt.layers.attention.dsa.dsa_attn_tp_row_head import (
    local_row_slice,
    merge_output_rows,
    merge_query_heads,
    pack_output_heads,
    pack_query_rows,
    plan_dsa_attn_tp_row_head,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDsaAttnTpRowHeadLayout(unittest.TestCase):
    def test_two_rank_exchange_and_inverse_preserve_rows_and_heads(self):
        rows, heads, dim, tp = 8, 3, 2, 2
        plans = [plan_dsa_attn_tp_row_head(rows, tp, rank) for rank in range(tp)]
        self.assertTrue(all(plan is not None for plan in plans))

        inputs = []
        for source_rank in range(tp):
            values = torch.empty((rows, heads, dim), dtype=torch.int64)
            for row in range(rows):
                for head in range(heads):
                    values[row, head] = torch.tensor(
                        [row, source_rank * heads + head]
                    )
            inputs.append(values)

        packed = [pack_query_rows(inputs[rank], plans[rank]) for rank in range(tp)]
        full_heads = []
        for owner_rank in range(tp):
            received = torch.stack(
                [packed[source_rank][owner_rank] for source_rank in range(tp)]
            )
            merged = merge_query_heads(received, plans[owner_rank])
            want = torch.cat(
                [
                    values[
                        plans[owner_rank].local_start : plans[owner_rank].local_end
                    ]
                    for values in inputs
                ],
                dim=1,
            )
            torch.testing.assert_close(merged, want)
            full_heads.append(merged)

        packed_back = [
            pack_output_heads(full_heads[rank], plans[rank]) for rank in range(tp)
        ]
        for head_owner in range(tp):
            received = torch.stack(
                [packed_back[row_owner][head_owner] for row_owner in range(tp)]
            )
            restored = merge_output_rows(received, plans[head_owner])
            torch.testing.assert_close(restored, inputs[head_owner])

    def test_plan_falls_back_for_empty_uneven_or_non_pair_rows(self):
        self.assertIsNone(plan_dsa_attn_tp_row_head(0, 2, 0))
        self.assertIsNone(plan_dsa_attn_tp_row_head(7, 2, 0))
        self.assertIsNone(plan_dsa_attn_tp_row_head(8, 1, 0))
        self.assertIsNone(plan_dsa_attn_tp_row_head(8, 4, 0))
        with self.assertRaisesRegex(ValueError, "outside"):
            plan_dsa_attn_tp_row_head(8, 2, 2)

    def test_row_aligned_metadata_uses_owner_slice(self):
        plan = plan_dsa_attn_tp_row_head(8, 2, 1)
        metadata = torch.arange(8 * 5).reshape(8, 5)
        torch.testing.assert_close(local_row_slice(metadata, plan), metadata[4:8])
        with self.assertRaisesRegex(ValueError, "expected 8"):
            local_row_slice(metadata[:7], plan)


if __name__ == "__main__":
    unittest.main()
