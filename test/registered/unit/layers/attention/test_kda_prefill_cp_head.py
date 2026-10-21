import unittest

import torch
from sglang.srt.layers.attention.linear.kda_prefill_cp_head import (
    gather_full_head_axis,
    merge_head_shards_for_token_owner,
    merge_token_shards_for_head_owner,
    pack_global_rows_for_token_owners,
    pack_local_rows_for_head_owners,
    plan_kda_prefill_cp_head,
    select_flat_head_range,
    select_packed_qkv_head_range,
    write_flat_head_range,
    write_packed_qkv_head_range,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKdaPrefillCpHeadLayout(unittest.TestCase):
    def _plans(self, rows=4, heads=6):
        return [
            plan_kda_prefill_cp_head(
                local_rows=rows,
                local_heads=heads,
                logical_rank_rows=[rows, rows],
                physical_rank_rows=[rows, rows],
                cp_size=2,
                cp_rank=rank,
            )
            for rank in range(2)
        ]

    def test_input_exchange_and_output_inverse_preserve_rows_and_heads(self):
        plans = self._plans()
        inputs = []
        for token_owner in range(2):
            value = torch.empty(4, 6, 2, dtype=torch.int64)
            for local_row in range(4):
                global_row = local_row * 2 + token_owner
                for head in range(6):
                    value[local_row, head] = torch.tensor([global_row, head])
            inputs.append(value)

        packed = [
            pack_local_rows_for_head_owners(inputs[rank], plans[rank])
            for rank in range(2)
        ]
        owned = []
        for head_owner in range(2):
            received = torch.stack([packed[src][head_owner] for src in range(2)])
            result = merge_token_shards_for_head_owner(received, plans[head_owner])
            expected = torch.stack(
                [
                    torch.tensor(
                        [row, head],
                        dtype=torch.int64,
                    )
                    for row in range(8)
                    for head in range(
                        plans[head_owner].head_start, plans[head_owner].head_end
                    )
                ]
            ).reshape(8, 3, 2)
            torch.testing.assert_close(result, expected)
            owned.append(result)

        packed_back = [
            pack_global_rows_for_token_owners(owned[rank], plans[rank])
            for rank in range(2)
        ]
        for token_owner in range(2):
            received = torch.stack(
                [packed_back[head_owner][token_owner] for head_owner in range(2)]
            )
            restored = merge_head_shards_for_token_owner(received, plans[token_owner])
            torch.testing.assert_close(restored, inputs[token_owner])

    def test_plan_falls_back_for_empty_uneven_padding_or_wrong_topology(self):
        common = dict(
            local_rows=4,
            local_heads=6,
            logical_rank_rows=[4, 4],
            physical_rank_rows=[4, 4],
            cp_size=2,
            cp_rank=0,
        )
        self.assertIsNotNone(plan_kda_prefill_cp_head(**common))
        for changes in (
            {
                "local_rows": 0,
                "logical_rank_rows": [0, 0],
                "physical_rank_rows": [0, 0],
            },
            {"logical_rank_rows": [4, 3]},
            {"physical_rank_rows": [5, 5]},
            {"cp_size": 1, "logical_rank_rows": [4], "physical_rank_rows": [4]},
            {"local_heads": 5},
        ):
            args = dict(common)
            args.update(changes)
            self.assertIsNone(plan_kda_prefill_cp_head(**args))
        with self.assertRaisesRegex(ValueError, "outside"):
            plan_kda_prefill_cp_head(**{**common, "cp_rank": 2})

    def test_flat_and_packed_head_slices_round_trip(self):
        plan = self._plans(rows=2, heads=6)[1]
        flat = torch.arange(2 * 6 * 2).reshape(2, 12)
        selected = select_flat_head_range(flat, plan, head_dim=2)
        torch.testing.assert_close(selected, flat.reshape(2, 6, 2)[:, 3:].reshape(2, 6))
        flat_out = torch.zeros_like(flat).reshape(2, 6, 2)
        write_flat_head_range(flat_out, selected.reshape(2, 3, 2), plan, head_axis=1)
        torch.testing.assert_close(flat_out[:, 3:], selected.reshape(2, 3, 2))
        self.assertEqual(torch.count_nonzero(flat_out[:, :3]), 0)

        packed = torch.arange(2 * 3 * 6 * 2).reshape(2, 36)
        packed_selected = select_packed_qkv_head_range(packed, plan, head_dim=2)
        expected = packed.reshape(2, 3, 6, 2)[:, :, 3:].reshape(2, 18)
        torch.testing.assert_close(packed_selected, expected)
        packed_out = torch.zeros_like(packed)
        write_packed_qkv_head_range(packed_out, packed_selected, plan, head_dim=2)
        torch.testing.assert_close(
            packed_out.reshape(2, 3, 6, 2)[:, :, 3:],
            expected.reshape(2, 3, 3, 2),
        )
        self.assertEqual(
            torch.count_nonzero(packed_out.reshape(2, 3, 6, 2)[:, :, :3]), 0
        )

    def test_state_gather_contract_concatenates_rank_order(self):
        plan = self._plans(rows=2, heads=6)[0]
        rank0 = torch.arange(2 * 3 * 2).reshape(2, 3, 2)
        rank1 = rank0 + 100

        class Group:
            @staticmethod
            def all_gather(value, dim):
                self.assertEqual(dim, 1)
                torch.testing.assert_close(value, rank0)
                return torch.cat((rank0, rank1), dim=1)

        from sglang.srt.runtime_context import get_parallel

        with get_parallel().override(attn_cp_group=Group()):
            gathered = gather_full_head_axis(rank0, plan, head_axis=1)
        torch.testing.assert_close(gathered, torch.cat((rank0, rank1), dim=1))


if __name__ == "__main__":
    unittest.main()
