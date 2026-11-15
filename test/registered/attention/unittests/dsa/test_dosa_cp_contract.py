import dataclasses
import unittest

from sglang.srt.layers.attention.dsa.dosa_cp import (
    DosaCarryKey,
    DosaCarryRegistry,
    build_dosa_cp_layout,
    map_logical_topk_to_physical,
    validate_dosa_cp_layouts,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeEvent:
    def __init__(self, ready=False):
        self.ready = ready

    def query(self):
        return self.ready


class TestDosaCPLayoutContract(unittest.TestCase):
    def _build_cp8_layouts(self):
        kwargs = dict(
            cp_size=8,
            request_generations=[101, 202, 303],
            rids=["req-a", "req-b", "req-c"],
            seq_lens=[29, 37, 19],
            extend_seq_lens=[17, 18, 5],
            query_layout_version="zigzag-2cp-v1",
            cache_versions=[7, 7, 8],
            page_table_rows=[2, 4, 6],
        )
        return [build_dosa_cp_layout(cp_rank=rank, **kwargs) for rank in range(8)]

    def test_cp8_multibatch_nondivisible_prefix_contract(self):
        layouts = self._build_cp8_layouts()
        validate_dosa_cp_layouts(layouts)

        self.assertEqual(sum(layout.local_row_count for layout in layouts), 40)
        rank0 = layouts[0]
        self.assertEqual(
            [segment.segment_kind for segment in rank0.segments],
            ["prev", "prev", "prev", "next", "next", "next"],
        )
        self.assertEqual(
            [segment.batch_idx for segment in rank0.segments],
            [0, 1, 2, 0, 1, 2],
        )
        self.assertEqual(rank0.prefix_lens, (12, 19, 14))
        self.assertEqual(rank0.segments[0].absolute_positions, (12, 13))
        self.assertEqual(rank0.segments[0].visible_ends, (13, 14))
        self.assertEqual(rank0.segments[0].output_rows, (0, 1))
        self.assertEqual(rank0.segments[3].segment_index, 15)
        self.assertEqual(rank0.segments[3].output_rows, (16,))
        self.assertEqual(rank0.segments[3].absolute_positions, (28,))

    def test_zero_length_segments_keep_local_order_contract(self):
        layouts = [
            build_dosa_cp_layout(
                cp_rank=rank,
                cp_size=8,
                request_generations=[1],
                rids=["short"],
                seq_lens=[3],
                extend_seq_lens=[3],
                query_layout_version="zigzag-2cp-v1",
            )
            for rank in range(8)
        ]
        validate_dosa_cp_layouts(layouts)
        self.assertEqual(sum(layout.local_row_count for layout in layouts), 3)
        self.assertTrue(any(layout.local_row_count == 0 for layout in layouts))

    def test_duplicate_output_row_fails_closed(self):
        layouts = self._build_cp8_layouts()
        target = layouts[1]
        segment = target.segments[0]
        duplicate = dataclasses.replace(
            segment,
            output_rows=(layouts[0].segments[0].output_rows[0],)
            * len(segment.output_rows),
        )
        layouts[1] = dataclasses.replace(
            target, segments=(duplicate,) + target.segments[1:]
        )
        with self.assertRaisesRegex(ValueError, "coverage is not exact"):
            validate_dosa_cp_layouts(layouts)


class TestDosaPagedMappingContract(unittest.TestCase):
    def test_shuffled_pages_match_runtime_formula(self):
        result = map_logical_topk_to_physical(
            [[0, 3, 4, 9, -1], [1, 7, 8, -1, -1]],
            page_table=[[9, 2, 12], [7, 4, 11]],
            row_batch=[0, 1],
            page_size=4,
        )
        self.assertEqual(result[0], (36, 39, 8, 49, -1))
        self.assertEqual(result[1], (29, 19, 44, -1, -1))

    def test_out_of_bounds_position_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "exceeds page-table"):
            map_logical_topk_to_physical(
                [[8]], page_table=[[3, 4]], row_batch=[0], page_size=4
            )


class TestDosaCarryLifecycleContract(unittest.TestCase):
    def test_release_waits_for_every_recorded_event(self):
        registry = DosaCarryRegistry[str]()
        key = DosaCarryKey("req-a", 11, 4, 0, "zigzag-v1", 7)
        calling_stream = _FakeEvent(ready=True)
        side_stream = _FakeEvent(ready=False)
        registry.put(key, "threshold")
        registry.record_use(key, calling_stream)
        registry.record_use(key, side_stream)

        registry.release_request("req-a", 11)
        self.assertIsNone(registry.lookup(key))
        self.assertEqual(registry.collect_retired(), ())
        self.assertEqual(registry.retired_count, 1)

        side_stream.ready = True
        self.assertEqual(registry.collect_retired(), (key,))
        self.assertEqual(registry.retired_count, 0)

    def test_same_rid_new_generation_cannot_reuse_old_carry(self):
        registry = DosaCarryRegistry[str]()
        old = DosaCarryKey("same-rid", 100, 9, 3, "zigzag-v1", 5)
        new = DosaCarryKey("same-rid", 101, 9, 3, "zigzag-v1", 5)
        registry.put(old, "old")

        self.assertIsNone(registry.lookup(new))
        registry.release_request("same-rid", 100)
        self.assertEqual(registry.collect_retired(), (old,))
        registry.put(new, "new")
        self.assertEqual(registry.lookup(new), "new")

    def test_same_generation_other_rid_is_not_released(self):
        registry = DosaCarryRegistry[str]()
        first = DosaCarryKey("req-a", 9, 1, 0, "zigzag-v1", 3)
        other = DosaCarryKey("req-b", 9, 1, 0, "zigzag-v1", 3)
        registry.put(first, "first")
        registry.put(other, "other")

        registry.release_request("req-a", 9)
        self.assertIsNone(registry.lookup(first))
        self.assertEqual(registry.lookup(other), "other")
        self.assertEqual(registry.collect_retired(), (first,))

    def test_cache_version_change_invalidates_stale_carry(self):
        registry = DosaCarryRegistry[str]()
        stale = DosaCarryKey("stale", 1, 0, 0, "zigzag-v1", 7)
        current = DosaCarryKey("current", 2, 0, 0, "zigzag-v1", 8)
        registry.put(stale, "stale")
        registry.put(current, "current")

        registry.invalidate_cache_version(8)
        self.assertIsNone(registry.lookup(stale))
        self.assertEqual(registry.lookup(current), "current")
        self.assertEqual(registry.collect_retired(), (stale,))


if __name__ == "__main__":
    unittest.main()
