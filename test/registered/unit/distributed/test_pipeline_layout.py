import unittest

from sglang.srt.distributed.pipeline_layout import (
    PipelineLayout,
    PipelineWavefrontSchedule,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPipelineLayout(unittest.TestCase):
    def test_vpp2_maps_two_non_contiguous_chunks_per_rank(self):
        layout = PipelineLayout.build(
            num_hidden_layers=40,
            physical_size=4,
            virtual_stages=2,
        )

        self.assertEqual(
            layout.layer_ids_for_rank(0),
            tuple(range(0, 5)) + tuple(range(20, 25)),
        )
        self.assertEqual(
            layout.layer_ids_for_rank(3),
            tuple(range(15, 20)) + tuple(range(35, 40)),
        )
        self.assertEqual(
            [stage.physical_rank for stage in layout.stages],
            [0, 1, 2, 3, 0, 1, 2, 3],
        )

    def test_digest_is_stable_and_changes_with_partition(self):
        default = PipelineLayout.build(40, 4, 2)
        same = PipelineLayout.build(40, 4, 2)
        uneven = PipelineLayout.build(
            40,
            4,
            2,
            partition=(4, 5, 5, 6, 5, 5, 5, 5),
        )

        self.assertEqual(default.digest, same.digest)
        self.assertNotEqual(default.digest, uneven.digest)

    def test_rejects_incomplete_partition_manifest(self):
        with self.assertRaisesRegex(ValueError, "does not match logical pipeline size"):
            PipelineLayout.build(40, 4, 2, partition=(10, 10, 10, 10))

        with self.assertRaisesRegex(ValueError, "does not match num_hidden_layers"):
            PipelineLayout.build(40, 4, 2, partition=(5,) * 7 + (4,))

    def test_vpp2_wavefront_schedule(self):
        schedule = PipelineWavefrontSchedule.build(4, 2)
        expected = (
            ((0, 0), None, None, None),
            ((1, 0), (0, 1), None, None),
            ((2, 0), (1, 1), (0, 2), None),
            ((3, 0), (2, 1), (1, 2), (0, 3)),
            ((0, 4), (3, 1), (2, 2), (1, 3)),
            ((1, 4), (0, 5), (3, 2), (2, 3)),
            ((2, 4), (1, 5), (0, 6), (3, 3)),
            ((3, 4), (2, 5), (1, 6), (0, 7)),
            (None, (3, 5), (2, 6), (1, 7)),
            (None, None, (3, 6), (2, 7)),
            (None, None, None, (3, 7)),
        )

        for tick, expected_actions in enumerate(expected):
            actual = []
            for rank in range(4):
                action = schedule.action(tick, rank)
                actual.append(
                    None if action is None else (action.batch_seq, action.stage_id)
                )
            self.assertEqual(tuple(actual), expected_actions)

    def test_vpp2_wavefront_preserves_dependencies_and_reuses_slots(self):
        schedule = PipelineWavefrontSchedule.build(4, 2)
        stage_ticks = {}
        first_batch_seq = 8
        for tick in range(schedule.num_ticks):
            for rank in range(4):
                action = schedule.action(tick, rank, first_batch_seq)
                if action is None:
                    continue
                self.assertEqual(action.physical_rank, action.stage_id % 4)
                self.assertEqual(action.slot_id, action.batch_seq % 4)
                stage_ticks[(action.batch_seq, action.stage_id)] = tick

        self.assertEqual(list(schedule.batch_seqs(first_batch_seq)), list(range(8, 12)))
        for batch_seq in schedule.batch_seqs(first_batch_seq):
            ticks = [stage_ticks[(batch_seq, stage_id)] for stage_id in range(8)]
            batch_offset = batch_seq - first_batch_seq
            self.assertEqual(ticks, list(range(batch_offset, batch_offset + 8)))
            self.assertEqual(
                schedule.completion_batch_seq(ticks[-1], first_batch_seq),
                batch_seq,
            )

    def test_vpp2_wavefront_rejects_unsupported_shapes(self):
        with self.assertRaisesRegex(ValueError, "VPP2 only"):
            PipelineWavefrontSchedule.build(4, 3)
        with self.assertRaisesRegex(ValueError, "one slot per PP rank"):
            PipelineWavefrontSchedule.build(4, 2, wave_size=8)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            PipelineWavefrontSchedule.build(4, 2).batch_seqs(-1)
        with self.assertRaisesRegex(ValueError, "must be in"):
            PipelineWavefrontSchedule.build(4, 2).action(-1, 0)


if __name__ == "__main__":
    unittest.main()
