import unittest

from sglang.srt.distributed.pipeline_layout import PipelineLayout
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


if __name__ == "__main__":
    unittest.main()
