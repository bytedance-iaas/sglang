import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

from sglang.srt.server_args import ServerArgs


class TestPipelineParallelSpecServerArgs(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        kwargs = dict(
            model_path="dummy",
            pp_size=2,
            disable_overlap_schedule=True,
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
        )
        kwargs.update(overrides)
        return ServerArgs(**kwargs)

    def test_static_topk_one_eagle_is_legal(self):
        self._args()._check_pipeline_parallel_args()

    def test_overlap_schedule_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "overlap schedule"):
            self._args(disable_overlap_schedule=False)._check_pipeline_parallel_args()

    def test_topk_greater_than_one_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "topk 1"):
            self._args(speculative_eagle_topk=2)._check_pipeline_parallel_args()

    def test_adaptive_speculation_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "adaptive"):
            self._args(speculative_adaptive=True)._check_pipeline_parallel_args()

    def test_non_eagle_algorithm_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "only the EAGLE"):
            self._args(speculative_algorithm="NGRAM")._check_pipeline_parallel_args()

    def test_non_spec_pp_remains_legal(self):
        self._args(speculative_algorithm=None)._check_pipeline_parallel_args()


if __name__ == "__main__":
    unittest.main()
