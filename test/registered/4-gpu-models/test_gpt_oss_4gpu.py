import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.gpt_oss_common import BaseTestGptOss

register_cuda_ci(est_time=392, stage="stage-c", runner_config="4-gpu-h100")
register_cuda_ci(est_time=350, stage="stage-c", runner_config="4-gpu-b200")


class TestGptOss4Gpu(BaseTestGptOss):
    def test_bf16_120b(self):
        self.run_test(
            model_variant="120b",
            quantization="bf16",
            expected_score_of_reasoning_effort={
                "low": 0.58,
            },
            other_args=["--tp", "4", "--cuda-graph-max-bs", "200"],
        )

    @unittest.skipIf(
        is_pr_ci(),
        "Temporarily skip unstable GPT-OSS 120B MXFP4 4-GPU test in PR UT",
    )
    def test_mxfp4_120b(self):
        self.run_test(
            model_variant="120b",
            quantization="mxfp4",
            expected_score_of_reasoning_effort={
                "low": 0.58,
            },
            other_args=[
                "--tp",
                "4",
                "--cuda-graph-max-bs",
                "200",
            ],
        )


if __name__ == "__main__":
    unittest.main()
