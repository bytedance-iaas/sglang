import os
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=160, suite="stage-b-test-1-gpu-large")


def _has_hf_cache_entry(repo_id: str, repo_type: str = "model") -> bool:
    cache_root = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not cache_root:
        return False
    prefix = "datasets--" if repo_type == "dataset" else "models--"
    repo_cache = os.path.join(cache_root, prefix + repo_id.replace("/", "--"))
    if not os.path.isdir(repo_cache):
        return False
    if repo_type != "model":
        return True

    snapshot_root = os.path.join(repo_cache, "snapshots")
    if not os.path.isdir(snapshot_root):
        return False

    weight_suffixes = (".safetensors", ".bin", ".pt")
    for root, _, files in os.walk(snapshot_root):
        if any(filename.endswith(weight_suffixes) for filename in files):
            return True
    return False


class BaseW8A8Test(CustomTestCase):
    model: str = None
    quantization: str = None
    gsm8k_accuracy_threshold: float = None
    throughput_threshold: float = None

    @classmethod
    def setUpClass(cls):
        if cls is BaseW8A8Test:
            raise unittest.SkipTest("Skip base test class")

        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = []
        if cls.quantization:
            other_args.extend(["--quantization", cls.quantization])

        if not _has_hf_cache_entry(cls.model):
            raise unittest.SkipTest(f"Model cache missing: {cls.model}")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        if cls is BaseW8A8Test:
            return
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        if self.gsm8k_accuracy_threshold is None:
            self.skipTest("gsm8k_accuracy_threshold not set for this test")

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(metrics)
        self.assertGreater(metrics["score"], self.gsm8k_accuracy_threshold)

    def run_decode(self, max_new_tokens):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
                "ignore_eos": True,
            },
        )
        return response.json()

    def test_throughput(self):

        max_tokens = 256
        tic = time.perf_counter()
        res = self.run_decode(max_tokens)
        tok = time.perf_counter()
        print(res["text"])
        throughput = max_tokens / (tok - tic)
        print(f"Throughput: {throughput} tokens/s")
        self.assertGreaterEqual(throughput, self.throughput_threshold)


@unittest.skipIf(
    is_in_ci() and os.getenv("GITHUB_EVENT_NAME") == "pull_request",
    "Meta-Llama-3 W8A8 INT8 throughput is below the PR UT threshold on current H20 runners",
)
class TestW8A8Int8(BaseW8A8Test):
    model = "neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8"
    quantization = "w8a8_int8"
    gsm8k_accuracy_threshold = 0.69
    throughput_threshold = 200


class TestW8A8Fp8(BaseW8A8Test):
    model = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"
    quantization = "w8a8_fp8"
    gsm8k_accuracy_threshold = 0.69
    throughput_threshold = 200

    @unittest.skipIf(
        is_in_ci() and os.getenv("GITHUB_EVENT_NAME") == "pull_request",
        "Meta-Llama-3.1-8B FP8 GSM8K accuracy regresses on current CUDA PR UT runners",
    )
    def test_gsm8k(self):
        super().test_gsm8k()


@unittest.skipIf(
    is_in_ci() and os.getenv("GITHUB_EVENT_NAME") == "pull_request",
    "Qwen3 FP8 MoE accuracy/throughput is unstable on current CUDA PR UT runners",
)
class TestW8A8Fp8MoE(BaseW8A8Test):
    model = "RedHatAI/Qwen3-30B-A3B-FP8-dynamic"
    quantization = "w8a8_fp8"
    gsm8k_accuracy_threshold = 0.88
    throughput_threshold = 180


if __name__ == "__main__":
    unittest.main()
