import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

DEEPSEEK_R1_W4AFP8_MODEL_PATH = "Barrrrry/DeepSeek-R1-W4AFP8"

class TestDeepseekV3W4afp8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_R1_W4AFP8_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--trust-remote-code", "--tp", "8","--enable-ep-moe"]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1200,
            parallel=1200,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        self.assertGreater(metrics["accuracy"], 0.92)


class TestDeepseekV3W4Afp8Mtp(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_R1_W4AFP8_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "8",
            "--trust-remote-code",
            "--enable-ep-moe",
            "--cuda-graph-bs",
            "256",
            "--disable-radix-cache",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "2",
            "--speculative-num-draft-tokens",
            "4",
        ]
        if not is_in_amd_ci():
            other_args += ["--mem-frac", "0.7"]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        server_info = requests.get(self.base_url + "/get_server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v3 mtp)\n"
                f'{metrics["accuracy"]=:.3f}\n'
                f"{avg_spec_accept_length=:.2f}\n"
            )
            self.assertGreater(metrics["accuracy"], 0.935)
            self.assertGreater(avg_spec_accept_length, 2.9)

if __name__ == "__main__":
    unittest.main()
