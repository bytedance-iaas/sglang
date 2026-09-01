"""Local PD-disaggregation validation for Qwen3.8-27B (hybrid GDN) on 8xH20.

Run:  python3 test/manual/pd_qwen38_gdn.py
Model path overridable via SGLANG_QWEN38_MODEL. Prefill TP2 on GPU 0-1,
decode TP2 on GPU 2-3, nixl transfer (single node, no RDMA needed).
Validates that GDN/mamba state survives a real P->D transfer: gsm8k accuracy
through the LB must match the non-PD numbers (>= 0.90 gate, ~0.97 expected).
"""

import os
import unittest
from types import SimpleNamespace

from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)


class TestPDQwen38GDN(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = os.environ.get(
            "SGLANG_QWEN38_MODEL", "/data02/models/Qwen3.8-27B"
        )
        cls.prefill_tp_size = 2
        cls.decode_tp_size = 2
        cls.decode_base_gpu_id = 2

        cls.start_prefill()
        cls.start_decode()
        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)
        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-bootstrap-port",
                cls.bootstrap_port,
                "--tp",
                str(cls.prefill_tp_size),
            ]
            + cls.transfer_backend
            + cls.rdma_devices,
        )

    @classmethod
    def start_decode(cls):
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--disaggregation-mode",
                "decode",
                "--disaggregation-bootstrap-port",
                cls.bootstrap_port,
                "--tp",
                str(cls.decode_tp_size),
                "--base-gpu-id",
                str(cls.decode_base_gpu_id),
            ]
            + cls.transfer_backend
            + cls.rdma_devices,
        )

    def test_smoke(self):
        import requests

        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "The capital of Australia is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
        )
        print("smoke output:", response.json()["text"])
        self.assertIn("Canberra", response.json()["text"])

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=int(os.environ.get("SGLANG_QWEN38_GSM8K_N", "100")),
            num_threads=64,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")
        self.assertGreater(metrics["score"], 0.90)


if __name__ == "__main__":
    unittest.main()
