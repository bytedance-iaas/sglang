import asyncio
import base64
import os
import time
import unittest

import httpx

from sglang.srt.utils import get_free_port, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

# Use a default model if not provided
DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"


class TestEncodeServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encode_port = "30000"
        cls.encode_url = f"http://127.0.0.1:{cls.encode_port}"
        cls.model = os.getenv("SGLANG_TEST_MODEL", DEFAULT_MODEL)

        print(
            f"Setting up Encode Server on port {cls.encode_port} with model {cls.model}"
        )

        # Enable batching for test
        os.environ["SGLANG_ENCODER_ENABLE_BATCHING"] = "1"
        os.environ["SGLANG_ENCODER_MAX_BATCH_SIZE"] = "8"

        cls.start_encode()

        # Wait for server ready
        # Simple wait loop since we don't have the full Disaggregation fixture here
        print("Waiting for server to be ready...")
        time.sleep(10)  # Give it some time to start uvicorn
        # Ideally check /health but we keep it simple as requested

    @classmethod
    def start_encode(cls):
        """Start encode server for multimodal processing"""
        encode_args = [
            "--trust-remote-code",
            "--encoder-only",
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--tp",
            "1",
            "--port",
            cls.encode_port,
        ]
        cls.process_encode = popen_launch_server(
            cls.model,
            base_url=cls.encode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process_encode:
            kill_process_tree(cls.process_encode.pid)

    def test_concurrent_requests(self):
        asyncio.run(run_client())


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_base64_urls(image_path: str, repeat_count: int = 5) -> list:
    img_url = encode_image(image_path)
    base64_data = f"data:image/jpg;base64,{img_url}"

    return [base64_data] * repeat_count


async def send_request(client: httpx.AsyncClient, req_id: int, mm_items: list):
    # 构造符合接口规范的请求体
    port = get_free_port()
    payload = {
        "mm_items": mm_items,
        "req_id": f"req_{req_id}",
        "num_parts": 1,
        "part_idx": 0,
        "embedding_port": [port],
        "prefill_host": "localhost",
    }

    start_time = time.time()
    print(f"[Client] 发送请求 {req_id} at {start_time:.2f}")
    try:
        response = await client.post(
            "http://127.0.0.1:30000/encode",
            json=payload,
            timeout=60.0,  # 设置较短超时以观察阻塞
        )
        end_time = time.time()
        print(
            f"[Client] 收到请求 {req_id} 响应 at {end_time:.2f}, 耗时: {end_time - start_time:.2f}s"
        )
    except httpx.TimeoutException:
        print(f"[Client] 请求 {req_id} 超时! (说明被前面的请求阻塞了)")
    except Exception as e:
        print(f"[Client] 请求 {req_id} 错误: {e}")


async def run_client():
    # Use a dummy path or environment variable for image
    image_path = os.getenv("SGLANG_TEST_IMAGE", "tmp/w1280_h720.jpg")
    img_count = 4
    img_urls = get_image_base64_urls(image_path, img_count)

    req_count = 3
    async with httpx.AsyncClient() as client:
        # 并发发送 10 个请求
        print(f"\n--- Client: 并发发送 {req_count} 个请求 ---")
        tasks = [send_request(client, i, img_urls) for i in range(1, req_count + 1)]
        await asyncio.gather(*tasks)
        print("--- Client: 测试结束 ---\n")


if __name__ == "__main__":
    unittest.main()
