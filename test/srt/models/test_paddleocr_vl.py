import base64
import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Replace with the actual model path or ID if available publicly
# For now, we use the reference from the source code, assuming it might be available in the test env
PADDLEOCR_VL_MODEL_PATH = "PaddlePaddle/PaddleOCR-VL"


class TestPaddleOCRVLBasic(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = PADDLEOCR_VL_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Adjust arguments as needed for PaddleOCR_VL
        # Using trust-remote-code as custom models often require it
        other_args = [
            "--trust-remote-code",
            "--tp",
            "1",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _encode_image(self, image_path):
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def test_with_local_images(self):
        """Test inference by reading images from a specified directory."""
        # Assume images are located in a specific directory
        # You can change this path or set it via environment variable
        image_dir = os.getenv("SGLANG_TEST_IMAGE_DIR", "tmp/paddleocr_images")

        if not os.path.exists(image_dir):
            print(f"Image directory {image_dir} does not exist. Skipping test.")
            return

        image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not image_paths:
            print(f"No images found in {image_dir}. Skipping test.")
            return
        else:
            print(f"found {len(image_paths)} images in {image_dir}")

        from openai import OpenAI

        client = OpenAI(api_key="EMPTY", base_url=f"{self.base_url}/v1", timeout=3600)

        # Task-specific base prompts
        TASKS = {
            "ocr": "OCR:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "chart": "Chart Recognition:",
        }

        # Prepare content list with all images
        content = []
        for img_path in image_paths:
            image_base64 = self._encode_image(img_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )

        # Add text prompt at the end
        content.append({"type": "text", "text": TASKS["ocr"]})

        messages = [{"role": "user", "content": content}]

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
            generated_text = response.choices[0].message.content
            print(f"Generated text: {generated_text}")

            # Basic validation
            self.assertIsInstance(generated_text, str)
            self.assertGreater(len(generated_text), 0)

        except Exception as e:
            self.fail(f"Inference failed with error: {e}")


if __name__ == "__main__":
    unittest.main()
