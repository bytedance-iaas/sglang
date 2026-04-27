import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models import utils as model_utils


register_cpu_ci(est_time=4, suite="stage-a-test-cpu")


class TestEnableFusedSetKVBufferDCP(CustomTestCase):
    def test_disable_fused_path_when_dcp_enabled(self):
        forward_batch = SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(dtype=torch.bfloat16)
        )

        with (
            patch.object(model_utils, "_is_cuda", True),
            patch.object(model_utils, "_is_hip", False),
            patch(
                "sglang.srt.models.utils.is_prefill_context_parallel_enabled",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.utils.get_dcp_group",
                return_value=SimpleNamespace(world_size=2),
            ),
        ):
            self.assertFalse(model_utils.enable_fused_set_kv_buffer(forward_batch))

    def test_enable_fused_path_when_dcp_disabled(self):
        forward_batch = SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(dtype=torch.bfloat16)
        )

        with (
            patch.object(model_utils, "_is_cuda", True),
            patch.object(model_utils, "_is_hip", False),
            patch(
                "sglang.srt.models.utils.is_prefill_context_parallel_enabled",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.utils.get_dcp_group",
                return_value=SimpleNamespace(world_size=1),
            ),
        ):
            self.assertTrue(model_utils.enable_fused_set_kv_buffer(forward_batch))


if __name__ == "__main__":
    unittest.main()