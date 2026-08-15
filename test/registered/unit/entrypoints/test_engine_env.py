import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints.engine import _set_envs_and_config


def _server_args():
    return SimpleNamespace(
        attention_backend=None,
        custom_sigquit_handler=None,
        dcp_size=1,
        enable_metrics=False,
        enable_nccl_nvls=False,
        enable_symm_mem=False,
        nnodes=1,
    )


class TestEngineEnvironment(unittest.TestCase):
    @patch("sglang.srt.entrypoints.engine.signal.signal")
    @patch("sglang.srt.entrypoints.engine.set_ulimit")
    def test_preserves_explicit_cuda_module_loading(self, _set_ulimit, _signal):
        with patch.dict(
            os.environ,
            {
                "CUDA_MODULE_LOADING": "EAGER",
                "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
            },
            clear=True,
        ):
            _set_envs_and_config(_server_args())
            self.assertEqual(os.environ["CUDA_MODULE_LOADING"], "EAGER")

    @patch("sglang.srt.entrypoints.engine.signal.signal")
    @patch("sglang.srt.entrypoints.engine.set_ulimit")
    def test_defaults_cuda_module_loading_to_auto(self, _set_ulimit, _signal):
        with patch.dict(
            os.environ,
            {"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1"},
            clear=True,
        ):
            _set_envs_and_config(_server_args())
            self.assertEqual(os.environ["CUDA_MODULE_LOADING"], "AUTO")


if __name__ == "__main__":
    unittest.main()
