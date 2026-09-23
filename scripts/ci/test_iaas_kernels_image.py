"""CPU regressions for image provenance/ABI and plugin failures."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import check_iaas_kernels_image as checks


class TestIaasKernelsImage(unittest.TestCase):
    def setUp(self):
        self.lock_path = (
            Path(__file__).resolve().parents[2] / "docker/iaas-kernels.lock.json"
        )
        self.lock = checks.load_lock(self.lock_path)
        self.info = dict(
            torch="2.13.0+cu130",
            cuda="13.0",
            cxx11_abi=True,
            checkout=self.lock["revision"],
        )

    def test_matching_abi(self):
        checks.check_abi(self.lock, self.info, "2.13.0+cu130", "13.0", True)

    def test_reject_incompatible_image(self):
        for torch, cuda in [("2.15.0+cu134", "13.4"), ("2.13.0+cu129", "12.9")]:
            with self.subTest(torch=torch), self.assertRaisesRegex(
                RuntimeError, "Unsupported image ABI"
            ):
                checks.check_abi(self.lock, self.info, torch, cuda, True)

    def test_reject_stale_wheel(self):
        for key, value in [
            ("torch", "2.13.0a0+git123"),
            ("cuda", "12.9"),
            ("cxx11_abi", False),
            ("checkout", "f" * 40),
        ]:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                checks.check_abi(
                    self.lock,
                    dict(self.info, **{key: value}),
                    "2.13.0+cu130",
                    "13.0",
                    True,
                )

    def test_reject_moving_refs_and_unvalidated_variants(self):
        for key, value in [
            ("revision", "main"),
            ("repository", "https://example.com/kernels.git"),
            ("platform", "linux/arm64"),
            ("cuda", "13.4"),
            ("transport", "rdma"),
        ]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                lock_path = Path(root) / "lock.json"
                lock_path.write_text(json.dumps(dict(self.lock, **{key: value})))
                with self.assertRaises(RuntimeError):
                    checks.load_lock(lock_path)

    def plugin_modules(self, *, registered=True, initialized=False):
        factory = Mock(__module__="iaas_kernels.integrations.sglang")
        fused_moe = types.SimpleNamespace(
            _factories={("megamoe", (9, 0)): factory} if registered else {},
            _worker_initializers={"iaas_kernels": Mock()},
        )
        modules = {
            "torch": types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_initialized=lambda: initialized)
            ),
            "sglang.srt.plugins": types.SimpleNamespace(
                fused_moe=fused_moe, load_plugins=Mock()
            ),
            "iaas_kernels": types.SimpleNamespace(
                _native=types.SimpleNamespace(_module=None)
            ),
        }
        return patch.dict(sys.modules, modules)

    def test_registration_without_cuda(self):
        with self.plugin_modules():
            checks.check_plugin()

    def test_swallowed_plugin_exception_is_a_smoke_failure(self):
        # load_plugins() can return successfully after logging a load exception.
        with self.plugin_modules(registered=False), self.assertRaisesRegex(
            RuntimeError, "factory was not registered"
        ):
            checks.check_plugin()

    def test_eager_cuda_initialization_is_a_smoke_failure(self):
        with self.plugin_modules(initialized=True), self.assertRaisesRegex(
            RuntimeError, "CUDA was initialized"
        ):
            checks.check_plugin()


if __name__ == "__main__":
    unittest.main()
