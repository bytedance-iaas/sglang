"""CPU regressions for image provenance/ABI and plugin failures."""

import hashlib
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
            with (
                self.subTest(torch=torch),
                self.assertRaisesRegex(RuntimeError, "Unsupported image ABI"),
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
            ("transports", ["nvlink"]),
        ]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                lock_path = Path(root) / "lock.json"
                lock_path.write_text(json.dumps(dict(self.lock, **{key: value})))
                with self.assertRaises(RuntimeError):
                    checks.load_lock(lock_path)

    def test_reject_unpinned_or_incompatible_nvshmem(self):
        for key, value in [
            ("api_version", 2),
            ("nvshmem_version", "3.5.0"),
            ("nvshmem_revision", "main"),
            ("nvshmem_repository", "https://example.com/nvshmem.git"),
        ]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                path = Path(root) / "lock.json"
                path.write_text(
                    json.dumps(
                        dict(self.lock, rdma=dict(self.lock["rdma"], **{key: value}))
                    )
                )
                with self.assertRaisesRegex(RuntimeError, "RDMA requires"):
                    checks.load_lock(path)

    def test_rdma_must_match_single_node_wheel_and_api(self):
        info = dict(self.info, source_tree_sha256="a" * 64)
        rdma = dict(info, rdma_api_version=3, nvshmem="3.4.5", topology="runtime")
        checks.check_rdma_info(self.lock, info, rdma)
        for key, value in [
            ("rdma_api_version", 2),
            ("nvshmem", "3.5.0"),
            ("topology", "fixed"),
            ("source_tree_sha256", "b" * 64),
            ("checkout", "f" * 40),
            ("cxx11_abi", False),
        ]:
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(RuntimeError, "RDMA wheel"),
            ):
                checks.check_rdma_info(self.lock, info, dict(rdma, **{key: value}))

    def test_lto_missing_corrupt_or_nonexecutable_artifacts_fail(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            files = {}
            for name in (
                "link_object",
                "libnvshmem_device_lto.a",
                "License.nvshmem.txt",
            ):
                path = root / name
                path.write_bytes(name.encode())
                path.chmod(0o755)
                files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = dict(
                files=files,
                cuda="13.0",
                cpu_link_probe="passed",
                nvshmem_revision=self.lock["rdma"]["nvshmem_revision"],
                nvshmem_version="3.4.5",
            )
            checks.check_lto(root, manifest, self.lock)
            with self.assertRaisesRegex(RuntimeError, "LTO source/toolchain"):
                checks.check_lto(root, dict(manifest, cuda="13.4"), self.lock)
            with self.assertRaisesRegex(RuntimeError, "Invalid LTO artifact path"):
                checks.check_lto(
                    root,
                    dict(manifest, files=dict(files, **{"../escape": "x"})),
                    self.lock,
                )
            path = root / "link_object"
            path.chmod(0o644)
            with self.assertRaisesRegex(RuntimeError, "not executable"):
                checks.check_lto(root, manifest, self.lock)
            path.chmod(0o755)
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "LTO hash mismatch"):
                checks.check_lto(root, manifest, self.lock)
            path.unlink()
            with self.assertRaisesRegex(RuntimeError, "Missing LTO artifact"):
                checks.check_lto(root, manifest, self.lock)

    def test_nvshmem_headers_must_match_package(self):
        with tempfile.TemporaryDirectory() as root:
            home = Path(root)
            header = home / "include/non_abi/nvshmem_version.h"
            header.parent.mkdir(parents=True)
            header.write_text(
                "#define NVSHMEM_VENDOR_MAJOR_VERSION 3\n#define NVSHMEM_VENDOR_MINOR_VERSION 4\n#define NVSHMEM_VENDOR_PATCH_VERSION 5\n"
            )
            for name in (
                "include/nvshmem.h",
                "lib/libnvshmem_device.a",
                "lib/libnvshmem_host.so.3",
            ):
                path = home / name
                path.parent.mkdir(exist_ok=True)
                path.touch()
            dist = Mock(version="3.4.5", locate_file=Mock(return_value=home))
            with (
                patch.object(checks.metadata, "distribution", return_value=dist),
                patch.dict(checks.os.environ, {}, clear=True),
            ):
                self.assertEqual(checks.nvshmem_home(self.lock), home.resolve())
                dist.version = "3.5.0"
                with self.assertRaisesRegex(RuntimeError, "package mismatch"):
                    checks.nvshmem_home(self.lock)
                dist.version = "3.4.5"
                header.write_text(
                    header.read_text().replace("PATCH_VERSION 5", "PATCH_VERSION 4")
                )
                with self.assertRaisesRegex(RuntimeError, "headers mismatch"):
                    checks.nvshmem_home(self.lock)

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
            "iaas_kernels.rdma": types.SimpleNamespace(
                _native=types.SimpleNamespace(_module=None)
            ),
        }
        return patch.dict(sys.modules, modules)

    def test_registration_without_cuda(self):
        with self.plugin_modules():
            checks.check_plugin()

    def test_swallowed_plugin_exception_is_a_smoke_failure(self):
        # load_plugins() can return successfully after logging a load exception.
        with (
            self.plugin_modules(registered=False),
            self.assertRaisesRegex(RuntimeError, "factory was not registered"),
        ):
            checks.check_plugin()

    def test_eager_cuda_initialization_is_a_smoke_failure(self):
        with (
            self.plugin_modules(initialized=True),
            self.assertRaisesRegex(RuntimeError, "CUDA was initialized"),
        ):
            checks.check_plugin()


if __name__ == "__main__":
    unittest.main()
