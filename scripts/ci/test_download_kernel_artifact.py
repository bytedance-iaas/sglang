import base64
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "download_kernel_artifact", Path(__file__).with_name("download_kernel_artifact.py")
)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)

WHEEL = "sglang_kernel-0.4.6.post1+cu130-cp310-abi3-manylinux2014_x86_64.whl"


def make_wheel():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as wheel:
        wheel.writestr(
            "sglang_kernel-0.4.6.post1+cu130.dist-info/WHEEL", "Wheel-Version: 1.0\n"
        )
        wheel.writestr(
            "sglang_kernel-0.4.6.post1+cu130.dist-info/METADATA",
            "Name: sglang-kernel\n",
        )
        wheel.writestr("sgl_kernel/payload.bin", b"unique-kernel-payload")
    return buffer.getvalue()


class KernelArtifactTests(unittest.TestCase):
    def setUp(self):
        token = patch.dict(download.os.environ, {"GH_TOKEN": "test-token"})
        token.start()
        self.addCleanup(token.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        self.output.mkdir()

    def bundle(self, entries):
        archive = self.root / "artifact.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for name, data in entries:
                bundle.writestr(name, data)
        data = archive.read_bytes()
        metadata = {
            "id": 123,
            "name": "kernel",
            "expired": False,
            "size_in_bytes": len(data),
            "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
        }
        return archive, metadata

    def test_verified_wheel_is_published_without_extracting_member_paths(self):
        wheel = make_wheel()
        archive, metadata = self.bundle([("../../" + WHEEL, wheel)])
        result = download.verify_and_extract(archive, metadata, self.output)
        self.assertEqual(result, self.output / WHEEL)
        self.assertEqual(result.read_bytes(), wheel)
        self.assertEqual(list(self.output.iterdir()), [result])

    def test_partial_download_is_rejected_without_publishing(self):
        archive, metadata = self.bundle([(WHEEL, make_wheel())])
        archive.write_bytes(archive.read_bytes()[:100])
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            download.verify_and_extract(archive, metadata, self.output)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_wrong_digest_is_rejected(self):
        archive, metadata = self.bundle([(WHEEL, make_wheel())])
        metadata["digest"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            download.verify_and_extract(archive, metadata, self.output)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_missing_authoritative_digest_is_rejected(self):
        archive, metadata = self.bundle([(WHEEL, make_wheel())])
        metadata.pop("digest")
        with self.assertRaisesRegex(ValueError, "authoritative"):
            download.verify_and_extract(archive, metadata, self.output)

    def test_corrupt_inner_wheel_is_rejected_even_with_valid_artifact_digest(self):
        wheel = make_wheel().replace(
            b"unique-kernel-payload", b"broken-kernel-payload", 1
        )
        archive, metadata = self.bundle([(WHEEL, wheel)])
        with self.assertRaisesRegex(ValueError, "wheel ZIP CRC"):
            download.verify_and_extract(archive, metadata, self.output)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_ambiguous_wheels_are_rejected(self):
        archive, metadata = self.bundle(
            [(WHEEL, make_wheel()), ("nested/" + WHEEL, make_wheel())]
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            download.verify_and_extract(archive, metadata, self.output)

    def test_download_retries_partial_result_before_publishing(self):
        wheel = make_wheel()
        archive, metadata = self.bundle([(WHEEL, wheel)])
        data = archive.read_bytes()
        downloads = []

        def gh(command, **kwargs):
            self.assertEqual(command[0], "curl")
            self.assertNotIn("test-token", " ".join(command))
            if "stdout" not in kwargs:
                return SimpleNamespace(stdout=json.dumps({"artifacts": [metadata]}))
            downloads.append(command)
            self.assertFalse((self.output / WHEEL).exists())
            if len(downloads) == 1:
                kwargs["stdout"].write(data[:100])
            else:
                self.assertEqual(command[-3:-1], ["--continue-at", "100"])
                kwargs["stdout"].write(data[100:])
            self.assertIn("--max-time", command)
            self.assertIn("840", command)
            self.assertEqual(kwargs["timeout"], 870)
            return SimpleNamespace(returncode=0)

        argv = [
            "download",
            "--repository",
            "owner/repo",
            "--run-id",
            "12",
            "--name",
            "kernel",
            "--output",
            str(self.output),
        ]
        with patch("sys.argv", argv), patch.object(
            download.subprocess, "run", side_effect=gh
        ):
            download.main()
        self.assertEqual(len(downloads), 2)
        self.assertEqual((self.output / WHEEL).read_bytes(), wheel)

    def test_explicit_proxy_is_passed_without_exposing_token(self):
        with patch.dict(
            download.os.environ,
            {"KERNEL_ARTIFACT_HTTPS_PROXY": "http://proxy.example:3128"},
        ), patch.object(
            download.subprocess,
            "run",
            return_value=SimpleNamespace(stdout='{"artifacts": []}'),
        ) as run:
            download.github_api("repos/owner/repo/actions/runs/1/artifacts", text=True)
        command = run.call_args.args[0]
        self.assertIn("--proxy", command)
        self.assertIn("http://proxy.example:3128", command)
        self.assertNotIn("test-token", " ".join(command))

    def test_tos_wheel_is_downloaded_and_verified(self):
        wheel_data = make_wheel()
        wheel_sha256 = hashlib.sha256(wheel_data).hexdigest()
        tos_uri = f"tos://ai-infra/sglang-ci/kernel-wheels/1/2/1/kernel/{WHEEL}"

        def tos(command, **kwargs):
            self.assertEqual(command[:4], ["timeout", "20m", "tosutil", "cp"])
            self.assertEqual(command[4], tos_uri)
            self.assertIn("-e=tos-cn-beijing.volces.com", command)
            self.assertIn("-re=cn-beijing", command)
            config_arg = next(arg for arg in command if arg.startswith("-conf="))
            config = Path(config_arg.removeprefix("-conf="))
            self.assertEqual(config.stat().st_mode & 0o777, 0o600)
            self.assertEqual(config.read_bytes(), b"encrypted-config")
            self.assertNotIn("encrypted-config", " ".join(command))
            Path(command[5]).write_bytes(wheel_data)
            return SimpleNamespace(returncode=0)

        with patch.dict(
            download.os.environ,
            {
                "KERNEL_ARTIFACT_TOS_ENDPOINT": "tos-cn-beijing.volces.com",
                "KERNEL_ARTIFACT_TOS_REGION": "cn-beijing",
                "KERNEL_ARTIFACT_TOS_BUCKET": "ai-infra",
                "KERNEL_ARTIFACT_TOS_PREFIX": "sglang-ci/kernel-wheels",
                "TOSUTIL_CONFIG_B64": base64.b64encode(b"encrypted-config").decode(),
            },
        ), patch.object(download.subprocess, "run", side_effect=tos):
            result = download.recover_tos_wheel(tos_uri, wheel_sha256, self.output)
        self.assertEqual(result, self.output / WHEEL)
        self.assertEqual(result.read_bytes(), wheel_data)

    def test_tos_transport_requires_configuration_before_download(self):
        tos_uri = f"tos://ai-infra/sglang-ci/kernel-wheels/1/{WHEEL}"
        with patch.dict(
            download.os.environ,
            {
                "KERNEL_ARTIFACT_TOS_BUCKET": "ai-infra",
                "KERNEL_ARTIFACT_TOS_PREFIX": "sglang-ci/kernel-wheels",
                "KERNEL_ARTIFACT_TOS_ENDPOINT": "tos-cn-beijing.volces.com",
                "KERNEL_ARTIFACT_TOS_REGION": "cn-beijing",
            },
            clear=True,
        ), patch.object(download.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "configuration"):
                download.recover_tos_wheel(tos_uri, "0" * 64, self.output)
            run.assert_not_called()

    def test_tos_transport_rejects_unpinned_or_unverified_input(self):
        for tos_uri, wheel_sha256, error in (
            ("https://example.com/kernel.whl", "0" * 64, "pinned TOS"),
            ("tos://ai-infra/kernel.whl", "0" * 64, "pinned TOS"),
            (
                f"tos://other-bucket/sglang-ci/kernel-wheels/1/{WHEEL}",
                "0" * 64,
                "configured prefix",
            ),
            (
                f"tos://ai-infra/unrelated/1/{WHEEL}",
                "0" * 64,
                "configured prefix",
            ),
            (
                f"tos://ai-infra/sglang-ci/kernel-wheels/1/{WHEEL}",
                "missing",
                "authoritative SHA-256",
            ),
        ):
            with patch.object(download.subprocess, "run") as run:
                with patch.dict(
                    download.os.environ,
                    {
                        "KERNEL_ARTIFACT_TOS_BUCKET": "ai-infra",
                        "KERNEL_ARTIFACT_TOS_PREFIX": "sglang-ci/kernel-wheels",
                    },
                ):
                    with self.assertRaisesRegex(ValueError, error):
                        download.recover_tos_wheel(tos_uri, wheel_sha256, self.output)
                run.assert_not_called()

    def test_download_timeouts_are_bounded_and_publish_nothing(self):
        _, metadata = self.bundle([(WHEEL, make_wheel())])
        argv = [
            "download",
            "--repository",
            "owner/repo",
            "--run-id",
            "12",
            "--name",
            "kernel",
            "--output",
            str(self.output),
        ]
        with patch("sys.argv", argv), patch.object(
            download, "find_artifact", return_value=metadata
        ), patch.object(
            download.subprocess,
            "run",
            side_effect=download.subprocess.TimeoutExpired("gh", 300),
        ) as run:
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                download.main()
        self.assertEqual(run.call_count, 6)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_expired_artifact_is_rejected_before_download(self):
        _, metadata = self.bundle([(WHEEL, make_wheel())])
        metadata["expired"] = True
        with patch.object(
            download.subprocess,
            "run",
            return_value=SimpleNamespace(stdout=json.dumps({"artifacts": [metadata]})),
        ):
            with self.assertRaisesRegex(ValueError, "unexpired"):
                download.find_artifact("owner/repo", 12, "kernel")


if __name__ == "__main__":
    unittest.main()
