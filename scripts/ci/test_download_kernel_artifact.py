import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

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
            if "stdout" not in kwargs:
                return SimpleNamespace(stdout=json.dumps({"artifacts": [metadata]}))
            downloads.append(command)
            self.assertFalse((self.output / WHEEL).exists())
            kwargs["stdout"].write(data[:100] if len(downloads) == 1 else data)
            self.assertEqual(kwargs["timeout"], 300)
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
        self.assertEqual(run.call_count, 3)
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
