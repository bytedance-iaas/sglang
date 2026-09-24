"""Download a GitHub kernel artifact and reject partial or corrupt wheels."""

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path


def github_api(endpoint, **kwargs):
    # Supply credentials on stdin, never in process arguments or logs. curl
    # strips Authorization on cross-host redirects (do not use location-trusted).
    header = f"Authorization: Bearer {os.environ['GH_TOKEN']}\n"
    extra_args = kwargs.pop("extra_args", [])
    proxy = os.environ.get("KERNEL_ARTIFACT_HTTPS_PROXY", "")
    proxy_args = ["--proxy", proxy] if proxy else []
    return subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--connect-timeout",
            "20",
            "--speed-limit",
            "1024",
            "--speed-time",
            "30",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--header",
            "Accept: application/vnd.github+json",
            "--header",
            "@-",
            *proxy_args,
            *extra_args,
            f"{os.environ.get('GITHUB_API_URL', 'https://api.github.com')}/{endpoint}",
        ],
        input=header if kwargs.get("text") else header.encode(),
        **kwargs,
    )


def find_artifact(repository, run_id, name):
    matches = []
    page = 1
    while True:
        response = github_api(
            f"repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100&page={page}",
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        artifacts = json.loads(response.stdout)["artifacts"]
        matches.extend(a for a in artifacts if a["name"] == name)
        if len(artifacts) < 100:
            break
        page += 1
    if len(matches) != 1 or matches[0]["expired"]:
        raise ValueError("Expected exactly one unexpired kernel artifact")
    return matches[0]


def verify_and_extract(archive, artifact, output):
    """Publish one wheel only after archive identity and both CRC checks pass."""
    expected = artifact.get("digest", "")
    if not expected.startswith("sha256:") or len(expected) != 71:
        raise ValueError("Artifact has no authoritative SHA-256 digest")
    if archive.stat().st_size != artifact["size_in_bytes"]:
        raise ValueError("Artifact size mismatch")
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = "sha256:" + digest.hexdigest()
    if actual != expected:
        raise ValueError("Artifact SHA-256 mismatch")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Artifact ZIP CRC mismatch")
        wheels = [n for n in bundle.namelist() if n.endswith(".whl")]
        if len(wheels) != 1 or not Path(wheels[0]).name.startswith("sglang_kernel-"):
            raise ValueError("Expected exactly one sglang_kernel wheel")
        # Never extract artifact-controlled paths into the workspace.
        with tempfile.TemporaryDirectory(dir=output) as staging:
            wheel = Path(staging) / Path(wheels[0]).name
            with bundle.open(wheels[0]) as src, wheel.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            with zipfile.ZipFile(wheel) as package:
                if package.testzip() is not None:
                    raise ValueError("Kernel wheel ZIP CRC mismatch")
                names = package.namelist()
                for suffix in (".dist-info/WHEEL", ".dist-info/METADATA"):
                    if not any(n.endswith(suffix) for n in names):
                        raise ValueError(f"Kernel wheel missing {suffix}")
            destination = output / wheel.name
            os.replace(wheel, destination)
    print(f"Verified kernel artifact {artifact['id']}: {actual}")
    return destination


def verify_wheel(wheel, expected_sha256):
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Kernel wheel has no authoritative SHA-256")
    if not wheel.name.startswith("sglang_kernel-") or wheel.suffix != ".whl":
        raise ValueError("Expected one sglang_kernel wheel")
    digest = hashlib.sha256()
    with wheel.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError("Kernel wheel SHA-256 mismatch")
    with zipfile.ZipFile(wheel) as package:
        if package.testzip() is not None:
            raise ValueError("Kernel wheel ZIP CRC mismatch")
        names = package.namelist()
        for suffix in (".dist-info/WHEEL", ".dist-info/METADATA"):
            if not any(name.endswith(suffix) for name in names):
                raise ValueError(f"Kernel wheel missing {suffix}")


def recover_tos_wheel(tos_uri, expected_sha256, output):
    match = re.fullmatch(
        r"tos://([a-z0-9][a-z0-9.-]{1,61}[a-z0-9])/"
        r"([0-9A-Za-z._/-]+/sglang_kernel-[0-9A-Za-z.+_-]+\.whl)",
        tos_uri,
    )
    if match is None:
        raise ValueError("Require a pinned TOS sglang_kernel wheel URI")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Kernel wheel has no authoritative SHA-256")
    expected_bucket = os.environ.get("KERNEL_ARTIFACT_TOS_BUCKET", "")
    expected_prefix = os.environ.get("KERNEL_ARTIFACT_TOS_PREFIX", "").strip("/")
    if not expected_bucket or not expected_prefix:
        raise ValueError("TOS bucket and prefix are required")
    if match.group(1) != expected_bucket or not match.group(2).startswith(
        expected_prefix + "/"
    ):
        raise ValueError("TOS kernel wheel URI is outside the configured prefix")
    endpoint = os.environ.get("KERNEL_ARTIFACT_TOS_ENDPOINT", "")
    region = os.environ.get("KERNEL_ARTIFACT_TOS_REGION", "")
    if not endpoint or not region:
        raise ValueError("TOS endpoint and region are required")
    config_b64 = os.environ.get("TOSUTIL_CONFIG_B64", "")
    if not config_b64:
        raise ValueError("TOS configuration is required")
    with tempfile.TemporaryDirectory(dir=output) as staging:
        config = Path(staging) / "tosutil.conf"
        config.write_bytes(base64.b64decode(config_b64, validate=True))
        config.chmod(0o600)
        wheel = Path(staging) / Path(match.group(2)).name
        subprocess.run(
            [
                "timeout",
                "20m",
                "tosutil",
                "cp",
                tos_uri,
                str(wheel),
                f"-e={endpoint}",
                f"-re={region}",
                "-p=8",
                "-threshold=52428800",
                "-ps=16777216",
                "-vchecksum",
                "-f",
                f"-conf={config}",
            ],
            check=True,
            timeout=1230,
        )
        verify_wheel(wheel, expected_sha256)
        destination = output / wheel.name
        os.replace(wheel, destination)
    print(f"Verified TOS kernel wheel sha256:{expected_sha256}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tos-uri", default="")
    parser.add_argument("--wheel-sha256", default="")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob("*.whl")):
        raise ValueError("Refusing to reuse existing kernel wheels")
    if args.tos_uri:
        recover_tos_wheel(args.tos_uri, args.wheel_sha256, args.output)
        return
    if args.wheel_sha256:
        raise ValueError("--wheel-sha256 requires --tos-uri")
    artifact = find_artifact(args.repository, args.run_id, args.name)
    print(
        "Kernel artifact transport: "
        + (
            "explicit HTTPS proxy"
            if os.environ.get("KERNEL_ARTIFACT_HTTPS_PROXY")
            else "runner default"
        ),
        flush=True,
    )
    with tempfile.TemporaryDirectory(dir=args.output) as staging:
        archive = Path(staging) / "artifact.zip"
        for attempt in range(1, 7):
            started = time.monotonic()
            received_before = archive.stat().st_size if archive.exists() else 0
            try:
                try:
                    with archive.open("ab") as stream:
                        github_api(
                            f"repos/{args.repository}/actions/artifacts/{artifact['id']}/zip",
                            stdout=stream,
                            stderr=subprocess.PIPE,
                            check=True,
                            timeout=870,
                            extra_args=(
                                ["--max-time", "840"]
                                + (
                                    ["--continue-at", str(received_before)]
                                    if received_before
                                    else []
                                )
                            ),
                        )
                finally:
                    received = archive.stat().st_size
                verify_and_extract(archive, artifact, args.output)
                return
            except (subprocess.SubprocessError, ValueError, zipfile.BadZipFile) as exc:
                # Do not emit signed URLs or response bodies from download errors.
                print(
                    f"Kernel artifact attempt {attempt}/6 failed: {type(exc).__name__}; "
                    f"curl_exit={getattr(exc, 'returncode', None)}; "
                    f"bytes={received}/{artifact['size_in_bytes']}; "
                    f"resumed_from={received_before}; "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
                if received > artifact["size_in_bytes"]:
                    archive.unlink()
                if attempt == 6:
                    raise RuntimeError(
                        "Kernel artifact download/verification failed"
                    ) from None


if __name__ == "__main__":
    main()
