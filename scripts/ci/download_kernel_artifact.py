"""Download a GitHub kernel artifact and reject partial or corrupt wheels."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile


def github_api(endpoint, **kwargs):
    # Supply credentials on stdin, never in process arguments or logs. curl
    # strips Authorization on cross-host redirects (do not use location-trusted).
    header = f"Authorization: Bearer {os.environ['GH_TOKEN']}\n"
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


def recover_registry_wheel(layer_ref, expected_wheel_sha256, output):
    """Recover the exact wheel from a pinned, anonymously readable Harbor layer.

    The expected wheel hash must come from independent artifact verification.
    This is an alternate transport for the wheel, not an image build base.
    """
    match = re.fullmatch(
        r"([a-z0-9.-]+\.cr\.volces\.com)/([a-z0-9._/-]+)@sha256:([0-9a-f]{64})",
        layer_ref,
    )
    if match is None or not re.fullmatch(r"[0-9a-f]{64}", expected_wheel_sha256):
        raise ValueError("Require a pinned Volcengine layer and verified wheel SHA-256")
    host, repository, layer_sha256 = match.groups()
    query = urllib.parse.urlencode(
        {"service": "harbor-registry", "scope": f"repository:{repository}:pull"}
    )
    # Anonymous, pull-only token. No host credentials or new permissions needed.
    with urllib.request.urlopen(
        f"https://{host}/service/token?{query}", timeout=30
    ) as response:
        token = json.load(response)["token"]
    with tempfile.TemporaryDirectory(dir=output) as staging:
        archive = Path(staging) / "layer.tar.gz"
        with archive.open("wb") as stream:
            result = subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--proto",
                    "=https",
                    "--proto-redir",
                    "=https",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    "900",
                    "--header",
                    "@-",
                    f"https://{host}/v2/{repository}/blobs/sha256:{layer_sha256}",
                ],
                input=f"Authorization: Bearer {token}\n".encode(),
                stdout=stream,
                stderr=subprocess.PIPE,
                timeout=910,
            )
        if result.returncode:
            raise RuntimeError(
                f"Registry layer download failed: curl_exit={result.returncode}; "
                f"bytes={archive.stat().st_size}"
            )
        return verify_registry_layer(
            archive, layer_sha256, expected_wheel_sha256, output
        )


def verify_registry_layer(
    archive, expected_layer_sha256, expected_wheel_sha256, output
):
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_layer_sha256:
        raise ValueError("Registry layer SHA-256 mismatch")
    with tarfile.open(archive, "r:gz") as bundle:
        wheels = [m for m in bundle.getmembers() if m.name.endswith(".whl")]
        if (
            len(wheels) != 1
            or not wheels[0].isfile()
            or not Path(wheels[0].name).name.startswith("sglang_kernel-")
        ):
            raise ValueError("Expected exactly one regular sglang_kernel wheel")
        with tempfile.TemporaryDirectory(dir=output) as staging:
            wheel = Path(staging) / Path(wheels[0].name).name
            digest = hashlib.sha256()
            with bundle.extractfile(wheels[0]) as src, wheel.open("wb") as dst:
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    digest.update(chunk)
                    dst.write(chunk)
            if digest.hexdigest() != expected_wheel_sha256:
                raise ValueError("Kernel wheel SHA-256 mismatch")
            with zipfile.ZipFile(wheel) as package:
                if package.testzip() is not None:
                    raise ValueError("Kernel wheel ZIP CRC mismatch")
                for suffix in (".dist-info/WHEEL", ".dist-info/METADATA"):
                    if not any(n.endswith(suffix) for n in package.namelist()):
                        raise ValueError(f"Kernel wheel missing {suffix}")
            destination = output / wheel.name
            os.replace(wheel, destination)
    print(
        f"Verified registry layer sha256:{expected_layer_sha256}; "
        f"kernel wheel sha256:{expected_wheel_sha256}",
        flush=True,
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--registry-layer", default="")
    parser.add_argument("--wheel-sha256", default="")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob("*.whl")):
        raise ValueError("Refusing to reuse existing kernel wheels")
    if args.registry_layer:
        recover_registry_wheel(args.registry_layer, args.wheel_sha256, args.output)
        return
    if args.wheel_sha256:
        raise ValueError("--wheel-sha256 requires --registry-layer")
    artifact = find_artifact(args.repository, args.run_id, args.name)
    for attempt in range(1, 4):
        started = time.monotonic()
        received = 0
        try:
            with tempfile.TemporaryDirectory(dir=args.output) as staging:
                archive = Path(staging) / "artifact.zip"
                try:
                    with archive.open("wb") as stream:
                        github_api(
                            f"repos/{args.repository}/actions/artifacts/{artifact['id']}/zip",
                            stdout=stream,
                            stderr=subprocess.PIPE,
                            check=True,
                            timeout=300,
                        )
                finally:
                    received = archive.stat().st_size
                verify_and_extract(archive, artifact, args.output)
            return
        except (subprocess.SubprocessError, ValueError, zipfile.BadZipFile) as exc:
            # Do not emit signed URLs or response bodies from download errors.
            print(
                f"Kernel artifact attempt {attempt}/3 failed: {type(exc).__name__}; "
                f"curl_exit={getattr(exc, 'returncode', None)}; "
                f"bytes={received}/{artifact['size_in_bytes']}; "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
            if attempt == 3:
                raise RuntimeError(
                    "Kernel artifact download/verification failed"
                ) from None


if __name__ == "__main__":
    main()
