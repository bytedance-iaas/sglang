"""Download a GitHub kernel artifact and reject partial or corrupt wheels."""

import argparse
import hashlib
import json
import os
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob("*.whl")):
        raise ValueError("Refusing to reuse existing kernel wheels")
    artifact = find_artifact(args.repository, args.run_id, args.name)
    print(
        "Kernel artifact transport: "
        + ("explicit HTTPS proxy" if os.environ.get("KERNEL_ARTIFACT_HTTPS_PROXY") else "runner default"),
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
                            timeout=900,
                            extra_args=(
                                ["--continue-at", str(received_before)]
                                if received_before
                                else []
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
