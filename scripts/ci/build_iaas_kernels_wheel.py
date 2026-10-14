"""Build the pinned NVLink wheel with the consuming image's Torch/CUDA ABI."""

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path

from check_iaas_kernels_image import check_environment, load_lock, require


def build(lock_path, source, output, token_file):
    lock = load_lock(lock_path)
    check_environment(lock)
    # A fresh checkout prevents local source/binary leftovers entering the wheel.
    source.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=True)
    require(not list(output.glob("*.whl")), "Wheel output must be empty")
    subprocess.run(["git", "init", str(source)], check=True)
    git = ["git", "-C", str(source)]
    subprocess.run(git + ["remote", "add", "origin", lock["repository"]], check=True)
    git_env = os.environ.copy()
    git_env["GIT_TERMINAL_PROMPT"] = "0"
    if token_file.is_file():
        token = token_file.read_text().strip()
        require(bool(token), "Empty iaas-kernels read token")
        # Environment-only, URL-scoped auth: no token in argv, git config, wheel,
        # build args, or image layers. BuildKit supplies the temporary secret.
        auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        git_env.update(
            GIT_CONFIG_COUNT="1",
            GIT_CONFIG_KEY_0=f"http.{lock['repository']}.extraheader",
            GIT_CONFIG_VALUE_0=f"AUTHORIZATION: basic {auth}",
        )
    subprocess.run(
        git + ["fetch", "--depth=1", "origin", lock["revision"]],
        env=git_env,
        check=True,
    )
    subprocess.run(git + ["checkout", "--detach", "FETCH_HEAD"], check=True)
    head = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    require(head == lock["revision"], "Fetched iaas-kernels commit mismatch")
    project = tomllib.loads((source / "pyproject.toml").read_text())
    require(project["project"]["version"] == lock["version"], "Source version mismatch")
    env = dict(os.environ, IAAS_KERNELS_BUILD_RDMA="0", TVM_FFI_CUDA_ARCH_LIST="9.0a")
    subprocess.run(["bash", "build_sgl_deep_gemm.sh"], cwd=source, env=env, check=True)
    submodules = subprocess.check_output(
        git + ["submodule", "status", "--recursive"], text=True
    ).splitlines()
    require(
        bool(submodules) and all(line.startswith(" ") for line in submodules),
        "Missing or modified iaas-kernels submodules",
    )
    wheels = list((source / "dist").glob("iaas_kernels-*.whl"))
    require(len(wheels) == 1, "Expected exactly one iaas-kernels wheel")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        info = json.loads(archive.read("iaas_kernels/BUILD_INFO.json"))
        native = archive.read("iaas_kernels/_iaas_kernels.so")
        require(
            not any("_rdma.so" in name for name in archive.namelist()),
            "Unexpected RDMA extension in NVLink wheel",
        )
    manifest = dict(
        lock=lock,
        build_info=info,
        submodules=submodules,
        wheel=wheel.name,
        wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        native_sha256=hashlib.sha256(native).hexdigest(),
    )
    shutil.copy2(wheel, output / wheel.name)
    shutil.copy2(lock_path, output / "iaas-kernels.lock.json")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--source", default="/build/iaas-kernels", type=Path)
    parser.add_argument("--output", default="/wheels", type=Path)
    parser.add_argument(
        "--token-file", default="/run/secrets/iaas_kernels_token", type=Path
    )
    args = parser.parse_args()
    build(args.lock, args.source, args.output, args.token_file)
