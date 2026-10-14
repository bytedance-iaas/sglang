"""Check the installed NVLink/RDMA wheel and LTO tools without a CUDA context."""

import argparse
import ctypes
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_lock(path):
    lock = json.loads(Path(path).read_text())
    require(
        lock["repository"] == "https://github.com/bytedance-iaas/iaas-kernels.git"
        and re.fullmatch(r"[0-9a-f]{40}", lock["revision"]),
        "iaas-kernels source must use the approved repository and a full commit SHA",
    )
    require(
        lock["platform"] == "linux/amd64"
        and lock["cuda"] == "13.0"
        and lock.get("transports") == ["nvlink", "rdma"],
        "Only linux/amd64, cu130, NVLink/RDMA images are supported",
    )
    rdma = lock.get("rdma", {})
    require(
        rdma.get("api_version") == 3
        and rdma.get("nvshmem_version") == "3.4.5"
        and rdma.get("nvshmem_repository") == "https://github.com/NVIDIA/nvshmem.git"
        and re.fullmatch(r"[0-9a-f]{40}", rdma.get("nvshmem_revision", "")),
        "RDMA requires API v3 and pinned NVSHMEM 3.4.5 sources",
    )
    return lock


def check_rdma_info(lock, info, rdma_info):
    require(
        all(rdma_info.get(key) == value for key, value in info.items())
        and rdma_info.get("rdma_api_version") == lock["rdma"]["api_version"]
        and rdma_info.get("nvshmem") == lock["rdma"]["nvshmem_version"]
        and rdma_info.get("topology") == "runtime",
        "RDMA wheel source/ABI/API/NVSHMEM mismatch",
    )


def nvshmem_home(lock):
    dist = metadata.distribution("nvidia-nvshmem-cu13")
    require(dist.version == lock["rdma"]["nvshmem_version"], "NVSHMEM package mismatch")
    home = Path(dist.locate_file("nvidia/nvshmem")).resolve()
    override = os.environ.get("IAAS_KERNELS_RDMA_NVSHMEM_HOME")
    require(
        not override or Path(override).resolve() == home, "Unexpected NVSHMEM override"
    )
    header = home / "include/non_abi/nvshmem_version.h"
    version = ".".join(
        re.search(
            rf"#define NVSHMEM_VENDOR_{part}_VERSION\s+(\d+)", header.read_text()
        )[1]
        for part in ("MAJOR", "MINOR", "PATCH")
    )
    require(version == lock["rdma"]["nvshmem_version"], "NVSHMEM headers mismatch")
    for name in (
        "include/nvshmem.h",
        "lib/libnvshmem_device.a",
        "lib/libnvshmem_host.so.3",
    ):
        require((home / name).is_file(), f"Missing NVSHMEM dependency: {name}")
    return home


def check_lto(root, manifest, lock):
    require(
        manifest["nvshmem_revision"] == lock["rdma"]["nvshmem_revision"]
        and manifest["nvshmem_version"] == lock["rdma"]["nvshmem_version"]
        and manifest["cuda"] == lock["cuda"],
        "LTO source/toolchain mismatch",
    )
    require(
        manifest.get("cpu_link_probe") == "passed", "LTO CPU link probe did not pass"
    )
    require(
        {"link_object", "libnvshmem_device_lto.a", "License.nvshmem.txt"}
        <= manifest["files"].keys(),
        "Missing LTO artifacts in manifest",
    )
    for name, digest in manifest["files"].items():
        require(Path(name).name == name, "Invalid LTO artifact path")
        path = root / name
        require(path.is_file(), f"Missing LTO artifact: {name}")
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == digest,
            f"LTO hash mismatch: {name}",
        )
    require(os.access(root / "link_object", os.X_OK), "LTO linker is not executable")


def check_abi(lock, info, torch_version, cuda_version, cxx11_abi):
    require(
        torch_version.split("+")[0] == lock["torch"] and cuda_version == lock["cuda"],
        f"Unsupported image ABI: Torch {torch_version}, CUDA {cuda_version}",
    )
    require(
        info["torch"] == torch_version
        and info["cuda"] == cuda_version
        and info["cxx11_abi"] == cxx11_abi,
        "iaas-kernels wheel build/runtime ABI mismatch",
    )
    require(info["checkout"] == lock["revision"], "iaas-kernels checkout mismatch")


def check_environment(lock):
    import torch

    require(
        platform.system() == "Linux" and platform.machine() == "x86_64",
        "iaas-kernels image requires Linux x86_64",
    )
    require(
        ".".join(platform.python_version_tuple()[:2]) == lock["python"],
        "iaas-kernels image Python version mismatch",
    )
    require(
        metadata.version("apache-tvm-ffi") == lock["apache-tvm-ffi"],
        "iaas-kernels image apache-tvm-ffi version mismatch",
    )
    check_abi(
        lock,
        dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cxx11_abi=torch.compiled_with_cxx11_abi(),
            checkout=lock["revision"],
        ),
        torch.__version__,
        torch.version.cuda,
        torch.compiled_with_cxx11_abi(),
    )
    nvshmem_home(lock)


def check_plugin():
    import torch

    from sglang.srt.plugins import fused_moe, load_plugins

    require(not torch.cuda.is_initialized(), "CUDA was initialized before plugin load")
    load_plugins()
    factory = fused_moe._factories.get(("megamoe", (9, 0)))
    require(
        factory is not None
        and factory.__module__ == "iaas_kernels.integrations.sglang",
        "iaas-kernels SM90 factory was not registered (check plugin load errors/whitelist)",
    )
    require(
        "iaas_kernels" in fused_moe._worker_initializers,
        "iaas-kernels worker initializer was not registered",
    )
    require(not torch.cuda.is_initialized(), "Plugin registration initialized CUDA")
    from iaas_kernels import _native

    require(_native._module is None, "Plugin registration loaded the CUDA backend")
    from iaas_kernels.rdma import _native as rdma_native

    require(rdma_native._module is None, "Plugin registration loaded the RDMA backend")


def check_image(manifest_path):
    import torch

    manifest = json.loads(Path(manifest_path).read_text())
    lock = manifest["lock"]
    # Revalidate the lock saved with the image, not a moving external checkout.
    lock_path = Path(manifest_path).with_name("iaas-kernels.lock.json")
    require(load_lock(lock_path) == lock, "Image manifest/lock mismatch")
    check_environment(lock)
    dist = metadata.distribution("iaas-kernels")
    require(dist.version == lock["version"], "iaas-kernels package version mismatch")
    package = Path(dist.locate_file("iaas_kernels"))
    library = package / "_iaas_kernels.so"
    info = json.loads((package / "BUILD_INFO.json").read_text())
    require(info == manifest["build_info"], "Installed BUILD_INFO differs from wheel")
    check_abi(
        lock,
        info,
        torch.__version__,
        torch.version.cuda,
        torch.compiled_with_cxx11_abi(),
    )
    require(
        hashlib.sha256(library.read_bytes()).hexdigest() == manifest["native_sha256"],
        "Installed native library differs from wheel",
    )
    for header in (
        "include/iaas_kernels",
        "include/cutlass/cutlass.h",
        "include/cute/tensor.hpp",
    ):
        require((package / header).exists(), f"Missing JIT headers: {header}")
    rdma_library = package / "_iaas_kernels_rdma.so"
    rdma_info = json.loads((package / "RDMA_BUILD_INFO.json").read_text())
    require(
        rdma_info == manifest["rdma_build_info"],
        "Installed RDMA_BUILD_INFO differs from wheel",
    )
    check_rdma_info(lock, info, rdma_info)
    require(
        hashlib.sha256(rdma_library.read_bytes()).hexdigest()
        == manifest["rdma_native_sha256"],
        "Installed RDMA native library differs from wheel",
    )
    require(
        (package / "include/iaas_kernels/rdma").is_dir(), "Missing RDMA JIT headers"
    )
    lto_root = Path(manifest_path).parent / "lto"
    require(
        (lto_root.parent / "rdma-fp4.env").is_file(), "Missing FP4 RDMA startup profile"
    )
    check_lto(lto_root, manifest["lto"], lock)
    # Exercises the installed executable and its dynamic dependencies on CPU.
    subprocess.run(
        [str(lto_root / "link_object"), "--version"],
        check=True,
        stdout=subprocess.PIPE,
    )
    require(shutil.which("nvcc") is not None, "Runtime JIT requires nvcc")
    require(shutil.which("ninja") is not None, "Runtime JIT requires ninja")
    eps = [
        ep
        for ep in dist.entry_points
        if ep.group == "sglang.srt.plugins" and ep.name == "iaas_kernels"
    ]
    require(
        len(eps) == 1 and eps[0].value == "iaas_kernels.integrations.sglang:register",
        "Missing or unexpected iaas-kernels entry point",
    )
    deep_gemm = importlib.util.find_spec("deep_gemm")
    original = metadata.distribution("sgl-deep-gemm")
    require(
        deep_gemm is not None
        and Path(deep_gemm.origin).resolve()
        == Path(original.locate_file("deep_gemm/__init__.py")).resolve(),
        "deep_gemm is missing or shadowed",
    )
    check_plugin()
    # Match load_native(): importing tvm_ffi loads its host runtime library.
    # A bare CDLL otherwise cannot resolve libtvm_ffi.so in site-packages.
    import tvm_ffi  # noqa: F401

    # Resolve host dependencies/symbols, including libdw/libelf, without calling
    # the native init function (which would query the GPU).
    ctypes.CDLL(str(library))
    home = nvshmem_home(lock)
    ctypes.CDLL(str(home / "lib/libnvshmem_host.so.3"), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(rdma_library))
    from torch.distributed import _symmetric_memory

    require(
        _symmetric_memory.is_nvshmem_available(),
        "Torch lacks NVSHMEM symmetric memory support",
    )
    require(not torch.cuda.is_initialized(), "Image smoke check initialized CUDA")
    print(json.dumps({"iaas_kernels": manifest, "sgl-deep-gemm": original.version}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="/opt/sglang/share/iaas-kernels/manifest.json"
    )
    check_image(parser.parse_args().manifest)
