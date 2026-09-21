"""Check the installed NVLink wheel without a GPU or a CUDA context."""

import argparse
import ctypes
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import platform
import re
import shutil
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
        and lock["transport"] == "nvlink",
        "Only linux/amd64, cu130, NVLink images are supported",
    )
    return lock


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
    for filename in ("_iaas_kernels_rdma.so", "RDMA_BUILD_INFO.json"):
        require(
            not (package / filename).exists(), "NVLink image contains an RDMA wheel"
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
    # Resolve host dependencies/symbols, including libdw/libelf, without calling
    # the native init function (which would query the GPU).
    ctypes.CDLL(str(library))
    require(not torch.cuda.is_initialized(), "Image smoke check initialized CUDA")
    print(json.dumps({"iaas_kernels": manifest, "sgl-deep-gemm": original.version}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="/opt/sglang/share/iaas-kernels/manifest.json"
    )
    check_image(parser.parse_args().manifest)
