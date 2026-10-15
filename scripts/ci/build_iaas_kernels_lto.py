"""Build the pinned SM90 NVSHMEM device LTO archive; no GPU is needed."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from check_iaas_kernels_image import load_lock, nvshmem_home, require


def build_lto(lock, source, output):
    home = nvshmem_home(lock)
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    commands = []

    def run(command):
        command = list(map(str, command))
        commands.append(command)
        subprocess.run(command, check=True)

    source.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    git = ["git", "-C", source]
    run(["git", "init", source])
    run(git + ["remote", "add", "origin", lock["rdma"]["nvshmem_repository"]])
    run(git + ["fetch", "--depth=1", "origin", lock["rdma"]["nvshmem_revision"]])
    run(git + ["checkout", "--detach", "FETCH_HEAD"])
    head = subprocess.check_output(
        list(map(str, git + ["rev-parse", "HEAD"])), text=True
    ).strip()
    require(head == lock["rdma"]["nvshmem_revision"], "NVSHMEM source mismatch")
    # Build device IR against the wheel's generated headers/options. These
    # private ABI definitions must match the pinned source and host library.
    for name in (
        "device_host/nvshmem_common.cuh",
        "device_host_transport/nvshmem_common_ibgda.h",
    ):
        require(
            (source / "src/include" / name).read_bytes()
            == (home / "include" / name).read_bytes(),
            f"NVSHMEM source/wheel header mismatch: {name}",
        )
    transfer = "non_abi/device/pt-to-pt/transfer_device.cuh"
    require(
        (source / "src/include" / (transfer + ".in")).read_bytes()
        == (home / "include" / transfer).read_bytes(),
        "NVSHMEM transfer source mismatch",
    )
    build = source / "lto-build"
    build.mkdir()
    shutil.copy2(source / "src/device/init/init_device.cu", build / "init_device.cu")
    shutil.copy2(home / "include" / transfer, build / "transfer_device.cu")
    flags = [
        "-dc",
        "-rdc=true",
        "-gencode",
        "arch=compute_90a,code=lto_90a",
        "-std=c++17",
        "-O3",
        "-lineinfo",
        "--maxrregcount=32",
        "-Xcompiler=-fPIC",
        "-DNVSHMEM_X86_64",
        "-D__STDC_LIMIT_MACROS",
        "-D__STDC_CONSTANT_MACROS",
        f"-I{home}/include",
        f"-I{source}/src/include",
    ]
    for unit in ("init_device", "transfer_device"):
        run(
            [cuda / "bin/nvcc", build / f"{unit}.cu", *flags, "-o", build / f"{unit}.o"]
        )
    archive = output / "libnvshmem_device_lto.a"
    run(["ar", "rcsD", archive, build / "init_device.o", build / "transfer_device.o"])
    linker_source = Path(__file__).with_name("iaas_kernels_lto_link.cpp")
    linker = output / "link_object"
    run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            f"-I{cuda}/include",
            linker_source,
            f"-L{cuda}/lib64",
            f"-Wl,-rpath,{cuda}/lib64",
            "-lnvJitLink",
            "-o",
            linker,
        ]
    )
    run([linker, "--version"])
    # Link a real NVSHMEM device call on CPU so an empty/broken archive or an
    # OBJECT-vs-LTOIR mismatch cannot pass the image build just by existing.
    probe = build / "probe.cu"
    probe.write_text(
        '#include <nvshmem.h>\nextern "C" __global__ void iaas_lto_probe(int* dst, int pe) { nvshmem_int_p(dst, 1, pe); }\n'
    )
    run([cuda / "bin/nvcc", probe, *flags, "-o", build / "probe.o"])
    run([linker, build / "probe.o", archive, "archive", build / "probe.cubin"])
    require((build / "probe.cubin").stat().st_size > 0, "Empty LTO probe cubin")
    shutil.copy2(source / "License.txt", output / "License.nvshmem.txt")
    manifest = dict(
        nvshmem_revision=head,
        nvshmem_version=lock["rdma"]["nvshmem_version"],
        cuda=lock["cuda"],
        nvcc=subprocess.check_output(
            [str(cuda / "bin/nvcc"), "--version"], text=True
        ).strip(),
        linker_source_sha256=hashlib.sha256(linker_source.read_bytes()).hexdigest(),
        files={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.iterdir())
        },
        commands=commands,
        cpu_link_probe="passed",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build_lto(load_lock(args.lock), args.source, args.output)
