"""Explicit compiler entrypoints for the migration graph and CPU references."""

import os
import shutil

from sglang.kernels.ops.moe._native_build import SOURCE, compile_library, nvcc


def build_library(name, flags=(), *, cpu=False):
    names = ("multi_swap_cpu",) if cpu else ("plan_loop",)
    if name not in names:
        raise ValueError(f"unsupported native helper: {name}")
    compiler = shutil.which(os.getenv("CXX", "g++")) if cpu else nvcc()
    if not compiler:
        raise RuntimeError("C++ compiler is required for the independent CPU oracle")
    options = (
        ("-O3", "-std=c++17", "-shared", "-fPIC")
        if cpu
        else (
            "-O3",
            "-std=c++17",
            "-arch=sm_90",
            "--shared",
            "--cudart=shared",
            "-Xcompiler=-fPIC",
            "-rdc=true",
        )
    )
    source = SOURCE / "lplb" / (name + (".cpp" if cpu else ".cu"))
    return compile_library(
        name,
        str(source),
        compiler,
        (*options, *flags),
        (),
        () if cpu else ("-lcudadevrt",),
    )
