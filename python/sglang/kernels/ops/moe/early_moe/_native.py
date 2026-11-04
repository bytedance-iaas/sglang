"""Build one explicit geometry from the shared SM90 implementation."""

from sglang.kernels.ops.moe._native_build import (
    SOURCE,
    compile_library,
    cutlass_include,
    nvcc,
)


def build_library(geometry):
    geometry.validate()
    flags = (
        "-std=c++20",
        "-O3",
        "-lineinfo",
        "-gencode=arch=compute_90a,code=sm_90a",
        "--shared",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-Xcompiler=-fPIC",
        "-Xptxas=-v,-warn-spills",
        *geometry.cuda_flags,
    )
    return compile_library(
        "early_moe_" + geometry.key,
        str(SOURCE / "early_moe/early_moe.cu"),
        nvcc(),
        flags,
        (str(cutlass_include()),),
        ("-lcuda",),
    )
