"""Content-addressed native builds shared by early-MoE and EPLB.

Compilation is explicit/lazy and never writes to the installed package. Cache
identity includes source/header contents, flags and compiler version. Publishing
is locked and atomic so concurrent workers cannot load a partial shared object.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

KERNELS = Path(__file__).resolve().parents[2]
INCLUDE = KERNELS / "jit/include"
SOURCE = KERNELS / "jit/csrc"


def cache_root():
    return Path(
        os.getenv("SGLANG_MOE_NATIVE_CACHE_DIR", "~/.cache/sglang/moe_native")
    ).expanduser()


def nvcc():
    configured = os.getenv("SGLANG_MOE_NVCC")
    path = configured or str(
        Path(os.getenv("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"
    )
    found = shutil.which(path) or shutil.which("nvcc")
    if not found:
        raise RuntimeError("CUDA toolkit/nvcc is required to build SGLang MoE kernels")
    return found


@lru_cache(maxsize=None)
def tree_digest(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def cutlass_include():
    from sglang.kernels.jit.utils.deps import get_cutlass_include_paths

    for candidate in get_cutlass_include_paths():
        path = Path(candidate)
        if (path / "cute/tensor.hpp").is_file():
            return path
    raise RuntimeError("CUTLASS headers do not contain cute/tensor.hpp")


@lru_cache(maxsize=None)
def compile_library(name, source, compiler, flags=(), include_dirs=(), link_flags=()):
    import fcntl

    source = Path(source)
    version = subprocess.check_output([compiler, "--version"], text=True)
    roots = tuple(dict.fromkeys((str(INCLUDE), str(SOURCE), *map(str, include_dirs))))
    manifest = dict(
        name=name,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        compiler=compiler,
        compiler_version=version,
        flags=flags,
        link_flags=link_flags,
        headers={root: tree_digest(root) for root in roots},
    )
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:24]
    directory = cache_root() / name / key
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"lib{name}.so"
    with (directory / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.is_file():
            temporary = directory / f"{os.getpid()}.tmp.so"
            command = [
                compiler,
                *flags,
                *["-I" + p for p in roots],
                str(source),
                *link_flags,
                "-o",
                str(temporary),
            ]
            try:
                run = subprocess.run(
                    command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                )
                (directory / "compile.log").write_text(run.stdout)
                if run.returncode:
                    raise RuntimeError(
                        f"native build failed; inspect {directory / 'compile.log'}"
                    )
                manifest["command"] = command
                manifest["library_sha256"] = hashlib.sha256(
                    temporary.read_bytes()
                ).hexdigest()
                (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
    return output
