import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

BASE_IMAGE = "iaas-gpu-cn-beijing.cr.volces.com/serving/sglang@sha256:d33d932aee374f3884e7f49c54447c82ce61ac9e58749ebf5aa3a50d25c9cb28"
DEEP_GEMM_SHA256 = "f4e67086dc685ddcfcbb7833cc9770afd850cab23e173e77b9b18c19de0c2836"
DEEP_GEMM_URL = "https://files.pythonhosted.org/packages/7a/ad/6f9aa43796dc2f028cbe62b43c276dbce8091b955eb918d42a147d97b377/sgl_deep_gemm-0.1.7-py3-none-manylinux2014_x86_64.whl"
SOURCE_ROOT = Path("/sgl-workspace/sglang")
EVIDENCE_ROOT = Path("/usr/local/share/sglang/glm53-flash")
NATIVE_MODULES = (
    "sglang.srt.rust_extensions._grpc",
    "sglang.srt.rust_extensions._multimodal",
    "sglang.srt.rust_extensions._server",
    "sglang.srt.mem_cache.rust_tree_core.mem_cache",
)
CORE_VERSIONS = {
    "torch": "2.13.0+cu130",
    "triton": "3.7.1",
    "sglang-kernel": "0.4.6.post1",
    "flashinfer-python": "0.6.18",
    "transformers": "5.12.1",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_hash(kind, data):
    return hashlib.sha1(f"{kind} {len(data)}\0".encode() + data).hexdigest()


def tree_hash(files):
    root = {}
    for entry in files:
        node = root
        parts = entry["path"].split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = entry

    def digest(node):
        data = bytearray()
        for name, value in sorted(
            node.items(),
            key=lambda item: item[0] + ("/" if "path" not in item[1] else ""),
        ):
            if "path" in value:
                mode, oid = value["mode"], value["blob"]
            else:
                mode, oid = "40000", digest(value)
            data.extend(f"{mode} {name}\0".encode() + bytes.fromhex(oid))
        return git_hash("tree", data)

    return digest(root)


def verify_sources(root, document):
    require(tree_hash(document["files"]) == document["tree"], "Git tree mismatch")
    commit_data = document["commit_object"].encode()
    require(
        git_hash("commit", commit_data) == document["commit"], "Git commit mismatch"
    )
    require(
        commit_data.splitlines()[0] == f"tree {document['tree']}".encode(),
        "Commit/tree mismatch",
    )
    for entry in document["files"]:
        path = root / entry["path"]
        if entry["mode"] == "120000":
            require(path.is_symlink(), f"Missing symlink: {path}")
            data = os.readlink(path).encode()
        else:
            require(path.is_file() and not path.is_symlink(), f"Missing source: {path}")
            data = path.read_bytes()
            require(
                bool(path.stat().st_mode & 0o111) == (entry["mode"] == "100755"),
                f"Mode mismatch: {path}",
            )
        require(
            hashlib.sha256(data).hexdigest() == entry["sha256"],
            f"Source hash mismatch: {path}",
        )
        require(git_hash("blob", data) == entry["blob"], f"Git blob mismatch: {path}")


def prepare(context):
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    require(
        not git("status", "--porcelain").strip(),
        "Build requires a clean committed checkout",
    )
    require(not context.exists(), f"Context already exists: {context}")
    context.mkdir(parents=True)
    commit = git("rev-parse", "HEAD").decode().strip()
    tree = git("rev-parse", "HEAD^{tree}").decode().strip()
    archive = context / "source.tar"
    subprocess.run(
        ["git", "-C", str(root), "archive", "--format=tar", "HEAD", "-o", str(archive)],
        check=True,
    )
    entries = []
    with tarfile.open(archive) as stream:
        members = {member.name: member for member in stream.getmembers()}
        for record in git("ls-tree", "-rz", "--full-tree", "HEAD").split(b"\0"):
            if not record:
                continue
            attributes, name = record.split(b"\t", 1)
            mode, kind, blob = attributes.decode().split()
            require(
                kind == "blob", "Submodules must be explicitly packaged before building"
            )
            relative = name.decode()
            member = members[relative]
            data = (
                member.linkname.encode()
                if mode == "120000"
                else stream.extractfile(member).read()
            )
            require(
                git_hash("blob", data) == blob, f"Archive blob mismatch: {relative}"
            )
            entries.append(
                {
                    "path": relative,
                    "mode": mode,
                    "blob": blob,
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        document = {
            "commit": commit,
            "tree": tree,
            "commit_object": git("cat-file", "commit", "HEAD").decode(),
            "base_image": BASE_IMAGE,
            "deep_gemm_sha256": DEEP_GEMM_SHA256,
            "files": entries,
        }
        verify_sources(root, document)
    (context / "source.json").write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n"
    )
    shutil.copyfile(root / "docker/glm53-flash.Dockerfile", context / "Dockerfile")
    shutil.copyfile(
        root / "docker/glm53-flash-build-requirements.txt",
        context / "build-requirements.txt",
    )
    shutil.copyfile(Path(__file__), context / "verify.py")
    print(
        json.dumps(
            {
                "commit": commit,
                "tree": tree,
                "manifest_sha256": sha256(context / "source.json"),
            }
        )
    )


def packages():
    result = {}
    for dist in metadata.distributions():
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        result.setdefault(name, []).append(dist.version)
    return dict(sorted(result.items()))


def snapshot(path):
    installed = packages()
    for name, version in CORE_VERSIONS.items():
        require(
            metadata.version(name) == version,
            f"Unexpected base {name}: {installed.get(name)}",
        )
    path.write_text(json.dumps(installed, sort_keys=True, indent=2) + "\n")


def verify_install(runtime=False):
    manifest = EVIDENCE_ROOT / "source.json"
    require(
        sha256(manifest) == os.environ["SGLANG_SOURCE_MANIFEST_SHA256"],
        "Manifest digest mismatch",
    )
    document = json.loads(manifest.read_text())
    require(
        document["commit"] == os.environ["SGLANG_BUILD_COMMIT"],
        "Source commit mismatch",
    )
    require(document["tree"] == os.environ["SGLANG_BUILD_TREE"], "Source tree mismatch")
    require(document["base_image"] == BASE_IMAGE, "Base image mismatch")
    require(
        document["deep_gemm_sha256"] == DEEP_GEMM_SHA256,
        "DeepGEMM wheel identity mismatch",
    )
    verify_sources(SOURCE_ROOT, document)
    before = json.loads((EVIDENCE_ROOT / "base-packages.json").read_text())
    after = packages()
    allowed = {"sglang", "sgl-deep-gemm"}
    drift = {
        name: [before.get(name), after.get(name)]
        for name in before.keys() | after.keys()
        if name not in allowed and before.get(name) != after.get(name)
    }
    require(not drift, f"Base dependency drift: {drift}")
    require(after["sgl-deep-gemm"] == ["0.1.7"], "Wrong DeepGEMM version")
    require(
        after["sglang"] == [f"0.0.0.dev0+glm53.{document['commit'][:12]}"],
        "Wrong SGLang build version",
    )
    distribution = metadata.distribution("sglang")
    installed_root = Path(distribution.locate_file("sglang")).resolve()
    require(
        "site-packages" in installed_root.parts,
        "SGLang must be installed, not externally editable",
    )
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    require(
        not direct.get("dir_info", {}).get("editable"), "Unexpected editable install"
    )
    for entry in document["files"]:
        if entry["path"].startswith("python/sglang/") and entry["path"].endswith(".py"):
            relative = entry["path"].removeprefix("python/sglang/")
            if relative.startswith("kernels/aot/") or any(
                part.startswith(".") for part in Path(relative).parts
            ):
                continue
            require(
                sha256(installed_root / relative) == entry["sha256"],
                f"Installed Python source mismatch: {relative}",
            )
    native = {}
    for module in NATIVE_MODULES:
        relative = module.removeprefix("sglang.").replace(".", "/")
        candidates = list(
            (installed_root / relative).parent.glob(Path(relative).name + ".*.so")
        )
        require(len(candidates) == 1, f"Missing or ambiguous native module: {module}")
        native[module] = sha256(candidates[0])
    require(
        not any(
            value
            for key, value in os.environ.items()
            if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}
        ),
        "Build proxy leaked into runtime",
    )
    require(not os.environ.get("PYTHONPATH"), "External PYTHONPATH is not supported")
    result = {
        "commit": document["commit"],
        "tree": document["tree"],
        "manifest_sha256": sha256(manifest),
        "native_sha256": native,
        "packages": after,
        "installation": "PASS",
    }
    if runtime:
        import torch
        import deep_gemm
        import sglang

        require(
            Path(sglang.__file__).resolve().parent == installed_root,
            "Wrong SGLang import path",
        )
        for symbol in (
            "get_symm_buffer_for_mega_moe",
            "fp8_mega_moe",
            "mega_moe_pre_dispatch_sm90",
        ):
            require(hasattr(deep_gemm, symbol), f"Missing DeepGEMM symbol: {symbol}")
        for module in NATIVE_MODULES:
            loaded = importlib.import_module(module)
            require(
                Path(loaded.__file__).resolve().is_relative_to(installed_root),
                f"External native import: {module}",
            )
        importlib.import_module("sgl_kernel")
        importlib.import_module("sglang.srt.models.glm5_next")
        importlib.import_module("sglang.srt.models.glm5_next_nextn")
        require(torch.tensor([2, 3]).sum().item() == 5, "Torch CPU smoke failed")
        result["runtime_imports"] = "PASS"
        result["cuda_available"] = torch.cuda.is_available()
        result["full_h20_pd_inference"] = "NOT_RUN"
    print(json.dumps(result, sort_keys=True, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "snapshot", "install", "runtime"))
    parser.add_argument("--path", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        require(args.path is not None, "prepare requires --path")
        prepare(args.path.resolve())
    elif args.mode == "snapshot":
        require(args.path is not None, "snapshot requires --path")
        snapshot(args.path)
    else:
        verify_install(runtime=args.mode == "runtime")


if __name__ == "__main__":
    main()
