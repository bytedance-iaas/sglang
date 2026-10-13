# iaas-kernels in internal images

The internal release and development workflows preinstall the SM90 MegaMoE
provider for **linux/amd64, CUDA 13.0.3, Python 3.12, Torch 2.13.0,
NVLink and RDMA**.
`docker/iaas-kernels.lock.json` pins the source commit, package version and ABI.
The pin includes the fixed-M64 FP4 RDMA RS implementation (RDMA API v3).
The pinned commit must be available in the private source repository before
starting a release; a commit present only in a developer's local main cannot
be fetched by CI. There is no fallback to a moving branch or an older wheel.

The shared Dockerfile defaults to `INSTALL_IAAS_KERNELS=0`; unrelated upstream,
ARM, CUDA 12 and other workflow callers retain their existing behavior. Internal
release builds use the selected internal tag's local source, not an upstream
SGLang clone. Internal development builds use the selected branch/PR checkout.
The lightweight `private_dependency_smoke` target does not install this package.

## Credentials and build

Configure `IAAS_KERNELS_READ_TOKEN` in the internal CI environment/repository with
Contents read access to the private `bytedance-iaas/iaas-kernels` repository. The
standard `GITHUB_TOKEN` is scoped to SGLang and cannot read that private repository.
The workflow passes this credential as a BuildKit secret. It is not stored in
build arguments, the Git remote URL, wheel metadata, or image layers.

For a local Linux build, export `IAAS_KERNELS_READ_TOKEN` securely, then run:

```bash
docker buildx build --platform linux/amd64 --target framework_final \
  --build-arg CUDA_VERSION=13.0.3 \
  --build-arg BRANCH_TYPE=local --build-arg INSTALL_IAAS_KERNELS=1 \
  --secret id=iaas_kernels_token,env=IAAS_KERNELS_READ_TOKEN \
  -f docker/Dockerfile --load -t sglang:iaas-rdma .
```

The `iaas_kernels_builder` stage inherits `torch_deps`, fetches the pinned commit
and submodules, and invokes `build_sgl_deep_gemm.sh` with
`IAAS_KERNELS_BUILD_RDMA=1`. It installs the wheel with
`--force-reinstall --no-deps`. It does not replace `sgl-deep-gemm` or upgrade Torch.
Both `framework_final` and `runtime` contain the wheel, libdw/libelf dependencies,
JIT headers and CUDA compilation tools. NVSHMEM **3.4.5** is installed before
freezing the consuming image's dependency constraints and retained in both final
stages. Its headers, ordinary device archive and host library are runtime JIT
dependencies, not just build dependencies.

The builder also fetches the pinned NVIDIA NVSHMEM source and builds the SM90a
device LTO archive and a small nvJitLink executable. It checks the source against
the wheel's private ABI headers and uses the wheel's generated build options.
A CPU-only compile/link probe with a real NVSHMEM device call must pass. The
resulting tools and NVIDIA license are installed under
`/opt/sglang/share/iaas-kernels/lto/`; they do not rely on files copied from a
benchmark container. This archive is for iaas-kernels FP4 JIT, and does not replace
Torch/DeepEP's ordinary NVSHMEM archive or host library.

The image saves source/submodule SHAs, build ABI, wheel SHA256 and native library
SHA256 in `/opt/sglang/share/iaas-kernels/manifest.json`, alongside its lock file.
The manifest records both native extensions, RDMA API/NVSHMEM versions, LTO source
revision, build commands, compiler version, linker/archive hashes and link probe.
Use target `iaas_kernels_wheel` with `--output type=local,dest=<directory>` and the
same build arguments to export the wheel and provenance from the builder cache.
The package currently uses version `0.1.0.dev0`; identify binaries by the pinned
commit and wheel digest, not by the version string alone.

## Verification and publishing

The final stages run `scripts/ci/check_iaas_kernels_image.py` without a GPU. The
check validates the installed source/ABI/binaries, RDMA API v3, NVSHMEM 3.4.5,
JIT headers/tools, LTO hashes/executable, Torch NVSHMEM support, continued ownership
of `deep_gemm` by `sgl-deep-gemm`, both native host libraries, and plugin registration
without CUDA initialization.
Registration is asserted explicitly because SGLang logs and catches plugin errors.

For OCI and zstd candidates, CI runs the same check on the pushed **digest** before
assigning release tags, then uploads the result and image digest metadata. Nydus
uses the already checked OCI source through the existing conversion workflow.
CPU regressions can also be run locally:

```bash
python3 -m unittest discover -s scripts/ci -p test_iaas_kernels_image.py -v
docker run --rm --entrypoint python3 sglang:iaas-rdma \
  /sgl-workspace/sglang/scripts/ci/check_iaas_kernels_image.py
```

These checks validate packaging, not SM90 kernel JIT, numerical accuracy or
throughput. Before qualifying a new source/ABI pin, run the companion repository's
FP4/FP8 correctness and repeatability regressions, a TP8/EP8 service smoke test,
and a two-node TP16/EP16 FP4 RDMA service smoke test
on the candidate image with a fresh task-specific JIT cache. Existing operator
validation is not validation of a newly built image. This change does not add a
GPU release gate or claim that ID119 output randomness is resolved.

## Runtime selection

Plugins are discovered automatically. The image does not set `SGLANG_PLUGINS`
(a whitelist) or change the default MoE backend. If a deployment already sets a
whitelist, add `iaas_kernels` to its existing list. To select the single-node
provider on SM90, use `--moe-a2a-backend megamoe --megamoe-transport nvlink
--disable-shared-experts-fusion`, together with the normal TP/EP settings and
`SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` capacity for the workload.

For two-node FP4 RDMA, source the installed profile **in every worker's launch
environment**, then supply the normal node address/rank and model arguments:

```bash
source /opt/sglang/share/iaas-kernels/rdma-fp4.env
export NVSHMEM_IBGDA_NUM_RC_PER_PE=25  # E384 / EP16 + 1; recompute for other models
# Configure NIC/PE mapping, socket interface, GID and traffic class for your cluster.
python3 -m sglang.launch_server --model-path /path/to/model \
  --tp-size 16 --ep-size 16 --nnodes 2 --node-rank "$NODE_RANK" \
  --dist-init-addr "$DIST_INIT_ADDR" \
  --moe-a2a-backend megamoe --megamoe-transport rdma \
  --disable-shared-experts-fusion
```

The FP4 profile enables IBGDA, disables relaxed ordering, selects `RS_GENERAL=1`
and FP4 RS LTO, and points JIT at the image's linker/archive. The GENERAL=1,
LTO=0, B1 path has a known misaligned-address failure on H20, so LTO is required
for this validated profile. Network-specific settings are deliberately supplied
by the deployment. The image does not globally change the allocator, MoE backend,
FP4 policy or network configuration for other workloads.
Buffer capacity also remains a deployment setting. The optimized EP16/NVL8
same-rail push path uses requested capacity <=256, not the current forward's
token count; choose `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` for the
workload instead of assuming that every small forward uses this path.

FP8 RDMA uses `IAAS_KERNELS_RDMA_RS_GENERAL=0`; do not source the FP4 profile for
FP8. It still requires IBGDA/relaxed-ordering settings and enough RC QPs.
RDMA currently requires EP=WORLD in identical rank order, equal contiguous NVLink
domains of 1–8 ranks (at least two domains, at most 64 ranks), MoE TP=1, SwiGLU,
and separate shared experts. PP/EP subgroups, TBO and live EPLB are unsupported.
FP4 RS requires hidden width divisible by 512, intermediate width divisible by
256 and at most 4096, and matching RS-transformed weights. Set capacities and
protocol environment settings consistently before allocating buffers; do not
change them between forwards. See the pinned iaas-kernels `docs/sm90-rdma.md` for
the full direct-API contract.
