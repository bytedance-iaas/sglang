# iaas-kernels in internal images

The internal release and development workflows preinstall the SM90 MegaMoE
provider for **linux/amd64, CUDA 13.0.3, Python 3.12, Torch 2.13.0, NVLink**.
`docker/iaas-kernels.lock.json` pins the source commit, package version and ABI.
The initial pin includes the FP4 and FP8 swapAB synchronization fixes.

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
  -f docker/Dockerfile --load -t sglang:iaas-nvlink .
```

The `iaas_kernels_builder` stage inherits `torch_deps`, fetches the pinned commit
and submodules, and invokes `build_sgl_deep_gemm.sh` with
`IAAS_KERNELS_BUILD_RDMA=0`. It installs the wheel with
`--force-reinstall --no-deps`. It does not replace `sgl-deep-gemm` or upgrade Torch.
Both `framework_final` and `runtime` contain the wheel, libdw/libelf dependencies,
JIT headers and CUDA compilation tools.

The image saves source/submodule SHAs, build ABI, wheel SHA256 and native library
SHA256 in `/opt/sglang/share/iaas-kernels/manifest.json`, alongside its lock file.
Use target `iaas_kernels_wheel` with `--output type=local,dest=<directory>` and the
same build arguments to export the wheel and provenance from the builder cache.
The package currently uses version `0.1.0.dev0`; identify binaries by the pinned
commit and wheel digest, not by the version string alone.

## Verification and publishing

The final stages run `scripts/ci/check_iaas_kernels_image.py` without a GPU. The
check validates the installed source/ABI/binary, JIT headers/tools, absence of the
optional RDMA extension, continued ownership of `deep_gemm` by `sgl-deep-gemm`,
native host-library loading, and plugin registration without CUDA initialization.
Registration is asserted explicitly because SGLang logs and catches plugin errors.

For OCI and zstd candidates, CI runs the same check on the pushed **digest** before
assigning release tags, then uploads the result and image digest metadata. Nydus
uses the already checked OCI source through the existing conversion workflow.
CPU regressions can also be run locally:

```bash
python3 -m unittest discover -s scripts/ci -p test_iaas_kernels_image.py -v
docker run --rm --entrypoint python3 sglang:iaas-nvlink \
  /sgl-workspace/sglang/scripts/ci/check_iaas_kernels_image.py
```

These checks validate packaging, not SM90 kernel JIT, numerical accuracy or
throughput. Before qualifying a new source/ABI pin, run the companion repository's
FP4/FP8 correctness and repeatability regressions and a TP8/EP8 service smoke test
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
This wheel does not contain the optional multi-node RDMA extension.
