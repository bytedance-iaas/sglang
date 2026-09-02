ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG SGLANG_BUILD_COMMIT
ARG SGLANG_SOURCE_TREE
ARG SGLANG_SOURCE_ARCHIVE_SHA256
ARG SGLANG_BUILD_URL
ARG SGLANG_IMAGE_TAG
ARG SGL_DEEP_GEMM_COMMIT
ARG SGLANG_KERNEL_VERSION=0.4.6.post1

USER root

COPY .glm53-build/source.tar /tmp/glm53-source.tar

RUN set -eux; \
    test -n "${SGLANG_BUILD_COMMIT}"; \
    test -n "${SGLANG_SOURCE_TREE}"; \
    test -n "${SGLANG_SOURCE_ARCHIVE_SHA256}"; \
    test -n "${SGL_DEEP_GEMM_COMMIT}"; \
    test "$(sha256sum /tmp/glm53-source.tar | awk '{print $1}')" = "${SGLANG_SOURCE_ARCHIVE_SHA256}"; \
    rm -rf /glm53-community/sglang /sgl-workspace/sglang; \
    mkdir -p /glm53-community/sglang; \
    tar -xf /tmp/glm53-source.tar -C /glm53-community/sglang; \
    rm -f /tmp/glm53-source.tar; \
    test "$(git -C /glm53-community/sglang rev-parse HEAD)" = "${SGLANG_BUILD_COMMIT}"; \
    test "$(git -C /glm53-community/sglang write-tree)" = "${SGLANG_SOURCE_TREE}"; \
    git -C /glm53-community/sglang diff --quiet -- .; \
    git -C /glm53-community/sglang diff --cached --quiet -- .; \
    test -z "$(git -C /glm53-community/sglang status --porcelain=v1 --untracked-files=all)"; \
    ln -s /glm53-community/sglang /sgl-workspace/sglang; \
    test "$(readlink -f /sgl-workspace/sglang)" = /glm53-community/sglang; \
    test "$(cat /opt/sgl-deep-gemm/source-commit)" = "${SGL_DEEP_GEMM_COMMIT}"; \
    python3 -m pip install --no-deps --force-reinstall \
      "sglang-kernel==${SGLANG_KERNEL_VERSION}"; \
    test "$(python3 -c 'from importlib.metadata import version; print(version("sglang-kernel"))')" = "${SGLANG_KERNEL_VERSION}"; \
    python3 -c 'from importlib.metadata import distribution; files = tuple(map(str, distribution("sglang-kernel").files or ())); assert any(path.startswith("sgl_kernel/sm90/common_ops.") and path.endswith(".so") for path in files), files'; \
    printf '%s\n' "${SGLANG_SOURCE_ARCHIVE_SHA256}" > /opt/sglang-source-archive-sha256; \
    PYTHONPATH=/glm53-community/sglang/python python3 -c 'import importlib.machinery, pathlib; expected = pathlib.Path("/glm53-community/sglang/python/sglang/__init__.py"); actual = pathlib.Path(importlib.machinery.PathFinder.find_spec("sglang").origin).resolve(); print(f"SGLANG_SPEC_ORIGIN={actual}"); assert actual == expected'

ENV SGLANG_SOURCE_ROOT=/glm53-community/sglang \
    SGLANG_SOURCE_COMMIT=${SGLANG_BUILD_COMMIT} \
    SGLANG_SOURCE_TREE=${SGLANG_SOURCE_TREE}

LABEL org.opencontainers.image.revision=${SGLANG_BUILD_COMMIT} \
      org.opencontainers.image.source="https://github.com/bytedance-iaas/sglang" \
      org.opencontainers.image.url=${SGLANG_BUILD_URL} \
      org.opencontainers.image.version=${SGLANG_IMAGE_TAG} \
      ai.sglang.build.commit=${SGLANG_BUILD_COMMIT} \
      ai.sglang.build.tree=${SGLANG_SOURCE_TREE} \
      ai.sglang.build.source-archive-sha256=${SGLANG_SOURCE_ARCHIVE_SHA256} \
      ai.sglang.build.url=${SGLANG_BUILD_URL} \
      ai.sglang.build.base-image=${BASE_IMAGE} \
      ai.sglang.kernel.version=${SGLANG_KERNEL_VERSION} \
      ai.sglang.deepgemm.commit=${SGL_DEEP_GEMM_COMMIT} \
      ai.sglang.source.delivery="immutable-source-overlay"

WORKDIR /glm53-community/sglang
