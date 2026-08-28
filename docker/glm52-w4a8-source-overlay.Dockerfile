ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG SGLANG_BUILD_COMMIT
ARG SGLANG_SOURCE_TREE
ARG SGLANG_SOURCE_ARCHIVE_SHA256
ARG SGLANG_BUILD_URL
ARG SGLANG_IMAGE_TAG
ARG SGL_DEEP_GEMM_COMMIT

USER root

COPY .glm52-build/source.tar /tmp/glm52-source.tar

RUN set -eux; \
    test -n "${SGLANG_BUILD_COMMIT}"; \
    test -n "${SGLANG_SOURCE_TREE}"; \
    test -n "${SGLANG_SOURCE_ARCHIVE_SHA256}"; \
    test -n "${SGL_DEEP_GEMM_COMMIT}"; \
    test "$(sha256sum /tmp/glm52-source.tar | awk '{print $1}')" = "${SGLANG_SOURCE_ARCHIVE_SHA256}"; \
    rm -rf /glm52/sglang /sgl-workspace/sglang; \
    mkdir -p /glm52/sglang; \
    tar -xf /tmp/glm52-source.tar -C /glm52/sglang; \
    rm -f /tmp/glm52-source.tar; \
    test "$(git -C /glm52/sglang rev-parse HEAD)" = "${SGLANG_BUILD_COMMIT}"; \
    test "$(git -C /glm52/sglang write-tree)" = "${SGLANG_SOURCE_TREE}"; \
    git -C /glm52/sglang diff --quiet -- .; \
    git -C /glm52/sglang diff --cached --quiet -- .; \
    test -z "$(git -C /glm52/sglang status --porcelain=v1 --untracked-files=all)"; \
    ln -s /glm52/sglang /sgl-workspace/sglang; \
    test "$(readlink -f /sgl-workspace/sglang)" = /glm52/sglang; \
    test "$(cat /opt/sgl-deep-gemm/source-commit)" = "${SGL_DEEP_GEMM_COMMIT}"; \
    printf '%s\n' "${SGLANG_SOURCE_ARCHIVE_SHA256}" > /opt/sglang-source-archive-sha256; \
    PYTHONPATH=/glm52/sglang/python python3 -c 'import importlib.machinery, pathlib; expected = pathlib.Path("/glm52/sglang/python/sglang/__init__.py"); actual = pathlib.Path(importlib.machinery.PathFinder.find_spec("sglang").origin).resolve(); print(f"SGLANG_SPEC_ORIGIN={actual}"); assert actual == expected'

ENV SGLANG_SOURCE_ROOT=/glm52/sglang \
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
      ai.sglang.deepgemm.commit=${SGL_DEEP_GEMM_COMMIT} \
      ai.sglang.source.delivery="immutable-source-overlay"

WORKDIR /glm52/sglang
