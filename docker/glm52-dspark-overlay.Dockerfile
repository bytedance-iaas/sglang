ARG BASE_IMAGE=iaas-gpu-cn-beijing.cr.volces.com/serving/sglang:glm52-dp8-pppoll-5668722414-eic1.5.2@sha256:a322afaa9aa032810aa559d2497ed57fe91b71ac6d470c8b843923fadf7075d3
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG SOURCE_COMMIT
ARG SOURCE_TREE
ARG SOURCE_PYTHON_SHA256
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

LABEL org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      ai.byteiaas.sglang.source-tree="${SOURCE_TREE}" \
      ai.byteiaas.sglang.tracked-python-sha256="${SOURCE_PYTHON_SHA256}" \
      ai.byteiaas.sglang.base-image="${BASE_IMAGE}"

COPY python/ /tmp/glm52-dspark-source/python/
COPY .ci-artifacts/glm52-dspark-python-files.txt /tmp/glm52-dspark-source/python-files.txt

RUN set -eux; \
    test -n "${SOURCE_COMMIT}"; \
    test -n "${SOURCE_TREE}"; \
    test -n "${SOURCE_PYTHON_SHA256}"; \
    cd /tmp/glm52-dspark-source; \
    actual_python_sha256="$( \
      while IFS= read -r source_file; do \
        source_hash="$(sha256sum "${source_file}" | awk '{print $1}')"; \
        printf '%s %s\n' "${source_hash}" "${source_file}"; \
      done < python-files.txt | sha256sum | awk '{print $1}' \
    )"; \
    test "${actual_python_sha256}" = "${SOURCE_PYTHON_SHA256}"; \
    rm -rf /sgl-workspace/sglang/python; \
    cp -a /tmp/glm52-dspark-source/python /sgl-workspace/sglang/python; \
    SGLANG_BUILD_RUST_EXTS=none python3 -m pip install \
      --break-system-packages --no-cache-dir --no-deps --force-reinstall \
      -e /sgl-workspace/sglang/python; \
    python3 -m pip install \
      --break-system-packages --no-cache-dir --no-deps --force-reinstall \
      'sglang-kernel==0.4.5'; \
    printf 'SOURCE_COMMIT=%s\nSOURCE_TREE=%s\nSOURCE_PYTHON_SHA256=%s\nBASE_IMAGE=%s\n' \
      "${SOURCE_COMMIT}" "${SOURCE_TREE}" "${SOURCE_PYTHON_SHA256}" "${BASE_IMAGE}" \
      > /etc/sglang-glm52-dspark-provenance.env; \
    cd /tmp; \
    python3 -c 'import pathlib, sglang; expected = pathlib.Path("/sgl-workspace/sglang/python/sglang/__init__.py"); actual = pathlib.Path(sglang.__file__).resolve(); print(f"SGLANG_IMPORT_FILE={actual}"); assert actual == expected'; \
    python3 -c 'import importlib.metadata as m; version=m.version("sglang-kernel"); print(f"SGLANG_KERNEL_VERSION={version}"); assert version == "0.4.5"'; \
    rm -rf /tmp/glm52-dspark-source

ENV SGLANG_SOURCE_STACK_ID=glm52-dspark-deepep-baked \
    SGLANG_SOURCE_COMMIT=${SOURCE_COMMIT} \
    SGLANG_SOURCE_TREE=${SOURCE_TREE} \
    SGLANG_SOURCE_TRACKED_PYTHON_MANIFEST_SHA256=${SOURCE_PYTHON_SHA256}
