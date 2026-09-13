FROM iaas-gpu-cn-beijing.cr.volces.com/serving/sglang@sha256:d33d932aee374f3884e7f49c54447c82ce61ac9e58749ebf5aa3a50d25c9cb28 AS builder
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ARG SOURCE_COMMIT
ARG BUILD_JOBS=8
COPY build-requirements.txt /tmp/build-requirements.txt
RUN timeout 600 python3 -m pip install --no-deps --require-hashes --retries 3 --timeout 60 \
      -r /tmp/build-requirements.txt
RUN curl --fail --location --retry 3 --retry-all-errors --connect-timeout 30 --max-time 900 \
      https://static.rust-lang.org/dist/rust-1.92.0-x86_64-unknown-linux-gnu.tar.xz \
      -o /tmp/rust.tar.xz && \
    printf '%s  %s\n' d2ccef59dd9f7439f2c694948069f789a044dc1addcc0803613232af8f88ee0c /tmp/rust.tar.xz | sha256sum -c - && \
    tar -xJf /tmp/rust.tar.xz -C /tmp && \
    /tmp/rust-1.92.0-x86_64-unknown-linux-gnu/install.sh --prefix=/opt/glm53-rust \
      --components=rustc,cargo,rust-std-x86_64-unknown-linux-gnu --disable-ldconfig && \
    rm -rf /tmp/rust.tar.xz /tmp/rust-1.92.0-x86_64-unknown-linux-gnu
ENV PATH="/opt/glm53-rust/bin:${PATH}" \
    CARGO_NET_RETRY=3 \
    CARGO_HTTP_TIMEOUT=60 \
    CARGO_BUILD_JOBS=${BUILD_JOBS}
COPY source.tar /tmp/source.tar
RUN rm -rf /sgl-workspace/sglang && mkdir -p /sgl-workspace/sglang && \
    tar -xf /tmp/source.tar -C /sgl-workspace/sglang && rm /tmp/source.tar
WORKDIR /sgl-workspace/sglang
RUN timeout 3600 env SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0.dev0+glm53.${SOURCE_COMMIT:0:12}" \
      SGLANG_BUILD_RUST_EXTS=all \
      python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/output ./python

FROM iaas-gpu-cn-beijing.cr.volces.com/serving/sglang@sha256:d33d932aee374f3884e7f49c54447c82ce61ac9e58749ebf5aa3a50d25c9cb28
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ARG SOURCE_COMMIT
ARG SOURCE_TREE
ARG SOURCE_MANIFEST_SHA256
ARG BUILD_URL
ARG IMAGE_TAG
COPY verify.py source.json /usr/local/share/sglang/glm53-flash/
RUN python3 /usr/local/share/sglang/glm53-flash/verify.py snapshot \
      --path /usr/local/share/sglang/glm53-flash/base-packages.json
COPY source.tar /tmp/source.tar
COPY deep-gemm.whl /tmp/sgl_deep_gemm-0.1.7-py3-none-manylinux2014_x86_64.whl
COPY --from=builder /tmp/output/ /tmp/glm53-wheels/
RUN printf '%s  %s\n' f4e67086dc685ddcfcbb7833cc9770afd850cab23e173e77b9b18c19de0c2836 \
      /tmp/sgl_deep_gemm-0.1.7-py3-none-manylinux2014_x86_64.whl | sha256sum -c - && \
    timeout 300 python3 -m pip install --no-index --no-deps --force-reinstall \
      /tmp/glm53-wheels/*.whl /tmp/sgl_deep_gemm-0.1.7-py3-none-manylinux2014_x86_64.whl && \
    rm -rf /sgl-workspace/sglang && mkdir -p /sgl-workspace/sglang && \
    tar -xf /tmp/source.tar -C /sgl-workspace/sglang && \
    rm -rf /tmp/source.tar /tmp/glm53-wheels /tmp/sgl_deep_gemm-0.1.7-py3-none-manylinux2014_x86_64.whl
ENV SGLANG_BUILD_COMMIT=${SOURCE_COMMIT} \
    SGLANG_BUILD_TREE=${SOURCE_TREE} \
    SGLANG_SOURCE_MANIFEST_SHA256=${SOURCE_MANIFEST_SHA256} \
    SGLANG_BUILD_URL=${BUILD_URL} \
    SGLANG_IMAGE_TAG=${IMAGE_TAG} \
    SGLANG_RUST_BUILD_MODE=never
LABEL org.opencontainers.image.source="https://github.com/bytedance-iaas/sglang" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.url="${BUILD_URL}" \
      org.opencontainers.image.version="${IMAGE_TAG}" \
      ai.sglang.build.commit="${SOURCE_COMMIT}" \
      ai.sglang.build.tree="${SOURCE_TREE}" \
      ai.sglang.build.source-manifest-sha256="${SOURCE_MANIFEST_SHA256}" \
      ai.sglang.build.base-image="iaas-gpu-cn-beijing.cr.volces.com/serving/sglang@sha256:d33d932aee374f3884e7f49c54447c82ce61ac9e58749ebf5aa3a50d25c9cb28" \
      ai.sglang.build.deep-gemm-wheel-sha256="f4e67086dc685ddcfcbb7833cc9770afd850cab23e173e77b9b18c19de0c2836" \
      ai.sglang.source.delivery="image-baked-no-runtime-install" \
      ai.sglang.image.tag="${IMAGE_TAG}"
WORKDIR /sgl-workspace/sglang
RUN env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    python3 /usr/local/share/sglang/glm53-flash/verify.py install \
      > /usr/local/share/sglang/glm53-flash/install-check.json
ENTRYPOINT []
CMD ["python3", "-m", "sglang.launch_server", "--help"]
