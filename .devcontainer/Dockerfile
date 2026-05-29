ARG BASE_IMAGE=lmsysorg/sglang:dev

FROM ${BASE_IMAGE} AS deepgemm_builder

ARG DEEPGEMM_REPO=https://github.com/sgl-project/DeepGEMM.git
ARG DEEPGEMM_REF=1554ec1d52010e49cdd18b75f57c4d04fee838c7
ARG SGL_DEEP_GEMM_VERSION=0.1.0+dev.pr36

USER root

RUN set -eux; \
    rm -rf /tmp/DeepGEMM; \
    git clone --no-checkout "${DEEPGEMM_REPO}" /tmp/DeepGEMM; \
    cd /tmp/DeepGEMM; \
    git fetch --depth=1 origin refs/pull/36/merge; \
    git checkout --detach FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = "${DEEPGEMM_REF}"; \
    git submodule update --init --recursive; \
    python3 -c 'from pathlib import Path; p = Path("sgl_deep_gemm/__init__.py"); text = p.read_text(); needle = "    mega_moe_pre_dispatch,\n)"; replacement = "    mega_moe_pre_dispatch,\n    transform_weights_for_mega_moe_sm90,\n    fp8_mega_moe,\n)"; assert needle in text; p.write_text(text.replace(needle, replacement))'; \
    echo -n "${SGL_DEEP_GEMM_VERSION}" > sgl_deep_gemm/VERSION; \
    bash build_sgl_deep_gemm.sh

FROM ${BASE_IMAGE}

COPY --from=deepgemm_builder /tmp/DeepGEMM/dist/sgl_deep_gemm-*.whl /tmp/

RUN set -eux; \
    python3 -m pip uninstall -y sgl-deep-gemm deep-gemm || true; \
    python3 -m pip install --no-deps --force-reinstall /tmp/sgl_deep_gemm-*.whl; \
    rm -f /tmp/sgl_deep_gemm-*.whl; \
    rm -rf /root/.cache/deep_gemm; \
    python3 -c 'import deep_gemm, importlib.metadata as md; required = ("fp8_mega_moe", "fp8_fp4_mega_moe", "mega_moe_pre_dispatch", "transform_weights_for_mega_moe_sm90"); missing = [name for name in required if not hasattr(deep_gemm, name)]; assert not missing, missing; print("sgl-deep-gemm:", md.version("sgl-deep-gemm")); print("deep_gemm:", deep_gemm.__file__)'

# Create non-root user with specified UID and GID
# NOTE: Replace with your own UID and GID. This is a workaround from https://github.com/microsoft/vscode-remote-release/issues/49#issuecomment-489060908.
ARG HOST_UID=1003
ARG HOST_GID=1003
RUN groupadd -g $HOST_GID devuser && \
    useradd -m -u $HOST_UID -g $HOST_GID -s /bin/zsh devuser

# Give devuser sudo access
RUN apt-get update && apt-get install -y sudo && \
    echo "devuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/devuser && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Set up oh-my-zsh for devuser
RUN cp -r /root/.oh-my-zsh /home/devuser/.oh-my-zsh && \
    cp /root/.zshrc /home/devuser/.zshrc && \
    sed -i 's|/root/.oh-my-zsh|/home/devuser/.oh-my-zsh|g' /home/devuser/.zshrc && \
    chown -R devuser:devuser /home/devuser/

# Set workspace directory and ownership
WORKDIR /sgl-workspace/sglang
RUN chown -R devuser:devuser /sgl-workspace

# Switch to devuser
USER devuser

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
