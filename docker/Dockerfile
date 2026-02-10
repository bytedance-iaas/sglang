ARG CUDA_VERSION=12.9.1
# 使用已缓存好的base镜像
FROM hub.byted.org/iaas/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 as base
ARG TARGETARCH

ARG GRACE_BLACKWELL=0
# 配置 apt 将缓存放到 /root/.cache/apt 目录
RUN mkdir -p /root/.cache/apt/archives/partial /root/.cache/apt/lists/partial \
 && echo 'Dir::Cache "/root/.cache/apt";' > /etc/apt/apt.conf.d/99custom-cache \
 && echo 'Dir::Cache::archives "/root/.cache/apt/archives";' >> /etc/apt/apt.conf.d/99custom-cache \
 && echo 'Dir::State::lists "/root/.cache/apt/lists";' >> /etc/apt/apt.conf.d/99custom-cache \
 && echo 'APT::Sandbox::User "root";' >> /etc/apt/apt.conf.d/99custom-cache
# 编译之前，将源统一设置成内网的
RUN sed -i 's/archive.ubuntu.com/mirrors.byted.org/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.byted.org/g' /etc/apt/sources.list

ARG BUILD_TYPE=all
ARG BRANCH_TYPE=remote
ARG GRACE_BLACKWELL=0
ARG HOPPER_SBO=0

ARG GRACE_BLACKWELL_DEEPEP_BRANCH=gb200_blog_part_2
ARG HOPPER_SBO=1
ARG HOPPER_SBO_DEEPEP_COMMIT=9f2fc4b3182a51044ae7ecb6610f7c9c3258c4d6
ARG W4A8_PER_TENSOR_TRANSFER=1
ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG BUILD_AND_DOWNLOAD_PARALLEL=8
ARG SGL_KERNEL_VERSION=0.3.21
ARG CUSTOM_CACHE_TOS_AK=""
ARG CUSTOM_CACHE_TOS_BUCKET="iaas-servingkit"
ARG SGL_VERSION
ARG USE_LATEST_SGLANG=0
ARG GDRCOPY_VERSION=2.5.1
ARG PIP_DEFAULT_INDEX
ARG UBUNTU_MIRROR
ARG GITHUB_ARTIFACTORY=github.com
ARG INSTALL_FLASHINFER_JIT_CACHE=0
ARG FLASHINFER_VERSION=0.6.1

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-${GDRCOPY_VERSION}/ \
    FLASHINFER_VERSION=${FLASHINFER_VERSION}

# Add GKE default lib and bin locations
ENV PATH="${PATH}:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Replace Ubuntu sources if specified
RUN if [ -n "$UBUNTU_MIRROR" ]; then \
    sed -i "s|http://.*archive.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list && \
    sed -i "s|http://.*security.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list; \
fi

# Python setup (combined with apt update to reduce layers)
RUN --mount=type=cache,target=/var/cache/apt,id=base-apt \
    apt update && apt install -y --no-install-recommends wget software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt install -y --no-install-recommends python3.12-full python3.12-dev python3.10-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && wget -q https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py --break-system-packages \
    && rm get-pip.py \
    # Allow pip to install packages globally (PEP 668 workaround for Ubuntu 24.04)
    && python3 -m pip config set global.break-system-packages true \
    # Fix for apt-add-repository
    && cd /usr/lib/python3/dist-packages/ \
    && ln -s apt_pkg.cpython-310-*-linux-gnu.so apt_pkg.so

# 在安装pip后，设置pip源为火山内网源
RUN python3 -m pip config set global.index-url https://bytedpypi.byted.org/simple

# Set timezone and install all packages
RUN --mount=type=cache,target=/var/cache/apt,id=base-apt echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
 && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
 && apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    ca-certificates \
    software-properties-common \
    netcat-openbsd \
    kmod \
    unzip \
    openssh-server \
    curl \
    wget \
    lsof \
    locales \
    # Build essentials (needed for framework stage)
    build-essential \
    cmake \
    perl \
    patchelf \
    ccache \
    git-lfs \
    # MPI and NUMA
    libopenmpi-dev \
    libnuma1 \
    libnuma-dev \
    numactl \
    # transformers multimodal VLM
    ffmpeg \
    # InfiniBand/RDMA
    libibverbs-dev \
    libibverbs1 \
    libibumad3 \
    librdmacm1 \
    libnl-3-200 \
    libnl-route-3-200 \
    libnl-route-3-dev \
    libnl-3-dev \
    ibverbs-providers \
    infiniband-diags \
    perftest \
    # Development libraries
    libgoogle-glog-dev \
    libgtest-dev \
    libjsoncpp-dev \
    libunwind-dev \
    libboost-all-dev \
    libssl-dev \
    libgrpc-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    pybind11-dev \
    libhiredis-dev \
    libcurl4-openssl-dev \
    libczmq4 \
    libczmq-dev \
    libfabric-dev \
    # Package building tools
    devscripts \
    debhelper \
    fakeroot \
    dkms \
    check \
    libsubunit0 \
    libsubunit-dev \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Replace pip global cache if specified
RUN if [ -n "${PIP_DEFAULT_INDEX}" ]; then \
    python3 -m pip config set global.index-url ${PIP_DEFAULT_INDEX}; \
fi

# GDRCopy installation
RUN mkdir -p /tmp/gdrcopy && cd /tmp \
    && curl --retry 3 --retry-delay 2 -fsSL -o v${GDRCOPY_VERSION}.tar.gz \
        https://${GITHUB_ARTIFACTORY}/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz \
    && tar -xzf v${GDRCOPY_VERSION}.tar.gz && rm v${GDRCOPY_VERSION}.tar.gz \
    && cd gdrcopy-${GDRCOPY_VERSION}/packages \
    && CUDA=/usr/local/cuda ./build-deb-packages.sh \
    && dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb gdrcopy-tests_*.deb gdrcopy_*.deb \
    && cd / && rm -rf /tmp/gdrcopy

# Fix DeepEP IBGDA symlink
RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

########################################################
########## Framework Development Image ################
########################################################

# Copy local source if building from local
FROM scratch AS local_src
COPY . /src

FROM base AS framework

ARG BRANCH_TYPE
ARG BUILD_TYPE
ARG CUDA_VERSION
ARG BUILD_AND_DOWNLOAD_PARALLEL
ARG SGL_KERNEL_VERSION
ARG SGL_VERSION
ARG USE_LATEST_SGLANG
ARG INSTALL_FLASHINFER_JIT_CACHE
ARG FLASHINFER_VERSION
ARG GRACE_BLACKWELL
ARG GRACE_BLACKWELL_DEEPEP_BRANCH
ARG DEEPEP_COMMIT
ARG TRITON_LANG_COMMIT
ARG GITHUB_ARTIFACTORY

WORKDIR /sgl-workspace
ARG CUSTOM_CACHE_TOS_AK
ARG CUSTOM_CACHE_TOS_BUCKET
ARG CUDA_VERSION
ARG SGLANG_BRANCH

# 设置 ccache 环境变量
ENV CCACHE_DIR=/root/.cache/ccache
ENV PATH="/usr/lib/ccache:${PATH}"
ENV CMAKE_C_COMPILER_LAUNCHER=ccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# 安装 toscli 工具并下载 ccache 缓存
RUN if [ -n "$CUSTOM_CACHE_TOS_AK" ]; then \
      curl -s https://luban-source.byted.org/repository/scm/toutiao.tos.toscli_1.0.0.20.tar.gz -o - | tar xz -C . ./toscli \
      && chmod +x ./toscli \
      && mv ./toscli /usr/bin/ \
      && CACHE_TAR_NAME="ccache-${CUDA_VERSION}-${SGLANG_BRANCH}.tar.gz" \
      && echo "Downloading cache from TOS: ${CACHE_TAR_NAME}" \
      && http_proxy= https_proxy= toscli -endpoint tos-cn-north.byted.org -timeout 30m -bucket ${CUSTOM_CACHE_TOS_BUCKET} -accessKey ${CUSTOM_CACHE_TOS_AK} get -filename /tmp/${CACHE_TAR_NAME} cache/sglang-image/${CACHE_TAR_NAME} || true \
      && if [ -f "/tmp/${CACHE_TAR_NAME}" ]; then \
           echo "Cache file downloaded, size: $(du -h /tmp/${CACHE_TAR_NAME} | cut -f1)" \
           && mkdir -p /root/.cache \
           && tar -xzf /tmp/${CACHE_TAR_NAME} -C /root \
           && echo "Cache extracted successfully" \
           && echo "Cache contents after extraction:" \
           && du -sh /root/.cache/* 2>/dev/null || echo "  /root/.cache is empty or not accessible" \
           && rm -f /tmp/${CACHE_TAR_NAME}; \
         else \
           echo "No cache found, starting fresh build" \
           && mkdir -p /root/.cache; \
         fi \
    else \
      echo "CUSTOM_CACHE_TOS_AK not set, skipping cache download" \
      && mkdir -p /root/.cache; \
    fi


# 使用bytedance-iaas代码仓进行sglang安装
ARG SGLANG_BRANCH
COPY --from=local_src /src /tmp/local_src
RUN if [ "$BRANCH_TYPE" = "local" ]; then \
        cp -r /tmp/local_src /sgl-workspace/sglang; \
    else \
        git clone --depth=1 https://github.com/bytedance-iaas/sglang.git -b ${SGLANG_BRANCH} /sgl-workspace/sglang; \
    fi \
    && rm -rf /tmp/local_src

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel html5lib six \
    && cd sglang \
    && case "$CUDA_VERSION" in \
        12.6.1) CUINDEX=126 ;; \
        12.8.1) CUINDEX=128 ;; \
        12.9.1) CUINDEX=129 ;; \
        13.0.1) CUINDEX=130 ;; \
        *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
    && if [ "$CUDA_VERSION" = "12.6.1" ]; then \
        python3 -m pip install https://${GITHUB_ARTIFACTORY}/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu124-cp310-abi3-manylinux2014_$(uname -m).whl --force-reinstall --no-deps \
    ; \
    elif [ "$CUDA_VERSION" = "12.8.1" ] || [ "$CUDA_VERSION" = "12.9.1" ]; then \
        python3 -m pip install sgl-kernel==${SGL_KERNEL_VERSION} \
    ; \
    elif [ "$CUDA_VERSION" = "13.0.1" ]; then \
        python3 -m pip install https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu130-cp310-abi3-manylinux2014_$(uname -m).whl --force-reinstall --no-deps \
    ; \
    else \
        echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 \
    ; \
    fi \
    && python3 -m pip install -e "python[${BUILD_TYPE}]" --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX} \
    && if [ "$INSTALL_FLASHINFER_JIT_CACHE" = "1" ]; then \
        python3 -m pip install flashinfer-jit-cache==${FLASHINFER_VERSION} --index-url https://flashinfer.ai/whl/cu${CUINDEX} ; \
    fi \
    && FLASHINFER_CUBIN_DOWNLOAD_THREADS=${BUILD_AND_DOWNLOAD_PARALLEL} FLASHINFER_LOGGING_LEVEL=warning python3 -m flashinfer --download-cubin

# DeepEP
# We use Tom's DeepEP fork for GB200 for now; the 1fd57b0276311d035d16176bb0076426166e52f3 commit is https://github.com/fzyzcjy/DeepEP/tree/gb200_blog_part_2
# TODO: move from Tom's branch to DeepEP hybrid-ep branch
# We use the nvshmem version that ships with torch 2.9.1
# CU12 uses 3.3.20 and CU13 uses 3.3.24
RUN set -eux; \
    if [ "$GRACE_BLACKWELL" = "1" ]; then \
      git clone https://github.com/fzyzcjy/DeepEP.git && \
      cd DeepEP && \
      git checkout ${GRACE_BLACKWELL_DEEPEP_BRANCH} && \
      sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh && \
      cd .. ; \
    elif [ "$HOPPER_SBO" = "1" ]; then \
      git clone https://github.com/deepseek-ai/DeepEP.git -b antgroup-opt && \
      cd DeepEP && \
      git checkout ${HOPPER_SBO_DEEPEP_COMMIT} && \
      sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh && \
      cd .. ; \
    elif [ "$W4A8_PER_TENSOR_TRANSFER" = "1" ]; then \
      git clone https://github.com/bytedance-iaas/DeepEP.git -b support_per_tensor_transfer && \
      cd DeepEP && \
      sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh && \
      cd .. ; \
    else \
        curl --retry 3 --retry-delay 2 -fsSL -o ${DEEPEP_COMMIT}.zip \
            https://${GITHUB_ARTIFACTORY}/deepseek-ai/DeepEP/archive/${DEEPEP_COMMIT}.zip && \
        unzip -q ${DEEPEP_COMMIT}.zip && rm ${DEEPEP_COMMIT}.zip && mv DeepEP-${DEEPEP_COMMIT} DeepEP && cd DeepEP && \
        sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh && \
        cd .. ; \
    fi

# Install DeepEP
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /sgl-workspace/DeepEP && \
    case "$CUDA_VERSION" in \
        12.6.1) \
            CHOSEN_TORCH_CUDA_ARCH_LIST='9.0' \
            ;; \
        12.8.1) \
            # FIXED: 12.8.1 does NOT support Blackwell 10.3 \
            CHOSEN_TORCH_CUDA_ARCH_LIST='9.0;10.0' \
            ;; \
        12.9.1|13.0.1) \
            # 12.9.1+ properly supports Blackwell 10.3 \
            CHOSEN_TORCH_CUDA_ARCH_LIST='9.0;10.0;10.3' \
            ;; \
        *) \
            echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 \
            ;; \
    esac && \
    if [ "${CUDA_VERSION%%.*}" = "13" ]; then \
        sed -i "/^    include_dirs = \['csrc\/'\]/a\    include_dirs.append('${CUDA_HOME}/include/cccl')" setup.py; \
    fi && \
    TORCH_CUDA_ARCH_LIST="${CHOSEN_TORCH_CUDA_ARCH_LIST}" MAX_JOBS=${BUILD_AND_DOWNLOAD_PARALLEL} pip install --no-build-isolation .

# Install essential Python packages
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
    datamodel_code_generator \
    mooncake-transfer-engine==0.3.8.post1 \
    pre-commit \
    pytest \
    black \
    isort \
    icdiff \
    uv \
    wheel \
    scikit-build-core \
    nixl \
    py-spy \
    cubloaty \
    google-cloud-storage

# Build and install sgl-model-gateway (install Rust, build, then remove to save space)
RUN --mount=type=cache,target=/root/.cache/pip \
    curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && export PATH="/root/.cargo/bin:${PATH}" \
    && rustc --version && cargo --version \
    && python3 -m pip install maturin \
    && cd /sgl-workspace/sglang/sgl-model-gateway/bindings/python \
    && ulimit -n 65536 && maturin build --release --features vendored-openssl --out dist \
    && python3 -m pip install --force-reinstall dist/*.whl \
    && cd /sgl-workspace/sglang/sgl-model-gateway \
    && cargo build --release --bin sglang-router --features vendored-openssl \
    && cp target/release/sglang-router /usr/local/bin/sglang-router \
    && rm -rf /root/.cargo /root/.rustup target dist ~/.cargo \
    && sed -i '/\.cargo\/env/d' /root/.profile /root/.bashrc 2>/dev/null || true

# Patching packages for CUDA 12/13 compatibility
# TODO: Remove when torch version covers these packages
RUN --mount=type=cache,target=/root/.cache/pip if [ "${CUDA_VERSION%%.*}" = "12" ]; then \
    python3 -m pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps ; \
    python3 -m pip install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall --no-deps ; \
elif [ "${CUDA_VERSION%%.*}" = "13" ]; then \
    python3 -m pip install nvidia-nccl-cu13==2.28.3 --force-reinstall --no-deps ; \
    python3 -m pip install nvidia-cublas==13.1.0.3 --force-reinstall --no-deps ; \
    python3 -m pip install nixl-cu13 --no-deps ; \
    python3 -m pip install cuda-python==13.1.1 ; \
fi

# Install development tools
RUN --mount=type=cache,target=/var/cache/apt,id=framework-apt \
    apt-get update && apt-get install -y --no-install-recommends \
    gdb \
    ninja-build \
    vim \
    tmux \
    htop \
    zsh \
    tree \
    silversearcher-ag \
    cloc \
    pkg-config \
    bear \
    less \
    rdma-core \
    openssh-server \
    gnuplot \
    infiniband-diags \
    perftest \
    ibverbs-providers \
    libibumad3 \
    libibverbs1 \
    libnl-3-200 \
    libnl-route-3-200 \
    librdmacm1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 配置 apt 将缓存放到 /root/.cache/apt 目录
RUN mkdir -p /root/.cache/apt/archives/partial /root/.cache/apt/lists/partial \
 && echo 'Dir::Cache "/root/.cache/apt";' > /etc/apt/apt.conf.d/99custom-cache \
 && echo 'Dir::Cache::archives "/root/.cache/apt/archives";' >> /etc/apt/apt.conf.d/99custom-cache \
 && echo 'Dir::State::lists "/root/.cache/apt/lists";' >> /etc/apt/apt.conf.d/99custom-cache \
 && echo 'APT::Sandbox::User "root";' >> /etc/apt/apt.conf.d/99custom-cache

RUN --mount=type=cache,target=/var/cache/apt apt update -y \
    && apt install -y --no-install-recommends gnupg \
    && echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu2004/$(if [ "$(uname -m)" = "aarch64" ]; then echo "arm64"; else echo "amd64"; fi) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/$(if [ "$(uname -m)" = "aarch64" ]; then echo "arm64"; else echo "x86_64"; fi)/7fa2af80.pub \
    && apt update -y \
    && apt install -y --no-install-recommends nsight-systems-cli \
    && rm -rf /var/lib/apt/lists/*

# Install minimal Python dev packages
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --break-system-packages \
    pytest \
    black \
    isort \
    icdiff \
    scikit-build-core \
    uv \
    pre-commit \
    pandas \
    matplotlib \
    tabulate \
    termplotlib

# diff-so-fancy
RUN curl --retry 3 --retry-delay 2 -LSso /usr/local/bin/diff-so-fancy \
        https://${GITHUB_ARTIFACTORY}/so-fancy/diff-so-fancy/releases/download/v1.4.4/diff-so-fancy \
    && chmod +x /usr/local/bin/diff-so-fancy

# clang-format
RUN curl --retry 3 --retry-delay 2 -LSso /usr/local/bin/clang-format \
        https://${GITHUB_ARTIFACTORY}/muttleyxd/clang-tools-static-binaries/releases/download/master-32d3ac78/clang-format-16_linux-amd64 \
    && chmod +x /usr/local/bin/clang-format

# Install clangd
# 使用clangd的缓存包加速编译
RUN wget https://tosv.byted.org/obj/iaas-servingkit/clangd-linux-18.1.3.zip \
    && mv clangd-linux-18.1.3.zip clangd.zip \
#RUN curl -L https://github.com/clangd/clangd/releases/download/18.1.3/clangd-linux-18.1.3.zip -o clangd.zip \
    && unzip clangd.zip \
    && cp -r clangd_18.1.3/bin/* /usr/local/bin/ \
    && cp -r clangd_18.1.3/lib/* /usr/local/lib/ \
    && rm -rf clangd_18.1.3 clangd.zip

# CMake
RUN CMAKE_VERSION=3.31.1 \
    && ARCH=$(uname -m) \
    && CMAKE_INSTALLER="cmake-${CMAKE_VERSION}-linux-${ARCH}" \
    && wget "https://tosv.byted.org/obj/iaas-servingkit/${CMAKE_INSTALLER}.tar.gz" \
    && tar -xzf "${CMAKE_INSTALLER}.tar.gz" \
    && cp -r "${CMAKE_INSTALLER}/bin/"* /usr/local/bin/ \
    && cp -r "${CMAKE_INSTALLER}/share/"* /usr/local/share/ \
    && rm -rf "${CMAKE_INSTALLER}" "${CMAKE_INSTALLER}.tar.gz"

# Install just
RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://just.systems/install.sh | \
    sed "s|https://github.com|https://${GITHUB_ARTIFACTORY}|g" | \
    bash -s -- --tag 1.42.4 --to /usr/local/bin

# Add yank script
COPY --chown=root:root --chmod=755 docker/configs/yank /usr/local/bin/yank

# Install oh-my-zsh and plugins
RUN sh -c "$(curl --retry 3 --retry-delay 2 -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && git clone --depth 1 https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && git clone --depth 1 https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# These configs are optional; users can override them by mounting their own files
COPY docker/configs/opt/.vimrc /opt/sglang/.vimrc
COPY docker/configs/opt/.tmux.conf /opt/sglang/.tmux.conf
COPY docker/configs/opt/.gitconfig /opt/sglang/.gitconfig

# Configure development environment
COPY docker/configs/.zshrc /root/.zshrc

# 使用pip安装just，加速编译流程
#RUN set -euxo ; \
#    curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
RUN pip install rust-just

# 上传 ccache 缓存到 TOS
ARG CUSTOM_CACHE_TOS_AK
ARG CUSTOM_CACHE_TOS_BUCKET
ARG CUDA_VERSION
RUN if [ -n "$CUSTOM_CACHE_TOS_AK" ] ; then \
      CACHE_TAR_NAME="ccache-${CUDA_VERSION}-${SGLANG_BRANCH}.tar.gz" \
      && echo "=== Preparing to upload cache to TOS: ${CACHE_TAR_NAME} ===" \
      && rm -rf /root/.cache/cargo-target /root/.cache/flashinfer \
      && echo "Cache contents before archiving:" \
      && du -sh /root/.cache/* 2>/dev/null || echo "  /root/.cache is empty" \
      && cd /root \
      && tar -czf /tmp/${CACHE_TAR_NAME} .cache \
      && echo "Cache archive created, size: $(du -h /tmp/${CACHE_TAR_NAME} | cut -f1)" \
      && echo "Archive contents:" \
      && tar -tzf /tmp/${CACHE_TAR_NAME} | head -20 \
      && echo "..." \
      && http_proxy= https_proxy= toscli -endpoint tos-cn-north.byted.org -timeout 10m -bucket ${CUSTOM_CACHE_TOS_BUCKET} -accessKey ${CUSTOM_CACHE_TOS_AK} del cache/sglang-image/${CACHE_TAR_NAME} || true \
      && http_proxy= https_proxy= toscli -endpoint tos-cn-north.byted.org -timeout 30m -bucket ${CUSTOM_CACHE_TOS_BUCKET} -accessKey ${CUSTOM_CACHE_TOS_AK} put -verbose -prefix cache/sglang-image/ -name ${CACHE_TAR_NAME} /tmp/${CACHE_TAR_NAME} \
      && echo "=== Cache uploaded successfully ===" \
      && rm -f /tmp/${CACHE_TAR_NAME}; \
    else \
      echo "CUSTOM_CACHE_TOS_AK not set, skipping cache upload"; \
    fi

RUN python3 -m pip install --upgrade "urllib3>=2.6.3"

# Set workspace directory
WORKDIR /sgl-workspace/sglang

# 将源统一切换成火山源
RUN pip config set global.index-url https://mirrors.ivolces.com/pypi/simple/
RUN rm -rf /var/lib/apt/lists/* && \
    sed -i 's/byted.org/ivolces.com/g' /etc/apt/sources.list.d/* && \
    sed -i 's/byted.org/ivolces.com/g' /etc/apt/sources.list