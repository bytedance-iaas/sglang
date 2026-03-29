ARG BASE_IMAGE=lmsysorg/sglang:latest
FROM ${BASE_IMAGE}

# 将当前代码仓库复制到官方镜像的源码目录并重新安装
COPY . /sgl-workspace/sglang
WORKDIR /sgl-workspace/sglang
RUN pip3 install -e "python[all]"

WORKDIR /sgl-workspace/
RUN git clone https://github.com/ByteDance-Seed/Triton-distributed.git
WORKDIR /sgl-workspace/Triton-distributed

RUN git submodule deinit --all -f && \
    rm -rf 3rdparty/triton && \
    git submodule update --init --recursive

RUN pip3 install -v cuda-python ninja cmake wheel pybind11 numpy chardet pytest "nvidia-ml-py>=12" && \
    pip3 install -v 'nvidia-nvshmem-cu12' 'cuda.core==0.2.0' "Cython>=0.29.24" && \
    CPPFLAGS="-I/usr/local/cuda/include" pip3 install https://developer.download.nvidia.com/compute/nvshmem/redist/nvshmem_python/source/nvshmem_python-source-0.1.0.36132199_cuda12-archive.tar.xz -v

RUN pip3 install black "clang-format==19.1.2" pre-commit ruff yapf==0.43

RUN apt-get update && \
    apt-get install -y wget gnupg && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | gpg -o /etc/apt/keyrings/llvm-snapshot.gpg --dearmor && \
    echo "deb [signed-by=/etc/apt/keyrings/llvm-snapshot.gpg] https://mirrors.tuna.tsinghua.edu.cn/llvm-apt/jammy/ llvm-toolchain-jammy-19 main" >> /etc/apt/sources.list.d/llvm-apt.list && \
    apt-get update && apt-get install -y clang-19 llvm-19 libclang-19-dev

RUN pip uninstall triton -y && pip uninstall triton_dist -y
RUN USE_TRITON_DISTRIBUTED_AOT=0 pip3 install -e python --verbose --no-build-isolation --use-pep517

WORKDIR /sgl-workspace/sglang
# Default command
CMD ["/bin/bash"]
