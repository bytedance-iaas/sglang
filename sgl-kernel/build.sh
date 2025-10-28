#!/bin/bash
set -ex

if [ $# -lt 2 ]; then
  echo "Usage: $0 <PYTHON_VERSION> <CUDA_VERSION> [ARCH]"
  exit 1
fi

PYTHON_VERSION="$1"          # e.g. 3.10
CUDA_VERSION="$2"            # e.g. 12.9
ARCH="${3:-$(uname -i)}"     # optional override

if [ "${ARCH}" = "aarch64" ]; then
  BASE_IMG="pytorch/manylinuxaarch64-builder"
else
  BASE_IMG="pytorch/manylinux2_28-builder"
fi

# Create cache directories for persistent build artifacts in home directory
# Using home directory to persist across workspace cleanups/checkouts
CACHE_DIR="${HOME}/.cache/sgl-kernel"
BUILDX_CACHE_DIR="${CACHE_DIR}/buildx"
mkdir -p "${BUILDX_CACHE_DIR}"

# Ensure a buildx builder with docker-container driver (required for cache export)
BUILDER_NAME="sgl-kernel-builder"
if ! docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
  docker buildx create --name "${BUILDER_NAME}" --driver docker-container --use --bootstrap
else
  docker buildx use "${BUILDER_NAME}"
fi

PY_TAG="cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}"

# Output directory for wheels
DIST_DIR="dist"
mkdir -p "${DIST_DIR}"
docker run --rm \
   -v $(pwd):/sgl-kernel \
   -v ${CMAKE_DOWNLOAD_CACHE}:/cmake-downloads \
   -v ${CCACHE_DIR}:/ccache \
   -e ENABLE_CMAKE_PROFILE="${ENABLE_CMAKE_PROFILE:-}" \
   -e ENABLE_BUILD_PROFILE="${ENABLE_BUILD_PROFILE:-}" \
   -e USE_CCACHE="${USE_CCACHE:-1}" \
   ${DOCKER_IMAGE} \
   bash -c "
   set -e
   # Install CMake (version >= 3.26) - Robust Installation with caching
   echo \"==================================\"
   echo \"Installing CMake\"
   echo \"==================================\"
   # 使用内网源加速编译
   pip config set global.index-url https://bytedpypi.byted.org/simple
   # Install CMake (version >= 3.26) - Robust Installation
   export CMAKE_VERSION_MAJOR=3.31
   export CMAKE_VERSION_MINOR=1
   # Setting these flags to reduce OOM chance only on ARM
   export CMAKE_BUILD_PARALLEL_LEVEL=$(( $(nproc)/3 < 48 ? $(nproc)/3 : 48 ))
   if [ \"${ARCH}\" = \"aarch64\" ]; then
      export CUDA_NVCC_FLAGS=\"-Xcudafe --threads=2\"
      export MAKEFLAGS='-j2'
      export CMAKE_BUILD_PARALLEL_LEVEL=2
      export NINJAFLAGS='-j2'
      echo \"ARM detected: Using extra conservative settings (2 parallel jobs)\"
   fi

echo "----------------------------------------"
echo "Build configuration"
echo "PYTHON_VERSION: ${PYTHON_VERSION}"
echo "CUDA_VERSION:   ${CUDA_VERSION}"
echo "ARCH:           ${ARCH}"
echo "BASE_IMG:       ${BASE_IMG}"
echo "PYTHON_TAG:     ${PY_TAG}"
echo "Output:         ${DIST_DIR}/"
echo "Buildx cache:   ${BUILDX_CACHE_DIR}"
echo "Builder:        ${BUILDER_NAME}"
echo "----------------------------------------"

# Optional profiling build-args (empty string disables)
BUILD_ARGS=()
[ -n "${ENABLE_CMAKE_PROFILE:-}" ] && BUILD_ARGS+=(--build-arg ENABLE_CMAKE_PROFILE="${ENABLE_CMAKE_PROFILE}")
[ -n "${ENABLE_BUILD_PROFILE:-}" ] && BUILD_ARGS+=(--build-arg ENABLE_BUILD_PROFILE="${ENABLE_BUILD_PROFILE}")

docker buildx build \
  --builder "${BUILDER_NAME}" \
  -f Dockerfile . \
  --build-arg BASE_IMG="${BASE_IMG}" \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg ARCH="${ARCH}" \
  --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
  --build-arg PYTHON_TAG="${PY_TAG}" \
  "${BUILD_ARGS[@]}" \
  --cache-from type=local,src=${BUILDX_CACHE_DIR} \
  --cache-to type=local,dest=${BUILDX_CACHE_DIR},mode=max \
  --target artifact \
  --output "type=local,dest=${DIST_DIR}" \
  --network=host

echo "Done. Wheels are in ${DIST_DIR}/"
   # Debugging CMake
   echo \"PATH: \$PATH\"
   which cmake
   cmake --version

   if [ \"${USE_CCACHE}\" = \"1\" ]; then
      echo \"==================================\"
      echo \"Installing and configuring ccache\"
      echo \"==================================\"

      # Install ccache 4.12.1 from source for CUDA support (yum provides old 3.7.7)
      echo \"Installing ccache 4.12.1 from source...\"

      # Install build dependencies
      yum install -y gcc gcc-c++ make wget tar

      # Download and build ccache 4.12.1
      cd /tmp
      wget -q https://github.com/ccache/ccache/releases/download/v4.12.1/ccache-4.12.1.tar.xz
      tar -xf ccache-4.12.1.tar.xz
      cd ccache-4.12.1

      # Build and install (uses already-installed CMake 3.31)
      mkdir build && cd build
      /opt/cmake/bin/cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr .. >/dev/null
      make -j\$(nproc) >/dev/null
      make install >/dev/null

      # Verify installation
      ccache --version
      echo \"ccache 4.12.1 installed successfully\"
      cd /sgl-kernel

      # Configure ccache
      export CCACHE_DIR=/ccache
      export CCACHE_BASEDIR=/sgl-kernel
      export CCACHE_MAXSIZE=10G
      export CCACHE_COMPILERCHECK=content
      export CCACHE_COMPRESS=true
      export CCACHE_SLOPPINESS=file_macro,time_macros,include_file_mtime,include_file_ctime

      # Set up ccache as compiler launcher (don't use PATH to avoid -ccbin conflicts)
      export CMAKE_C_COMPILER_LAUNCHER=ccache
      export CMAKE_CXX_COMPILER_LAUNCHER=ccache
      export CMAKE_CUDA_COMPILER_LAUNCHER=ccache

      # Show ccache stats before build
      ccache -sV || true
      echo \"\"
   else
      echo \"==================================\"
      echo \"ccache disabled (USE_CCACHE=0)\"
      echo \"==================================\"
      echo \"\"
   fi

   yum install numactl-devel -y --nogpgcheck && \
   yum install libibverbs -y --nogpgcheck && \
   ln -sv /usr/lib64/libibverbs.so.1 /usr/lib64/libibverbs.so && \
   ${PYTHON_ROOT_PATH}/bin/${TORCH_INSTALL} && \
   ${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir ninja setuptools==75.0.0 wheel==0.41.0 numpy uv scikit-build-core && \
   export FLASHINFER_CUDA_ARCH_LIST='8.0 8.9 9.0a 10.0a 12.0a' && \
   export CUDA_VERSION=${CUDA_VERSION} && \
   mkdir -p /usr/lib/${ARCH}-linux-gnu/ && \
   ln -s /usr/local/cuda-${CUDA_VERSION}/targets/${LIBCUDA_ARCH}-linux/lib/stubs/libcuda.so /usr/lib/${ARCH}-linux-gnu/libcuda.so && \
   export CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}} && \
   export C_INCLUDE_PATH=/usr/local/cuda/include/cccl${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}} && \

   cd /sgl-kernel && \
   ls -la ${PYTHON_ROOT_PATH}/lib/python${PYTHON_VERSION}/site-packages/wheel/ && \

   # Enable CMake profiling if requested
   if [ -n \"${ENABLE_CMAKE_PROFILE}\" ]; then
      echo \"CMake profiling enabled - will save to /sgl-kernel/cmake-profile.json\"
      export CMAKE_ARGS=\"--profiling-output=/sgl-kernel/cmake-profile.json --profiling-format=google-trace\"
   fi

   export NINJA_STATUS=\"[%f/%t %es] \"
   # Enable Ninja build profiling if requested
   if [ -n \"${ENABLE_BUILD_PROFILE}\" ]; then
      echo \"Ninja build profiling enabled - will save to /sgl-kernel/build-trace.json\"
   fi

   # 修复ptxas编译失败
   bash ./replace_ptxas.sh ${CUDA_VERSION} && \
   PYTHONPATH=${PYTHON_ROOT_PATH}/lib/python${PYTHON_VERSION}/site-packages ${PYTHON_ROOT_PATH}/bin/python -m uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation && \
   ./rename_wheels.sh

   # Show profile location if profiling was enabled
   if [ -n \"${ENABLE_CMAKE_PROFILE}\" ] && [ -f /sgl-kernel/cmake-profile.json ]; then
      echo \"\"
      echo \"==================================\"
      echo \"CMake Profile Generated\"
      echo \"==================================\"
      echo \"Profile saved to: cmake-profile.json\"
      echo \"View in browser: chrome://tracing or edge://tracing\"
      echo \"\"
   fi

   # Generate Ninja build trace if profiling enabled
   if [ -n \"${ENABLE_BUILD_PROFILE}\" ] && [ -f /sgl-kernel/build/.ninja_log ]; then
      echo \"\"
      echo \"==================================\"
      echo \"Generating Ninja Build Trace\"
      echo \"==================================\"

      # Download ninjatracing script from GitHub (using PR #39 branch for ninja log v7 support)
      wget -q https://raw.githubusercontent.com/cradleapps/ninjatracing/084212eaf68f25c70579958a2ed67fb4ec2a9ca4/ninjatracing -O /tmp/ninjatracing || echo \"Note: Failed to download ninjatracing, skipping build trace\"

      # Convert .ninja_log to Chrome trace (JSON format)
      if [ -f /tmp/ninjatracing ]; then
         ${PYTHON_ROOT_PATH}/bin/python /tmp/ninjatracing /sgl-kernel/build/.ninja_log > /sgl-kernel/build-trace.json || true

         if [ -f /sgl-kernel/build-trace.json ]; then
            # Compress the trace for smaller file size and faster loading
            gzip -9 -k /sgl-kernel/build-trace.json 2>/dev/null || true

            echo \"Build trace saved to: build-trace.json\"
            if [ -f /sgl-kernel/build-trace.json.gz ]; then
               ORIGINAL_SIZE=\$(stat -f%z /sgl-kernel/build-trace.json 2>/dev/null || stat -c%s /sgl-kernel/build-trace.json)
               COMPRESSED_SIZE=\$(stat -f%z /sgl-kernel/build-trace.json.gz 2>/dev/null || stat -c%s /sgl-kernel/build-trace.json.gz)
               echo \"Compressed to: build-trace.json.gz (\${RATIO}% smaller)\"
            fi
            echo \"\"
            echo \"View in browser:\"
            echo \"  - chrome://tracing (load JSON file)\"
            echo \"  - ui.perfetto.dev (recommended, supports .gz files)\"
            echo \"\"
            echo \"Shows:\"
            echo \"  - Compilation time per file\"
            echo \"  - Parallelism utilization\"
            echo \"  - Critical path (longest dependency chain)\"
            echo \"  - Where the 2-hour build time went\"
         fi
      fi
      echo \"\"
   fi

   # Show ccache statistics after build
   if [ \"${USE_CCACHE}\" = \"1\" ]; then
      echo \"\"
      echo \"==================================\"
      echo \"ccache Statistics\"
      echo \"==================================\"
      ccache -s
      echo \"\"
   fi
   "