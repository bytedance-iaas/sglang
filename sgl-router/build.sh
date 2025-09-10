#!/bin/bash
set -ex

PYTHON_VERSION=$1
VIRTUAL_ENV="/opt/venv"
PATH="/root/.cargo/bin:$VIRTUAL_ENV/bin:/root/.local/bin:$PATH"

# install dependencies
apt update -y \
    && apt install -y git build-essential libssl-dev pkg-config curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

curl -LsSf https://astral.sh/uv/install.sh | sh

# install python
uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}


# install rustup from rustup.rs
export RUSTUP_DIST_SERVER="https://rsproxy.cn"
export RUSTUP_UPDATE_ROOT="https://rsproxy.cn/rustup"
curl --proto '=https' --tlsv1.2 -sSf https://rsproxy.cn/rustup-init.sh | sh -s -- -y \
    && rustc --version && cargo --version

cargo build --release \
    && uv build \
    && rm -rf /root/.cache
