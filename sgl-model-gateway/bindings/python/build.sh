#!/bin/bash
set -ex

PYTHON_VERSION=$1
VIRTUAL_ENV="/opt/venv"
PATH="/root/.cargo/bin:$VIRTUAL_ENV/bin:/root/.local/bin:$PATH"

# install dependencies
apt update -y \
    && apt install -y git build-essential libssl-dev pkg-config curl zip\
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

VER=25.3
cd /tmp
curl -L -o protoc.zip "https://github.com/protocolbuffers/protobuf/releases/download/v${VER}/protoc-${VER}-linux-x86_64.zip"
unzip -o protoc.zip -d /usr/local
chmod +x /usr/local/bin/protoc
/usr/local/bin/protoc --version
export PROTOC=/usr/local/bin/protoc
export PATH=/usr/local/bin:$PATH

# install rustup from rustup.rs
export RUSTUP_DIST_SERVER="https://rsproxy.cn"
export RUSTUP_UPDATE_ROOT="https://rsproxy.cn/rustup"
curl --proto '=https' --tlsv1.2 -sSf https://rsproxy.cn/rustup-init.sh | sh -s -- -y \
    && rustc --version && cargo --version


pip install maturin
maturin build --release --out dist --features vendored-openssl