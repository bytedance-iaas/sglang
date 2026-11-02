#!/usr/bin/env bash
# Install the hash-locked agent toolchain into the current Python environment.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python3}"
"$PYTHON" -m pip install --require-hashes -r scripts/agent/requirements-agent.lock
pre-commit install --install-hooks >/dev/null 2>&1 || pre-commit install
echo "agent toolchain ready: $(pre-commit --version)"
