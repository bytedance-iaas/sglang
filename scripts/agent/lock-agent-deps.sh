#!/usr/bin/env bash
# Regenerate scripts/agent/requirements-agent.lock (hash-pinned) from requirements-agent.in.
set -euo pipefail
cd "$(dirname "$0")"
uv pip compile requirements-agent.in \
  --python-version 3.12 --universal --generate-hashes \
  --output-file requirements-agent.lock
