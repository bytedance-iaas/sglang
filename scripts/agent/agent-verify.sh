#!/usr/bin/env bash
# Full agent gate, identical locally and in CI (.github/workflows/agent-verify.yml).
# Runs without GPUs or the sglang runtime installed; GPU suites stay in pr-test.yml.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python3}"

scripts/agent/agent-check.sh

echo "==> lint-script unit tests"
"$PYTHON" -m unittest discover -s scripts/lint -p 'test_check_*.py'
echo "agent-verify: OK"
