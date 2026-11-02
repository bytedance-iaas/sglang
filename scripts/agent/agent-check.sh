#!/usr/bin/env bash
# Fast agent gate: repository governance checks + pre-commit on changed files.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python3}"

echo "==> governance checks"
"$PYTHON" scripts/lint/check_agent_governance.py

echo "==> pre-commit on changed files (base: ${AGENT_BASE_REF:-origin/main})"
changed="$(scripts/agent/changed-files.sh)"
if [ -z "$changed" ]; then
  echo "no changed files"
else
  # shellcheck disable=SC2086
  SKIP="${SKIP:-no-commit-to-branch}" pre-commit run --files $changed
fi
echo "agent-check: OK"
