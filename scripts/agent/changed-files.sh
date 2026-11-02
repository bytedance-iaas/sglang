#!/usr/bin/env bash
# Print files changed relative to the merge-base with AGENT_BASE_REF, including
# uncommitted and untracked files. Deleted files are omitted.
set -euo pipefail
cd "$(dirname "$0")/../.."
base_ref="${AGENT_BASE_REF:-origin/main}"
if ! git rev-parse --verify --quiet "$base_ref^{commit}" >/dev/null; then
  echo "changed-files: base ref '$base_ref' not found; run 'git fetch origin' or set AGENT_BASE_REF" >&2
  exit 2
fi
merge_base="$(git merge-base "$base_ref" HEAD)"
{
  git diff --name-only --diff-filter=d "$merge_base"
  git ls-files --others --exclude-standard
} | sort -u
