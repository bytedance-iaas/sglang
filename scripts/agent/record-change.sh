#!/usr/bin/env bash
# Append a key-change record to governance/AGENT_CHANGELOG.md.
# Usage: scripts/agent/record-change.sh "<summary>" [low|medium|high]
set -euo pipefail
cd "$(dirname "$0")/../.."
summary="${1:-}"
risk="${2:-low}"
if [ -z "$summary" ]; then
  echo "usage: make agent-record MSG='<summary>' [RISK=low|medium|high]" >&2
  exit 2
fi
case "$risk" in low|medium|high) ;; *) echo "RISK must be low, medium or high" >&2; exit 2 ;; esac
date="$(date +%Y-%m-%d)"
branch="$(git rev-parse --abbrev-ref HEAD)"
commit="$(git rev-parse --short HEAD)"
author="$(git config user.name || echo unknown)"
printf -- '- %s | %s | risk=%s | branch=%s | base=%s | %s\n' \
  "$date" "$author" "$risk" "$branch" "$commit" "$summary" >> governance/AGENT_CHANGELOG.md
echo "recorded in governance/AGENT_CHANGELOG.md"
