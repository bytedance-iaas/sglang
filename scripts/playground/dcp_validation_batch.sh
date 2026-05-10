#!/usr/bin/env bash
set -euo pipefail

# Runs a small batch of /v1/completions requests against a running server and
# writes JSONL outputs for quick diffing.
#
# Usage:
#   PORT=30000 OUT=/data01/code/foo/resp.jsonl bash scripts/playground/dcp_validation_batch.sh
#
# Optional:
#   HOST=127.0.0.1
#   MODEL=/data00/models/MiniMax-M2.7
#   TEMP=0
#   MAX_TOKENS=16

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
MODEL="${MODEL:-/data00/models/MiniMax-M2.7}"
TEMP="${TEMP:-0}"
MAX_TOKENS="${MAX_TOKENS:-16}"
OUT="${OUT:?OUT must be set to an absolute path for the output JSONL}"

prompts=(
  "Hello"
  "Hello DCP probe 050905"
  "Write a short answer: 1+1="
  "Explain in one sentence what a transformer is."
  "Give 3 keywords about airplanes."
)

mkdir -p "$(dirname "$OUT")"
: > "$OUT"

for p in "${prompts[@]}"; do
  payload=$(
    PROMPT_TEXT="$p" MODEL_PATH_FOR_REQ="$MODEL" MAX_TOKENS_FOR_REQ="$MAX_TOKENS" TEMP_FOR_REQ="$TEMP" \
    python3 - <<PY
import json
import os
print(json.dumps({
  "model": os.environ["MODEL_PATH_FOR_REQ"],
  "prompt": os.environ["PROMPT_TEXT"],
  "max_tokens": int(os.environ["MAX_TOKENS_FOR_REQ"]),
  "temperature": float(os.environ["TEMP_FOR_REQ"]),
}))
PY
  )

  curl -s "http://${HOST}:${PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "${payload}" >> "$OUT"
  echo >> "$OUT"
done

echo "wrote ${OUT}"
