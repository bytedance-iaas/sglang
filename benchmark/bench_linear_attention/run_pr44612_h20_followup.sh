#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT=${RESULT_ROOT:-/results/followup}
export RESULT_ROOT
source "$(dirname "$0")/run_pr44612_h20_service.sh"

GSM8K_PATH=${GSM8K_PATH:-/workload/gsm8k-test.jsonl}
mkdir -p "$RESULT_ROOT"

run_t6_confirmation() {
  local label=$1
  local backend=$2
  local run_dir="$RESULT_ROOT/$label"
  start_server "$label" "$backend" 5 6
  run_benchmark "$run_dir" warmup-1 64 64 8301 1024
  run_benchmark "$run_dir" warmup-2 64 64 8302 1024
  run_benchmark "$run_dir" measured-128 128 128 8303 1024
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader > "$run_dir/gpus-after.csv"
  stop_server
}

run_quality() {
  local label=$1
  local backend=$2
  local run_dir="$RESULT_ROOT/$label"
  mkdir -p "$run_dir"
  start_server "$label" "$backend" 3 4
  local command=(
    python3 -m sglang.test.run_eval
    --base-url "$BASE_URL"
    --model "$MODEL_NAME"
    --eval-name gsm8k
    --api completion
    --num-examples 100
    --num-threads 32
    --max-tokens 2048
    --temperature 0
    --top-p 0.95
    --num-shots 5
    --gsm8k-data-path "$GSM8K_PATH"
  )
  printf '%q ' "${command[@]}" > "$run_dir/gsm8k-100.command.txt"
  printf '\n' >> "$run_dir/gsm8k-100.command.txt"
  date --iso-8601=ns > "$run_dir/gsm8k-100.start"
  OPENAI_API_KEY=EMPTY "${command[@]}" \
    > "$run_dir/gsm8k-100.stdout" 2> "$run_dir/gsm8k-100.stderr"
  date --iso-8601=ns > "$run_dir/gsm8k-100.end"
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader > "$run_dir/gpus-after.csv"
  stop_server
}

test -s "$GSM8K_PATH"
sha256sum "$GSM8K_PATH" > "$RESULT_ROOT/gsm8k-test.sha256"
run_t6_confirmation b0t6b triton
run_t6_confirmation c0t6b nv_cutedsl
run_quality b0q triton
run_quality c0q nv_cutedsl
touch "$RESULT_ROOT/followup-complete"
