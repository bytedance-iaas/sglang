#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/GLM-5.3-Flash}
DATASET_PATH=${DATASET_PATH:-/workload/longbench_v2_8k.json}
RESULT_ROOT=${RESULT_ROOT:-/results/service}
PORT=${PORT:-30000}
BASE_URL="http://127.0.0.1:${PORT}"
MODEL_NAME=GLM-5.3-Flash
server_pid=

mkdir -p "$RESULT_ROOT"

stop_server() {
  if [[ -z "${server_pid:-}" ]]; then
    return
  fi
  if kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM -- "-$server_pid" 2>/dev/null || kill -TERM "$server_pid" || true
    for _ in $(seq 1 60); do
      if ! kill -0 "$server_pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
  fi
  if kill -0 "$server_pid" 2>/dev/null; then
    kill -KILL -- "-$server_pid" 2>/dev/null || kill -KILL "$server_pid" || true
  fi
  wait "$server_pid" 2>/dev/null || true
  server_pid=
  sleep 5
}

trap stop_server EXIT

wait_ready() {
  local server_log=$1
  for _ in $(seq 1 360); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -200 "$server_log" >&2 || true
      return 1
    fi
    if curl -fsS "$BASE_URL/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  tail -200 "$server_log" >&2 || true
  return 1
}

real_smoke() {
  local run_dir=$1
  local code
  code=$(curl -sS -o "$run_dir/smoke.json" -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_NAME\",\"prompt\":\"Return one word: ready\",\"max_tokens\":8,\"temperature\":0}" \
    "$BASE_URL/v1/completions")
  [[ "$code" = 200 ]]
  python3 - "$run_dir/smoke.json" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1]))
text = data["choices"][0].get("text", "")
if not text.strip():
    raise SystemExit("empty smoke completion")
PY
}

start_server() {
  local label=$1
  local backend=$2
  local steps=$3
  local draft_tokens=$4
  local run_dir="$RESULT_ROOT/$label"
  mkdir -p "$run_dir"

  local command=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL_NAME"
    --trust-remote-code
    --host 0.0.0.0
    --port "$PORT"
    --tp-size 8
    --ep-size 8
    --attn-dp-size 8
    --mm-enable-dp-encoder
    --dsa-prefill-backend tilelang
    --dsa-decode-backend tilelang
    --kv-cache-dtype bfloat16
    --moe-runner-backend deep_gemm
    --moe-a2a-backend megamoe
    --mem-fraction-static 0.73
    --cuda-graph-max-bs-decode 128
    --max-running-requests 256
    --disable-radix-cache
    --speculative-algorithm EAGLE
    --speculative-num-steps "$steps"
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens "$draft_tokens"
    --speculative-moe-a2a-backend megamoe
    --speculative-moe-runner-backend deep_gemm
    --enable-linear-replayssm-spec
    --linear-attn-verify-backend "$backend"
  )
  printf '%q ' "${command[@]}" > "$run_dir/server-command.txt"
  printf '\n' >> "$run_dir/server-command.txt"
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader > "$run_dir/gpus-before.csv"
  setsid "${command[@]}" > "$run_dir/server.log" 2>&1 &
  server_pid=$!
  printf '%s\n' "$server_pid" > "$run_dir/server.pid"
  wait_ready "$run_dir/server.log"
  real_smoke "$run_dir"
  curl -fsS "$BASE_URL/metrics" > "$run_dir/metrics-ready.txt"
}

run_benchmark() {
  local run_dir=$1
  local name=$2
  local prompts=$3
  local concurrency=$4
  local seed=$5
  local output_len=${6:-1024}
  local command=(
    python3 -m sglang.benchmark.serving
    --backend sglang
    --base-url "$BASE_URL"
    --model "$MODEL_NAME"
    --dataset-name sharegpt
    --dataset-path "$DATASET_PATH"
    --apply-chat-template
    --sharegpt-output-len "$output_len"
    --num-prompts "$prompts"
    --max-concurrency "$concurrency"
    --request-rate inf
    --seed "$seed"
    --warmup-requests 0
    --extra-request-body '{"ignore_eos":true}'
    --output-details
    --disable-tqdm
    --ready-check-timeout-sec 0
    --output-file "$run_dir/$name.jsonl"
  )
  printf '%q ' "${command[@]}" > "$run_dir/$name.command.txt"
  printf '\n' >> "$run_dir/$name.command.txt"
  date --iso-8601=ns > "$run_dir/$name.start"
  "${command[@]}" > "$run_dir/$name.stdout" 2> "$run_dir/$name.stderr"
  date --iso-8601=ns > "$run_dir/$name.end"
  curl -fsS "$BASE_URL/metrics" > "$run_dir/$name.metrics.txt"
}

profile_path() {
  local run_dir=$1
  local label=$2
  local profile_dir="$run_dir/profile"
  mkdir -p "$profile_dir"
  curl -fsS -X POST "$BASE_URL/start_profile" \
    -H 'Content-Type: application/json' \
    -d "{\"output_dir\":\"$profile_dir\",\"num_steps\":16,\"activities\":[\"GPU\"],\"with_stack\":false,\"record_shapes\":false,\"profile_prefix\":\"$label\"}" \
    > "$run_dir/profile-start.txt"
  run_benchmark "$run_dir" profile-request 1 1 8099 64
  for _ in $(seq 1 60); do
    if find "$profile_dir" -type f -name '*.json.gz' -o -name '*.json' | grep -q .; then
      break
    fi
    sleep 2
  done
  find "$profile_dir" -type f -printf '%P\t%s\n' | sort \
    > "$run_dir/profile-files.txt"
  test -s "$run_dir/profile-files.txt"
}

run_t4_variant() {
  local label=$1
  local backend=$2
  local with_profile=$3
  local run_dir="$RESULT_ROOT/$label"
  start_server "$label" "$backend" 3 4
  run_benchmark "$run_dir" warmup-1 64 64 8001 1024
  run_benchmark "$run_dir" warmup-2 64 64 8002 1024
  if [[ "$with_profile" = yes ]]; then
    profile_path "$run_dir" "$label"
  fi
  run_benchmark "$run_dir" measured-64 64 64 8101 1024
  run_benchmark "$run_dir" measured-128 128 128 8102 1024
  run_benchmark "$run_dir" measured-256 256 256 8103 1024
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader > "$run_dir/gpus-after.csv"
  stop_server
}

run_t6_screen() {
  local label=$1
  local backend=$2
  local run_dir="$RESULT_ROOT/$label"
  start_server "$label" "$backend" 5 6
  run_benchmark "$run_dir" warmup-1 64 64 8201 1024
  run_benchmark "$run_dir" warmup-2 64 64 8202 1024
  run_benchmark "$run_dir" measured-128 128 128 8203 1024
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader > "$run_dir/gpus-after.csv"
  stop_server
}

test -s "$DATASET_PATH"
cp "$DATASET_PATH" "$RESULT_ROOT/workload.json"
cp /workload/longbench_v2_8k.metadata.json "$RESULT_ROOT/workload.metadata.json"

run_t4_variant b0a triton yes
run_t4_variant c0a nv_cutedsl yes
run_t4_variant b0b triton no
run_t4_variant c0b nv_cutedsl no
run_t6_screen b0t6 triton
run_t6_screen c0t6 nv_cutedsl

touch "$RESULT_ROOT/matrix-complete"
