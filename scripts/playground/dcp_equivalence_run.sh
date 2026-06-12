#!/usr/bin/env bash
# DSv4 DCP equivalence regression launcher.
#
# Spins up two sglang servers (dcp_size=1 baseline, dcp_size=N candidate)
# back-to-back on a single H200/B200 node and compares their outputs via
# scripts/playground/dcp_equivalence_check.py.
#
# Required env:
#   MODEL_PATH        — path to the DSv4 checkpoint
# Optional env:
#   TP_SIZE           — default 16
#   DP_SIZE           — default 8
#   DCP_SIZE          — candidate dcp size, default 2
#   PORT_BASELINE     — default 30000
#   PORT_CANDIDATE    — default 30001
#   NUM_PROMPTS       — default 8
#   MAX_TOKENS        — default 64
#   EXTRA_ARGS        — passed to both servers (e.g. "--enable-hisparse")
#
# Example:
#   MODEL_PATH=/data/dsv4 bash scripts/playground/dcp_equivalence_run.sh
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
TP_SIZE="${TP_SIZE:-16}"
DP_SIZE="${DP_SIZE:-8}"
DCP_SIZE="${DCP_SIZE:-2}"
PORT_BASELINE="${PORT_BASELINE:-30000}"
PORT_CANDIDATE="${PORT_CANDIDATE:-30001}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
MAX_TOKENS="${MAX_TOKENS:-64}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/dcp_equiv_logs}"
mkdir -p "${LOG_DIR}"

cleanup() {
  local pid
  for pid in "${BASE_PID:-}" "${CAND_PID:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "[cleanup] kill ${pid}"
      kill -INT "${pid}" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

launch_server() {
  local port="$1"; shift
  local dcp="$1"; shift
  local logfile="$1"; shift
  local enable_dcp_env=()
  if [[ "${dcp}" -gt 1 ]]; then
    enable_dcp_env=(SGLANG_DSV4_ENABLE_DCP=1)
  fi
  echo "[launch] port=${port} dcp_size=${dcp} log=${logfile}"
  # shellcheck disable=SC2086
  env "${enable_dcp_env[@]}" \
      NCCL_GRAPH_MIXING_SUPPORT=0 \
      python -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --tp-size "${TP_SIZE}" \
      --dp-size "${DP_SIZE}" \
      --dcp-size "${dcp}" \
      --host 127.0.0.1 \
      --port "${port}" \
      ${EXTRA_ARGS} \
      > "${logfile}" 2>&1 &
  echo $!
}

BASE_LOG="${LOG_DIR}/baseline_dcp1_${PORT_BASELINE}.log"
CAND_LOG="${LOG_DIR}/candidate_dcp${DCP_SIZE}_${PORT_CANDIDATE}.log"

BASE_PID=$(launch_server "${PORT_BASELINE}" 1 "${BASE_LOG}")
CAND_PID=$(launch_server "${PORT_CANDIDATE}" "${DCP_SIZE}" "${CAND_LOG}")

echo "[wait] BASE_PID=${BASE_PID} CAND_PID=${CAND_PID}; tail logs in ${LOG_DIR}"

python "${WORK_DIR}/scripts/playground/dcp_equivalence_check.py" \
  --baseline-url "http://127.0.0.1:${PORT_BASELINE}" \
  --candidate-url "http://127.0.0.1:${PORT_CANDIDATE}" \
  --model-path "${MODEL_PATH}" \
  --num-prompts "${NUM_PROMPTS}" \
  --max-tokens "${MAX_TOKENS}"

ec=$?
echo "[done] equivalence-check exit=${ec}; logs preserved in ${LOG_DIR}"
exit "${ec}"
