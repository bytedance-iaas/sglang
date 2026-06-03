#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-recreate}"
POD="${POD:-sglang-eic-test}"
NAMESPACE="${NAMESPACE:-default}"
NODE_NAME="${NODE_NAME:-192.168.12.62}"
IMAGE="${IMAGE:-iaas-gpu-cn-beijing.cr.volces.com/serving/sglang:deepseek-v4-community}"
EIC_CONFIG_MAP="${EIC_CONFIG_MAP:-sglang-dsv4-eic-cm}"
REMOTE_EIC_YAML="${REMOTE_EIC_YAML:-/sgl-workspace/config/remote-eic.yaml}"
APPLY_EIC_CONFIG="${APPLY_EIC_CONFIG:-1}"
EIC_REMOTE_URL="${EIC_REMOTE_URL:-eic://192.168.12.62:12500}"
EIC_MASTER_ADDR="${EIC_MASTER_ADDR:-${EIC_REMOTE_URL#eic://}}"
EIC_INSTANCE_ID="${EIC_INSTANCE_ID:-h20-perf}"
EIC_REGION_NAME="${EIC_REGION_NAME:-cn-shanghai}"
EIC_CLUSTER_ZONE="${EIC_CLUSTER_ZONE:-cn-shanghai-d}"
EIC_TRANS_TYPE="${EIC_TRANS_TYPE:-0}"
EIC_THREAD_NUM="${EIC_THREAD_NUM:-2}"
EIC_LOG_DIR="${EIC_LOG_DIR:-/sgl-workspace/log}"
EIC_CLIENT_RPC_TIMEOUT_MS="${EIC_CLIENT_RPC_TIMEOUT_MS:-30000}"
EIC_CLIENT_KV_REQ_TIMEOUT_MS="${EIC_CLIENT_KV_REQ_TIMEOUT_MS:-30000}"
EIC_CLIENT_SLICE_QOS_TIMEOUT_MS="${EIC_CLIENT_SLICE_QOS_TIMEOUT_MS:-30000}"
EIC_CLIENT_MIN_RPC_TIMEOUT_MS="${EIC_CLIENT_MIN_RPC_TIMEOUT_MS:-30000}"
EIC_CLIENT_KV_ENABLE_SMART_TIMEOUT="${EIC_CLIENT_KV_ENABLE_SMART_TIMEOUT:-false}"
EIC_CLIENT_ENABLE_SLICE_TASK_COPY_BYPASS="${EIC_CLIENT_ENABLE_SLICE_TASK_COPY_BYPASS:-false}"
EIC_CLIENT_ENABLE_SLICE_TASK_COPY_THREAD="${EIC_CLIENT_ENABLE_SLICE_TASK_COPY_THREAD:-false}"
EIC_CLIENT_ENABLE_KV_SET_CRC="${EIC_CLIENT_ENABLE_KV_SET_CRC:-true}"
EIC_CLIENT_KV_GET_ENABLE_GDR="${EIC_CLIENT_KV_GET_ENABLE_GDR:-false}"
EIC_CLIENT_KV_SET_ENABLE_GDR="${EIC_CLIENT_KV_SET_ENABLE_GDR:-false}"
EIC_USE_POLLING_MODE="${EIC_USE_POLLING_MODE:-true}"
EIC_USE_BYTE_EXPRESS="${EIC_USE_BYTE_EXPRESS:-false}"
EIC_ENABLE_MULTI_NIC="${EIC_ENABLE_MULTI_NIC:-false}"
EIC_CLIENT_MULTI_NET_LOCAL_INTERFACE_NAMES="${EIC_CLIENT_MULTI_NET_LOCAL_INTERFACE_NAMES:-eth0}"
EIC_CLIENT_DRAM_MEMPOOL_LIMIT_BYTES="${EIC_CLIENT_DRAM_MEMPOOL_LIMIT_BYTES:-17179869184}"
EIC_CLIENT_SPLIT_KV_SLICE_SIZE_BYTE="${EIC_CLIENT_SPLIT_KV_SLICE_SIZE_BYTE:-65536}"
EIC_ENABLE_KVSET_GPU_DIRECT="${EIC_ENABLE_KVSET_GPU_DIRECT:-False}"
EIC_ENABLE_KVGET_GPU_DIRECT="${EIC_ENABLE_KVGET_GPU_DIRECT:-False}"
EIC_ENABLE_KVSET_DIRECT="${EIC_ENABLE_KVSET_DIRECT:-False}"
EIC_ENABLE_GPU_NIC_AFFINITY="${EIC_ENABLE_GPU_NIC_AFFINITY:-False}"
EIC_MAX_BATCH_SIZE="${EIC_MAX_BATCH_SIZE:-1}"
GPU_COUNT="${GPU_COUNT:-8}"
MODEL_PATH="${MODEL_PATH:-/data01/models/DeepSeek-V4-Flash}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SERVER_LOG="${SERVER_LOG:-/tmp/dsv4-eic-server.log}"
ENABLE_EIC="${ENABLE_EIC:-1}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-marlin}"
WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-1800}"
BASE_SERVER_ARGS="${BASE_SERVER_ARGS:---tp-size ${GPU_COUNT} --trust-remote-code --mem-fraction-static 0.8 --disable-cuda-graph --disable-overlap-schedule --watchdog-timeout ${WATCHDOG_TIMEOUT} --moe-runner-backend ${MOE_RUNNER_BACKEND}}"
if [[ "${ENABLE_EIC}" == "1" ]]; then
  DEFAULT_SERVER_ARGS="${BASE_SERVER_ARGS} --enable-hierarchical-cache --enable-eic-cache --hicache-io-backend kernel"
else
  DEFAULT_SERVER_ARGS="${BASE_SERVER_ARGS}"
fi
SERVER_ARGS="${SERVER_ARGS:-${DEFAULT_SERVER_ARGS}}"
SERVER_PROC_PATTERN='[s]glang.launch_server|[s]glang serve'
SKIP_SGL_KERNEL_VERSION_CHECK="${SKIP_SGL_KERNEL_VERSION_CHECK:-1}"
SGLANG_NUMA_BIND_V2="${SGLANG_NUMA_BIND_V2:-0}"
SGLANG_OPT_DEEPGEMM_HC_PRENORM="${SGLANG_OPT_DEEPGEMM_HC_PRENORM:-false}"
SGLANG_ENABLE_UNIFIED_RADIX_TREE="${SGLANG_ENABLE_UNIFIED_RADIX_TREE:-${ENABLE_EIC}}"
SGLANG_WARMUP_TIMEOUT="${SGLANG_WARMUP_TIMEOUT:-${WATCHDOG_TIMEOUT}}"

eic_config_manifest() {
  cat <<YAML
apiVersion: v1
kind: ConfigMap
metadata:
  name: ${EIC_CONFIG_MAP}
  namespace: ${NAMESPACE}
data:
  eic_flag_file: |-
    --eic_region_name=${EIC_REGION_NAME}
    --eic_cluster_zone=${EIC_CLUSTER_ZONE}
    --eic_cluster_uuid=${EIC_INSTANCE_ID}
    --eic_client_master_addr_list=${EIC_MASTER_ADDR}
    --eic_client_using_fixed_master=true
    --eic_route_view_type=1
    --block_group_space_name=eic
    --eic_client_log_file_size_mb=200
    --eic_client_log_file_num=20
    --eic_client_log_dir=${EIC_LOG_DIR}
    --eic_enable_io_audit_log=false
    --byterpc_enable_time_profiler=false
    --byterpc_enable_loop_metrics=false
    --eic_client_rpc_timeout_in_ms=${EIC_CLIENT_RPC_TIMEOUT_MS}
    --eic_client_kv_req_timeout_in_ms=${EIC_CLIENT_KV_REQ_TIMEOUT_MS}
    --eic_client_enable_slice_task_copy_bypass=${EIC_CLIENT_ENABLE_SLICE_TASK_COPY_BYPASS}
    --eic_client_enable_slice_task_copy_thread=${EIC_CLIENT_ENABLE_SLICE_TASK_COPY_THREAD}
    --eic_client_slice_qos_mode=1
    --eic_client_slice_qos_sliding_window_tx_size=32
    --eic_client_slice_qos_sliding_window_rx_size=32
    --eic_client_slice_qos_tx_through_kb=1000000000
    --eic_client_slice_qos_rx_through_kb=1000000000
    --eic_client_enable_kv_set_crc=${EIC_CLIENT_ENABLE_KV_SET_CRC}
    --eic_client_kv_get_check_crc_type=0
    --eic_client_kv_get_enable_gdr=${EIC_CLIENT_KV_GET_ENABLE_GDR}
    --eic_client_kv_set_enable_gdr=${EIC_CLIENT_KV_SET_ENABLE_GDR}
    --eic_use_polling_mode=${EIC_USE_POLLING_MODE}
    --eic_use_byte_express=${EIC_USE_BYTE_EXPRESS}
    --eic_enable_multi_nic=${EIC_ENABLE_MULTI_NIC}
    --eic_client_multi_net_local_interface_names=${EIC_CLIENT_MULTI_NET_LOCAL_INTERFACE_NAMES}
    --eic_client_dram_mempool_limit_bytes=${EIC_CLIENT_DRAM_MEMPOOL_LIMIT_BYTES}
    --eic_client_slice_qos_timeout_ms=${EIC_CLIENT_SLICE_QOS_TIMEOUT_MS}
    --eic_client_min_rpc_timeout_in_ms=${EIC_CLIENT_MIN_RPC_TIMEOUT_MS}
    --eic_client_kv_enable_smart_timeout=${EIC_CLIENT_KV_ENABLE_SMART_TIMEOUT}
    --eic_client_split_kv_slice_size_byte=${EIC_CLIENT_SPLIT_KV_SLICE_SIZE_BYTE}
  remote-eic.yaml: |-
    remote_url: "${EIC_REMOTE_URL}"
    eic_instance_id: "${EIC_INSTANCE_ID}"
    chunk_size: 256
    local_device: null
    max_local_cache_size: 5
    eic_thread_num: ${EIC_THREAD_NUM}
    eic_log_dir: "${EIC_LOG_DIR}"
    eic_log_level: 2
    eic_trans_type: ${EIC_TRANS_TYPE}
    eic_flag_file: "/sgl-workspace/config/eic_flag_file"
    remote_serde: null
    pipelined_backend: False
    save_decode_cache: False
    enable_blending: False
    blend_recompute_ratio: 0.5
    blend_min_tokens: 256
    enable_kvset_gpu_direct: ${EIC_ENABLE_KVSET_GPU_DIRECT}
    enable_kvget_gpu_direct: ${EIC_ENABLE_KVGET_GPU_DIRECT}
    enable_kvset_direct: ${EIC_ENABLE_KVSET_DIRECT}
    enable_async_kvset: False
    enable_gpu_nic_affinity: ${EIC_ENABLE_GPU_NIC_AFFINITY}
    eic_direct_backup: False
    eic_direct_writeback: False
    eic_max_batch_size: ${EIC_MAX_BATCH_SIZE}
    load_remote_threshold: 1000
    load_back_check: True
YAML
}

apply_eic_config() {
  eic_config_manifest | kubectl -n "${NAMESPACE}" apply -f -
}

manifest() {
  cat <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: ${POD}
  namespace: ${NAMESPACE}
  labels:
    app: sglang
    component: prefill
spec:
  nodeName: ${NODE_NAME}
  hostNetwork: true
  hostIPC: true
  dnsPolicy: ClusterFirstWithHostNet
  restartPolicy: Always
  containers:
    - name: worker
      image: ${IMAGE}
      imagePullPolicy: Always
      command: ["/bin/sh", "-c"]
      args: ["sleep infinity"]
      workingDir: /sgl-workspace
      securityContext:
        privileged: true
      env:
        - name: NCCL_IB_GID_INDEX
          value: "3"
        - name: NCCL_IB_DISABLE
          value: "0"
        - name: NCCL_IB_HCA
          value: mlx5_
        - name: NCCL_SOCKET_IFNAME
          value: eth0
        - name: GLOO_SOCKET_IFNAME
          value: eth0
        - name: REMOTE_EIC_YAML
          value: ${REMOTE_EIC_YAML}
        - name: SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK
          value: "${SKIP_SGL_KERNEL_VERSION_CHECK}"
        - name: SGLANG_NUMA_BIND_V2
          value: "${SGLANG_NUMA_BIND_V2}"
        - name: SGLANG_OPT_DEEPGEMM_HC_PRENORM
          value: "${SGLANG_OPT_DEEPGEMM_HC_PRENORM}"
        - name: SGLANG_ENABLE_UNIFIED_RADIX_TREE
          value: "${SGLANG_ENABLE_UNIFIED_RADIX_TREE}"
        - name: SGLANG_WARMUP_TIMEOUT
          value: "${SGLANG_WARMUP_TIMEOUT}"
        - name: MY_HOST_IP
          valueFrom:
            fieldRef:
              fieldPath: status.hostIP
      resources:
        limits:
          nvidia.com/gpu: ${GPU_COUNT}
        requests:
          nvidia.com/gpu: ${GPU_COUNT}
      volumeMounts:
        - name: data01
          mountPath: /data01
        - name: data02
          mountPath: /data02
        - name: shared-mem
          mountPath: /dev/shm
        - name: nvidia-topologyd
          mountPath: /var/run/nvidia-topologyd
        - name: eic-config
          mountPath: /sgl-workspace/config
  volumes:
    - name: shared-mem
      emptyDir:
        medium: Memory
    - name: data01
      hostPath:
        path: /data01
        type: DirectoryOrCreate
    - name: data02
      hostPath:
        path: /data02
        type: DirectoryOrCreate
    - name: nvidia-topologyd
      hostPath:
        path: /var/run/nvidia-topologyd
        type: DirectoryOrCreate
    - name: eic-config
      configMap:
        name: ${EIC_CONFIG_MAP}
        defaultMode: 0755
YAML
}

status() {
  kubectl -n "${NAMESPACE}" get pod "${POD}" -o wide
  kubectl -n "${NAMESPACE}" get pod "${POD}" \
    -o jsonpath='{.spec.containers[0].image}{"\n"}{.status.containerStatuses[0].imageID}{"\n"}' || true
  kubectl -n "${NAMESPACE}" get events \
    --field-selector involvedObject.name="${POD}" \
    --sort-by=.lastTimestamp | tail -20 || true
}

smoke() {
  kubectl -n "${NAMESPACE}" wait --for=condition=Ready "pod/${POD}" --timeout=20m
  kubectl -n "${NAMESPACE}" exec "${POD}" -- bash -lc '
set -euo pipefail
python3 - <<PY
import importlib
import importlib.metadata as metadata

for dist in ("sglang", "eic"):
    try:
        print(f"{dist}={metadata.version(dist)}")
    except Exception as exc:
        print(f"{dist}=missing:{type(exc).__name__}")

for mod in (
    "sglang.srt.configs.deepseek_v4",
    "sglang.srt.models.deepseek_v4",
    "sglang.srt.mem_cache.eic_memory_pool",
    "sglang.srt.mem_cache.eic_hiradix_cache",
    "sglang.srt.mem_cache.eic_chunk_cache",
):
    importlib.import_module(mod)
    print("OK", mod)
PY
test -s "${REMOTE_EIC_YAML:-/sgl-workspace/config/remote-eic.yaml}"
test -s /sgl-workspace/config/eic_flag_file
python3 -m sglang.launch_server --help 2>&1 | grep -E -- "--enable-eic-cache|--hicache-storage-backend"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
'
}

serve() {
  kubectl -n "${NAMESPACE}" wait --for=condition=Ready "pod/${POD}" --timeout=20m
  kubectl -n "${NAMESPACE}" exec "${POD}" -- bash -lc "
set -euo pipefail
pkill -f '${SERVER_PROC_PATTERN}' || true
rm -f '${SERVER_LOG}'
"
  kubectl -n "${NAMESPACE}" exec "${POD}" -- bash -lc "
set -euo pipefail
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK='${SKIP_SGL_KERNEL_VERSION_CHECK}' \
REMOTE_EIC_YAML='${REMOTE_EIC_YAML}' \
SGLANG_NUMA_BIND_V2='${SGLANG_NUMA_BIND_V2}' \
SGLANG_OPT_DEEPGEMM_HC_PRENORM='${SGLANG_OPT_DEEPGEMM_HC_PRENORM}' \
SGLANG_ENABLE_UNIFIED_RADIX_TREE='${SGLANG_ENABLE_UNIFIED_RADIX_TREE}' \
SGLANG_WARMUP_TIMEOUT='${SGLANG_WARMUP_TIMEOUT}' \
nohup python3 -m sglang.launch_server \
  --model-path '${MODEL_PATH}' \
  --host 0.0.0.0 \
  --port '${SGLANG_PORT}' \
  ${SERVER_ARGS} \
  >'${SERVER_LOG}' 2>&1 &
echo \$! >/tmp/dsv4-eic-server.pid
"
  logs
}

eic_check() {
  kubectl -n "${NAMESPACE}" exec "${POD}" -- bash -lc "
set -euo pipefail
REMOTE_EIC_YAML='${REMOTE_EIC_YAML}' python3 - <<'PY'
import time
import yaml

import eic
import torch

config_file = '${REMOTE_EIC_YAML}'
config = yaml.safe_load(open(config_file))
endpoint = config['remote_url'][len('eic://'):]
init_option = eic.InitOption()
init_option.log_dir = config.get('eic_log_dir', '/tmp')
init_option.log_level = eic.LogLevel(config.get('eic_log_level', 2))
init_option.transport_type = eic.TransportType(config.get('eic_trans_type', 3))
init_option.flag_file = config.get('eic_flag_file')
if config.get('enable_gpu_nic_affinity', False):
    init_option.multi_net_local_interface_names = 'eth1'

client = eic.Client()
start = time.time()
print(f\"config={config_file}\")
print(f\"instance={config.get('eic_instance_id')} endpoint={endpoint}\")
ret = client.init(config.get('eic_instance_id'), endpoint, init_option)
print(f\"ret={ret} cost={time.time() - start:.2f}s\")
if ret != 0:
    raise SystemExit(1)

key = f'sglang-dsv4-eic-check-{int(time.time() * 1000)}'
value = b'hello-eic'
keys = eic.StringVector()
keys.append(key)
vals = eic.IOBuffers()
value_tensor = torch.tensor(list(value), dtype=torch.uint8, device='cpu')
vals.append(value_tensor.data_ptr(), value_tensor.numel(), False)
set_option = eic.SetOption()
set_option.ns = config.get('eic_namespace', '')
set_code, set_status = client.mset(keys, vals, set_option)
print(f'mset code={set_code} status={[str(x) for x in set_status.status_codes]}')

out = eic.IOBuffers()
out_tensor = torch.empty_like(value_tensor)
out.append(out_tensor.data_ptr(), out_tensor.numel(), False)
get_option = eic.GetOption()
get_option.ns = config.get('eic_namespace', '')
get_code, _, get_status = client.mget(keys, get_option, out)
got = bytes(out_tensor.tolist()).decode()
print(f'mget code={get_code} status={[str(x) for x in get_status.status_codes]} value={got!r}')
ok = (
    set_code == eic.StatusCode.SUCCESS
    and get_code == eic.StatusCode.SUCCESS
    and got == value.decode()
)
raise SystemExit(0 if ok else 1)
PY
"
}

wait_server() {
  kubectl -n "${NAMESPACE}" exec "${POD}" -- bash -lc "
set -euo pipefail
for _ in \$(seq 1 180); do
  if curl -fsS http://127.0.0.1:${SGLANG_PORT}/health >/dev/null 2>&1; then
    echo ready
    exit 0
  fi
  if ! pgrep -f '${SERVER_PROC_PATTERN}' >/dev/null; then
    tail -200 '${SERVER_LOG}' || true
    exit 1
  fi
  sleep 5
done
tail -200 '${SERVER_LOG}' || true
exit 1
"
}

request() {
  wait_server
  kubectl -n "${NAMESPACE}" exec "${POD}" -- bash -lc "
set -euo pipefail
curl -fsS http://127.0.0.1:${SGLANG_PORT}/generate \
  -H 'Content-Type: application/json' \
  -d '{\"text\":\"hello\",\"sampling_params\":{\"max_new_tokens\":8,\"temperature\":0}}'
echo
"
}

logs() {
  kubectl -n "${NAMESPACE}" exec "${POD}" -- bash -lc "tail -200 '${SERVER_LOG}' 2>/dev/null || true"
}

stop() {
  kubectl -n "${NAMESPACE}" exec "${POD}" -- bash -lc "
pkill -f '${SERVER_PROC_PATTERN}' || true
rm -f /tmp/dsv4-eic-server.pid
"
}

case "${ACTION}" in
  config)
    apply_eic_config
    ;;
  config-manifest)
    eic_config_manifest
    ;;
  manifest)
    manifest
    ;;
  recreate)
    if [[ "${APPLY_EIC_CONFIG}" == "1" ]]; then
      apply_eic_config
    fi
    kubectl -n "${NAMESPACE}" delete pod "${POD}" --ignore-not-found --wait=true
    manifest | kubectl apply -f -
    status
    ;;
  apply)
    if [[ "${APPLY_EIC_CONFIG}" == "1" ]]; then
      apply_eic_config
    fi
    manifest | kubectl apply -f -
    status
    ;;
  status)
    status
    ;;
  smoke)
    smoke
    ;;
  serve)
    serve
    ;;
  wait-server)
    wait_server
    ;;
  request)
    request
    ;;
  eic-check)
    eic_check
    ;;
  logs)
    logs
    ;;
  stop)
    stop
    ;;
  *)
    echo "usage: $0 [config|config-manifest|manifest|recreate|apply|status|smoke|serve|wait-server|request|eic-check|logs|stop]" >&2
    exit 2
    ;;
esac
