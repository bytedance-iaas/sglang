#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-recreate}"
POD="${POD:-sglang-eic-test}"
NAMESPACE="${NAMESPACE:-default}"
NODE_NAME="${NODE_NAME:-192.168.12.62}"
IMAGE="${IMAGE:-iaas-gpu-cn-beijing.cr.volces.com/serving/sglang:deepseek-v4-community}"
EIC_CONFIG_MAP="${EIC_CONFIG_MAP:-sglang-eic-cm}"
REMOTE_EIC_YAML="${REMOTE_EIC_YAML:-/sgl-workspace/config/remote-eic.yaml}"
GPU_COUNT="${GPU_COUNT:-8}"
MODEL_PATH="${MODEL_PATH:-/data01/models/DeepSeek-V4-Flash}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SERVER_LOG="${SERVER_LOG:-/tmp/dsv4-eic-server.log}"
ENABLE_EIC="${ENABLE_EIC:-1}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-marlin}"
BASE_SERVER_ARGS="${BASE_SERVER_ARGS:---tp-size ${GPU_COUNT} --trust-remote-code --mem-fraction-static 0.8 --disable-cuda-graph --moe-runner-backend ${MOE_RUNNER_BACKEND}}"
if [[ "${ENABLE_EIC}" == "1" ]]; then
  DEFAULT_SERVER_ARGS="${BASE_SERVER_ARGS} --enable-hierarchical-cache --enable-eic-cache"
else
  DEFAULT_SERVER_ARGS="${BASE_SERVER_ARGS}"
fi
SERVER_ARGS="${SERVER_ARGS:-${DEFAULT_SERVER_ARGS}}"
SERVER_PROC_PATTERN='[s]glang.launch_server|[s]glang serve'
SKIP_SGL_KERNEL_VERSION_CHECK="${SKIP_SGL_KERNEL_VERSION_CHECK:-1}"
SGLANG_NUMA_BIND_V2="${SGLANG_NUMA_BIND_V2:-0}"
SGLANG_OPT_DEEPGEMM_HC_PRENORM="${SGLANG_OPT_DEEPGEMM_HC_PRENORM:-false}"
SGLANG_ENABLE_UNIFIED_RADIX_TREE="${SGLANG_ENABLE_UNIFIED_RADIX_TREE:-${ENABLE_EIC}}"

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
raise SystemExit(0 if ret == 0 else 1)
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
  manifest)
    manifest
    ;;
  recreate)
    kubectl -n "${NAMESPACE}" delete pod "${POD}" --ignore-not-found --wait=true
    manifest | kubectl apply -f -
    status
    ;;
  apply)
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
    echo "usage: $0 [manifest|recreate|apply|status|smoke|serve|wait-server|request|eic-check|logs|stop]" >&2
    exit 2
    ;;
esac
