#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-recreate}"
POD="${POD:-sglang-eic-test}"
NAMESPACE="${NAMESPACE:-default}"
NODE_NAME="${NODE_NAME:-192.168.12.62}"
IMAGE="${IMAGE:-iaas-gpu-cn-beijing.cr.volces.com/serving/sglang:deepseek-v4-community}"
EIC_CONFIG_MAP="${EIC_CONFIG_MAP:-sglang-eic-cm}"
GPU_COUNT="${GPU_COUNT:-8}"
MODEL_PATH="${MODEL_PATH:-/data01/models/DeepSeek-V4-Flash}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SERVER_LOG="${SERVER_LOG:-/tmp/dsv4-eic-server.log}"
SERVER_ARGS="${SERVER_ARGS:---tp-size ${GPU_COUNT} --trust-remote-code --enable-hierarchical-cache --hicache-storage-backend eic --enable-eic-cache --mem-fraction-static 0.8}"
SERVER_PROC_PATTERN='[s]glang.launch_server|[s]glang serve'
SKIP_SGL_KERNEL_VERSION_CHECK="${SKIP_SGL_KERNEL_VERSION_CHECK:-1}"

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
          value: /sgl-workspace/config/remote-eic.yaml
        - name: SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK
          value: "${SKIP_SGL_KERNEL_VERSION_CHECK}"
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
test -s /sgl-workspace/config/remote-eic.yaml
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
  logs)
    logs
    ;;
  stop)
    stop
    ;;
  *)
    echo "usage: $0 [manifest|recreate|apply|status|smoke|serve|wait-server|request|logs|stop]" >&2
    exit 2
    ;;
esac
