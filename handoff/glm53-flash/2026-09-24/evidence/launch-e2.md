# 已验证 E2 启动组合：命令与解析边界

这是先前固定 C80 场景的**已验证起始对照**，不是本轮设计推荐，也不是当前在线服务。来源是 2026-09-23 的最终 `off-b10.yaml` 渲染、启动日志和 `poc_glm5.3-flash@57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c`。内部挂载、地址、设备名和 registry 已换成变量；性能相关数值、开关和角色差异与渲染命令一致。需要两台各 8×H20 的同 RDMA fabric Pod、已放置的模型与 DeepGEMM 0.1.7。以下命令**只作研究材料，本轮没有执行**。

在 P、D、Router 各自容器中提供 `SGLANG_SOURCE`、`DEEP_GEMM_SITE`、`MODEL_DIR`、`POD_IP`、`NCCL_IFNAME`、`RDMA_DEVICES`、`RDMA_GID_INDEX`。其中 `SGLANG_SOURCE` 必须是上述精确源码 commit/tree 的干净 checkout；`MODEL_DIR` 指向已核验的 GLM-5.3-Flash 权重，`DEEP_GEMM_SITE` 指向 ABI 匹配的 0.1.7 site-packages。P/D 分别提供本地可写 `CACHE_DIR`。Router 另提供 `PREFILL_URL`、`DECODE_URL`，例如集群内 HTTP 地址。变量不应从已结束任务的地址/Permit 猜测。

各容器先执行以下公共环境段：

```bash
set -euo pipefail
: "${SGLANG_SOURCE:?}" "${DEEP_GEMM_SITE:?}" "${MODEL_DIR:?}" "${POD_IP:?}"
: "${NCCL_IFNAME:?}" "${RDMA_DEVICES:?}" "${RDMA_GID_INDEX:?}"
: "${CACHE_DIR:?}"
test "$(git -C "$SGLANG_SOURCE" rev-parse HEAD)" = 57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c
test "$(git -C "$SGLANG_SOURCE" rev-parse 'HEAD^{tree}')" = 17ea448109e9ac7b7674b1b13fa64528380632d1
export PYTHONPATH="$SGLANG_SOURCE/python:$DEEP_GEMM_SITE:${PYTHONPATH:-}"
export NCCL_SOCKET_IFNAME="$NCCL_IFNAME" GLOO_SOCKET_IFNAME="$NCCL_IFNAME"
export NCCL_IB_HCA="$RDMA_DEVICES" NCCL_IB_GID_INDEX="$RDMA_GID_INDEX" NCCL_IB_DISABLE=0
export MC_GID_INDEX="$RDMA_GID_INDEX" MOONCAKE_PROTOCOL=rdma
export SGLANG_DSA_FP8_KV_LAYOUT=raw512
export SGLANG_OPT_KDA_ACCEPTED_STATE=0 SGLANG_HOST_IP="$POD_IP"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
export SGLANG_ENABLE_JIT_DEEPGEMM=1 SGLANG_DEEPGEMM_ON_H20=1
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
```

原渲染 entrypoint 还核对源码 clean tree、Python import 路径、DeepGEMM wheel SHA/ABI、模型 inventory、CUDA compat，并创建缓存/日志目录；上段仅保留可共享的精确源码门和性能环境。新运行须恢复这些实载检查，不能仅靠环境变量声明。

## Prefill：单个 8 GPU Pod

接公共环境段执行：

```bash
export SGLANG_DG_CACHE_DIR="$CACHE_DIR/deep_gemm"
export SGLANG_OPT_GLM5_NEXT_KDA_PROJECTION_FUSION=0
export SGLANG_DEEPGEMM_STANDARD_LAYOUT=auto SGLANG_PP_LAYER_PARTITION=24,21
python3 -m sglang.launch_server \
  --model-path "$MODEL_DIR" --served-model-name GLM-5.3-Flash \
  --chat-template "$MODEL_DIR/chat_template.jinja" \
  --dcp-size 1 --mem-fraction-static 0.70 --context-length 69632 \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 128 --disable-shared-experts-fusion \
  --reasoning-parser glm45 --tool-call-parser glm47 --skip-server-warmup \
  --speculative-algorithm EAGLE --speculative-num-steps 1 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \
  --enable-metrics --enable-metrics-for-all-schedulers \
  --tp-size 4 --ep-size 4 --pp-size 2 --max-running-requests 32 \
  --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
  --pp-max-micro-batch-size 1 --disable-overlap-schedule \
  --moe-runner-backend triton --speculative-moe-runner-backend triton \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 --disaggregation-ib-device "$RDMA_DEVICES" \
  --host 0.0.0.0 --port 31201 --nccl-port 21201
```

## Decode：单个 8 GPU Pod，Replay 关闭

接公共环境段执行：

```bash
export SGLANG_DG_CACHE_DIR="$CACHE_DIR/deep_gemm/decode-megamoe-cap1024"
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=1024
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_FAIL_CLOSED=1
export SGLANG_DEEPGEMM_STANDARD_LAYOUT=compact
export SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION=1
export SGLANG_OPT_FUSED_KDA_VERIFY=0
export SGLANG_OPT_GLM5_NEXT_KDA_PROJECTION_FUSION=1
export SGLANG_OPT_GLM5_NEXT_KDA_PROJECTION_FUSION_MODE=a_only
export SGLANG_OPT_FUSE_MHC_POST_PRE=0 SGLANG_RAGGED_VERIFY_MODE=static
python3 -m sglang.launch_server \
  --model-path "$MODEL_DIR" --served-model-name GLM-5.3-Flash \
  --chat-template "$MODEL_DIR/chat_template.jinja" \
  --dcp-size 1 --mem-fraction-static 0.90 --context-length 69632 \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 128 --disable-shared-experts-fusion \
  --reasoning-parser glm45 --tool-call-parser glm47 --skip-server-warmup \
  --speculative-algorithm EAGLE --speculative-num-steps 5 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  --enable-metrics --enable-metrics-for-all-schedulers \
  --tp-size 8 --dp-size 8 --ep-size 8 --pp-size 1 \
  --max-running-requests 80 --max-total-tokens 680000 \
  --disaggregation-decode-extra-slots 10 \
  --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
  --enable-dp-attention --enable-dp-attention-local-control-broadcast \
  --moe-runner-backend deep_gemm --speculative-moe-runner-backend deep_gemm \
  --moe-a2a-backend megamoe --speculative-moe-a2a-backend megamoe \
  --linear-attn-decode-backend triton --linear-attn-verify-backend triton \
  --mamba-ssm-dtype float32 --linear-replayssm-cache-len 16 \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "$RDMA_DEVICES" \
  --host 0.0.0.0 --port 31202 --nccl-port 21202
```

`--linear-replayssm-cache-len 16` 在这个命令中只是声明，未传 `--enable-linear-replayssm-spec`，故 Replay **关闭**。最终 D 日志解析为 `max_running_requests=80`、`max_total_tokens=680000`、`disaggregation_decode_extra_slots=10`、`enable_linear_replayssm_spec=false`、`mamba_ssm_dtype=float32`、`chunked_prefill_size=1024`（DP8 hook 从命令 8192 分配）、`disable_radix_cache=true`（PD Decode hook）。这是实际解析值，不等同于只看 YAML。

## Router：P/D 就绪并注册后启动

Router 容器需加载同一固定 SGLang Python 源码及其路由包；此处 `PREFILL_URL` 和 `DECODE_URL` 是已注册的集群内 worker URL，不含旧任务 DNS。

```bash
: "${PREFILL_URL:?}" "${DECODE_URL:?}"
python3 -m sglang_router.launch_router \
  --pd-disaggregation --prefill-policy round_robin --decode-policy round_robin \
  --prefill "$PREFILL_URL" 8998 --decode "$DECODE_URL" \
  --host 0.0.0.0 --port 30000 --prometheus-port 29000 \
  --request-timeout-secs 1800 --disable-retries \
  --worker-startup-timeout-secs 3600 --model-path "$MODEL_DIR"
```

Router、P、D 的 `/health`/Ready 不能证明生成正确；先前验证使用了经 Router 的真实请求。以上端口、拓扑、PD 超时和 RDMA 配置在新环境须与网络/模型许可重新核对。

## 已实现但未默认采用的 Replay 组合

固定源码支持在 D 上加 `--enable-linear-replayssm-spec`，保持 `T6/topk1/L16/static/Triton verify/FP32 state`；P 不开启。历史容量档位的 D `max-running-requests / max-total-tokens / extra slots` 分别为 `96 / 814144 / 10`、`128 / 1082432 / 10`、`160 / 1350720 / 10`。这只是已实现、曾验证可运行的组合，不是“最佳启动命令”的替代推荐。固定 C80 未证明吞吐改善，已有质量门未通过；不把容量扩张直接当作算法加速或本轮唯一研究方向。
