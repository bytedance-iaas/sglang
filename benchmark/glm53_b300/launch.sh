#!/usr/bin/env bash
set -euo pipefail
# Run inside the pinned image on 8 B300 GPUs; model/cache mounts are in deployment.yaml.
python3 -m sglang.launch_server \
  --model-path /models/GLM-5.3 \
  --served-model-name glm-5.3 \
  --tp 8 \
  --dtype bfloat16 \
  --quantization fp8 \
  --attention-backend dsa \
  --kv-cache-dtype fp8_e4m3 \
  --dsa-prefill-backend trtllm \
  --dsa-decode-backend trtllm \
  --dsa-topk-backend sgl-kernel \
  --moe-runner-backend deep_gemm \
  --context-length 262144 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 32768 \
  --max-running-requests 256 \
  --cuda-graph-max-bs-decode 256 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-cache-report \
  --stream-response-default-include-usage \
  --enable-metrics \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-prefill-cp \
  --cp-strategy interleave \
  --dsa-paged-mqa-logits-backend cutedsl
