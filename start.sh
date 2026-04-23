#!/usr/bin/env bash
# GR00T-N1.7 launch script (mirrors /data/dongmao_dev/sglang/start.sh for
# alpamayo_r1).  Triton attention + disabled CUDA graphs, per the plan's
# F8 recipe — fa3 has flaky mem-efficient SDPA dispatch on sm121 and the
# custom DiT Euler loop composes more cleanly without graph capture.
FLASHINFER_DISABLE_VERSION_CHECK=1 \
CUDA_VISIBLE_DEVICES=0 \
python3 -m sglang.launch_server \
    --model-path /data/models/GR00T-N1.7-3B/ \
    --tokenizer-path /data/models/Cosmos-Reason2-2B/ \
    --port 30000 \
    --tp 1 \
    --attention-backend triton \
    --disable-cuda-graph \
    --disable-radix-cache \
    --skip-server-warmup
