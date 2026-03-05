SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0 \
CUDA_VISIBLE_DEVICES=7 \
SGLANG_ALPAMAYO_FM_DEBUG=0 \
SGLANG_ALPAMAYO_FM_DEBUG_MAX_STEPS=10 \
SGLANG_ALPAMAYO_FM_DISABLE_VLM_KV=0 \
python3 -m sglang.launch_server --model-path /data/models/Alpamayo-R1-10B \
 --port 29003 \
 --tp 1 \
 --disable-cuda-graph \
 --disable-overlap-schedule \
 --attention-backend triton \
 --skip-server-warmup 
 # flashinfer has bug
 # triton GOOD
 # torch_native BUG
 # trtllm_mha has bug
