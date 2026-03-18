SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
CUDA_VISIBLE_DEVICES=6 \
python3 -m sglang.launch_server --model-path /data/models/Alpamayo-R1-10B-Origin \
 --tokenizer-path Qwen/Qwen3-VL-8B-Instruct \
 --port 29003 \
 --tp 1 \
 --disable-cuda-graph \
 --enable-deterministic-inference \
 --attention-backend triton \  
# --tokenizer-path Qwen/Qwen3-VL-8B-Instruct \
# --tokenizer-path /data/models/Qwen3-VL-8B-Instruct \
# --enable-deterministic-inference \
# --skip-server-warmup 
 # flashinfer GOOD
 # triton GOOD
 # torch_native GOOD
 # fa3 GOOD
 # trtllm_mha has unknown
