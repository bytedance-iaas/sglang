#SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
CUDA_VISIBLE_DEVICES=4 \
python3 -m sglang.launch_server --model-path /data/models/Alpamayo-R1-10B-Origin \
 --tokenizer-path Qwen/Qwen3-VL-8B-Instruct \
 --port 30000 \
 --tp 1 \
 --disable-cuda-graph \
 --attention-backend triton  
# --tokenizer-path Qwen/Qwen3-VL-8B-Instruct \
# --tokenizer-path /data/models/Qwen3-VL-8B-Instruct \
# --skip-server-warmup 
 # flashinfer GOOD
 # triton GOOD
 # torch_native BUG
 # trtllm_mha has bug
