#8B-FP8

#with cache
CUDA_VISIBLE_DEVICES=0  SGLANG_VLM_CACHE_SIZE_MB=1024 NCCL_MIN_NCHANNELS=24 NCCL_IB_QPS_PER_CONNECTION=8 SGLANG_USE_MODELSCOPE=1 python3 -m sglang.launch_server --prefill-attention-backend fa3 --decode-attention-backend flashinfer --model-path /data01/models/Qwen3-VL-8B-Instruct-FP8/ --host 127.0.0.1 --port 8010 --mem-fraction-static 0.5 --cuda-graph-max-bs 128 --tensor-parallel-size 1 --mm-attention-backend fa3 --cuda-graph-bs 128 120 112 104 96 88 80 72 64 56 48 40 32 24 16 8 4 2 1 &

#wocache
CUDA_VISIBLE_DEVICES=1  SGLANG_VLM_CACHE_SIZE_MB=0 NCCL_MIN_NCHANNELS=24 NCCL_IB_QPS_PER_CONNECTION=8 SGLANG_USE_MODELSCOPE=1 python3 -m sglang.launch_server --prefill-attention-backend fa3 --decode-attention-backend flashinfer --model-path /data01/models/Qwen3-VL-8B-Instruct-FP8/ --host 127.0.0.1 --port 8020 --mem-fraction-static 0.5 --cuda-graph-max-bs 128 --tensor-parallel-size 1 --mm-attention-backend fa3 --cuda-graph-bs 128 120 112 104 96 88 80 72 64 56 48 40 32 24 16 8 4 2 1 --disable-radidx-cache &


#30B-Moe-FP8

#with cache
CUDA_VISIBLE_DEVICES=2  SGLANG_VLM_CACHE_SIZE_MB=1024 NCCL_MIN_NCHANNELS=24 NCCL_IB_QPS_PER_CONNECTION=8 SGLANG_USE_MODELSCOPE=1 python3 -m sglang.launch_server --prefill-attention-backend fa3 --decode-attention-backend flashinfer --model-path /data01/models/Qwen3-VL-30B-A3B-Instruct-FP8 --host 127.0.0.1 --port 8030 --mem-fraction-static 0.8 --cuda-graph-max-bs 128 --tensor-parallel-size 1 --mm-attention-backend fa3 --cuda-graph-bs 128 120 112 104 96 88 80 72 64 56 48 40 32 24 16 8 4 2 1 &

#wocache
CUDA_VISIBLE_DEVICES=3  SGLANG_VLM_CACHE_SIZE_MB=0 NCCL_MIN_NCHANNELS=24 NCCL_IB_QPS_PER_CONNECTION=8 SGLANG_USE_MODELSCOPE=1 python3 -m sglang.launch_server --prefill-attention-backend fa3 --decode-attention-backend flashinfer --model-path /data01/models/Qwen3-VL-30B-A3B-Instruct-FP8 --host 127.0.0.1 --port 8040 --mem-fraction-static 0.8 --cuda-graph-max-bs 128 --tensor-parallel-size 1 --mm-attention-backend fa3 --cuda-graph-bs 128 120 112 104 96 88 80 72 64 56 48 40 32 24 16 8 4 2 1 --disable-radix-cache &
