export SGLANG_DEBUGPY=1
export SGLANG_DEBUGPY_PORT=5678     # base port
export SGLANG_DEBUGPY_ONLY_RANK=0   # optional, break only in rank0

python local_prompt_demo.py \
  --model deepseek-ai/DeepSeek-V3.2-Exp \
  --prompt "Explain radix attention in 3 bullet points." \
  --tp 1 --dp 1 --enable-dp-attention \
  --moe-runner-backend asym_comp \
  --json-model-override-args '{"num_hidden_layers": 5}' \
  --context-length 8192 \
  --max-total-tokens 8192 \
  --temperature 0.0 \
  --nccl-port 29500 \
  --chunked-prefill-size 2048 \
  --mem-fraction-static 0.90 \
  --moe-expert-offload cpu