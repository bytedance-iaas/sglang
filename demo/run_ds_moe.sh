# export PYTHONPATH=/sgl-workspace/sglang/python:$PYTHONPATH
# export PYTHON_EXECUTABLE="python -Xfrozen_modules=off"
# export SGLANG_DEBUGPY=1
# export DEBUGPY_BASE_PORT=5678
# export DEBUGPY_WAIT=1
# export DEBUGPY_WAIT_RANK=0   # only rank0 pauses waiting for VSCode
# export PYDEVD_DISABLE_FILE_VALIDATION=1
# export SGLANG_DEBUGPY_MATCH=sglang.launch_server
# asym_comp deep_gemm
#    --moe-a2a-backend deepep \

python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp \
 --tp 1 --dp 1 --enable-dp-attention \
 --moe-runner-backend asym_comp \
 --load-format dummy \
 --json-model-override-args '{"num_hidden_layers": 5}' \
 --context-length 8192 \
 --max-total-tokens 8192 \
 --chunked-prefill-size 2048 \
 --mem-fraction-static 0.90 \
 --moe-expert-offload cpu

