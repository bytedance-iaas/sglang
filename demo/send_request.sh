python3 -m sglang.bench_serving \
  --backend sglang-oai-chat \
  --base-url http://127.0.0.1:30000\
  --dataset-name sharegpt \
  --num-prompts 200 \
  --request-rate 4 \
  --max-concurrency 64 \
  --output-file results.jsonl \
  --output-details