CONCURRENCY=$1
PROT_NUM=$2

for i in {1..5};
do
    python3 -m sglang.bench_serving --backend sglang-oai-chat --dataset-name image --num-prompts $PROT_NUM --apply-chat-template --random-output-len 64 --random-input-len 32 --image-resolution 448x448 --image-format jpeg --image-count 1 --image-content blank --random-range-ratio 1 --max-concurrency $CONCURRENCY --host=127.0.0.1 --port=8080;
done > /root/benchmark_fp8/8080_$CONCURRENCY.log

for i in {1..5};
do
    python3 -m sglang.bench_serving --backend sglang-oai-chat --dataset-name image --num-prompts $PROT_NUM --apply-chat-template --random-output-len 64 --random-input-len 32 --image-resolution 448x448 --image-format jpeg --image-count 1 --image-content blank --random-range-ratio 1 --max-concurrency $CONCURRENCY --host=127.0.0.1 --port=8070;
done > /root/benchmark_fp8/8070_$CONCURRENCY.log

# for i in {1..3};
# do
#     python3 -m sglang.bench_serving --backend sglang-oai-chat --dataset-name image --num-prompts $PROT_NUM --apply-chat-template --random-output-len 64 --random-input-len 32 --image-resolution 448x448 --image-format jpeg --image-count 1 --image-content blank --random-range-ratio 1 --max-concurrency $CONCURRENCY --host=127.0.0.1 --port=8060;
# done > 8060_$CONCURRENCY.log

# for i in {1..3};
# do
#     python3 -m sglang.bench_serving --backend sglang-oai-chat --dataset-name image --num-prompts $PROT_NUM --apply-chat-template --random-output-len 64 --random-input-len 32 --image-resolution 448x448 --image-format jpeg --image-count 1 --image-content blank --random-range-ratio 1 --max-concurrency $CONCURRENCY --host=127.0.0.1 --port=8050;
# done > 8050_$CONCURRENCY.log

# for i in {1..3};
# do
#     python3 -m sglang.bench_serving --backend sglang-oai-chat --dataset-name image --num-prompts $PROT_NUM --apply-chat-template --random-output-len 64 --random-input-len 32 --image-resolution 448x448 --image-format jpeg --image-count 1 --image-content blank --random-range-ratio 1 --max-concurrency $CONCURRENCY --host=127.0.0.1 --port=8040;
# done > 8040_$CONCURRENCY.log


# for i in {1..3};
# do
#     python3 -m sglang.bench_serving --backend sglang-oai-chat --dataset-name image --num-prompts $PROT_NUM --apply-chat-template --random-output-len 64 --random-input-len 32 --image-resolution 448x448 --image-format jpeg --image-count 1 --image-content blank --random-range-ratio 1 --max-concurrency $CONCURRENCY --host=127.0.0.1 --port=8030;
# done > 8030_$CONCURRENCY.log


# for i in {1..3};
# do
#     python3 -m sglang.bench_serving --backend sglang-oai-chat --dataset-name image --num-prompts $PROT_NUM --apply-chat-template --random-output-len 64 --random-input-len 32 --image-resolution 448x448 --image-format jpeg --image-count 1 --image-content blank --random-range-ratio 1 --max-concurrency $CONCURRENCY --host=127.0.0.1 --port=8020;
# done > 8020_$CONCURRENCY.log

# for i in {1..3};
# do
#     python3 -m sglang.bench_serving --backend sglang-oai-chat --dataset-name image --num-prompts $PROT_NUM --apply-chat-template --random-output-len 64 --random-input-len 32 --image-resolution 448x448 --image-format jpeg --image-count 1 --image-content blank --random-range-ratio 1 --max-concurrency $CONCURRENCY --host=127.0.0.1 --port=8010;
# done > 8010_$CONCURRENCY.log