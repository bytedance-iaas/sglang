CUDA_VISIBLE_DEVICES=7 \
python3 -m sglang.launch_server --model-path /data/models/Alpamayo-R1-10B --port 29003 --tp 1 --disable-cuda-graph --disable-overlap-schedule