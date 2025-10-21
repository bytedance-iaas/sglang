# EPD Disaggregation

## Debug Command

You can try developing with the following commands.

```
docker run -it \
--name h100_{your_name} \
--gpus all  \
--shm-size 32g  \
--privileged  \
--network host  \
--ipc=host  \
-v ~/.cache:/root/.cache  \
-v /dev/infiniband:/dev/infiniband  \
-v /sys/class/infiniband:/sys/class/infiniband  \
-v /usr/sbin/show_gids:/usr/sbin/show_gids:ro  \
lmsysorg/sglang:latest /bin/bash


python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0  --disaggregation-mode encode --port 60001 --disaggregation-ib-device  mlx5_0


python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0  --disaggregation-mode text --port 60002 --encoder-disaggregated --disaggregation-ib-device  mlx5_0 --disable-cuda-graph


python -m sglang.srt.disaggregation.mini_lb --encode http://127.0.0.1:60001 --text http://127.0.0.1:60002 --host 0.0.0.0 --port 9080

wget https://gist.github.com/yhyang201/1dc8143c1476b58aad3ab0ea613741e2 -o send.py && python send.py
```
## Why and What is EPD Disaggregation?

Multimodal Large Language Model (MLLM) inference comprises two distinct phases: **Prefill** and **Decode**. The Prefill phase is computation-intensive, processing the entire input sequence, while the Decode phase is memory-intensive, managing the Key-Value (KV) cache for token generation. Traditionally, these phases are handled within a unified engine, where combined scheduling of prefill and decode batches introduces inefficiencies. To address these challenges, we introduce **Prefill and Decoding (PD) Disaggregation** in SGLang.

### Issues with Unified Scheduling

The conventional unified engine, which processes prefill and decode batches together, results in two significant problems:

1. **Prefill Interruption**: Incoming prefill batches frequently interrupt ongoing decode batches, causing substantial delays in token generation.
2. **DP Attention Imbalance**: In data-parallel (DP) attention, one DP worker may process a prefill batch while another handles a decode batch simultaneously, leading to increased decode latency.

PD Disaggregation resolves these by separating the two stages, enabling tailored optimizations for each.

For the design details, please refer to [link](https://docs.google.com/document/d/1rQXJwKd5b9b1aOzLh98mnyMhBMhlxXA5ATZTHoQrwvc/edit?tab=t.0).

Currently, we support Mooncake and NIXL as the transfer engine.

### Usage

### QwenVL, PD not disaggregated

```bash
# start an encode server
$ python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --host 0.0.0.0  --disaggregation-mode encode --port 60001 --disaggregation-ib-device ${device_name}
# start a text server
$ python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --host 0.0.0.0  --disaggregation-mode text --port 60002 --encoder-disaggregated --disaggregation-ib-device ${device_name}
# mini_lb
$ python -m sglang.srt.disaggregation.mini_lb --encode http://127.0.0.1:60001 --text http://127.0.0.1:60002 --host 0.0.0.0 --port 9080
```


### QwenVL, EPD all disaggregated

```bash
# start an encode server
$ python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --host 0.0.0.0  --disaggregation-mode encode --port 60001 --disaggregation-ib-device ${device_name}
# start a prefill server, make sure to set --encoder-disaggregated
$ python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --host 0.0.0.0  --disaggregation-mode prefill --port 60002 --encoder-disaggregated --disaggregation-ib-device ${device_name}
# start a decode server
$ python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --host 0.0.0.0  --disaggregation-mode decode --port 60003 --disaggregation-ib-device ${device_name}
# mini_lb
$ python -m sglang.srt.disaggregation.mini_lb --encode http://127.0.0.1:60001 --prefill http://127.0.0.1:60002 --decode http://127.0.0.1:60003 --host 0.0.0.0 --port 9080
```
