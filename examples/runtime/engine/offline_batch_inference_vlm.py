"""
Usage:
python offline_batch_inference_vlm.py --model-path Qwen/Qwen2-VL-7B-Instruct --chat-template=qwen2-vl
"""

import argparse
import dataclasses
import os
import time

import torch
from PIL import Image

import sglang as sgl
from sglang.srt.conversation import chat_templates
from sglang.srt.server_args import ServerArgs


def main(
    server_args: ServerArgs,
):
    vlm = sgl.Engine(**dataclasses.asdict(server_args))

    conv = chat_templates[server_args.chat_template].copy()
    image_token = conv.image_token

    img_s_path = "./img_s.jpg"
    img_m_path = "./img_m.jpg"
    img_l_path = "./img_l.jpg"
    img_warmup_path = "./warmup.jpg"

    hoshi_url_large = "/root/test_img/keihann.jpg"
    # wget https://m.media-amazon.com/images/I/91vPFJ1EdYL._AC_SL1500_.jpg  （1500 * 1001）
    # wget https://cdn-ak.f.st-hatena.com/images/fotolife/r/ritocamp/20250218/20250218070400.jpg （1200 * 800）
    # wget https://kakimoto-office.com/wp-content/uploads/2014/09/R0021478.jpg （3648 * 2736）
    # wget https://newsatcl-pctr.c.yimg.jp/t/amd-img/20250610-00258339-sasahi-000-11-view.jpg  (1200 * 800)
    # img = Image.open(image_url)

    download_cmds = [
        "wget -O warmup.jpg https://newsatcl-pctr.c.yimg.jp/t/amd-img/20250610-00258339-sasahi-000-11-view.jpg",
        "wget -O img_s.jpg https://cdn-ak.f.st-hatena.com/images/fotolife/r/ritocamp/20250218/20250218070400.jpg",
        "wget -O img_m.jpg https://m.media-amazon.com/images/I/91vPFJ1EdYL._AC_SL1500_.jpg",
        "wget -O img_l.jpg https://kakimoto-office.com/wp-content/uploads/2014/09/R0021478.jpg",
    ]
    print("downloading test data.......")
    for cmd in download_cmds:
        os.system(cmd)
    print("|download OK|")

    warmup_img = Image.open(img_warmup_path)
    img_s = Image.open(img_s_path)
    img_m = Image.open(img_m_path)
    img_l = Image.open(img_l_path)

    prompt = f"Please Tell me what's in this image?\n{image_token}"
    sampling_params = {
        "temperature": 0.001,
        "max_new_tokens": 30,
    }

    # warmup
    output_warmup = vlm.generate(
        prompt=prompt,
        image_data=[warmup_img],
        sampling_params=sampling_params,
    )

    print("===============================")
    print(f"Prompt: {prompt}")
    print(f"Generated text for warmup img: {output_warmup['text']}")

    torch.cuda.synchronize()
    get_dur = lambda des, x, y: print(" {} during :{} ms".format(des, (x - y) * 1000))
    s_t = time.time()
    out_put_s = vlm.generate(
        prompt=prompt,
        image_data=[img_s],
        sampling_params=sampling_params,
    )

    torch.cuda.synchronize()
    e_t = time.time()
    get_dur("s_image_processing", e_t, s_t)
    print("===============================")
    print(f"Prompt: {prompt}")
    print(f"Generated text for s img: {out_put_s['text']}")

    s_t = time.time()
    out_put_m = vlm.generate(
        prompt=prompt,
        image_data=[img_m],
        sampling_params=sampling_params,
    )

    torch.cuda.synchronize()
    e_t = time.time()
    get_dur("m_image_processing", e_t, s_t)
    print("===============================")
    print(f"Prompt: {prompt}")
    print(f"Generated text for m img: {out_put_m['text']}")

    s_t = time.time()
    out_put_l = vlm.generate(
        prompt=prompt,
        image_data=[img_l],
        sampling_params=sampling_params,
    )

    torch.cuda.synchronize()
    e_t = time.time()
    get_dur("m_image_processing", e_t, s_t)
    print("===============================")
    print(f"Prompt: {prompt}")
    print(f"Generated text for l img: {out_put_l['text']}")

    os.system("rm *.jpg -rf")
    vlm.shutdown()


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    main(server_args)
