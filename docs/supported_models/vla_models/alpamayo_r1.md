# Alpamayo-R1

[Alpamayo-R1](https://huggingface.co/nvidia/Alpamayo-R1-10B) is NVIDIA's Vision-Language-Action (VLA) model for autonomous driving. Built on top of Qwen3-VL-8B, it takes multi-camera images and ego-vehicle trajectory history as input, performs chain-of-thought reasoning about the driving scene, and outputs future trajectory waypoints.

Key features:
- Multi-camera input (4 cameras, 4 frames each)
- Trajectory history conditioning via special `<|traj_history|>` tokens
- Chain-of-thought reasoning before trajectory prediction
- Outputs 64 future waypoints (6.4s at 10Hz) as (x, y, z) coordinates

## Launch Server

```{note}
Currently only the `triton` attention backend is supported for Alpamayo-R1. Other backends (`flashinfer`, `torch_native`, `trtllm_mha`) are not yet compatible.
```

The `--tokenizer-path` is optional. If omitted, SGLang will automatically download the tokenizer from `Qwen/Qwen3-VL-8B-Instruct`. You can also specify it explicitly if you have a local copy:

```shell
python3 -m sglang.launch_server \
  --model-path nvidia/Alpamayo-R1-10B \
  --tokenizer-path Qwen/Qwen3-VL-8B-Instruct \
  --port 30000 \
  --tp 1 \
  --disable-cuda-graph \
  --attention-backend triton
```


## Inference Example

The following script sends multi-camera images and trajectory history to the server via the `/generate` endpoint, then extracts the predicted trajectory from the response.

### Dependencies

Alpamayo-R1 requires the [physical-ai-av](https://huggingface.co/datasets/nvidia/physical-ai-av) dataset SDK for loading driving data:

```shell
pip install physical-ai-av einops scipy
```

### Inference Script

This example uses the OpenAI-compatible `/v1/chat/completions` endpoint. The data loading utility is provided at `python/sglang/srt/models/alpamayo/dataset.py` in the SGLang repo.

```python
import base64
import io

import numpy as np
import torch
from openai import OpenAI
from PIL import Image

from sglang.srt.models.alpamayo.dataset import load_physical_aiavdataset


def encode_image_to_base64(image: torch.Tensor) -> str:
    """Encode an RGB tensor (C,H,W) or (H,W,C) to a JPEG base64 string."""
    tensor = image.detach().cpu()
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D tensor (C,H,W), got shape={tuple(tensor.shape)}")

    if tensor.shape[0] == 3:
        tensor_chw = tensor
    elif tensor.shape[-1] == 3:
        tensor_chw = tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected RGB with 3 channels, got shape={tuple(tensor.shape)}")

    if tensor_chw.dtype != torch.uint8:
        tensor_chw = (tensor_chw.clamp(0, 1) * 255).to(torch.uint8)

    array_hwc = tensor_chw.permute(1, 2, 0).numpy()
    img = Image.fromarray(array_hwc)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --- Load data and encode images ---
clip_id = "06b483cf-6d9c-4b18-b54b-4429c80867e3"
data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)

frames = data["image_frames"].flatten(0, 1)  # (N_cameras * num_frames, C, H, W)
images_b64 = [encode_image_to_base64(frame) for frame in frames]

history_traj = {
    "ego_history_xyz": data["ego_history_xyz"].tolist(),
    "ego_history_rot": data["ego_history_rot"].tolist(),
}

# --- Build prompt via OpenAI-compatible chat API ---
num_traj_token = 48
hist_traj_placeholder = (
    f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
)

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

messages = [
    {
        "role": "system",
        "content": "You are a driving assistant that generates safe and accurate actions.",
    },
    {
        "role": "user",
        "content": [
            *[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                for b64 in images_b64
            ],
            {
                "type": "text",
                "text": (
                    f"{hist_traj_placeholder}"
                    "output the chain-of-thought reasoning of the driving process, "
                    "then output the future trajectory"
                ),
            },
        ],
    },
    {"role": "assistant", "content": "<|cot_start|>"},
]

resp = client.chat.completions.create(
    model="nvidia/Alpamayo-R1-10B",
    messages=messages,
    max_tokens=256,
    temperature=0.6,
    top_p=0.98,
    extra_body={"history_traj": history_traj, "continue_final_message": True},
)

# --- Extract results ---
print("Generated text:", resp.choices[0].message.content)

sglext = getattr(resp, "sglext", None)
if sglext is None:
    sglext = getattr(resp, "model_extra", {}).get("sglext")

traj_xyz = np.asarray(sglext["traj_xyz"])

# Normalize to (n_samples, n_waypoints, 3)
while traj_xyz.ndim > 3:
    traj_xyz = traj_xyz[0]
if traj_xyz.ndim == 2:
    traj_xyz = traj_xyz[None]
pred_xy_all = traj_xyz[:, :, :2]

# --- Compute ADE (Average Displacement Error) ---
gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].numpy()
min_steps = min(pred_xy_all.shape[1], gt_xy.shape[0])
pred_xy_all = pred_xy_all[:, :min_steps, :]
gt_xy = gt_xy[:min_steps, :]

ade_each = np.linalg.norm(pred_xy_all - gt_xy[None, ...], axis=-1).mean(axis=-1)
best_idx = int(np.argmin(ade_each))
print(f"Best trajectory index: {best_idx}")
print(f"ADE: {ade_each[best_idx]:.4f} meters")
```
