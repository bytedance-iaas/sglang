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
