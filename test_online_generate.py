import base64
import io
import os
import pickle

import pandas as pd
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt


clip_ids = pd.read_parquet("clip_ids.parquet")["clip_id"].tolist()
clip_id = clip_ids[1]
pkl_filename = f"{clip_id}_data.pkl"

if not os.path.exists(pkl_filename):
    from sglang.srt.models.alpamayo.dataset import load_physical_aiavdataset

    print(f"Loading dataset for clip_id: {clip_id}...")
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    print("Dataset loaded.")
    with open(pkl_filename, "wb") as f:
        pickle.dump(data, f)
else:
    print(f"Loading pre-saved data from {pkl_filename}...")
    with open(pkl_filename, "rb") as f:
        data = pickle.load(f)


num_traj_token = 48
hist_traj_placeholder = (
    f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
)

frames = data["image_frames"].flatten(0, 1)


def _encode_rgb_tensor_to_png_base64(image: torch.Tensor) -> str:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(image)}")

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

    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Pillow is required to encode frames as PNG. Install it with `pip install pillow`."
        ) from e

    img = Image.fromarray(array_hwc)
    target_h = 200
    if img.height != target_h:
        target_w = max(1, round(img.width * target_h / img.height))
        img = img.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


images = [_encode_rgb_tensor_to_png_base64(frame) for frame in frames]
# /generate image_data can accept strings; use data URL for consistency with OpenAI chat path.
image_data = [f"data:image/png;base64,{b64}" for b64 in images]

history_traj = {
    "ego_history_xyz": data["ego_history_xyz"].tolist(),
    "ego_history_rot": data["ego_history_rot"].tolist(),
}

# Build a Qwen-style prompt string for /generate.
# Keep it aligned with test_online_full.py semantic content.
image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
user_text = (
    f"{image_placeholder * len(image_data)}"
    f"{hist_traj_placeholder}"
    "output the chain-of-thought reasoning of the driving process, "
    "then output the future trajectory"
)

prompt = (
    "<|im_start|>system\n"
    "You are a driving assistant that generates safe and accurate actions."
    "<|im_end|>\n"
    "<|im_start|>user\n"
    f"{user_text}"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<|cot_start|>"
)

payload = {
    "text": prompt,
    "image_data": image_data,
    "history_traj": history_traj,
    "sampling_params": {
        "temperature": 0.6,
        "top_p": 0.98,
        "max_new_tokens": 256,
    },
    "stream": False,
}

url = "http://127.0.0.1:30000/generate"
resp = requests.post(url, json=payload, timeout=300)
resp.raise_for_status()
ret = resp.json()




if isinstance(ret, list):
    ret = ret[0]

print(ret.get("text", ""))


meta_info = ret.get("meta_info", {})
#print("traj_xyz:", meta_info.get("traj_xyz"))
pred_xyz = np.asarray(meta_info.get("traj_xyz", []))

# print(f"first 5 gt_xy:\n{gt_xy[:5, :]}")

while pred_xyz.ndim > 2:
    pred_xyz = pred_xyz[0]
pred_xy = pred_xyz[..., :-1]


gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
gt_xy = gt_xy.T

print(f"first 5 gt_xy:\n{gt_xy[:5, :]}")
print(f"first 5 pred_xy:\n{pred_xy[:5, :]}")
diff = np.linalg.norm(pred_xy - gt_xy, axis=-1)
ade = diff.mean()
print("ADE:", ade, "meters")


def rotate_90cc(xy: np.ndarray) -> np.ndarray:
    return np.stack([-xy[:, 1], xy[:, 0]], axis=1)


pred_xy_rot = rotate_90cc(pred_xy)
gt_xy_rot = rotate_90cc(gt_xy)

plt.figure(figsize=(8, 8))
plt.plot(pred_xy_rot[:, 0], pred_xy_rot[:, 1], "o-", label="Predicted Trajectory")
plt.plot(gt_xy_rot[:, 0], gt_xy_rot[:, 1], "r-", label="Ground Truth Trajectory")
plt.ylabel("y coordinate (meters)")
plt.xlabel("x coordinate (meters)")
plt.legend(loc="best")
plt.axis("equal")
plt.tight_layout()
pic_name = f"{clip_id}_trajectory.png"
plt.savefig(pic_name, dpi=200)
plt.close()
print(f"Saved trajectory plot to {pic_name}")
# call viu command to view the saved plot (Linux-specific)
# os.system(f"viu {pic_name}")