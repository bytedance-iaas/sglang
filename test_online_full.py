# Example clip ID
import os

import base64
import io

import torch
from openai import OpenAI
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time



clip_ids = pd.read_parquet("clip_ids.parquet")["clip_id"].tolist()
clip_id = clip_ids[1]
# 774 clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
# 5 clip_id = "74d763f1-9c47-416e-a840-a3e4bd8cc6a7"
pkl_filename = f"{clip_id}_data.pkl"
if not os.path.exists(pkl_filename):
	from load_physical_aiavdataset import load_physical_aiavdataset
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


def _encode_rgb_tensor_to_base64(image: torch.Tensor, fmt: str = "JPEG", quality: int = 85) -> str:
	"""Encode an RGB image tensor to a base64 string.

	Expects `image` shaped (3, H, W). Accepts uint8 (0-255) or float (0-1).
	fmt: "PNG" or "JPEG". JPEG is much faster to encode.
	"""
	from PIL import Image

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
	if fmt == "JPEG":
		img.save(buf, format="JPEG", quality=quality)
	else:
		img.save(buf, format="PNG")
	return base64.b64encode(buf.getvalue()).decode("utf-8")

# Compare PNG vs JPEG encoding time
t_enc = time.perf_counter()
images_png = [_encode_rgb_tensor_to_base64(frame, fmt="PNG") for frame in frames]
t_png = time.perf_counter() - t_enc
print(f"PNG encoding: {t_png:.2f}s for {len(frames)} frames, total size: {sum(len(b) for b in images_png) / 1024:.0f} KB")

t_enc = time.perf_counter()
images_jpg = [_encode_rgb_tensor_to_base64(frame, fmt="JPEG", quality=85) for frame in frames]
t_jpg = time.perf_counter() - t_enc
print(f"JPEG encoding: {t_jpg:.2f}s for {len(frames)} frames, total size: {sum(len(b) for b in images_jpg) / 1024:.0f} KB")

# Use JPEG for actual inference
images = images_jpg
IMG_MIME = "image/jpeg"
# Extract trajectory history
history_traj = {
    "ego_history_xyz": data["ego_history_xyz"].tolist(),
    "ego_history_rot": data["ego_history_rot"].tolist(),
}
client = OpenAI(
	base_url="http://127.0.0.1:29003/v1",
	api_key="EMPTY",
)
# messages = [
# 	{
# 		"role": "system",
# 		"content": [
# 			{
# 				"type": "text",
# 				"text": "You are a driving assistant that generates safe and accurate actions.",
# 			}
# 		],
# 	},
# 	{
# 		"role": "user",
# 		"content": [
# 			*[
# 				{
# 					"type": "image_url",
# 					"image_url": {"url": f"data:image/png;base64,{image_b64}"},
# 				}
# 				for image_b64 in images
# 			],
# 			{
# 				"type": "text",
# 				"text": f"{hist_traj_placeholder}output the chain-of-thought reasoning of the driving process, then output the future trajectory",
# 			},
# 		],
# 	},
# 	{
# 		"role": "assistant",
# 		"content": [
# 			{
# 				"type": "text",
# 				"text": "<|cot_start|>",
# 			}
# 		],
# 	},
# ]

messages = [
  {"role": "system", "content": "You are a driving assistant that generates safe and accurate actions."},
  {"role": "user", "content": [
      *[{"type":"image_url","image_url":{"url": f"data:{IMG_MIME};base64,{b64}"}} for b64 in images],
      {"type":"text","text": f"{hist_traj_placeholder}output the chain-of-thought reasoning of the driving process, then output the future trajectory"},
  ]},
  {"role": "assistant", "content": "<|cot_start|>"},
]

now = time.perf_counter()
resp = client.chat.completions.create(
	model="Qwen/Qwen3-VL-8B-Instruct",
	messages=messages,
	max_tokens=256,
    extra_body={"history_traj": history_traj, "continue_final_message": True},
	temperature=0.6,
	top_p=0.98,
)
t_request = time.perf_counter() - now
print(f"HTTP request (JPEG): {t_request:.2f}s")
print(f"Total (encode + request): {t_jpg + t_request:.2f}s  (was PNG: {t_png + t_request:.2f}s estimated)")

print(resp.choices[0].message.content)
sglext = getattr(resp, "sglext", None)
if sglext is None:
	sglext = getattr(resp, "model_extra", {}).get("sglext")

traj_xyz=np.asarray(sglext['traj_xyz'])


print(f"first 5 pred_xy:\n{traj_xyz[0][:5][...,:-1]}")



gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].numpy()
gt_xy = gt_xy
print(f"first 5 gt_xy:\n{gt_xy[:5, :]}")


def rotate_90cc(xy: np.ndarray) -> np.ndarray:
    return np.stack([-xy[:, 1], xy[:, 0]], axis=1)


if traj_xyz.size == 0:
	raise ValueError("`sglext.traj_xyz` is empty in response")

if traj_xyz.ndim >= 5:
	pred_xy_all = traj_xyz[0, 0, :, :, :2]
elif traj_xyz.ndim == 3:
	pred_xy_all = traj_xyz[:, :, :2]
elif traj_xyz.ndim == 2:
	pred_xy_all = traj_xyz[None, :, :2]
else:
	raise ValueError(f"Unexpected `sglext.traj_xyz` shape: {traj_xyz.shape}")

pred_steps = pred_xy_all.shape[1]
gt_steps = gt_xy.shape[0]
min_steps = min(pred_steps, gt_steps)
pred_xy_all = pred_xy_all[:, :min_steps, :]
gt_xy = gt_xy[:min_steps, :]

ade_each = np.linalg.norm(pred_xy_all - gt_xy[None, ...], axis=-1).mean(axis=-1)
best_idx = int(np.argmin(ade_each))
pred_xy = pred_xy_all[best_idx]

print(f"Best trajectory index: {best_idx}")
print(f"ADE: {ade_each[best_idx]} meters")

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
plt.savefig("output.png", dpi=200)
plt.close()
print("Saved trajectory plot to output.png")

