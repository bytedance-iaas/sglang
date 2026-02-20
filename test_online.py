# Example clip ID
import os

import base64
import io

import torch
from openai import OpenAI

from load_physical_aiavdataset import load_physical_aiavdataset
clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
print(f"Loading dataset for clip_id: {clip_id}...")
data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
print("Dataset loaded.")


frames = data["image_frames"].flatten(0, 1)


def _encode_rgb_tensor_to_png_base64(image: torch.Tensor, save_path: str | None = None) -> str:
	"""Encode an RGB image tensor to a base64 PNG string.

	Expects `image` shaped (3, H, W). Accepts uint8 (0-255) or float (0-1).
	If save_path is provided, also writes the PNG to disk.
	"""
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

	img = Image.fromarray(array_hwc, mode="RGB")
	target_h = 336
	if img.height != target_h:
		target_w = max(1, round(img.width * target_h / img.height))
		img = img.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
	buf = io.BytesIO()
	img.save(buf, format="PNG")
	if save_path is not None:
		img.save(save_path, format="PNG")
	return base64.b64encode(buf.getvalue()).decode("utf-8")

images = [
	_encode_rgb_tensor_to_png_base64(frame, save_path="last_frame.png" if i == 0 else None)
	for i, frame in enumerate(frames)
]

# Extract trajectory history
history_traj = {
    "ego_history_xyz": data["ego_history_xyz"].tolist(),
    "ego_history_rot": data["ego_history_rot"].tolist(),
}

import pdb; pdb.set_trace()
client = OpenAI(
	base_url="http://127.0.0.1:29003/v1",
	api_key="EMPTY",
)

resp = client.chat.completions.create(
	model="Qwen/Qwen3-VL-8B-Instruct",
	messages=[
		{
			"role": "user",
			"content": [
				*[
					{
						"type": "image_url",
						"image_url": {"url": f"data:image/png;base64,{image_b64}"},
					}
					for image_b64 in images
				],
				{"type": "text", "text": "describe this image"},
			],
		}
	],
	max_tokens=512,
    extra_body={"history_traj": history_traj},
)

print(resp.choices[0].message.content)
