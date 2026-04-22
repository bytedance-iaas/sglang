"""GR00T-N1.7 manual online smoke test (F8).

Runs against a server started via `./start.sh`.  Sends one chat-completions
request with a fake 256x256 image + a zeroed proprio_state + the G1
embodiment tag, and prints the shape + head of the predicted action
trajectory carried on `sglext.pred_traj`.

Contract (F7): GR00T reuses alpamayo's VLA channel — robot payload goes
into the top-level `history_traj` field under two keys:
  history_traj["proprio_state"]: dict[joint_name -> list[float]]
  history_traj["embodiment"]:    str tag
Response `sglext.pred_traj` is a per-request list of nested lists shaped
[action_horizon=40, action_dim=132].

Usage:
  # terminal 1:
  ./start.sh
  # terminal 2:
  python3 test_online_full.py
"""

import asyncio
import base64
import io
import sys

import numpy as np
from openai import AsyncOpenAI
from PIL import Image


EMBODIMENT = "real_g1_relative_eef_relative_joints"
PROPRIO = {
    "left_wrist_eef_9d": [0.0] * 9,
    "right_wrist_eef_9d": [0.0] * 9,
    "left_hand": [0.0] * 6,
    "right_hand": [0.0] * 6,
    "left_arm": [0.0] * 7,
    "right_arm": [0.0] * 7,
    "waist": [0.0] * 3,
}


def _fake_image_b64() -> str:
    img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


async def main() -> int:
    client = AsyncOpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")
    img_url = _fake_image_b64()
    history_traj = {
        "proprio_state": PROPRIO,
        "embodiment": EMBODIMENT,
    }

    resp = await client.chat.completions.create(
        model="GR00T-N1.7",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": "pick up the red cube"},
                ],
            }
        ],
        max_tokens=1,
        # history_traj is a SGLang-extension field.  The OpenAI SDK doesn't
        # know it directly, so pass via extra_body — serving_chat.py also
        # accepts it as a top-level field when a raw HTTP client is used.
        extra_body={"history_traj": history_traj},
    )

    sglext = getattr(resp, "sglext", None)
    if sglext is None and hasattr(resp, "model_extra") and resp.model_extra:
        sglext = resp.model_extra.get("sglext")
    if sglext is None:
        print("no sglext block on response:", resp, file=sys.stderr)
        return 1

    # sglext may come back as a dict (pydantic model_extra) or pydantic model.
    pred = (
        sglext.get("pred_traj")
        if isinstance(sglext, dict)
        else getattr(sglext, "pred_traj", None)
    )
    if pred is None:
        print("no pred_traj in sglext:", sglext, file=sys.stderr)
        return 1

    arr = np.asarray(pred[0])  # first (and only) request
    if arr.shape != (40, 132):
        print(f"expected (40, 132), got {arr.shape}", file=sys.stderr)
        return 1

    print("pred_traj shape:", arr.shape)
    print("pred_traj[0, :8]:", arr[0, :8].tolist())
    print("pred_traj[-1, :8]:", arr[-1, :8].tolist())
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
