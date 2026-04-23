# GR00T-N1.7

[GR00T-N1.7](https://huggingface.co/nvidia/GR00T-N1.7-3B) is NVIDIA's Vision-Language-Action (VLA) model for humanoid and manipulator robots. Built on top of Qwen3-VL (Cosmos-Reason2-2B) with a flow-matching DiT action head, it takes multi-camera images, a natural-language instruction, an embodiment tag, and the current proprio state, and predicts a 40-step, 132-dim future action trajectory.

Key features:
- Multi-camera input (RGB images — one or more per request)
- Flow-matching action head (4-step Euler integration)
- 32 supported embodiments via an embodiment-id projector (DROID, Unitree G1, R1 Pro, LIBERO, SimplerEnv, ...)
- Output: `[action_horizon=40, max_action_dim=132]` tensor in the normalized/padded action space — client-side Isaac-GR00T processing converts it back to physical joint commands
- VL backbone hidden state is taken at `select_layer=16` and fed to the action head (matches Isaac-GR00T's `Gr00tPolicy`)

## Launch Server

```{note}
Only the `triton` attention backend is supported for GR00T-N1.7. `fa3` has flaky mem-efficient SDPA dispatch on sm121, and the custom DiT Euler loop composes more cleanly without CUDA graph capture — so `--disable-cuda-graph` is also required.
```

The `--tokenizer-path` is optional. If omitted, SGLang will automatically download the tokenizer from `nvidia/Cosmos-Reason2-2B`:

```shell
FLASHINFER_DISABLE_VERSION_CHECK=1 \
python3 -m sglang.launch_server \
    --model-path nvidia/GR00T-N1.7-3B \
    --tokenizer-path nvidia/Cosmos-Reason2-2B \
    --port 30000 \
    --tp 1 \
    --attention-backend triton \
    --disable-cuda-graph \
    --disable-radix-cache \
    --skip-server-warmup
```

A ready-to-run version lives at `start.sh` in the repo root.

## Inference Example

GR00T-N1.7 uses the SGLang VLA contract: the client sends raw images + language instruction via the standard OpenAI `chat.completions` message format, and sends proprio state + embodiment tag via `extra_body={"history_traj": {...}}`. The server returns the predicted trajectory under `sglext.pred_traj`.

### Dependencies

For loading LeRobot-format demonstrations and for client-side state/action normalization (the statistics must match those the model was trained on), GR00T-N1.7 relies on NVIDIA's [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) package:

```shell
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T && pip install -e .

pip install requests pillow numpy
```

Or run your script through Isaac's `uv` environment:

```shell
cd /path/to/Isaac-GR00T
uv run python your_script.py
```

```{note}
The sglang server itself does NOT need Isaac-GR00T. The dependency is purely client-side, only for convenience helpers (dataset loading, normalize/unnormalize). If you produce normalized proprio state yourself, you can skip the Isaac dependency.
```

### Inference Script

A full end-to-end tutorial that loads a DROID demonstration via `LeRobotEpisodeLoader`, drives the sglang server for 200 frames, and reports physical-space MSE/MAE vs ground truth lives at `test_online_full.py` in the repo root. The core request/response flow is:

```python
import base64
import io
import numpy as np
import requests
from PIL import Image

# 1. Images -> base64 JPEG data URLs.
def _img_to_data_url(arr: np.ndarray) -> str:
    pil = Image.fromarray(arr)  # HWC uint8
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# 2. Build the OpenAI-format chat message.  Send every image frame you
#    want the VLM to see, followed by the language instruction.
content = [
    {"type": "image_url", "image_url": {"url": _img_to_data_url(exterior_rgb)}},
    {"type": "image_url", "image_url": {"url": _img_to_data_url(wrist_rgb)}},
    {"type": "text", "text": "Put the blue block in the green bowl"},
]

# 3. Proprio state: per-key lists of floats already normalized with
#    the trained-model statistics (Isaac's `state_action_processor.apply_state`).
#    Embodiment: one of the tag strings in the table below.
history_traj = {
    "proprio_state": {
        "eef_9d": [...],              # 9 floats
        "gripper_position": [...],    # 1 float
        "joint_position": [...],      # 7 floats
    },
    "embodiment": "oxe_droid_relative_eef_relative_joint",
}

# 4. POST to sglang.  `history_traj` is accepted as a top-level field.
payload = {
    "model": "nvidia/GR00T-N1.7-3B",
    "messages": [{"role": "user", "content": content}],
    "max_tokens": 1,         # GR00T doesn't emit free-form text; pred lives in sglext
    "history_traj": history_traj,
}
resp = requests.post("http://127.0.0.1:30000/v1/chat/completions", json=payload).json()

# 5. Read the predicted action trajectory.  Shape: [40, 132].
pred_traj = np.asarray(resp["sglext"]["pred_traj"][0], dtype=np.float32)
assert pred_traj.shape == (40, 132)
```

To turn `pred_traj` back into physical joint commands, pass it through Isaac-GR00T's `policy.processor.decode_action(pred_traj[None, ...], embodiment_tag, state=raw_state)`. The canonical end-to-end example — including dataset loading and MSE/MAE against ground-truth — is `test_online_full.py`.

## Request Schema

| Field                            | Type                         | Notes |
|----------------------------------|------------------------------|-------|
| `messages[*].content[*]`         | OpenAI chat array            | Standard `image_url` / `text` parts. Send all temporal frames your embodiment uses. |
| `max_tokens`                     | `int`                        | Set to `1`. GR00T's response is delivered via `sglext`, not via generated text. |
| `history_traj.proprio_state`     | `dict[str, list[float]]`     | Per-modality key → 1-D list of already-normalized state values. The order of values per key must match the embodiment's modality config. |
| `history_traj.embodiment`        | `str`                        | One of the tags in the [embodiment table](#embodiment-tag-table) below. |

`history_traj` may be sent at the top level or under `extra_body`; both are accepted by `/v1/chat/completions`.

## Response Schema

GR00T attaches its prediction to the standard OpenAI response under a `sglext` block:

```text
{
  "choices": [...],
  "sglext": {
    "pred_traj": [[[ ... ]]]   // shape: [batch, action_horizon=40, max_action_dim=132]
  }
}
```

- `sglext.pred_traj` — `List[Optional[List[List[float]]]]`, one entry per request in the batch. Each non-null entry is a nested list of shape `[action_horizon=40, max_action_dim=132]` in the model's normalized+padded action space. For a request that did not carry `history_traj`, the entry is `null`.

## Embodiment Tag Table

The `embodiment` string selects the per-embodiment projector index inside the action head. Tags map to `EMBODIMENT_TAG_TO_PROJECTOR_INDEX` in `python/sglang/srt/multimodal/processors/groot_n1d7.py`:

| Embodiment tag                                       | Projector index |
|------------------------------------------------------|-----------------|
| `simpler_env_google`                                 | 0               |
| `simpler_env_widowx`                                 | 1               |
| `libero_sim`                                         | 2               |
| `new_embodiment`                                     | 10              |
| `oxe_droid_relative_eef_relative_joint`              | 24              |
| `real_g1_relative_eef_relative_joints`               | 25              |
| `unitree_g1_full_body_with_waist_height_nav_cmd`     | 25              |
| `real_r1_pro_sharpa_relative_eef`                    | 26              |
| `real_r1_pro_sharpa_relative_eef_human`              | 26              |
| `real_r1_pro_sharpa_relative_eef_maxinsights`        | 26              |
| `real_r1_pro_sharpa_relative_eef_mecka`              | 26              |
| `xdof_relative_eef_relative_joint`                   | 27              |
| `xdof_relative_eef_relative_joint_subtask`           | 27              |

Unknown tags raise `ValueError` at the processor layer with the full list of supported tags.
