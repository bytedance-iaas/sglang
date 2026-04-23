"""GR00T-N1.7 manual online smoke test — DROID closed-loop replay
with normalization parity against Isaac-GR00T's standalone_inference_script.py.

Hard-coded against `/data/dongmao_dev/Isaac-GR00T/demo_data/droid_sample`:
  - trajectory 1 ("Put the blue block in the green bowl", 266 frames)
  - embodiment = oxe_droid_relative_eef_relative_joint (DROID, 17-d action)
  - action_horizon = 8 (re-plan every 8 frames; trim model's 40-step
    output down to the first 8)

Pipeline (matches standalone_inference_script.py + Gr00tPolicy):
  1. Read raw physical state from parquet, NORMALIZE via StateActionProcessor
     (use_percentiles=True, clip_outliers=True, use_relative_action=True).
  2. Send normalized proprio + 2 images + language instruction to sglang.
  3. Receive `sglext.pred_traj` (40, 132) — normalized RELATIVE actions.
  4. UNNORMALIZE pred + convert RELATIVE→ABSOLUTE using `unapply_action(...)`
     (state-dependent for eef_9d / joint_position; gripper_position is ABS).
  5. Compare with GT physical actions from parquet (same units the standalone
     script labels "Unnormalized Action MSE").

Usage:
  ./start.sh                # terminal 1
  python3 test_online_full.py
"""

import asyncio
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from openai import AsyncOpenAI
from PIL import Image

# Pull in Isaac-GR00T's StateActionProcessor for normalize/unnormalize parity.
sys.path.insert(0, "/data/dongmao_dev/Isaac-GR00T")
from gr00t.data.state_action.state_action_processor import StateActionProcessor  # noqa: E402


EMBODIMENT = "oxe_droid_relative_eef_relative_joint"
TRAJ_ID = 1
ACTION_HORIZON = 8
MODEL_HORIZON = 40
DROID_STATE_KEYS = ["eef_9d", "gripper_position", "joint_position"]
DROID_ACTION_DIM = 17  # eef_9d(9) + gripper_position(1) + joint_position(7)
DROID_DIM_OFFSETS = {"eef_9d": (0, 9), "gripper_position": (9, 10), "joint_position": (10, 17)}

MODEL_DIR = Path("/data/models/GR00T-N1.7-3B")
DATASET_ROOT = Path("/data/dongmao_dev/Isaac-GR00T/demo_data/droid_sample")
PARQUET_PATH = DATASET_ROOT / "data" / "chunk-000" / f"episode_{TRAJ_ID:06d}.parquet"
EXTERIOR_VIDEO = (
    DATASET_ROOT
    / "videos"
    / "chunk-000"
    / "observation.images.exterior_1_left"
    / f"episode_{TRAJ_ID:06d}.mp4"
)
WRIST_VIDEO = (
    DATASET_ROOT
    / "videos"
    / "chunk-000"
    / "observation.images.wrist_left"
    / f"episode_{TRAJ_ID:06d}.mp4"
)


def _build_state_action_processor() -> StateActionProcessor:
    statistics = json.loads((MODEL_DIR / "statistics.json").read_text())
    pcfg = json.loads((MODEL_DIR / "processor_config.json").read_text())
    modality_configs = pcfg["processor_kwargs"]["modality_configs"]
    pkw = pcfg.get("processor_kwargs", {})
    return StateActionProcessor(
        modality_configs=modality_configs,
        statistics=statistics,
        use_percentiles=pkw.get("use_percentiles", True),
        clip_outliers=pkw.get("clip_outliers", True),
        apply_sincos_state_encoding=pkw.get("apply_sincos_state_encoding", False),
        use_relative_action=True,  # GR00T-N1.7 final_model_config.json: true
    )


def _img_b64(arr_rgb: np.ndarray) -> str:
    img = Image.fromarray(arr_rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _state_at(df: pd.DataFrame, idx: int) -> dict:
    """Raw physical state at frame `idx` as per-key 1D float32 arrays."""
    row = df.iloc[idx]
    return {
        "eef_9d": np.asarray(row["observation.state.eef_9d"], dtype=np.float32),
        "gripper_position": np.asarray(
            [row["observation.state.gripper_position"]], dtype=np.float32
        ),
        "joint_position": np.asarray(
            row["observation.state.joint_position"], dtype=np.float32
        ),
    }


def _gt_action_chunk(df: pd.DataFrame, j: int, n: int) -> np.ndarray:
    """GT physical action chunk over [j, j+n)."""
    end = min(j + n, len(df))
    rows = []
    for k in range(j, end):
        row = df.iloc[k]
        rows.append(
            np.concatenate(
                [
                    np.asarray(row["action.eef_9d"], dtype=np.float32),
                    np.asarray([row["action.gripper_position"]], dtype=np.float32),
                    np.asarray(row["action.joint_position"], dtype=np.float32),
                ]
            )
        )
    return np.stack(rows)  # (<=n, 17)


def _normalize_state(sap: StateActionProcessor, state: dict) -> dict:
    """Normalize raw physical state to [-1, 1] per StateActionProcessor.

    apply_state expects per-key arrays of shape (T, D); we pass T=1 (single
    frame) and return per-key python lists for JSON transport to the server.
    """
    state_in = {k: state[k][None, :] for k in DROID_STATE_KEYS}  # each (1, D)
    state_norm = sap.apply_state(state_in, EMBODIMENT)
    return {k: state_norm[k][0].astype(np.float32).tolist() for k in DROID_STATE_KEYS}


def _decode_pred_action(
    sap: StateActionProcessor, pred: np.ndarray, state: dict
) -> np.ndarray:
    """Unnormalize + convert RELATIVE→ABSOLUTE for the predicted action chunk.

    pred: (40, 132) — model output (normalized, padded action dim).
    state: per-key raw physical state at the current frame.
    Returns: (40, 17) physical absolute actions in DROID's
    [eef_9d | gripper_position | joint_position] layout.
    """
    pred_per_key = {}
    for key, (start, end) in DROID_DIM_OFFSETS.items():
        # unapply_action expects (B, T_action, D); we have B=1.
        pred_per_key[key] = pred[None, :, start:end]
    state_for_unapply = {k: state[k][None, None, :] for k in DROID_STATE_KEYS}  # (1,1,D)
    abs_action = sap.unapply_action(pred_per_key, EMBODIMENT, state=state_for_unapply)
    out = np.concatenate(
        [abs_action[k][0] for k in DROID_STATE_KEYS], axis=-1
    )  # (40, 17)
    return out.astype(np.float32)


async def _predict(client, ext_b64, wri_b64, instruction, proprio_norm):
    history_traj = {"proprio_state": proprio_norm, "embodiment": EMBODIMENT}
    resp = await client.chat.completions.create(
        model="GR00T-N1.7",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": ext_b64}},
                    {"type": "image_url", "image_url": {"url": wri_b64}},
                    {"type": "text", "text": instruction},
                ],
            }
        ],
        max_tokens=1,
        extra_body={"history_traj": history_traj},
    )
    sglext = getattr(resp, "sglext", None)
    if sglext is None and getattr(resp, "model_extra", None):
        sglext = resp.model_extra.get("sglext")
    if sglext is None:
        raise RuntimeError(f"no sglext on response: {resp}")
    pred = (
        sglext.get("pred_traj")
        if isinstance(sglext, dict)
        else getattr(sglext, "pred_traj", None)
    )
    if pred is None:
        raise RuntimeError(f"no pred_traj in sglext: {sglext}")
    arr = np.asarray(pred[0])
    if arr.shape != (MODEL_HORIZON, 132):
        raise RuntimeError(f"expected ({MODEL_HORIZON}, 132), got {arr.shape}")
    return arr


async def main() -> int:
    df = pd.read_parquet(PARQUET_PATH)
    instruction = str(df.iloc[0]["language_instruction"])
    num_steps = len(df)
    print(f"traj {TRAJ_ID}: {num_steps} frames, instruction={instruction!r}")
    print(
        f"action_horizon={ACTION_HORIZON} (re-plan every {ACTION_HORIZON} frames; "
        f"trim model's {MODEL_HORIZON}-step output)"
    )

    sap = _build_state_action_processor()
    print(
        f"normalizer: use_percentiles={sap.use_percentiles} "
        f"clip_outliers={sap.clip_outliers} use_relative_action={sap.use_relative_action}"
    )

    ext_vr = VideoReader(str(EXTERIOR_VIDEO), ctx=cpu(0))
    wri_vr = VideoReader(str(WRIST_VIDEO), ctx=cpu(0))

    client = AsyncOpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")
    sq_err_sum = np.zeros(DROID_ACTION_DIM, dtype=np.float64)
    count = 0

    for j in range(0, num_steps - 1, ACTION_HORIZON):
        ext_frame = ext_vr[j].asnumpy()
        wri_frame = wri_vr[j].asnumpy()
        state = _state_at(df, j)
        proprio_norm = _normalize_state(sap, state)
        pred_norm = await _predict(
            client, _img_b64(ext_frame), _img_b64(wri_frame), instruction, proprio_norm
        )
        pred_abs = _decode_pred_action(sap, pred_norm, state)  # (40, 17) physical
        gt = _gt_action_chunk(df, j, ACTION_HORIZON)            # (<=8, 17) physical
        n = gt.shape[0]
        sq_err_sum += ((pred_abs[:n] - gt) ** 2).sum(axis=0)
        count += n
        if j == 0:
            print(f"step {j}: pred_abs[0]={pred_abs[0].tolist()}")
            print(f"step {j}: gt[0]      ={gt[0].tolist()}")
        print(
            f"step {j:>4d}: chunk_mse={((pred_abs[:n]-gt)**2).mean():.6f}"
        )

    mse_per_dim = sq_err_sum / max(count, 1)
    print(f"\nDROID PHYSICAL-SPACE MSE over {count} GT-aligned action steps:")
    print(f"  eef_9d (0..9):           {mse_per_dim[:9].mean():.6f}")
    print(f"  gripper_position (9):    {mse_per_dim[9]:.6f}")
    print(f"  joint_position (10..17): {mse_per_dim[10:17].mean():.6f}")
    print(f"  overall:                 {mse_per_dim.mean():.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
