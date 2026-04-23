"""Open-loop eval of the sglang GR00T-N1.7 port against demo_data/droid_sample.

Runs inside Isaac-GR00T's uv env (Python 3.10) so we can reuse its
`LeRobotEpisodeLoader` + `StateActionProcessor` for data loading and
normalization / unnormalization.  The model inference itself is delegated
to a running sglang server (expected at http://127.0.0.1:30000).

This mirrors what `standalone_inference_script.py --inference-mode pytorch`
does (README's "Zero-Shot Inference" command), except the model-forward
step hits sglang instead of a local Gr00tPolicy.  On this box Isaac
reports Unnormalized Action MSE = 0.003289 on DROID traj 1; if our port
is correct, the sglang-driven eval should land in the same ballpark.

Usage:

    # terminal 1: start sglang server
    cd /data/dongmao_dev/sglang-groot && ./start.sh

    # terminal 2: run this eval
    cd /data/dongmao_dev/Isaac-GR00T
    uv run python /data/dongmao_dev/sglang-groot/scripts/groot_parity/open_loop_eval_sglang.py \\
        --traj-ids 1 --action-horizon 8

Reads:
  - /data/models/GR00T-N1.7-3B/statistics.json (via Isaac's processor)
  - /data/dongmao_dev/Isaac-GR00T/demo_data/droid_sample

Prints per-trajectory and overall MSE/MAE, exits 0 on success.
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch  # noqa: F401 — pulled in by Isaac imports
from PIL import Image


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("open_loop_eval_sglang")


MODEL_PATH = "/data/models/GR00T-N1.7-3B"
DATASET = "/data/dongmao_dev/Isaac-GR00T/demo_data/droid_sample"
DEFAULT_EMBODIMENT = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
SGLANG_URL = "http://127.0.0.1:30000/v1/chat/completions"


def _img_to_data_url(img_arr: np.ndarray) -> str:
    """`img_arr` is HWC uint8 (single frame).  JPEG-encode and return a data URL."""
    pil = Image.fromarray(img_arr)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _sglang_predict(
    images_by_view: dict,          # {view_name: (T, H, W, 3) uint8}
    pre_model_state_dict: dict,    # {state_key: list[float]} normalized per Isaac
    embodiment_tag_value: str,
    instruction: str,
    timeout_s: float = 120.0,
) -> np.ndarray:
    """Send one request to the sglang chat endpoint, return pred_traj
    np.ndarray of shape (action_horizon, max_action_dim) in normalized
    space.

    The message embeds the newest frame of each view (delta_index==0,
    i.e. the last T-slot).  Pre-model (normalized) state is passed via
    `extra_body={"history_traj": {...}}` using the sglang VLA contract.
    """
    # DROID modality has delta_indices=[-15, 0] per video view — 2 temporal
    # frames.  Send all frames so the VLM sees the same temporal context
    # Isaac's policy does.
    content = []
    for view_name, frames in images_by_view.items():
        for t in range(frames.shape[0]):
            content.append({
                "type": "image_url",
                "image_url": {"url": _img_to_data_url(frames[t])},
            })
    content.append({"type": "text", "text": instruction})

    history_traj = {
        "proprio_state": {k: [float(x) for x in np.asarray(v).flatten()]
                          for k, v in pre_model_state_dict.items()},
        "embodiment": embodiment_tag_value,
    }

    payload = {
        "model": "/data/models/GR00T-N1.7-3B/",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1,
        # serving_chat accepts `history_traj` as a top-level field.
        "history_traj": history_traj,
    }

    resp = requests.post(SGLANG_URL, json=payload, timeout=timeout_s)
    if not resp.ok:
        raise RuntimeError(
            f"sglang HTTP {resp.status_code} {resp.reason}: {resp.text[:500]}"
        )
    data = resp.json()
    sglext = data.get("sglext")
    if sglext is None:
        raise RuntimeError(f"sglang response missing sglext block: {data}")
    pred = sglext.get("pred_traj") if isinstance(sglext, dict) else getattr(sglext, "pred_traj", None)
    if pred is None:
        raise RuntimeError(f"sglang response missing pred_traj: {sglext}")

    # pred: per-request list; we sent one request so take [0].
    arr = np.asarray(pred[0], dtype=np.float32)
    if arr.shape != (40, 132):
        raise RuntimeError(f"expected pred_traj (40, 132), got {arr.shape}")
    return arr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--traj-ids", type=int, nargs="+", default=[1])
    p.add_argument("--embodiment-tag", default=DEFAULT_EMBODIMENT)
    p.add_argument("--action-horizon", type=int, default=8,
                   help="How many consecutive pred-action steps to use per inference call.")
    p.add_argument("--steps", type=int, default=200,
                   help="Cap on dataset steps per trajectory.")
    global SGLANG_URL
    p.add_argument("--sglang-url", default=SGLANG_URL)
    args = p.parse_args()
    SGLANG_URL = args.sglang_url

    # Isaac imports — only available under uv run.
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    emb = EmbodimentTag.resolve(args.embodiment_tag)
    log.info(f"[eval] loading Gr00tPolicy {MODEL_PATH} for helpers (embodiment={emb.value})")
    # We load the policy only for its modality_configs + state_action_processor;
    # the model is unused (inference goes through sglang).
    policy = Gr00tPolicy(
        embodiment_tag=emb,
        model_path=MODEL_PATH,
        device="cuda:0",
        strict=False,
    )
    modality_configs = {k: v for k, v in policy.get_modality_config().items()}
    sap = policy.processor.state_action_processor  # for (un)normalize

    loader = LeRobotEpisodeLoader(dataset_path=DATASET, modality_configs=modality_configs)
    log.info(f"[eval] dataset length: {len(loader)}")

    state_keys = modality_configs["state"].modality_keys
    action_keys = modality_configs["action"].modality_keys
    language_key = modality_configs["language"].modality_keys[0]
    video_keys = modality_configs["video"].modality_keys

    all_traj_mse = []
    all_traj_mae = []
    for traj_id in args.traj_ids:
        log.info("\n" + "=" * 80)
        log.info(f"=== Trajectory {traj_id} ===")
        log.info("=" * 80)

        traj = loader[traj_id]
        n = min(args.steps, len(traj))
        step_counts = list(range(0, n, args.action_horizon))
        log.info(f"[eval] running {len(step_counts)} inference calls over {n} steps")

        pred_across_time = []

        for idx, step_count in enumerate(step_counts):
            t0 = time.time()
            data_point = extract_step_data(traj, step_count, modality_configs, emb)

            # Raw state per key (T, D) — Isaac normalizes BEFORE the model.
            raw_state = {k: np.asarray(data_point.states[k], dtype=np.float32)
                         for k in state_keys}
            norm_state = sap.apply_state(state=raw_state, embodiment_tag=emb.value)
            # norm_state: {key: (T, D) np.ndarray}

            # Images: (T, H, W, 3) uint8 per view.
            imgs = {v: np.asarray(data_point.images[v], dtype=np.uint8)
                    for v in video_keys}

            instruction = data_point.text
            pred_norm = _sglang_predict(imgs, norm_state, emb.value, instruction)
            dt = time.time() - t0
            log.info(f"  [step {idx + 1}/{len(step_counts)}] sglang pred_traj "
                     f"max-abs={np.abs(pred_norm).max():.3f}  ({dt:.2f}s)")

            # Unnormalize via Isaac's processor: splits the 132-wide output
            # back into per-action-key chunks, does inverse normalization, and
            # converts relative→absolute using the current raw state.
            unnorm = policy.processor.decode_action(pred_norm[None, ...], emb, state=raw_state)
            # unnorm: dict with key names (no prefix); shape (1, action_horizon, D_key)
            for j in range(args.action_horizon):
                concat = np.concatenate(
                    [np.atleast_1d(np.atleast_1d(unnorm[k][0])[j]) for k in action_keys],
                    axis=0,
                )
                pred_across_time.append(concat)

        pred_arr = np.asarray(pred_across_time, dtype=np.float32)  # (n, D_joint)

        # Ground-truth action straight from the LeRobot DataFrame — mirrors
        # standalone_inference_script.py's extract_state_joints helper.
        gt_cols = []
        for key in action_keys:
            col = f"action.{key}"
            gt_cols.append(np.vstack([np.asarray(arr, dtype=np.float32)
                                     for arr in traj[col]]))
        gt_arr = np.concatenate(gt_cols, axis=-1)  # (traj_len, D_joint)

        # Trim / align
        m = min(pred_arr.shape[0], gt_arr.shape[0])
        pred_arr = pred_arr[:m]
        gt_arr = gt_arr[:m]

        mse = float(np.mean((pred_arr - gt_arr) ** 2))
        mae = float(np.mean(np.abs(pred_arr - gt_arr)))
        all_traj_mse.append(mse)
        all_traj_mae.append(mae)
        log.info(f"[traj {traj_id}] MSE={mse:.6f}  MAE={mae:.6f}  (over {m} steps)")

    log.info("\n" + "=" * 80)
    log.info(f"Summary across {len(all_traj_mse)} traj:")
    log.info(f"  avg MSE = {np.mean(all_traj_mse):.6f}")
    log.info(f"  avg MAE = {np.mean(all_traj_mae):.6f}")
    log.info("Reference (Gr00tPolicy base model, same cmd line) reported on this box: "
             "MSE=0.003289, MAE=0.037619")
    return 0


if __name__ == "__main__":
    sys.exit(main())
