from __future__ import annotations

import base64
import io
import logging
import os
import threading
import traceback
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import sys

# Ensure the repo root is on sys.path so load_physical_aiavdataset can be imported
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
import physical_ai_av
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

from load_physical_aiavdataset import load_physical_aiavdataset

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
CLIP_IDS_PATH = REPO_ROOT / "clip_ids.parquet"

MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "http://127.0.0.1:29003/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "EMPTY")
VIDEO_CACHE_MAX_BYTES = int(os.getenv("VIDEO_CACHE_MAX_BYTES", str(512 * 1024 * 1024)))
VIDEO_CACHE_MAX_ENTRIES = int(os.getenv("VIDEO_CACHE_MAX_ENTRIES", "16"))

import hashlib
import pickle

DATA_CACHE_DIR = Path(os.getenv("DATA_CACHE_DIR", str(APP_DIR / "data_cache")))
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CAMERA_NAMES = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]

CAMERA_NAME_TO_ATTR = {
    "camera_cross_left_120fov": "CAMERA_CROSS_LEFT_120FOV",
    "camera_front_wide_120fov": "CAMERA_FRONT_WIDE_120FOV",
    "camera_cross_right_120fov": "CAMERA_CROSS_RIGHT_120FOV",
    "camera_rear_left_70fov": "CAMERA_REAR_LEFT_70FOV",
    "camera_rear_tele_30fov": "CAMERA_REAR_TELE_30FOV",
    "camera_rear_right_70fov": "CAMERA_REAR_RIGHT_70FOV",
    "camera_front_tele_30fov": "CAMERA_FRONT_TELE_30FOV",
}

AVDI: physical_ai_av.PhysicalAIAVDatasetInterface | None = None
OPENAI_CLIENT: OpenAI | None = None
CLIP_IDS_CACHE: list[str] | None = None


class VideoClipCache:
    """Simple thread-safe LRU cache for (video bytes, timestamps) pairs."""

    def __init__(self, max_entries: int, max_bytes: int) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._cache: OrderedDict[tuple[str, str], tuple[bytes, np.ndarray]] = OrderedDict()
        self._size_bytes = 0
        self._lock = threading.Lock()

    @staticmethod
    def _entry_size(video_bytes: bytes, timestamps: np.ndarray) -> int:
        return len(video_bytes) + int(timestamps.nbytes)

    def get(self, key: tuple[str, str]) -> tuple[bytes, np.ndarray] | None:
        with self._lock:
            value = self._cache.pop(key, None)
            if value is None:
                return None
            self._cache[key] = value
            return value

    def put(self, key: tuple[str, str], value: tuple[bytes, np.ndarray]) -> None:
        video_bytes, timestamps = value
        entry_size = self._entry_size(video_bytes, timestamps)
        if entry_size > self.max_bytes:
            return
        with self._lock:
            old = self._cache.pop(key, None)
            if old is not None:
                self._size_bytes -= self._entry_size(old[0], old[1])
            self._cache[key] = value
            self._size_bytes += entry_size
            while self._cache and (
                len(self._cache) > self.max_entries or self._size_bytes > self.max_bytes
            ):
                _, evicted = self._cache.popitem(last=False)
                self._size_bytes -= self._entry_size(evicted[0], evicted[1])

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._cache),
                "bytes": self._size_bytes,
                "max_entries": self.max_entries,
                "max_bytes": self.max_bytes,
            }


VIDEO_CLIP_CACHE = VideoClipCache(
    max_entries=VIDEO_CACHE_MAX_ENTRIES, max_bytes=VIDEO_CACHE_MAX_BYTES
)


class PredictRequest(BaseModel):
    clip_id: str | None = None
    clip_num: int | None = Field(default=None, ge=0)
    t0_us: int = Field(ge=1)
    camera_names: list[str] | None = None
    num_frames: int = Field(default=4, ge=1, le=8)
    include_input_images: bool = False


class PredictFromVideoTimeRequest(BaseModel):
    clip_id: str | None = None
    clip_num: int | None = Field(default=None, ge=0)
    camera_name: str = "camera_front_wide_120fov"
    video_time_s: float = Field(ge=0.0)
    camera_names: list[str] | None = None
    num_frames: int = Field(default=4, ge=1, le=8)
    include_input_images: bool = False


app = FastAPI(title="Physical AI AV Trajectory Viewer")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_avdi() -> physical_ai_av.PhysicalAIAVDatasetInterface:
    global AVDI
    if AVDI is None:
        AVDI = physical_ai_av.PhysicalAIAVDatasetInterface()
    return AVDI


def get_openai_client() -> OpenAI:
    global OPENAI_CLIENT
    if OPENAI_CLIENT is None:
        OPENAI_CLIENT = OpenAI(base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY)
    return OPENAI_CLIENT


def get_clip_ids() -> list[str]:
    global CLIP_IDS_CACHE
    if CLIP_IDS_CACHE is not None:
        return CLIP_IDS_CACHE
    if CLIP_IDS_PATH.exists():
        CLIP_IDS_CACHE = pd.read_parquet(CLIP_IDS_PATH)["clip_id"].tolist()
    else:
        CLIP_IDS_CACHE = get_avdi().clip_index.index.tolist()
    return CLIP_IDS_CACHE


def resolve_clip_id(clip_id: str | None, clip_num: int | None) -> str:
    all_clip_ids = get_clip_ids()
    if clip_id is not None and clip_id != "":
        if clip_id not in all_clip_ids:
            raise HTTPException(status_code=404, detail=f"Unknown clip_id: {clip_id}")
        return clip_id
    if clip_num is None:
        raise HTTPException(status_code=400, detail="Provide either clip_id or clip_num")
    if clip_num < 0 or clip_num >= len(all_clip_ids):
        raise HTTPException(
            status_code=400,
            detail=f"clip_num out of range: {clip_num} (total={len(all_clip_ids)})",
        )
    return all_clip_ids[clip_num]


def resolve_clip_num(clip_id: str) -> int:
    try:
        return get_clip_ids().index(clip_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Unknown clip_id: {clip_id}")


def camera_names_to_features(camera_names: list[str] | None) -> list[str] | None:
    if camera_names is None:
        return None
    avdi = get_avdi()
    features: list[str] = []
    for camera_name in camera_names:
        attr_name = CAMERA_NAME_TO_ATTR.get(camera_name)
        if attr_name is None:
            raise HTTPException(status_code=400, detail=f"Unsupported camera: {camera_name}")
        features.append(getattr(avdi.features.CAMERA, attr_name))
    return features


def encode_rgb_tensor_to_base64(image: torch.Tensor, fmt: str = "JPEG", quality: int = 85) -> str:
    tensor = image.detach().cpu()
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape={tuple(tensor.shape)}")

    if tensor.shape[0] == 3:
        tensor_chw = tensor
    elif tensor.shape[-1] == 3:
        tensor_chw = tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected RGB tensor, got shape={tuple(tensor.shape)}")

    if tensor_chw.dtype != torch.uint8:
        tensor_chw = (tensor_chw.clamp(0, 1) * 255).to(torch.uint8)

    array_hwc = tensor_chw.permute(1, 2, 0).numpy()
    from PIL import Image

    img = Image.fromarray(array_hwc)
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=quality)
    else:
        img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _video_disk_cache_path(clip_id: str, camera_name: str, suffix: str) -> Path:
    return DATA_CACHE_DIR / f"video_{clip_id}_{camera_name}{suffix}"


def extract_camera_video_and_timestamps(clip_id: str, camera_name: str) -> tuple[bytes, np.ndarray]:
    cache_key = (clip_id, camera_name)
    cached = VIDEO_CLIP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Try disk cache first
    mp4_path = _video_disk_cache_path(clip_id, camera_name, ".mp4")
    ts_path = _video_disk_cache_path(clip_id, camera_name, "_ts.npy")
    if mp4_path.exists() and ts_path.exists():
        video_bytes = mp4_path.read_bytes()
        timestamps = np.load(ts_path)
        logger.warning("Video loaded from DISK CACHE: %s", mp4_path.name)
        out = (video_bytes, timestamps)
        VIDEO_CLIP_CACHE.put(cache_key, out)
        return out

    avdi = get_avdi()
    attr_name = CAMERA_NAME_TO_ATTR.get(camera_name)
    if attr_name is None:
        raise HTTPException(status_code=400, detail=f"Unsupported camera: {camera_name}")

    feature = getattr(avdi.features.CAMERA, attr_name)
    try:
        chunk_id = avdi.get_clip_chunk(clip_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Unknown clip_id: {clip_id}")
    chunk_filename = avdi.features.get_chunk_feature_filename(chunk_id, feature)
    files_in_zip = avdi.features.get_clip_files_in_zip(clip_id, feature)

    try:
        with avdi.open_file(chunk_filename, maybe_stream=True) as f:
            with zipfile.ZipFile(f, "r") as zf:
                video_bytes = zf.read(files_in_zip["video"])
                ts_bytes = zf.read(files_in_zip["frame_timestamps"])
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Missing camera file for {clip_id}: {exc}")

    timestamps = (
        pd.read_parquet(io.BytesIO(ts_bytes))["timestamp"].to_numpy(dtype=np.int64, copy=False)
    )
    if timestamps.size == 0:
        raise HTTPException(status_code=404, detail="No frame timestamps found for camera clip")
    timestamps = timestamps.copy()

    # Save to disk cache
    try:
        mp4_path.write_bytes(video_bytes)
        np.save(ts_path, timestamps)
        logger.warning("Video cached to disk: %s", mp4_path.name)
    except Exception as e:
        logger.warning("Video disk cache write failed: %s", e)

    out = (video_bytes, timestamps)
    VIDEO_CLIP_CACHE.put(cache_key, out)
    return out


def pick_nearest_timestamp(timestamps: np.ndarray, target_us: int) -> int:
    idx = int(np.searchsorted(timestamps, target_us, side="left"))
    if idx <= 0:
        return int(timestamps[0])
    if idx >= len(timestamps):
        return int(timestamps[-1])
    left = int(timestamps[idx - 1])
    right = int(timestamps[idx])
    if abs(target_us - left) <= abs(right - target_us):
        return left
    return right


def rotate_xy(xy: np.ndarray, angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return xy @ rot.T


def estimate_heading_from_history(history_xy: np.ndarray) -> float | None:
    """Estimate heading angle (rad) from history trajectory, in XY plane."""
    if history_xy.ndim != 2 or history_xy.shape[0] < 2 or history_xy.shape[1] != 2:
        return None

    diffs = np.diff(history_xy, axis=0)
    for i in range(len(diffs) - 1, -1, -1):
        dx, dy = float(diffs[i, 0]), float(diffs[i, 1])
        if np.hypot(dx, dy) > 1e-4:
            return float(np.arctan2(dy, dx))
    return None


def run_trajectory_inference(
    data: dict[str, Any], include_input_images: bool = False
) -> dict[str, Any]:
    import time as _time

    num_traj_token = 48
    hist_traj_placeholder = (
        f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    )

    t_start = _time.perf_counter()

    image_frames = data["image_frames"]
    num_cameras = int(image_frames.shape[0])
    num_frames_per_camera = int(image_frames.shape[1])
    frames = image_frames.flatten(0, 1)
    images = [encode_rgb_tensor_to_base64(frame, fmt="JPEG") for frame in frames]

    t_encode = _time.perf_counter()

    history_traj = {
        "ego_history_xyz": data["ego_history_xyz"].tolist(),
        "ego_history_rot": data["ego_history_rot"].tolist(),
    }

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
                    for b64 in images
                ],
                {
                    "type": "text",
                    "text": f"{hist_traj_placeholder}output the chain-of-thought reasoning of the driving process, then output the future trajectory",
                },
            ],
        },
        {"role": "assistant", "content": "<|cot_start|>"},
    ]

    client = get_openai_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=256,
        extra_body={"history_traj": history_traj, "continue_final_message": True},
        temperature=0.6,
        top_p=0.98,
    )
    t_request = _time.perf_counter()

    sglext = getattr(resp, "sglext", None)
    if sglext is None:
        sglext = getattr(resp, "model_extra", {}).get("sglext")
    if not sglext or "traj_xyz" not in sglext:
        raise RuntimeError("No `sglext.traj_xyz` in model response")

    traj_xyz = np.asarray(sglext["traj_xyz"])
    if traj_xyz.size == 0:
        raise RuntimeError("Model returned empty traj_xyz")

    if traj_xyz.ndim >= 5:
        pred_xy_all = traj_xyz[0, 0, :, :, :2]
    elif traj_xyz.ndim == 3:
        pred_xy_all = traj_xyz[:, :, :2]
    elif traj_xyz.ndim == 2:
        pred_xy_all = traj_xyz[None, :, :2]
    else:
        raise RuntimeError(f"Unexpected traj_xyz shape: {traj_xyz.shape}")

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].numpy()

    min_steps = min(pred_xy_all.shape[1], gt_xy.shape[0])
    pred_xy_all = pred_xy_all[:, :min_steps, :]
    gt_xy = gt_xy[:min_steps, :]

    ade_each = np.linalg.norm(pred_xy_all - gt_xy[None, ...], axis=-1).mean(axis=-1)
    best_idx = int(np.argmin(ade_each))
    pred_xy = pred_xy_all[best_idx]
    fde = float(np.linalg.norm(pred_xy[-1] - gt_xy[-1]))

    # Rotate trajectories so driving direction at t0 points upward (+Y on the plot).
    history_xy = data["ego_history_xyz"].cpu()[0, 0, :, :2].numpy()
    heading_rad = estimate_heading_from_history(history_xy)
    if heading_rad is None:
        # Fallback keeps current convention used in test_online_full.py plotting.
        rotation_rad = float(np.pi / 2.0)
        rotation_mode = "fallback_90cc"
    else:
        rotation_rad = float(np.pi / 2.0 - heading_rad)
        rotation_mode = "align_heading_to_up"

    pred_xy_rot = rotate_xy(pred_xy, rotation_rad)
    gt_xy_rot = rotate_xy(gt_xy, rotation_rad)

    encode_s = t_encode - t_start
    request_s = t_request - t_encode
    total_s = t_request - t_start
    timing_str = f"[encode={encode_s:.2f}s  request={request_s:.2f}s  total={total_s:.2f}s]"
    logger.warning("run_trajectory_inference timing: %s", timing_str)

    raw_text = resp.choices[0].message.content or ""
    assistant_text = f"{raw_text}\n\n{timing_str}"

    out = {
        "pred_xy_m_raw": pred_xy.tolist(),
        "gt_xy_m_raw": gt_xy.tolist(),
        "pred_xy_m": pred_xy_rot.tolist(),
        "gt_xy_m": gt_xy_rot.tolist(),
        "num_steps": int(min_steps),
        "best_traj_index": best_idx,
        "ade_m": float(ade_each[best_idx]),
        "fde_m": fde,
        "rotation_mode": rotation_mode,
        "rotation_angle_deg": float(rotation_rad * 180.0 / np.pi),
        "heading_angle_deg": None if heading_rad is None else float(heading_rad * 180.0 / np.pi),
        "num_input_images": int(frames.shape[0]),
        "num_cameras": num_cameras,
        "num_frames_per_camera": num_frames_per_camera,
        "camera_indices": data["camera_indices"].tolist(),
        "absolute_timestamps_us": data["absolute_timestamps"].tolist(),
        "assistant_text": assistant_text,
    }
    if include_input_images:
        out["input_images_b64"] = images
    return out


def predict_with_t0(
    clip_id: str,
    t0_us: int,
    camera_names: list[str] | None,
    num_frames: int,
    include_input_images: bool = False,
    num_history_steps: int = 16,
    time_step: float = 0.1,
) -> dict[str, Any]:
    # Ensure t0_us is large enough for the history window
    min_t0_us = int(num_history_steps * time_step * 1_000_000) + 1
    if t0_us < min_t0_us:
        logger.warning("Clamping t0_us from %d to %d (minimum for history window)", t0_us, min_t0_us)
        t0_us = min_t0_us
    import time as _time
    t_start = _time.perf_counter()

    # Build a cache key from all parameters that affect the loaded data
    cam_key = ",".join(sorted(camera_names)) if camera_names else "default"
    cache_hash = hashlib.md5(f"{clip_id}_{t0_us}_{cam_key}_{num_frames}".encode()).hexdigest()[:12]
    cache_path = DATA_CACHE_DIR / f"{clip_id}_{t0_us}_{cache_hash}.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        t_data = _time.perf_counter()
        logger.warning("Data loaded from CACHE in %.2fs: %s", t_data - t_start, cache_path.name)
    else:
        avdi = get_avdi()
        data = load_physical_aiavdataset(
            clip_id=clip_id,
            t0_us=t0_us,
            avdi=avdi,
            maybe_stream=True,
            camera_features=camera_names_to_features(camera_names),
            num_frames=num_frames,
        )
        t_data = _time.perf_counter()
        # Save to disk cache
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            logger.warning("Data loaded from NETWORK in %.2fs, cached to: %s", t_data - t_start, cache_path.name)
        except Exception as e:
            logger.warning("Data loaded from NETWORK in %.2fs, cache write failed: %s", t_data - t_start, e)

    pred = run_trajectory_inference(data, include_input_images=include_input_images)
    t_infer = _time.perf_counter()
    logger.warning(
        "predict_with_t0 timing: data_load=%.2fs  inference=%.2fs  total=%.2fs  t0_us=%d",
        t_data - t_start, t_infer - t_data, t_infer - t_start, t0_us,
    )
    pred["clip_id"] = clip_id
    pred["clip_num"] = resolve_clip_num(clip_id)
    pred["t0_us"] = t0_us
    pred["model_name"] = MODEL_NAME
    pred["camera_names"] = camera_names if camera_names is not None else DEFAULT_CAMERA_NAMES
    return pred


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/cache_stats")
async def cache_stats() -> dict[str, Any]:
    return {"video_clip_cache": VIDEO_CLIP_CACHE.stats()}


@app.get("/api/cameras")
async def cameras() -> dict[str, Any]:
    return {"cameras": list(CAMERA_NAME_TO_ATTR.keys()), "default_cameras": DEFAULT_CAMERA_NAMES}


@app.get("/api/clips")
async def clips(limit: int = 200, offset: int = 0) -> dict[str, Any]:
    all_clip_ids = get_clip_ids()
    limit = min(max(limit, 1), 2000)
    offset = max(offset, 0)
    clip_ids = all_clip_ids[offset : offset + limit]
    clips_with_num = [{"clip_num": offset + i, "clip_id": clip_id} for i, clip_id in enumerate(clip_ids)]
    return {
        "total": len(all_clip_ids),
        "clip_ids": clip_ids,
        "clips": clips_with_num,
    }


@app.get("/api/video_meta/{clip_id}/{camera_name}")
async def video_meta(clip_id: str, camera_name: str) -> dict[str, Any]:
    _, timestamps = extract_camera_video_and_timestamps(clip_id, camera_name)
    return {
        "clip_id": clip_id,
        "camera_name": camera_name,
        "num_frames": int(len(timestamps)),
        "start_timestamp_us": int(timestamps[0]),
        "end_timestamp_us": int(timestamps[-1]),
        "timestamps_us": timestamps.astype(np.int64).tolist(),
    }


@app.get("/api/video_meta_by_num/{clip_num}/{camera_name}")
async def video_meta_by_num(clip_num: int, camera_name: str) -> dict[str, Any]:
    clip_id = resolve_clip_id(None, clip_num)
    return await video_meta(clip_id=clip_id, camera_name=camera_name)


@app.get("/api/video/{clip_id}/{camera_name}.mp4")
async def video_file(clip_id: str, camera_name: str) -> Response:
    video_bytes, _ = extract_camera_video_and_timestamps(clip_id, camera_name)
    return Response(content=video_bytes, media_type="video/mp4")


@app.get("/api/video_by_num/{clip_num}/{camera_name}.mp4")
async def video_file_by_num(clip_num: int, camera_name: str) -> Response:
    clip_id = resolve_clip_id(None, clip_num)
    return await video_file(clip_id=clip_id, camera_name=camera_name)


@app.post("/api/predict")
async def predict(req: PredictRequest) -> dict[str, Any]:
    try:
        clip_id = resolve_clip_id(req.clip_id, req.clip_num)
        return predict_with_t0(
            clip_id=clip_id,
            t0_us=req.t0_us,
            camera_names=req.camera_names,
            num_frames=req.num_frames,
            include_input_images=req.include_input_images,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("predict failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/predict_from_video_time")
async def predict_from_video_time(req: PredictFromVideoTimeRequest) -> dict[str, Any]:
    try:
        clip_id = resolve_clip_id(req.clip_id, req.clip_num)
        _, timestamps = extract_camera_video_and_timestamps(clip_id, req.camera_name)
        # Need at least 1.6s of history data before t0; clamp video_time_s accordingly
        safe_video_time_s = max(req.video_time_s, 1.7)
        target_ts = int(timestamps[0] + safe_video_time_s * 1_000_000)
        t0_us = pick_nearest_timestamp(timestamps, target_ts)
        return predict_with_t0(
            clip_id=clip_id,
            t0_us=t0_us,
            camera_names=req.camera_names,
            num_frames=req.num_frames,
            include_input_images=req.include_input_images,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("predict_from_video_time failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8899, reload=False)
