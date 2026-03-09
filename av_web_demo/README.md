# Physical AI AV Web Demo

A minimal web service to:
- play a selected camera video for a `clip_id`
- run trajectory inference at current video time (or manual `t0_us`)
- show **Pred vs GT** trajectories and metrics (**ADE/FDE**)
- support synced playback inference: every `interval` seconds, sample images and update trajectories

By default, the web UI selects clips by numeric index (`clip_num`), not UUID.

## Run

From repo root (`/data/dongmao_dev/sglang`):

```bash
pip install fastapi uvicorn openai pillow
uvicorn av_web_demo.app:app --host 0.0.0.0 --port 8899
```

Open: <http://127.0.0.1:8899>

## Synced mode (with 16 images)

In UI:
1. Keep `Inference Cameras` as default 4 cameras.
2. Keep `Frames Per Camera = 4`.
3. Click `Start Sync` (it auto-plays video).

The service will infer every `Sync Interval (s)` using `4 x 4 = 16` images and refresh
`pred_traj / gt_traj / ADE / FDE` in sync with playback.

## Environment variables

- `MODEL_BASE_URL` (default: `http://127.0.0.1:29003/v1`)
- `MODEL_NAME` (default: `Qwen/Qwen3-VL-8B-Instruct`)
- `MODEL_API_KEY` (default: `EMPTY`)
- `VIDEO_CACHE_MAX_ENTRIES` (default: `16`)
- `VIDEO_CACHE_MAX_BYTES` (default: `536870912`, i.e. 512MB)

Example:

```bash
MODEL_BASE_URL=http://127.0.0.1:29003/v1 \
MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct \
uvicorn av_web_demo.app:app --host 0.0.0.0 --port 8899
```

## API summary

- `GET /api/health`
- `GET /api/cache_stats`
- `GET /api/clips`
- `GET /api/cameras`
- `GET /api/video_meta/{clip_id}/{camera_name}`
- `GET /api/video_meta_by_num/{clip_num}/{camera_name}`
- `GET /api/video/{clip_id}/{camera_name}.mp4`
- `GET /api/video_by_num/{clip_num}/{camera_name}.mp4`
- `POST /api/predict`
- `POST /api/predict_from_video_time`

`POST /api/predict*` accepts either:
- `clip_num` (recommended)
- or `clip_id` (backward compatible)

Both predict endpoints also support:
- `include_input_images` (`bool`, default `false`) to return input images in base64.

## Notes

- Ground truth trajectory is read from dataset `ego_future_xyz`.
- Prediction trajectory is read from model response `sglext.traj_xyz`.
- Comparison metrics are computed after step alignment with common length.
- Returned `pred_xy_m` / `gt_xy_m` are rotated to align with driving direction.
  Raw trajectories are also returned as `pred_xy_m_raw` / `gt_xy_m_raw`.
- Frontend pauses video before `/api/predict_from_video_time` request and resumes after trajectory is drawn.
