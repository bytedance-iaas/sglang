import importlib.metadata
import json
import pathlib

import sglang
import torch


actual = pathlib.Path(sglang.__file__).resolve()
expected = pathlib.Path("/sgl-workspace/sglang/python/sglang/__init__.py")
assert actual == expected, (actual, expected)
sglang_kernel = importlib.metadata.version("sglang-kernel")
flashinfer_python = importlib.metadata.version("flashinfer-python")
assert sglang_kernel == "0.4.5", sglang_kernel
assert flashinfer_python == "0.6.14", flashinfer_python
assert torch.__version__ == "2.11.0+cu130", torch.__version__
assert torch.version.cuda == "13.0", torch.version.cuda

# Observability delivery gate: the ported APMPlus gen-ai OTel provider must
# import and initialise inside the serving image, and must be wired into the
# OpenAI chat entrypoint. instrumentation.logging/system_metrics stay optional.
from sglang.srt import openai_observability as obs

obs_file = pathlib.Path(obs.__file__).resolve()
expected_obs = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/openai_observability.py"
)
assert obs_file == expected_obs, (obs_file, expected_obs)
assert hasattr(obs, "otel_provider"), "otel_provider missing"
assert hasattr(obs, "is_otel_available"), "is_otel_available missing"
otel_available = bool(obs.is_otel_available())
serving_chat_src = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py"
).read_text()
assert "openai_observability" in serving_chat_src, "chat entrypoint not wired"
assert "otel_provider.record" in serving_chat_src, "record() not wired"

# Draft-graph-vote fix gate (#32209 backport): the seedless MTP fallback must be
# rank-consistent under DP attention. Confirm the fix is baked into this image.
eagle_worker_src = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py"
).read_text()
assert (
    "def requires_dp_attention_eager_forward" in eagle_worker_src
), "MTP DP-vote fix missing: requires_dp_attention_eager_forward"
dp_attn_src = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_components/dp_attn.py"
).read_text()
assert "can_draft_cuda_graph" in dp_attn_src, "MTP DP-vote fix missing: can_draft_cuda_graph"
from sglang.srt.managers.schedule_batch import ScheduleBatch as _SB

assert hasattr(
    _SB, "force_disable_draft_cuda_graph"
), "MTP DP-vote fix missing: force_disable_draft_cuda_graph"
assert hasattr(
    _SB, "can_run_dp_draft_cuda_graph"
), "MTP DP-vote fix missing: can_run_dp_draft_cuda_graph"

# HiSparse retract-fix gate: (a) decode retract KV offload must be gated on the
# offload flag with a PD rebootstrap fallback, and (b) HiSparse + decode offload
# must be a hard startup guard. Confirm both are baked into this image.
server_args_src = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/server_args.py"
).read_text()
assert (
    "HiSparse compressed KV is owned by the" in server_args_src
), "HiSparse retract fix(b) missing: HiSparse+decode-offload guard"
scheduler_src = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py"
).read_text()
assert (
    "pd_rebootstrap_forced_output_id" in scheduler_src
), "HiSparse retract fix(a) missing: decode retract rebootstrap route"
schedule_batch_src = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/managers/schedule_batch.py"
).read_text()
assert (
    "disaggregation_decode_enable_offload_kvcache" in schedule_batch_src
), "HiSparse retract fix(a) missing: release_req offload gate"

# HiSparse retract-fix (c): the spec-verify draft-slot grow path must reclaim
# resident device pages before failing, matching the admission/dynamic-decode
# allocation sites. Confirm the reclaim-before-raise wiring is baked in.
coordinator_src = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/managers/hisparse_coordinator.py"
).read_text()
assert (
    "_compute_padded_grow" in coordinator_src
), "HiSparse retract fix(c) missing: _compute_padded_grow helper"
assert (
    "demote_until_hisparse_available" in coordinator_src
), "HiSparse retract fix(c) missing: reclaim-before-raise in _ensure_padded_buffer"

print(
    json.dumps(
        {
            "sglang_file": str(actual),
            "sglang_kernel": sglang_kernel,
            "flashinfer_python": flashinfer_python,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "observability_file": str(obs_file),
            "observability_available": otel_available,
            "mtp_dp_vote_fix": True,
            "hisparse_retract_fix": True,
        },
        sort_keys=True,
    )
)
