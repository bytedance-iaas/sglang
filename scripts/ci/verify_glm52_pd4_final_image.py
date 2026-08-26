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

eagle_worker_src = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py"
).read_text()
assert "def requires_dp_attention_eager_forward" in eagle_worker_src
dp_attn_src = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_components/dp_attn.py"
).read_text()
assert "can_draft_cuda_graph" in dp_attn_src
from sglang.srt.managers.schedule_batch import ScheduleBatch as _SB

assert hasattr(_SB, "force_disable_draft_cuda_graph")
assert hasattr(_SB, "can_run_dp_draft_cuda_graph")

server_args_src = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/server_args.py"
).read_text()
assert "HiSparse compressed KV is owned by the" in server_args_src
scheduler_src = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py"
).read_text()
assert "pd_rebootstrap_forced_output_id" in scheduler_src
schedule_batch_src = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/managers/schedule_batch.py"
).read_text()
assert "disaggregation_decode_enable_offload_kvcache" in schedule_batch_src

coordinator_src = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/managers/hisparse_coordinator.py"
).read_text()
assert "_compute_padded_grow" in coordinator_src
assert "demote_until_hisparse_available" in coordinator_src
allocator_src = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/mem_cache/allocator/hisparse.py"
).read_text()
assert "class _HiSparsePageOwnership" in allocator_src
assert "_stable_unique_page_ids" in allocator_src
assert "release_hisparse_ownership" in allocator_src
assert "Only the canonical HiSparse slot owner may abort staging requests" in (
    coordinator_src
)
assert "It must neither clear the canonical" in coordinator_src

eagle_common_src = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_common.py"
).read_text()
assert "def _finalize_hisparse_accepted_tokens" in eagle_common_src
assert "coordinator.finalize_accepted_tokens_spec_v2(" in eagle_common_src
assert eagle_common_src.count("_finalize_hisparse_accepted_tokens(") >= 2

disagg_utils_src = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/disaggregation/utils.py"
).read_text()
prefill_src = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/disaggregation/prefill.py"
).read_text()
assert "Every participant proved that this keyed queue is empty" in (
    disagg_utils_src
)
assert "ordered_keys=[req.rid for req in candidates]" in prefill_src
assert 'pp_disagg_prefill_poll_groups["bootstrap"]' in prefill_src
assert 'pp_disagg_prefill_poll_groups["transfer"]' in prefill_src
assert "PP already polled every local rank" in prefill_src
assert "Initialized PP prefill poll groups" in scheduler_src

common_conn_src = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/disaggregation/common/conn.py"
).read_text()
mooncake_conn_src = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/disaggregation/mooncake/conn.py"
).read_text()
assert "def begin_request(" in common_conn_src
assert "request_failure_history" in common_conn_src
assert "bootstrap_generation" in schedule_batch_src
assert "supports_request_generation = True" in mooncake_conn_src
assert "Ignoring stale status for room" in mooncake_conn_src

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
            "hisparse_page_ownership_fix": True,
            "hisparse_eagle_v2_finalizer_fix": True,
            "pp_empty_poll_consensus_fix": True,
            "pp_poll_phase_isolation_fix": True,
            "request_generation_isolation_fix": True,
        },
        sort_keys=True,
    )
)
