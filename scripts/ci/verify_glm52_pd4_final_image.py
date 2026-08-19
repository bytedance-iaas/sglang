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
        },
        sort_keys=True,
    )
)
