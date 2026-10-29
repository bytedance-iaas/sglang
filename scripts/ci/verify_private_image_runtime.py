"""Verify private-delivery packages without initializing the EIC client.

The EIC SDK aborts during import on a build runner without its runtime
environment. Real client initialization and read/write checks belong in a
configured task Pod; image CI only proves that the SDK module is installed.
"""

import importlib.metadata
import importlib.util
import os

for distribution in (
    "sglang",
    "sglang-kernel",
    "mooncake-transfer-engine-cuda13",
    "eic",
    "onion-ai-data",
    "deep-ep",
):
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = "not-installed-as-distribution"
    print(f"{distribution}={version}")

expected_mooncake = os.environ.get("EXPECTED_MOONCAKE_VERSION")
if expected_mooncake:
    actual_mooncake = importlib.metadata.version("mooncake-transfer-engine-cuda13")
    if actual_mooncake != expected_mooncake:
        raise SystemExit(
            f"Mooncake version mismatch: expected={expected_mooncake} actual={actual_mooncake}"
        )

eic_spec = importlib.util.find_spec("eic")
if eic_spec is None:
    raise SystemExit("EIC SDK module is not installed")
print(f"eic-module={eic_spec.origin}")

import torch

print(f"torch={torch.__version__} cuda={torch.version.cuda}")

import sglang

print(f"sglang-module={sglang.__file__}")
