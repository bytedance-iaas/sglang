import importlib.metadata
import json
import pathlib

import sglang
import torch


actual = pathlib.Path(sglang.__file__).resolve()
expected = pathlib.Path("/sgl-workspace/sglang/python/sglang/__init__.py")
assert actual == expected, (actual, expected)
print(
    json.dumps(
        {
            "sglang_file": str(actual),
            "sglang_kernel": importlib.metadata.version("sglang-kernel"),
            "flashinfer_python": importlib.metadata.version("flashinfer-python"),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        sort_keys=True,
    )
)
