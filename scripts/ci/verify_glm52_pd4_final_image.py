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
print(
    json.dumps(
        {
            "sglang_file": str(actual),
            "sglang_kernel": sglang_kernel,
            "flashinfer_python": flashinfer_python,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        sort_keys=True,
    )
)
