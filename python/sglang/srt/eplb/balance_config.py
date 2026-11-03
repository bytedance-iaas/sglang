"""Shared row-block unit for Block-LPLB and migration."""

import os

BLOCK_SIZE = int(os.getenv("SGLANG_LPLB_BLOCK_SIZE", "64"))
if BLOCK_SIZE not in (64, 128):
    raise ValueError(f"Unsupported load-balance block size: {BLOCK_SIZE}")
CUDA_FLAGS = [f"-DWAN22_BALANCE_BLOCK_SIZE={BLOCK_SIZE}"]
