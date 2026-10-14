# DSV4.1 SM90 decode tuning

Default-off optimization evaluated on 8 NVIDIA H20 GPUs, attention DP=8,
EP=8, BF16 residuals, Humming block-FP8 dense projections, and DSpark block size 5.
Set the corresponding environment variable before starting the server and
capturing CUDA graphs.

## mHC

Enable `SGLANG_OPT_DSV41_SM90_FUSED_MHC=1` before graph capture. For the admitted
SM90 V4.1 shapes, use the existing TF32x3 partial projection and fused
reduce/Sinkhorn with one pipeline stage. At measured small row counts use
2 warps; retain 4 warps for aligned M>=128 shapes and outside the measured
96..180 interval. K splits, BLOCK_K=64, dot precision and Sinkhorn math remain
unchanged. Unsupported shapes retain the original path.

At M=120, an 80-call stack with independent inputs/weights (40 layers x 2)
reduced partial-projection time from 3.192 to 2.606 ms. At M=100 it changed
from 2.687 to 1.931 ms. These are operator totals, not whole-model latency.
The small-batch launch change was bitwise checked against the preceding fused
baseline, including graph replay, weight/input updates, strided rows and
cross-batch agreement. This is not a claim of whole-model bitwise equivalence
between enabled and disabled mHC algorithms.

```bash
PYTHONPATH=python python3 test/registered/kernels/ops/layernorm/test_mhc_sm90_launch.py
```


In the prior original-command benchmark (concurrency 152, 2048 requests,
15984 system + 16 question tokens, 1500 output tokens, three runs per group),
this launch tuning raised mean output throughput from 13562.84 to 13800.50
tok/s (+1.75%); mean TPOT changed from 10.1215 to 9.9555 ms. Both versions
already enabled all three optimization flags. This measures the incremental
launch change, not fused mHC on/off. Sequential runs remain subject to drift.
