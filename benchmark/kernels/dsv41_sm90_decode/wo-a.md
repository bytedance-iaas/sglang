# DSV4.1 SM90 decode tuning

Default-off optimization evaluated on 8 NVIDIA H20 GPUs, attention DP=8,
EP=8, BF16 residuals, Humming block-FP8 dense projections, and DSpark block size 5.
Set the corresponding environment variable before starting the server and
capturing CUDA graphs.

## WO_A

Enable `SGLANG_OPT_DSV41_SM90_WO_A=1`. The admitted BF16 target-verify path has
8 local groups and M=96..192. M<160 uses token-major cuBLAS BMM output to avoid
an output-layout copy; M=160..192 uses the measured Triton GEMM. Ordinary
decode, unsupported shapes and disabled flags retain their existing paths.

For 40 independent layer weights at M=120, WO_A plus layout changed from
3.127 to 2.925 ms. In the full 160-request trace, the layout copy's 0.210 ms
was removed while GEMM time was essentially unchanged. Tests cover FP32
reference tolerances, dispatch fallback, and graph input/weight updates.
WO_A is BF16-tolerance validated; it is not claimed bitwise equal to einsum.

```bash
PYTHONPATH=python python3 test/registered/kernels/ops/attention/dsv4/test_sm90_wo_a.py
```
