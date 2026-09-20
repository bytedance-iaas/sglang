# DeepSeek-V4.1-Flash with SM90 MegaMoE

This recipe uses one Linux CUDA host with eight H20 GPUs in one NVLink domain,
TP8/EP8, and the `iaas-kernels` SGLang plugin. Routed experts use MXFP4 weights
and FP8 activations (W4A8). FlashInfer's default SM90 MXFP4 path uses BF16
activations (W4A16), so generated tokens need not match across these backends.

## Install

Use SGLang's `dsv4.1` branch with the fused MoE extension API v1 in this change.
Install it into your model-serving environment from the repository root:

```bash
python3 -m pip install -e python
```

Install an `iaas-kernels` wheel containing the SM90 plugin and built against
the serving environment's PyTorch/CUDA ABI. The validated kernel revision is
`6054ef39ba9a66180c7bd1ce862b5ac4cfe45ac3` from
`https://github.com/bytedance-iaas/iaas-kernels` (PyTorch 2.13). To build that revision:

```bash
git clone --recurse-submodules https://github.com/bytedance-iaas/iaas-kernels.git
cd iaas-kernels
git checkout 6054ef39ba9a66180c7bd1ce862b5ac4cfe45ac3
git submodule update --init --recursive
bash build_iaas_kernels.sh
python3 -m pip install --force-reinstall --no-deps dist/iaas_kernels-*.whl
```

Follow the kernel repository's build prerequisites (CUDA toolkit, C++20
compiler, `libdw-dev`, `apache-tvm-ffi==0.1.11`, `ninja`, and `build`). This wheel
coexists with `sgl-deep-gemm`; it does not replace that package, which other
model operations and the speculative draft may still use. Python-only changes
to an editable SGLang installation do not require a kernel rebuild.

## Configure

Set these variables before starting workers:

```bash
export SGLANG_PLUGINS=iaas_kernels
export SGLANG_DSV4_FP4_EXPERTS=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=2048
export SGLANG_DSV41_MEGAMOE_REDUCE_SCATTER=0
```

| Setting | Purpose |
|---|---|
| `SGLANG_PLUGINS=iaas_kernels` | Explicitly load the wheel's entry point. If you use other plugins, include their names in this comma-separated list. |
| `SGLANG_DSV4_FP4_EXPERTS=1` | Retain the model's packed FP4 expert weights; this is already the branch default. |
| `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=2048` | Allocate the per-rank token capacity at load time. Cover both prefill chunks and graph/verify token rows; this is not the number of concurrent requests. The branch default is 8192. Exceeding the capacity raises an error. |
| `SGLANG_DSV41_MEGAMOE_REDUCE_SCATTER=0` | Keep the optional token-sharded residual optimization off for the DSPARK recipe below. It defaults to off. |

`IAAS_KERNELS_JIT_CACHE_DIR` optionally selects a writable JIT cache directory.
For the measured setup, `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` disables eager
DeepGEMM precompilation; it is not the switch that selects MegaMoE. Keep
`SGLANG_RAGGED_VERIFY_MODE=static` for the DSPARK recipe (the branch default).
Select `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` only if your host needs an
explicit network interface. No token or secret is required for a local checkpoint.

## Launch TP8/EP8 with DSPARK

Run this from an environment with both packages installed. Replace `MODEL_PATH`
with your local DeepSeek-V4.1-Flash checkpoint directory, including its draft
weights. The target and draft use the same checkpoint path.

```bash
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_RAGGED_VERIFY_MODE=static

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name DeepSeek-V4.1-Flash \
  --trust-remote-code \
  --tp-size 8 --ep-size 8 --dp-size 1 \
  --attention-backend dsv4 --image-processor-backend pil \
  --moe-a2a-backend megamoe \
  --moe-runner-backend deep_gemm \
  --disable-shared-experts-fusion \
  --chunked-prefill-size 2048 --context-length 16384 \
  --max-running-requests 128 --cuda-graph-max-bs-decode 128 \
  --mem-fraction-static 0.90 --random-seed 12345 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path "$MODEL_PATH" \
  --speculative-dspark-block-size 5 \
  --reasoning-parser auto --tool-call-parser auto \
  --host 0.0.0.0 --port 30241
```

`--moe-a2a-backend megamoe` selects the SM90 plugin, which owns routed-expert
dispatch, computation, and combine. `--moe-runner-backend deep_gemm` is the
tested companion setting; it does not bypass the plugin. TP8/EP8 gives MoE
TP=1. `--disable-shared-experts-fusion` keeps the shared expert outside the
fused routed-expert kernel, as required by this adapter. Single-host transport
selection is automatic; the Docker container name does not select RDMA.

To reproduce the non-thinking evaluation, pass
`"chat_template_kwargs": {"thinking": false}` in each chat request. This
controls the model's thinking mode, not the MoE backend. Closing thinking does
not guarantee a short response or eliminate repeated generation.

## Optional ReduceScatter path

For ordinary decode without DSPARK, remove all three `--speculative-*`
arguments and set `SGLANG_DSV41_MEGAMOE_REDUCE_SCATTER=1`. The optimization
reduce-scatters attention's output projection, keeps the mHC residual and
shared/routed expert work local to each token shard, and gathers the next
attention input. The reduction uses FP32 accumulation before returning BF16.

The guard requires DeepSeek-V4.1, SM90, TP8/EP8, attention DP=CP=PP=1,
MoE TP=1, compatible pre-mix handoff, and a positive token-row count divisible
by eight. Prefill, DSPARK/hidden-state capture, incompatible input layouts,
and other models retain their existing path. Setting the variable to 1 does
not enable this optimization for DSPARK verification.

## Verify the selected backend

Check worker logs for `IAAS MegaMoE transport=nvlink EP=8 physical_nodes=1`.
If the wheel is missing or excluded by `SGLANG_PLUGINS`, SM90 MegaMoE reports a
plugin error instead of silently falling back. Do not enable TBO, live EPLB,
or DWDP expert offload with this adapter. This recipe validates text inference;
preserving vision-aware routing is not a multimodal accuracy certification.

Full GSM8K runs have shown occasional repetitive responses reaching the output
limit with this W4A8 path. The original triggering numerical difference remains
unresolved; neither enabling thinking nor FP32 LM-head output is a validated
fix. Record accuracy, length-limit hits, and throughput separately when
comparing with FlashInfer W4A16.
