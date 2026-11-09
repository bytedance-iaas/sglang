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

## Experimental compact PD decode

The compact extension supports DeepSeek-V4.1 Mooncake **decode** with MegaMoE,
one attention rank per DP group (`attn_dp_size == tp_size`), PP=CP=DCP=1,
and overlap scheduling. Prefill coordination and compact prefill remain
unsupported. Keep the prefill server on static verification and use the same
DSPARK block size and KV layout on both servers.

For a TP8/EP8 PD decode command with attention DP8, change the verify mode and
provide a cost table measured for the serving configuration:

```bash
export SGLANG_RAGGED_VERIFY_MODE=compact
# Add to the decode server's existing arguments:
# --speculative-dspark-sps-table-path /path/to/profiled-sps.json
```

Without a profiled SPS table, compact defaults to verify-all and does not
adapt the verify budget. Check the startup log for `graph_tier=dp-gathered`
and inspect `verify_tokens_local`, `verify_tokens_dp_synced`, and
`verify_tokens_graph_key` in DSPARK records when validating compaction.

The 1017 checkpoint completed graph capture and 64/256-concurrency PD runs on
eight H20s with this extension. Those runs used forced diagnostic budgets;
they do not establish adaptive throughput gains. Four instructed chat prompts,
repeated for 64 requests across full and trimmed budgets, matched static output.
Raw greedy completions varied even in the static control; broader accuracy
validation is still needed before enabling this path for production traffic.

### Calibrated B6 / 16K table on H20

The bundled [B6 cost table](dspark_sps_b6_16k_h20.json) was recalibrated for
commit `ea45913771` with the 1017 checkpoint on eight H20s. Decode uses
TP8/EP8/attention-DP8, `dsv4` attention, `deep_gemm`/MegaMoE, `humming`,
`--cuda-graph-max-bs-decode 32`, and `--max-running-requests 256`.
The [calibration metadata](dspark_sps_b6_16k_h20.metadata.json) records the
optimization environment, graph tiers, validation errors, and benchmark setup.

From the repository root, apply these settings to a PD decode command matching
that configuration. Use B6 and a compatible KV layout on the prefill server too:

```bash
export SGLANG_RAGGED_VERIFY_MODE=compact
unset SGLANG_SIMULATE_ACC_LEN
# Add to the decode server's existing arguments:
# --speculative-dspark-block-size 6
# --speculative-dspark-sps-table-path examples/runtime/deepseek_v41_megamoe/dspark_sps_b6_16k_h20.json
# --speculative-dspark-align-verify-tokens-to-graph-tier
```

The 12 held-out calibration windows had a mean absolute relative error of
1.00% (maximum 1.79%). Six sparse active-rank windows had 7.94% mean error
(maximum 10.48%). Revalidate the table when changing the serving configuration.

Three runs per mode used a generated shared-prefix workload with approximately
16.7K input tokens and 1K output tokens, 1,024 requests per run, and no simulated
acceptance. All 18,432 requests succeeded. Relative to the matched B6 static
baseline, the mean results were:

| Concurrency | Throughput change | Request P99 TPOT change |
|---|---:|---:|
| 64 | +8.58% | +2.61% |
| 128 | +8.09% | +2.59% |
| 256 | +5.05% | +7.80% |

Positive P99 changes mean higher latency. At concurrency 256, the three paired
throughput changes were +9.89%, -0.92%, and +6.02%. This configuration improved
mean throughput while leaving request-level tail latency worse in this workload.
The comparison measures the combined new-table and graph-fill configuration.

### Experimental compensation for repeatedly capped requests

`SGLANG_DSPARK_VERIFY_COMPENSATION_MAX_BOOST=0.5` enables an experimental
compact-only ranking policy. The default `0` keeps the original scheduler.
A request gains one credit when its verified window is shorter than B+1 and
all of that window is committed. Credit is capped at eight and halves on
other steps. Its candidate ranking weight is `1 + max_boost * credit / 8`;
raw survival still determines eligibility and preserves candidate prefix order.
SPS selects the same budget before compensation, and graph-tier alignment is
unchanged. This reallocates the existing budget rather than enlarging it.
A cap hit does not prove that an unverified candidate would have been accepted.

Credit lives on the GPU. Slot ownership is checked against request identity;
new/reused slots start at zero and requests absent from the current batch lose
ownership. No per-step GPU readback is added for compensation. The policy is
experimental, not an established tail-latency improvement.

For attribution, `SGLANG_DSPARK_DEBUG_DUMP=core,step_cpu_time,reqs` additionally
records the exact historical survival used to select the budget, the aligned
budget, pre-update credit and ranking weight, and `baseline_verify_len` (the
unweighted allocation at the same aligned budget). The baseline is a one-step
counterfactual, not a replay of the output trajectory without compensation.
Detailed recording performs extra copies and an extra top-k with compensation;
use `core,step_cpu_time` only for performance measurements.

### Follow-up comparison (2026-11-09)

Two mirrored-order repetitions compared six configurations on two eight-H20
hosts with the 1017 checkpoint, B6, attention-DP8 decode, the same cost table,
and graph-tier fill. Each run used 1,024 requests at concurrency 256 with about
16.7K input tokens and exactly 1,024 output tokens. The generated workload used
one shared prefix (seed 42), with a 99.5% prefix-cache hit rate. Each decode
process was restarted; only the shared prefix was warmed before measurement.
Core and CPU-step diagnostics were enabled, but detailed request recording was
disabled. All 12,288 requests succeeded.

| Configuration | Output tokens/s | Request P99 TPOT (ms) | Token P99 ITL (ms) |
|---|---:|---:|---:|
| Static | 15,756.54 | 25.16 | 40.32 |
| Compact | 16,373.17 | 17.95 | 38.06 |
| Compact + compensation 0.25 | 14,299.58 | 18.93 | 37.02 |
| Compact + compensation 0.5 | 15,794.08 | 17.70 | 36.04 |
| Compact + STS | 15,800.03 | 19.50 | 39.60 |
| Compact + STS + compensation 0.25 | 15,534.75 | 20.87 | 41.37 |

These are arithmetic means of two runs, including run-level P99s, not pooled
P99s or statistical significance claims. Compact improved mean throughput by
3.91% and request P99 TPOT by 28.68% against the same-version static control.
The compensation-0.25 second run had a large first-token delay with a prefill
bootstrap backlog and decode transfer waits; the exact cause remains unresolved,
so its full throughput deficit cannot be attributed to compensation. The run
was retained rather than replaced with a selected rerun.

Neither STS nor compensation improved both throughput and request P99 in these
means; keep both disabled by default. STS reduced request-held-out calibration
ECE, but training and holdout requests shared one long prefix. This does not
establish cross-prefix generalization or a serving-speed improvement. All
recorded final single-request compact steps already used the full width of
seven; tail differences primarily appeared in the number of remaining steps.
Core records cannot separate remaining output work from actual rejection rates.

Stable eight-step windows in the dominant 224-token graph bucket had cost-table
overestimation of 2.68% and 1.49% in the two compact runs. Smaller buckets had
too few stable windows to justify replacing the table. CPU step intervals
include host scheduling and overlap effects and are not GPU-event timings.
These synthetic results and functional generation checks do not replace a
representative workload benchmark or a full model-accuracy evaluation.

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

### Collecting STS labels on the compact path

Use a separate diagnostic run with `SGLANG_RAGGED_VERIFY_MODE=compact`,
`SGLANG_DSPARK_FOLDED_PROPOSAL=0`, and
`SGLANG_DSPARK_STS_COLLECT_PATH=/absolute/fresh-directory/sample`.
Omit `--speculative-dspark-sps-table-path` to select the built-in uniform
verify-all layout while retaining the confidence head. A forced budget fraction
of one with a calibrated cost table can still exclude negligible-probability
positions; check the actual verification widths.

The recorder uses the verifier's real acceptance counts and excludes rows whose
verification window is shorter than `gamma + 1`. It writes separate TP-rank
shards containing raw logits, prefix acceptance masks, request IDs, prefix
lengths and local batch sizes. Use a fresh directory for each run. Collection
rejects folded proposals and simulated acceptance; omit the STS calibration
argument so the confidence temperatures remain the identity.

After the workload finishes, fetch `/server_info` before reading the shards.
This flushes pending samples on every rank, including ranks that did not execute
the idle decode hook. Match recorded `(request ID, prefix length)` pairs against
the request-level diagnostic records when validating a collection run.

Split calibration training and validation by request ID, and use a separate
workload seed for performance evaluation. Collection performs synchronous CPU
copies and runs the proposal eagerly. Restore folded proposal execution and
turn collection off for throughput comparisons; a lower training ECE alone does
not establish a serving speedup.
