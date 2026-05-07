# Piecewise CUDA Graph

## Motivation

Standard CUDA graphs capture the entire model forward pass as a single graph. This works well for decode (fixed batch size), but not for extend/prefill where the number of tokens varies across iterations.

Piecewise CUDA Graph (PCG) solves this by splitting the model's computation graph into pieces (roughly one per layer) at "split points" (e.g., MoE dispatch ops). Each piece is captured as a separate CUDA graph for a set of pre-defined token lengths. At runtime, the input is padded to the nearest captured size, and each piece is replayed. This eliminates kernel launch overhead for prefill/extend while still supporting dynamic shapes.

Recently we **enabled PCG by default**, which means that the old `--enable-piecewise-cuda-graph` flag is deprecated. Use `--disable-piecewise-cuda-graph` to turn it off.

## Usage

PCG is enabled by default for supported configurations. No extra flags needed:

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct
```

### Disable PCG

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disable-piecewise-cuda-graph
```

### Custom capture sizes

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --piecewise-cuda-graph-max-tokens 2048
```

### Server Args

| Argument | Default | Description |
|---|---|---|
| `--disable-piecewise-cuda-graph` | `False` | Disable PCG for extend/prefill. |
| `--enforce-piecewise-cuda-graph` | `False` | Force-enable PCG, skipping all auto-disable conditions. For testing only. |
| `--piecewise-cuda-graph-max-tokens` | `None` (auto) | Maximum token count to capture. Defaults to `chunked_prefill_size` (non-MLA) or `2048` (MLA). |
| `--piecewise-cuda-graph-tokens` | `None` (auto) | Explicit list of token lengths to capture. Auto-generated if not set. |
| `--piecewise-cuda-graph-compiler` | `"eager"` | Compiler backend for the captured subgraphs. Choices: `eager`, `inductor`. |
| `--log-pcg-pad-stats` | `False` | Periodically log PCG padding statistics (raw vs padded token counts, avg/max pad ratio). Useful for tuning `--piecewise-cuda-graph-tokens`. See [Performance Tuning](#performance-tuning). |
| ~~`--enable-piecewise-cuda-graph`~~ | — | **Deprecated.** PCG is now enabled by default. Use `--enforce-piecewise-cuda-graph` to skip auto-disable conditions. |

## Bug Report

PCG is enabled by default but is still in an experimental stage. Since PCG relies on `torch.compile` to trace the model's forward pass, most bugs are introduced by torch compile tracing failures (e.g., untraceable ops, dynamic control flow, or graph breaks). If you encounter any issues related to PCG, please disable it by adding `--disable-piecewise-cuda-graph` to your launch command and report the bug at [GitHub Issues](https://github.com/sgl-project/sglang/issues/new/choose). We greatly appreciate your help in improving this feature.

### For Users

If you see an error message like the following during server startup, it is a PCG bug:

```
Piecewise CUDA Graph is enabled by default as an experimental feature.
To work around this error, add --disable-piecewise-cuda-graph to your launch command.
Please report this issue at https://github.com/sgl-project/sglang/issues/new/choose
```

To work around it, add `--disable-piecewise-cuda-graph` to your launch command. When filing a bug report, please include:
1. The full error traceback
2. Model name and quantization method
3. Launch command with all arguments
4. GPU type and driver version

### For Developers

Since PCG relies on `torch.compile` to trace the model's forward pass, newly developed CUDA kernels (both JIT kernels and sgl-kernels) are typically not compatible with `torch.compile` out of the box. The tracing will fail on untraceable operations such as JIT compilation, file I/O, or dynamic module loading inside the kernel.

To make a kernel compatible with PCG, you need to register it as a custom op using `register_custom_op` from `sglang.srt.utils.custom_op`. This wraps the kernel as an opaque node in the compiled graph so that `torch.compile` will not trace inside it.

**Example usage (JIT kernel):**

```python
from sglang.srt.utils.custom_op import register_custom_op

# Inplace operator (no return value)
@register_custom_op(mutates_args=["output_q", "output_s"])
def per_token_group_quant_8bit(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    # kernel implementation ...
```

**Example usage (operator with output):**

```python
# out_shape indicates which argument has the same shape as the output
@register_custom_op(mutates_args=["x"], out_shape=0)
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.add_(y)
```

For wrapping external library functions (e.g., FlashInfer kernels), use `register_custom_op_from_extern` instead. See `python/sglang/srt/utils/custom_op.py` for full API documentation.

## How it works
### Torch compile backend

PCG uses `torch.compile` with a custom backend (`SGLangBackend`) to split and compile the model's forward pass. The flow is:

```
model.forward wrapper
→ torch.compile(..., backend=SGLangBackend)
→ FX graph
→ split_graph() at registered split ops
→ split_gm (top-level graph that chains the pieces)
→ replace capturable submodules with CUDAPiecewiseBackend
→ runtime dispatch: eager split ops + per-piece capture/replay
```

- **Install**: `install_torch_compiled()` replaces `model.forward` with a wrapper function. When `is_in_piecewise_cuda_graph()` returns True, the wrapper dispatches to the compiled callable; otherwise it falls back to the original forward. The first invocation through this path triggers Dynamo tracing and graph compilation — CUDA graph replay only happens after the capture phase completes.

- **Split**: When `torch.compile` traces the model, `SGLangBackend` receives the FX graph and calls `split_graph()`. Ops listed in `CompilationConfig.split_ops` are treated as split points, so the graph is cut at each one. These split-op submodules are left to run eagerly at runtime, while the surrounding submodules are compiled and wrapped by `CUDAPiecewiseBackend`. The result is a top-level "stitching graph" (`split_gm`) with children such as `submod_0`, `submod_1`, … interleaving capturable subgraphs and eager split-op submodules.

- **Replace**: `PiecewiseCompileInterpreter` iterates over each capturable submodule in `split_gm`, compiles it for general (dynamic) shapes, and replaces it in-place with a `CUDAPiecewiseBackend` instance. Split-op submodules (e.g., attention, all-reduce) are left as-is and run eagerly at runtime.

- **Dispatch**: At runtime, calling `split_gm` executes the stitching graph, which calls each submodule in order. Split-op submodules run eagerly. Each `CUDAPiecewiseBackend` submodule goes through three phases:
  - **Compile warmup** — runs the general-shape compiled path.
  - **Capture** — for each capture size, runs one warmup pass then records a CUDA graph.
  - **Steady-state replay** — replays the captured CUDA graph for each forward pass.

### Piecewise cuda graph runner

`PiecewiseCudaGraphRunner` orchestrates the full lifecycle through three phases:

- **Compile** — Warms up JIT kernels with a dummy forward pass, then wraps the model with `torch.compile`, triggering Dynamo tracing to split the FX graph and create `CUDAPiecewiseBackend` instances for each subgraph piece.

- **Capture** — Iterates over capture sizes in reverse order (largest first). For each size, runs the forward pass twice (one warmup, one CUDA graph capture).

- **Replay** — At runtime, finds the smallest captured size >= actual token count via binary search, copies inputs into static buffers with zero-padding, replays the captured CUDA graphs, and slices outputs back to the actual token count.

### Memory optimization

The memory cost of PCG comes from two parts: **torch memory allocator** and **non-torch memory**.

The torch memory allocator overhead is trivial thanks to several optimizations: a global shared memory pool is reused across all CUDA graph runners and capture sizes, capture is done in reverse order (large to small) so smaller graphs reuse memory allocated by larger ones, and output tensors of the last subgraph are stored as weak references to maximize memory reuse.

The main memory overhead comes from non-torch memory — the CUDA graph objects themselves require GPU memory to store the recorded kernel launch parameters and internal state. This overhead scales with the number of captured sizes, which is why `piecewise_cuda_graph_max_tokens` is capped conservatively by default.

### Shape configuration
Piecewise CUDA graph pre-captures graphs for a set of token counts. At runtime, the actual token count is rounded up to the nearest captured size (via binary search), and the corresponding graph is replayed. If the token count exceeds the largest captured size, the runtime falls back to the normal (non-graph) forward path.

The default capture schedule is auto-generated with increasing granularity:

| Token range | Step size |
|-------------|-----------|
| 4 – 32      | 4         |
| 48 – 256    | 16        |
| 288 – 512   | 32        |
| 576 – 1024  | 64        |
| 1280 – 4096 | 256       |
| 4096+       | 512       |

For the auto-generated schedule, sizes are capped at `--piecewise-cuda-graph-max-tokens`. The default cap is `chunked_prefill_size` for non-MLA models and `2048` for MLA backend models. If `--max-total-tokens` is set, the cap is further limited to not exceed it. Additionally, Llama-2 models are auto-capped at 4096 tokens as a temporary workaround.

## Performance Tuning

PCG's benefit is **not universal** — whether it improves throughput depends on the interplay between kernel-launch overhead, kernel execution time, and padding overhead. This section gives an empirical framework for deciding when (and how) to use PCG.

### When PCG helps vs hurts

PCG throughput delta ≈ `Σ(launch overhead saved)` − `Σ(pad waste)` − `Σ(replay_prepare overhead)`

For the delta to be positive:

| Factor | Favors PCG (speedup) | Hurts PCG (regression) |
|---|---|---|
| **Model architecture** | Many small ops per layer (hybrid linear / SSM, MoE with frequent dispatch/combine, models with lots of normalization/residual fusions) | Dense Transformer with a few heavy ops per layer |
| **Token count per request** | Small prefill (≤ 256 tokens): launch overhead dominates wall time | Large prefill (> 1024 tokens): already compute-bound, launch savings are negligible |
| **Bucket alignment** | `raw_tokens` falls very close to a captured bucket (pad ratio ≈ 1.0) | `raw_tokens` falls just above a bucket boundary (pad ratio ≥ 1.1) |
| **GPU** | High FLOPs/launch ratio (H100, H200) | Memory-bound or older GPUs where pad wastes more time |

As a rule of thumb:

- **Hybrid-linear / Mamba-like / MoE models** (e.g. Qwen3.5-MoE) tend to see consistent speedup because per-layer kernel count is higher.
- **Dense VL models** (e.g. Qwen2.5-VL, Qwen3-VL) often show **neutral or slightly negative** impact on prefill benchmarks, especially once image tokens push per-request token counts into the hundreds or thousands.
- Even within a single model, a small prefill request may speed up by ~5% while a large one slows by ~1%, canceling out over a mixed workload.

We recommend **benchmarking on your actual workload** before deciding to keep PCG enabled in production.

### Observing padding overhead

Padding is the single most tunable source of PCG overhead. To quantify it for your workload, enable:

```bash
python3 -m sglang.launch_server \
    --model-path <your model> \
    --log-pcg-pad-stats
```

Every 100 replays the server logs a line such as:

```
[PCG pad stats] count=100 avg_pad_ratio=1.045 max_pad_ratio=1.180 (raw_sum=82341 padded_sum=86031)
```

Interpreting the numbers:

- `avg_pad_ratio` close to 1.02 — padding overhead is minimal; buckets are well-matched.
- `avg_pad_ratio` ≥ 1.10 — meaningful compute is being wasted on padded tokens; consider tuning `--piecewise-cuda-graph-tokens`.
- `max_pad_ratio` ≥ 1.30 — at least one request fell near the start of a wide bucket interval (most common around the 1024→1280 or 4096→4608 jumps).

This flag is off by default and introduces no graph-break risk because the accounting happens outside any compiled graph.

### Tuning `--piecewise-cuda-graph-tokens`

If `avg_pad_ratio` is high, override the default schedule to match your request distribution:

```bash
python3 -m sglang.launch_server \
    --model-path <your model> \
    --piecewise-cuda-graph-tokens 4 8 16 32 64 96 128 160 192 224 256 \
                                  288 320 384 448 512 640 768 896 1024
```

Guidelines:

1. **Analyze first** — collect real prefill token counts (the server's `Prefill batch` log line contains `#new-token`). Build a histogram and identify the hot ranges.
2. **Densify hot ranges** — where 80%+ of requests live, space buckets ≤ 5% apart (so `max pad ratio < 1.05`).
3. **Drop cold ranges** — each bucket costs one warmup + one captured cudagraph (≈ a few hundred MB of GPU memory and a few seconds of startup), so don't keep buckets you'll never hit.
4. **Watch the big jumps** — the default schedule transitions from step=64 to step=256 at 1024. If many of your requests fall in 1025–1536, inserting extra buckets (e.g. 1088, 1152, 1216, 1280) can noticeably reduce waste.
5. **Re-measure** — re-enable `--log-pcg-pad-stats` and confirm `avg_pad_ratio` actually dropped before trusting the new schedule.

### A concrete example

From a Qwen3-VL benchmark with 720-px images (525 raw tokens → 576 padded bucket):

| Config | `avg_pad_ratio` | Per-request latency |
|---|---|---|
| PCG on, default buckets | 1.097 | 278.5 ms |
| PCG off | — | 274.6 ms |
| PCG on, custom buckets with 528 inserted | ~1.01 | ~275 ms |

Here padding ate the ~5 ms of launch overhead that PCG would otherwise save; inserting a 528 bucket recovers the savings. The same benchmark on Qwen3.5-9B (a hybrid-linear model) stayed net-positive even at `pad_ratio=1.07`, because the launch savings from the denser kernel graph dominated the pad waste.

The takeaway: **treat PCG and its bucket schedule as a pair of knobs to tune together**, not as a global on/off switch.

## Compatibility

PCG is auto-disabled in the following scenarios. We are actively working on expanding compatibility — support for many of these will be coming soon.

- Disabled model architectures (e.g., `DeepseekV32ForCausalLM`)
- Speculative decoding
- DP attention
- Pipeline parallelism (`pp_size > 1`)
- Non-CUDA hardware (AMD ROCm, Ascend NPU)
- MoE A2A backend
- LoRA
- Multimodal / VLM models
- DLLM (diffusion LLM)
- Deterministic inference
- PD disaggregation
- Expert distribution recorder / EPLB

Use `--enforce-piecewise-cuda-graph` to skip all auto-disable checks (for testing/debugging only).

## Code Reference

| File | Description |
|---|---|
| `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` | Main runner: init, capture, replay |
| `python/sglang/srt/compilation/compile.py` | `install_torch_compiled` trampoline |
| `python/sglang/srt/compilation/backend.py` | `SGLangBackend`, graph splitting, piecewise compilation |
| `python/sglang/srt/compilation/cuda_piecewise_backend.py` | Per-subgraph CUDA graph capture/replay |
| `python/sglang/srt/compilation/piecewise_context_manager.py` | Global context flags and `ForwardContext` |
| `python/sglang/srt/compilation/compilation_config.py` | Capture sizes, split ops, compiler config |
| `python/sglang/srt/utils/custom_op.py` | `register_custom_op` for torch.compile compatibility |
| `python/sglang/srt/server_args.py` | Server arguments and auto-disable logic |
