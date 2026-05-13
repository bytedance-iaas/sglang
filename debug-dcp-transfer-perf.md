# DCP Transfer Performance Debug [OPEN]

## Session
- Session ID: `dcp-transfer-perf`
- Goal: Explain and optimize the large TTFT gap between `TP8 -> TP16 + DCP2 + page_size=64` and the non-DCP baseline.
- Constraint: Collect runtime evidence before changing business logic; remove temporary instrumentation before final merge.

## Symptom
- User-reported benchmark:
  - DCP setup with `page_size=64`: TTFT about `54132.05 ms`.
  - Baseline combination: TTFT about `1639.86 ms`.
- Previous quick single-request check showed low latency, which does not match the user's benchmark and is insufficient.

## Hypotheses
- H1: `send_kvcache_dcp()` dominates TTFT because it sends many token-level RDMA blocks across layers/KV tensors.
- H2: DCP chunk slicing or metadata causes repeated or amplified transfer work compared with expected page/token counts.
- H3: KV transfer is not dominant; TTFT is mainly decode backend / DCP FlashInfer / DeepEP / symmetric memory first-token overhead.
- H4: Router or benchmark queueing/cache policy contributes significant waiting time, making endpoint-level TTFT much larger than isolated request latency.

## Evidence Plan
- Instrument DCP transfer without changing logic:
  - Count pages, source tokens, destination logical tokens, selected DCP tokens, layers, transfer blocks.
  - Measure address-building time and per-layer transfer time.
  - Measure total `send_kvcache_dcp()` wall time.
  - Record chunk index slice and DCP rank/size.
- Run user-like `page_size=64` topology and benchmark prompt/concurrency if available.
- Compare transfer time against request-level TTFT.

## Status
- OPEN: pre-fix evidence collected; testing first optimization.

## Pre-Fix Evidence
- Topology tested:
  - Prefill `TP8 + EP8`
  - Decode `TP16 + DCP2 + EP16`
  - Mooncake PD transfer, `page_size=64`
  - Temporary DCP transfer instrumentation enabled on prefill.
- Single 8K prompt:
  - Request wall time: about `1.682s`
  - Response `e2e_latency`: about `1.657s`
  - Prompt tokens: `8239`
  - DCP large chunk events: `16` events with `selected_tokens=4096`
  - Large event DCP transfer wall: p50 about `228.89ms`, max about `314.55ms`
- Concurrent 8K prompts:
  - `concurrency=4`: e2e p50 about `1.98s`, max about `2.61s`
  - `concurrency=8`: e2e p50 about `3.22s`, max about `4.72s`
  - `concurrency=16`: e2e p50 about `5.83s`, max about `9.10s`
  - `concurrency=32`: e2e p50 about `10.18s`, max about `17.11s`
  - Reported queue time remained `0.0`, so the observed growth is not explained by returned router queue metadata.
- DCP transfer event aggregate:
  - Done events: `1920`
  - Sum of DCP `total_ms`: about `246727.1ms`
  - Largest single DCP event: about `383.47ms`
  - Typical large event block count: `selected_tokens * layers * 2 = 4096 * 62 * 2 = 507904` transfer blocks.

## Evidence Interpretation
- H1 is confirmed as a material bottleneck under concurrency: the current DCP path submits very many token-granular RDMA blocks and scales poorly.
- H2 is not confirmed as a correctness-level amplification bug: selected token counts match DCP rank filtering expectations, e.g. `8192 / 2 = 4096`.
- H3 remains possible for the user's `54132ms` benchmark because isolated request latency did not reproduce `54s`, but KV transfer is still large enough to be worth optimizing.
- H4 remains possible because the user benchmark setup may include different concurrency, warmup, routing, or measurement policy than the isolated checks.

## Optimization Plan
- Step 1, low-risk A/B:
  - Keep DCP logical-token mapping unchanged.
  - Change `send_kvcache_dcp()` from per-layer K/V parallel Mooncake calls to cross-layer grouped batch calls.
  - Default chunk size: `SGLANG_DCP_TRANSFER_BATCH_BLOCKS=65536`.
  - Expected effect: reduce Mooncake calls for a 4096-token DCP chunk from about `124` calls to about `8` calls, while keeping the same RDMA byte layout.
- Step 2, higher-impact follow-up if Step 1 is insufficient:
  - Add DCP-aware staging/pack.
  - Prefill gathers selected DCP-rank tokens into contiguous staging memory.
  - Mooncake transfers one contiguous staging region.
  - Decode scatters staging data into local DCP KV slots.
  - Expected effect: reduce block count from O(tokens * layers * KV) to O(chunks), but it needs protocol and decode scatter changes.

## Optimization Patch
- Implemented Step 1 in `send_kvcache_dcp()`:
  - Build all layer K/V transfer address lists once.
  - Submit them in grouped Mooncake batches.
  - Temporarily added post-fix instrumentation fields:
    - `batch_calls`
    - `batch_block_limit`
    - `batch_build_ms`
    - `batch_transfer_ms`
- Local checks:
  - `python3 -m py_compile python/sglang/srt/disaggregation/mooncake/conn.py`
  - `python3 -m compileall -q python/sglang/srt/disaggregation/mooncake/conn.py python/sglang/srt/disaggregation/decode.py`

## Post-Fix Test
- First attempt: grouped batch submitted serially.
  - Same 10.8K prompt benchmark:
    - `concurrency=1`: e2e about `2.38s`
    - `concurrency=32`: e2e max about `37.26s`
  - Result: rejected. It reduced Mooncake call count but removed per-layer transfer parallelism.
- Second attempt: grouped batch submitted through the existing executor.
  - Same 10.8K prompt benchmark:
    - `concurrency=1`: e2e about `1.98s`
    - `concurrency=4`: e2e p50 about `2.17s`, max about `3.14s`
    - `concurrency=8`: e2e p50 about `3.48s`, max about `5.74s`
    - `concurrency=16`: e2e p50 about `6.14s`, max about `11.01s`
    - `concurrency=32`: e2e p50 about `11.48s`, max about `21.81s`
  - Runtime events for the 10.8K prompt:
    - `batch_calls`: `3`, `8`, or `11` depending on selected-token count.
    - Large-event `total_ms` p50 improved from about `551.7ms` in serial grouped batch to about `347.0ms`.
- Comparable 8.2K prompt benchmark with parallel grouped batch:
  - Prompt tokens: `8225`
  - `concurrency=1`: e2e about `0.78s`
  - `concurrency=4`: e2e p50 about `1.72s`, max about `2.52s`
  - `concurrency=8`: e2e p50 about `2.84s`, max about `4.67s`
  - `concurrency=16`: e2e p50 about `5.00s`, max about `8.98s`
  - `concurrency=32`: e2e p50 about `9.33s`, max about `17.70s`
  - New DCP events: `976`
  - `batch_calls`: always `8`
  - `selected_tokens`: `4112` or `4113`
  - DCP `total_ms` p50: about `271.43ms`, max about `360.30ms`
  - `batch_transfer_ms` p50: about `250.87ms`, max about `331.26ms`
  - `batch_build_ms` p50: about `18.56ms`

## Current Conclusion
- Low-risk grouped parallel batching is safe and gives a modest improvement for request-level latency at comparable 8K prompt length.
- It does not remove the core O(tokens * layers * KV) RDMA block count, so it should not be expected to close a `54s -> 1.6s` gap by itself.
- The proper high-impact optimization remains DCP-aware staging/pack:
  - Pack selected DCP-rank tokens on prefill into contiguous staging memory.
  - Transfer contiguous staging region via Mooncake.
  - Scatter into decode local DCP KV slots.
  - This is expected to reduce transfer block count by orders of magnitude but needs protocol/scatter changes and separate correctness validation.

## DCP-Aware Staging Prototype
- Implemented DCP staging/packing:
  - `gather_dcp_tokens_to_staging()` packs DCP-selected source token rows into contiguous staging memory.
  - `scatter_dcp_staging_to_kv()` scatters packed rows into decode local DCP KV slots.
  - Decode staging scatter now filters logical token indices by `logical % dcp_size == dcp_rank` and writes `logical // dcp_size`.
  - DCP transfer dispatch now prefers staging when `SGLANG_DISAGG_STAGING_BUFFER=1` and falls back to `send_kvcache_dcp()` if staging is unavailable.
  - Staging prefetch now treats DCP metadata as logical-token metadata and converts it back to page-count chunks.
- Local checks:
  - `python3 -m compileall -q python/sglang/srt/disaggregation/mooncake/conn.py python/sglang/srt/disaggregation/common/staging_buffer.py python/sglang/srt/disaggregation/common/staging_handler.py python/sglang/srt/disaggregation/common/conn.py python/sglang/srt/disaggregation/decode.py`
  - VS Code diagnostics clean for edited files.

## DCP-Aware Staging Test
- Runtime configuration:
  - Prefill `TP8 + EP8`
  - Decode `TP16 + DCP2 + EP16`
  - `page_size=64`
  - `SGLANG_DISAGG_STAGING_BUFFER=1`
  - `SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB=1024`
  - `SGLANG_DISAGG_STAGING_POOL_SIZE_MB=8192`
- Correctness:
  - Chinese prompt returned coherent Chinese Forbidden City content.
  - No previous Go/JSON/code-style KV corruption was observed.
- 8.2K prompt benchmark:
  - Prompt tokens: `8225`
  - `concurrency=1`: e2e about `0.96s`
  - `concurrency=4`: e2e p50 about `0.45s`, max about `0.45s`
  - `concurrency=8`: e2e p50 about `0.38s`, max about `0.40s`
  - `concurrency=16`: e2e p50 about `0.54s`, max about `0.58s`
  - `concurrency=32`: e2e p50 about `0.69s`, max about `0.75s`
- Transfer evidence:
  - New `dcp_staging_transfer_done` events: `992`
  - Fallback `dcp_transfer_done` events: `0`
  - `blocks`: always `1`
  - `selected_tokens`: `4096`, `4112`, or `4113` for full chunks; `16/17` for short chunks.
  - DCP staging `total_ms`: p50 about `5.23ms`, max about `16.32ms`
  - DCP staging `gather_ms`: p50 about `2.77ms`, max about `12.48ms`
  - DCP staging `transfer_ms`: p50 about `1.84ms`, max about `4.81ms`
  - Status: all `0`
- Log scan:
  - No runtime traceback, transfer fallback, allocation failure, or staging buffer too-small warning was found.
  - Only pre-existing CUTE DSL package-walk warnings appeared during startup.

## DCP-Aware Staging Conclusion
- The high-impact optimization works as intended: DCP KV transfer block count is reduced from about `~508K` blocks per large chunk to `1` contiguous block per chunk/writer.
- Compared with grouped parallel DCP transfer at the same 8.2K prompt length:
  - `concurrency=32` e2e max improved from about `17.70s` to about `0.75s`.
  - DCP transfer p50 improved from about `271ms` to about `5ms`.
- This is the recommended performance optimization path. The remaining work is cleanup/refinement:
  - Temporary transfer instrumentation has been removed from source before final merge.
  - Consider replacing the torch DCP gather/scatter prototype with Triton kernels if further CPU/kernel-launch overhead matters.
  - Use realistic user benchmark settings to confirm whether the original `54132ms` TTFT is eliminated end-to-end.
