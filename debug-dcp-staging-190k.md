# DCP Staging 190K Debug Summary

Status: [CLOSED]

## Final State

- 190K generated-shared-prefix acceptance passed with DCP staging enabled.
- Final validated topology: prefill TP8, decode TP16 + DCP2, MiniMax-M2.7.
- Final workload: 300 prompts, input target 190K, output 1500, concurrency 60, request-rate 30.
- Final validation result: 300/300 successful requests.
- The original permanent transfer stall is resolved: no stuck `decode_transfer:39`, no DCP health timeout, no prefill `KVPoll.Bootstrapping` timeout, and no decode traceback in the passing run.

## Root Cause

- DCP staging relies on chunk-level `CHUNK_READY` notifications plus decode-side staging allocations.
- At 190K pressure, notification/allocation ordering can race:
  - `CHUNK_READY` can arrive before the decode room or chunk allocation is visible.
  - Replaying only at registration/final last-chunk scatter is insufficient.
  - Final all-prefill `Success` can arrive while intermediate chunks still have valid staging allocations.
- If an intermediate allocated chunk has no replayed ready record and final handling only scatters the last chunk, `staging_handler.is_done()` remains false forever and decode requests stay in `decode_transfer`.

## Fixes Kept

- `python/sglang/srt/disaggregation/common/staging_handler.py`
  - Replays pending ready chunks from `advance_scatter()` so preserved `CHUNK_READY` state is self-healing.
  - Adds `submit_all_remaining_scatter_async(room)` as a final all-prefill-success fallback.
  - Marks staging complete only after all submitted scatter events are done and no valid chunk allocation remains.
- `python/sglang/srt/disaggregation/mooncake/conn.py`
  - Keeps `_chunk_writer_counts` until `submit_chunk_scatter()` succeeds.
  - Makes pending-ready cleanup idempotent with guarded `pop` to avoid decode-thread / scheduler replay races.
  - Replays ready chunks and submits all remaining allocated chunk scatters when all prefill ranks report success.
- `python/sglang/srt/managers/scheduler.py`
  - Adds progress-aware DCP health reset using active batch state and `forward_ct`.
  - Keeps real-stall detection when pending queues remain non-empty and forward progress stops.
- `python/sglang/srt/disaggregation/decode.py`
  - Keeps DCP post-transfer queue length consistency checks across attention TP ranks.
  - Allows continuous decode overlap when queue lengths are consistent, instead of requiring the transfer queue to be fully drained.
- `sgl-model-gateway/src/routers/http/pd_router.rs`
  - Separates prefill/decode circuit-breaker outcomes in HTTP PD dual dispatch.
  - Prevents decode-only send failures from marking the selected prefill worker as failed.

## Debug Code Cleanup

- Removed temporary `SGLANG_DCP_STAGING_190K_DEBUG` HTTP debug reporting helpers and all `_debug_report_dcp_staging_190k(...)` call sites.
- Removed local `.dbg/` debug-server environment artifacts.
- Kept normal warnings/errors and production-safe control-flow fixes.

## Key Validation Evidence

### post-fix-6

- Workload reached 300/300 benchmark progress, and the previous `decode_transfer:39` staging stall did not recur.
- Successful requests were only 51 because the router circuit breaker opened after decode-side send failures.
- Router then misreported `No available prefill workers` because the old PD path recorded a final 5xx response to both prefill and decode circuits.

### post-fix-7

- Router was restarted with `--disable-circuit-breaker` to isolate DCP staging correctness.
- Benchmark result:
  - Successful requests: 300.
  - Benchmark duration: 1039.89s.
  - Total input tokens: 58,337,977.
  - Total generated tokens: 450,000.
  - Input token throughput: 56,100.23 tok/s.
  - Output token throughput: 432.74 tok/s.
  - Mean TTFT: 73,672.83ms.
  - Mean TPOT: 81.89ms.
- Queue/health counts:
  - Router `No available`: 0.
  - Router `status_code=503`: 0.
  - Router circuit open: 0.
  - Prefill DCP health timeout: 0.
  - Prefill `KVPoll` timeout: 0.
  - Decode DCP health warnings: 0.
  - Decode tracebacks: 0.
- Decode queue behavior was normal for this capacity point: `#running-req` around 37-38, `#transfer-req` mostly 0 with only short 1-request transients, and final drain reached `#prealloc-req: 0`, `#transfer-req: 0`.

## Performance Interpretation

- DCP2 is now stable, but it is not expected to beat decode DP2 on lowest TPOT/TTFT when DP2 can already fit the workload.
- `decode tp16+dp2+ep16` has two independent full decode replicas, simpler KV layout, independent queues, and no DCP shard synchronization.
- `decode tp16+dcp2+ep16` saves decode KV capacity and supports longer/larger concurrent contexts, but pays extra cost from DCP KV layout, attention synchronization/LSE correction, staging gather/scatter, and collective-order safety gates.
- The observed TPOT gap of roughly 10+ ms and prefill gap of roughly 200ms are consistent with this architecture cost rather than a remaining correctness bug.

## Recommended Service/Test Mode

- For validating DCP staging only, keep router `--disable-circuit-breaker` to avoid router policy masking backend behavior.
- For validating the router attribution fix, run the same 190K workload with circuit breaker enabled and confirm no mass `No available prefill workers` / 503 after decode-only transient failures.
- Keep staging enabled for DCP long-context performance:
  - `SGLANG_DISAGG_STAGING_BUFFER=1`
  - `SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB=1024`
  - `SGLANG_DISAGG_STAGING_POOL_SIZE_MB=8192`

## Next Optimization Directions

- Short term:
  - Prefer `decode tp16+dp2+ep16` when the goal is lowest TPOT and the workload fits in each decode replica.
  - Prefer `decode tp16+dcp2+ep16` when the goal is larger context capacity or higher long-context concurrency within decode KV limits.
- Medium term:
  - Tune staging chunk size and watermark frequency to reduce control-plane overhead.
  - Optimize DCP staging gather/scatter kernels and reduce per-chunk CUDA event overhead.
  - Revalidate router circuit behavior with circuit breaker enabled after a gateway build.
- Long term:
  - Implement the physical DCP KV layout optimization so prefill writes KV closer to decode DCP physical layout.
  - This is the path most likely to reduce or remove staging gather/scatter cost, but it is a larger architectural change touching allocator, req-to-token mapping, transfer protocol, and attention assumptions.
