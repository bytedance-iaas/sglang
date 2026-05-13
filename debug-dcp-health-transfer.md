# Debug: DCP Health Transfer

Status: [OPEN]

## Symptom

- In DCP decode mode, health warning appears during long random benchmark:
  - `DCP health check pending queues exceeded fast-return timeout`
  - `elapsed=180.0s timeout=60s counts={'decode_transfer': 1}`
- `/health` can still return `200 OK` after the warning.

## User Reproduction

- Prefill: MiniMax-M2.7 TP8 EP8, staging buffer enabled, disaggregation prefill.
- Decode: MiniMax-M2.7 TP16 DCP2 EP16, staging buffer enabled, disaggregation decode, max running requests 64.
- Router: PD disaggregation, `cache_aware` prefill policy, `round_robin` decode policy.
- Benchmark:
  - `dataset-name random`
  - `random-input-len 128000`
  - `random-output-len 1500`
  - `random-range-ratio 0.1`
  - `num-prompts 50000`
  - `max-concurrency 12`
  - `request-rate 2`

## Hypotheses

1. Real transfer long-tail: one decode transfer request remains legitimately in `disagg_decode_transfer_queue` for more than 60s because the 128k random prompt transfer is still in progress.
2. Queue accounting leak: the request has already reached `KVPoll.Success` or failed, but the queue entry is not removed from `disagg_decode_transfer_queue`.
3. Per-rank mismatch: only some TP ranks have the queue entry, but the health warning is emitted on every rank due to local stale state or DCP collective synchronization effects.
4. Health timer attribution bug: `_disagg_health_pending_since` is tracking aggregate DCP pending state rather than the specific queue entry, so elapsed time can overstate a newly arrived transfer item.
5. Router/external health behavior is benign: the warning is internal scheduler telemetry and does not correlate with HTTP health failures or request errors.

## Instrumentation Plan

- Add debug-server reporting only; do not change business logic.
- Capture DCP health pending snapshot:
  - rank identity
  - pending queue counts
  - pending elapsed
  - decode transfer queue length and representative request identifiers
  - receiver poll status and staging status when available
- Reproduce with the user random workload or a scaled-down equivalent if needed.

## Evidence

- Pre-fix reproduction:
  - Run directory: `/data01/code/dcp_health_transfer_debug_20260513/`
  - Workload: `random-input-len=128000`, `random-output-len=1500`, `max-concurrency=12`, `request-rate=2`, `num-prompts=300`.
  - Direct decode `/health` probe returned 200 during the observed window.
- Debug collector evidence:
  - At elapsed `60s`, transfer queue front request:
    - `rid=d4f35e651e7a4c609e1827ea0b090b41`
    - `bootstrap_room=4571132161003375190`
    - `origin_input_len=115868`
    - `kv_status=2` (`KVPoll.WaitingForInput`)
    - `live_staging_allocs=3`
    - `staging_done=false`
  - At elapsed `92s`, transfer queue front request had changed:
    - `rid=9534a246530c48d99cd83037031fc992`
    - `bootstrap_room=1202151485843355322`
    - `origin_input_len=105540`
    - `kv_status=2` (`KVPoll.WaitingForInput`)
    - `live_staging_allocs=13`
    - `staging_done=false`
- Interpretation:
  - The warning was not necessarily reporting the age of the same transfer request.
  - `_disagg_health_pending_since` measured aggregate "DCP queues have been non-empty" time.
  - Long 128k prompts can keep `decode_transfer` continuously non-empty while individual front requests rotate.
  - This can overstate per-request pending age and produce noisy warnings such as `elapsed=180s`.

## Conclusion

- Confirmed issue:
  - Health pending elapsed attribution was too coarse.
  - It should track the identity of the pending DCP work, not only whether any DCP queue is non-empty.
- Minimal fix:
  - Add `_disagg_health_pending_signature`.
  - Reset `_disagg_health_pending_since` when the active DCP pending signature changes.
  - Decode signature includes active counts and representative prealloc/transfer request IDs, rooms, sessions, and metadata buffer indices.
- Expected behavior after fix:
  - Warnings only fire when the same pending DCP work stays pending longer than `SGLANG_DISAGG_HEALTH_PENDING_TIMEOUT`.
  - Continuous but progressing long-input traffic should not accumulate elapsed across different requests.

## Post-Fix Validation

- Run directory: `/data01/code/dcp_health_transfer_fix_20260513/`
- Fix deployed:
  - Synced `scheduler.py` to prefill, decode0, and decode1.
  - Container-side `py_compile` passed.
  - Relaunched user topology with `TRAE_DEBUG_DCP_HEALTH_TRANSFER_URL=http://127.0.0.1:18081/` and `SGLANG_DISAGG_HEALTH_PENDING_TIMEOUT=60`.
- Workload:
  - `dataset-name=random`
  - `random-input-len=128000`
  - `random-output-len=1500`
  - `random-range-ratio=0.1`
  - `num-prompts=100`
  - `max-concurrency=12`
  - `request-rate=2`
- Result:
  - `Successful requests: 100/100`
  - `Benchmark duration: 1281.07s`
  - `Request throughput: 0.08 req/s`
  - `Output token throughput: 65.11 tok/s`
  - `Peak output token throughput: 421.00 tok/s`
  - `Peak concurrent requests: 14`
- Health / warning comparison:
  - Pre-fix: warning reproduced by about 26 completed requests; decode0 and decode1 each logged `DCP health check pending queues exceeded`.
  - Post-fix: no `DCP health check pending queues exceeded` warning in prefill, decode0, decode1, or router.
  - Post-fix debug collector lines: `0`, meaning no pending item exceeded the per-signature timeout.
  - Router `/health`: `1347/1347` returned `200`.
  - Direct decode `/health`: after decode became reachable, `652` returned `200`; startup-only probe errors were `44` connection failures and `2` `503`.
- Error counters after traffic:
  - prefill/decode0/decode1:
    - `timed out after=0`
    - `Decode transfer failed=0`
    - `Traceback=0`
    - `ERROR=0`
    - `Exception=0`
    - `CHUNK_READY received for unregistered=0`
    - `STAGING_REQ received for unregistered=0`
    - `transfer_queue_len_mismatch=0`
- Current status:
  - Root cause confirmed and minimal fix validated.
  - Extended post-fix soak with 300 random 128k/1500 requests completed successfully:
    - `Successful requests: 300/300`
    - `Benchmark duration: 3816.42s`
    - `Request throughput: 0.08 req/s`
    - `Input token throughput: 5743.90 tok/s`
    - `Output token throughput: 65.74 tok/s`
    - `Peak output token throughput: 435.00 tok/s`
    - `Peak concurrent requests: 15`
  - Post-soak router `/health`: `3840/3840` returned `200`.
  - Post-soak debug collector lines: `0`.
  - Post-soak prefill/decode0/decode1 counters were all `0` for:
    - `DCP health check pending queues exceeded`
    - `timed out after`
    - `Decode transfer failed`
    - `Traceback`
    - `ERROR`
    - `Exception`
    - `CHUNK_READY received for unregistered`
    - `STAGING_REQ received for unregistered`
    - `transfer_queue_len_mismatch`
  - Router log still contains startup-only readiness noise (`ERROR=30`, `Health check failed=25`), but the independent traffic-time router health probe had no failures.
  - Temporary TRAE debug reporting was removed from source after the fix was validated.
- Cleanup validation:
  - Run directory: `/data01/code/dcp_commit_validation_20260513/`.
  - Synced cleaned source without `TRAE_DEBUG`, `DCP_TRANSFER_PERF`, `debug-point`, or `DCP_STAGING_DBG` instrumentation.
  - Container-side `py_compile` passed on prefill, decode0, and decode1.
  - New-router validation with random 128k/1500 `num-prompts=10` completed successfully:
    - `Successful requests: 10/10`
    - `Benchmark duration: 42.66s`
    - `Request throughput: 0.23 req/s`
    - `Output token throughput: 245.84 tok/s`
    - `Peak output token throughput: 350.00 tok/s`
    - `Peak concurrent requests: 10`
  - Router `/health`: `66/66` returned `200`.
  - Prefill/router/decode0/decode1 counters remained `0` for DCP health pending warnings, transfer queue mismatch, transfer timeouts/failures, traceback/error/exception, unregistered staging messages, and temporary debug markers.
