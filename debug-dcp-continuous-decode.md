# DCP Continuous Decode Debug Summary

Status: [CLOSED]

## Final State

- The temporary continuous-decode instrumentation was removed before commit.
- The remaining `decode.py` change is a safety/performance fix, not debug code.
- Continuous decode now keeps cross-rank queue consistency checks while allowing overlap when the post-transfer queue is non-empty but consistent across attention TP ranks.

## Root Cause

- The previous post-transfer gate required the DCP decode transfer queue to drain to zero before scheduling more decode work.
- Under long-context PD+DCP load, one pending transfer item could block an already-running decode batch.
- This caused wave-style completion and degraded TPOT, even when queue lengths were consistent and collective-order safety could be preserved.

## Fix Kept

- `python/sglang/srt/disaggregation/decode.py`
  - Uses MIN/MAX all-reduce to verify post-transfer queue length consistency across attention TP ranks.
  - Returns false only if ranks disagree on queue length.
  - Allows continuous decode to proceed when ranks agree, even if the transfer queue is still non-empty.

## Validation Summary

- Random 128K/1500 quick run after removing instrumentation restored TPOT to about 22.46ms.
- Low-load control showed intrinsic single-request TTFT around 501.65ms.
- High-load TTFT remains queue/admission dominated and should not be interpreted as single-request transfer latency.
- No DCP health warnings, queue mismatches, timeouts, transfer failures, tracebacks, or unregistered staging messages were observed in the validated run.

## Relation To 190K Staging Fix

- The continuous decode change improves overlap and throughput smoothness.
- The 190K staging fix resolves a different issue: staged KV chunks could remain allocated forever after a ready/allocation race.
- Both fixes are needed for stable high-pressure PD+DCP long-context service.
