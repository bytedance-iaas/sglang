# Debug Session: vpp-admission-stall
- **Status**: [OPEN]
- **Issue**: VPP prefill runs for a while, then stops admitting and processing new requests without an exception.
- **Debug Server**: http://10.254.210.200:7777/event
- **Log File**: .dbg/trae-debug-log-vpp-admission-stall.ndjson

## Reproduction Steps
1. Start DeepSeek-V4.1 prefill with PP4 x TP2 x VPP2.
2. Send requests continuously until request processing stops.
3. Keep the server running long enough to capture the stalled scheduler state.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | bootstrap_round_active never receives its apply return | High | Low | Pending |
| B | pending_admit never receives its ADMIT return | High | Low | Pending |
| C | pending_chunk_batches never receives PREFIX_COMMIT return | High | Low | Pending |
| D | resource gate remains blocked or lacks a rank snapshot | Medium | Low | Pending |
| E | control outbox or pending send work stops progressing | Medium | Low | Pending |

## Log Evidence
Instrumentation added:
- A/B/C: PP0 admission blocker transitions.
- D: per-PP resource snapshot transitions.
- E: periodic control/activation backlog samples.

Pre-fix run ID: `pre-fix`.

## Verification Conclusion
Pending.
