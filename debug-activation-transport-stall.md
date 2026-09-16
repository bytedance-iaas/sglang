# Debug Session: activation-transport-stall
- **Status**: [OPEN]
- **Issue**: PP2 x TP4 x VPP2 prefill stalls with one pending activation send on each PP rank and no ready activation.
- **Collection**: Native low-frequency `[VPP-STALL]` logger, as required for the remote multi-node environment.

## Reproduction Steps
1. Launch DeepSeek-V4.1-Flash with PP2 x TP4 x VPP2 disaggregated prefill.
2. Submit enough requests to fill metadata slots.
3. Wait for `[VPP-STALL]` after scheduler progress stops.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| H1 | Activation receive is waiting for metadata. | Medium | Low | Pending recv remains in metadata state. |
| H2 | Metadata completed but GPU payload P2P work does not complete. | High | Low | Pending recv remains in payload state with incomplete payload works. |
| H3 | Payload completed but TP all-gather does not complete. | Medium | Low | Pending recv remains in all-gather state with incomplete collective work. |
| H4 | Send and receive FIFO/tag ordering diverged. | High | Medium | Both ranks retain sends with different batch/stage identities from pending receives. |

## Log Evidence
Pre-instrumentation: both PP ranks report `activation_sends=1`, `ready_proxies=0`, `arrivals=0`, and no control/bootstrap backlog.

Instrumentation added:
- Per-TP-lane receive state, waiter/work completion, buffered bytes, and activation identity.
- Per-TP-lane send work completion, payload bytes, and `(batch, src_stage, dst_stage)` identity.
- Focused CPU tests verify snapshots do not call `wait()` or advance transport state.

## Verification Conclusion
Instrumentation verification: 38 focused tests passed. Root cause remains pending a multi-node reproduction.
