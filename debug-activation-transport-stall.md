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

Multi-node reproduction at 2026-09-16 14:47:53:
- All eight PP/TP lanes completed metadata receive and remained in `recv.state=payload` with `payload_works=(False,)`.
- PP0 receives batch 7 stage 1->2 while its pending send is batch 6 stage 2->3.
- PP1 receives batch 6 stage 2->3 while its pending send is batch 7 stage 1->2.
- Send and receive identities match across peers, but both GPU send works and both matching GPU receive works remain incomplete for at least 20 seconds.
- No traceback, NCCL timeout, control backlog, bootstrap round, or TP-lane divergence is present.

## Verification Conclusion
H1 rejected: metadata size and body are complete.
H2 confirmed: all lanes are blocked in GPU payload P2P.
H3 rejected: no lane reaches TP all-gather.
H4 refined: message identities match, but opposite-direction sends are launched before
their matching receives on the shared PP NCCL communicator, creating a symmetric
cross-send ordering deadlock.

Minimal fix:
- Create a second PP transport group only for PP2 + VPP.
- Route PP0->PP1 activation traffic through the base PP group.
- Route PP1->PP0 activation traffic through the duplicate PP group.
- Keep the control ring and PP4 activation traffic unchanged.

Post-fix verification:
- Local focused tests pass, including PP2 directional routing and duplicate group
  rank construction.
- Multi-node evidence confirms the activation deadlock is gone: no activation send
  or payload work remains pending.

Second multi-node stall at 2026-09-17 03:05:04:
- All slots are empty and activation transport is idle.
- PP1 reports 3584 physical free KV tokens, below the 4096 token gate.
- The outer resource gate counts only allocator free space, while `PrefillAdder`
  admits against allocator free plus radix-evictable KV.
- Completed batches can therefore leave enough evictable KV to run the next batch
  while the outer gate permanently rejects it; metadata buffers then remain held.

Second minimal fix:
- Report schedulable KV (`free + evictable`) in the VPP resource snapshot using
  the same full/SWA pool accounting as `PrefillAdder`.
- Retain the inner `PrefillAdder` as the authoritative allocation check.
