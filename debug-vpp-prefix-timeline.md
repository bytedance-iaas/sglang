# Debug Session: vpp-prefix-timeline

- Status: [OPEN]
- Issue: Measure whether the VPP first-pass prefix commit barrier causes GPU idle time during chunked Prefill.
- Collection: Native logger and existing PyTorch profiler, per user preference; no HTTP debug server.
- Runtime: PP2 x TP4 x VPP2, chunk size 2048, burst size 6. Hardware run not yet collected.
- Scope: Observability only; preserve scheduling, communication order, and request lifecycle.

## Hypotheses

| ID | Hypothesis | Expected evidence | Status |
|----|------------|-------------------|--------|
| H1 | Prefix barrier prevents useful overlap | Prefix wait overlaps GPU compute gaps with outstanding input | Pending |
| H2 | Admission ACK or activation transport dominates | ACK/receive delays explain gaps after prefix wait ends | Pending |
| H3 | Compute saturation or rank imbalance limits throughput | Little GPU idle or substantially unequal rank compute loads | Pending |

## Collection requirements

- Correlate PP rank, TP lane, batch sequence, logical stage, and chunk ranges.
- Record actual GPU stage intervals separately from CPU launch intervals.
- Record admission blockers and prefix commit transitions without adding collectives or synchronous CUDA waits.
- Use a bounded capture and report clock alignment limits for cross-rank analysis.
- Do not include prompts or model output in diagnostic records.

## Evidence

No runtime timeline collected yet. Repository already has `run_vpp_stage_<id>` profiler ranges.

## Instrumentation

- Enable `SGLANG_VPP_TIMELINE=1` on every prefill worker before server startup.
- Use the existing profiler with `SGLANG_PROFILE_V2=0`. Markers are emitted only while its PyTorch profiler is active.
- `admission_gate` records all gate predicates on state changes, including queued work and continuation availability. A false resource/slot predicate is an independent blocker.
- `admit_queued`, control arrivals/returns, `prefix_commit_queued`, and `activation_ready` correlate by batch sequence.
- `stage_launch` records hashed request IDs and chunk ranges. CPU stage ranges include batch and slot IDs; correlate their CUDA launches with actual device kernels in the same rank trace.
- Each rank emits at most 10,000 diagnostic markers per unique profile ID; TP0 also emits native `[VPP_TIMELINE]` logs. A limit warning means the remaining timeline is incomplete.
- Trace marker names use `vpp_timeline/<event>` followed by URL-encoded JSON metadata (decode with `urllib.parse.unquote`). This avoids invalid JSON exports on PyTorch versions that do not escape quotes in user annotation names. Native log metadata is plain JSON.
- No scheduling policy, communication ordering, CUDA synchronization, or collective is added. Diagnostic resource checks are read-only.

## Capture on the GPU deployment

Keep the existing PP2 × TP4 × VPP2, chunk 2048, burst 6 launch arguments and workload unchanged. Add these environment variables to **both prefill nodes**, then restart using the usual deployment command:

```bash
export SGLANG_VPP_TIMELINE=1
export SGLANG_PROFILE_V2=0
```

Warm up the service. Run the existing long-input benchmark continuously through the normal PD/router endpoint, with inputs long enough for at least six 2048-token chunks. While that workload is active, run the following on a machine that can reach the **prefill HTTP endpoint**. Replace the example URL with that endpoint; do not send the profile request to the decode worker.

```bash
VPP_PREFILL_URL=http://127.0.0.1:30000
VPP_PROFILE_ID=vpp-prefix-$(date +%Y%m%d-%H%M%S)
VPP_PROFILE_DIR=/tmp/$VPP_PROFILE_ID
curl --fail-with-body -X POST "$VPP_PREFILL_URL/start_profile" \
  -H 'Content-Type: application/json' \
  -d "{\"output_dir\":\"$VPP_PROFILE_DIR\",\"profile_id\":\"$VPP_PROFILE_ID\",\"activities\":[\"CPU\",\"GPU\"],\"with_stack\":false,\"record_shapes\":false,\"merge_profiles\":false}"
sleep 15
curl --fail-with-body -X POST "$VPP_PREFILL_URL/stop_profile"
printf 'Collect traces on every prefill node: %s\n' "$VPP_PROFILE_DIR"
```

Stop and export before stopping the workload. If interrupted after start, run `/stop_profile` manually. Avoid `num_steps`, `start_step`, and `profile_by_stage`: VPP advances the forward counter per local stage, so step-based capture can stop ranks at different scheduling points. Start/stop/export overhead is outside the useful interior window; use a few complete chunks well inside it.

Collect:

- All `*.trace.json.gz` files under the printed `$VPP_PROFILE_DIR` on **each prefill node** (normally 8 rank files for PP2 × TP4).
- Both nodes' native server logs covering the same interval, including `[VPP_TIMELINE]` and profiler start/stop messages.
- Exact launch/benchmark commands, input lengths, concurrency, and throughput.

The local workspace is macOS and has no GPU deployment endpoint supplied; hardware capture has not been executed here.

## Analysis criteria

1. On PP0, bound prefix blocking using `admit_queued`, `admission_gate.prefix_batches`, and the returned `PREFIX_COMMIT`/`COMPLETION`. Split out overlapping admission ACK, bootstrap, slot, and resource blockers.
2. Compare those intervals with actual CUDA compute kernels on **all TP lanes**. Measure union of busy intervals, not the sum of overlapping kernels or CPU range duration. Report compute gaps and communication activity separately; active NCCL kernels do not establish useful compute saturation.
3. Correlate chunk identity and batch across S0 → S1 → S2 → S3. Check activation readiness and the delay from commit return to the next S0.
4. Use each rank's own clock for overlap/idle measurements. Cross-node absolute offsets are not calibrated by merging traces or subtracting the first event; control-message causality only constrains alignment.
5. A prefix-only gate with pending input overlapping a compute gap supports H1. It does not by itself prove that removing the barrier is safe or quantify the attainable throughput gain.

## Validation

- 58 focused scheduler/pipeline/chunk gate tests passed.
- Ruff and `git diff --check` passed.
- Real CPU profiler export smoke passed: valid Chrome JSON, decoded metadata, timestamps, disabled path, and event cap. The first smoke exposed quote escaping in annotation export; URL-encoded metadata fixed it.
- CUDA profiling remains to be validated on the deployment.
