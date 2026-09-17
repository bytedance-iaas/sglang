# Debug Session: vpp-prefix-timeline

- Status: [OPEN]
- Issue: Measure whether the VPP first-pass prefix commit barrier causes GPU idle time during chunked Prefill.
- Collection: Native logger and existing PyTorch profiler, per user preference; no HTTP debug server.
- Runtime: PP2 x TP4 x VPP2 on one 8-H20 prefill host, chunk size 2048, burst size 6. Hardware trace collected on 2026-09-17.
- Scope: Observability only; preserve scheduling, communication order, and request lifecycle.

## Hypotheses

| ID | Hypothesis | Expected evidence | Status |
|----|------------|-------------------|--------|
| H1 | Prefix barrier prevents useful overlap | Prefix wait overlaps GPU compute gaps with outstanding input | Supported: PP0 all-lane idle overlaps prefix-only gate |
| H2 | Admission ACK or activation transport dominates | ACK/receive delays explain gaps after prefix wait ends | ACK contribution observed; dominance not established |
| H3 | Compute saturation or rank imbalance limits throughput | Little GPU idle or substantially unequal rank compute loads | Continuous saturation contradicted in profiled window; stage imbalance observed |

## Collection requirements

- Correlate PP rank, TP lane, batch sequence, logical stage, and chunk ranges.
- Record actual GPU stage intervals separately from CPU launch intervals.
- Record admission blockers and prefix commit transitions without adding collectives or synchronous CUDA waits.
- Use a bounded capture and report clock alignment limits for cross-rank analysis.
- Do not include prompts or model output in diagnostic records.

## Evidence

Eight CPU/CUDA rank traces have been collected and downloaded to `.dbg/vpp-prefix-20260917/trace-8k/`. The analysis uses a common interior window before profiler stop, excluding the start/stop/export stalls.

## Instrumentation

- Enable `SGLANG_VPP_TIMELINE=1` on every prefill worker before server startup.
- Use the existing profiler with `SGLANG_PROFILE_V2=0`. Markers are emitted only while its PyTorch profiler is active.
- `admission_gate` records all gate predicates on state changes, including queued work and continuation availability. A false resource/slot predicate is an independent blocker.
- `admit_queued`, control arrivals/returns, `prefix_commit_queued`, and `activation_ready` correlate by batch sequence.
- `stage_launch` records hashed request IDs and chunk ranges. CPU stage ranges include batch and slot IDs; correlate their CUDA launches with actual device kernels in the same rank trace.
- Each rank emits at most 10,000 diagnostic markers per unique profile ID; TP0 also emits native `[VPP_TIMELINE]` logs. A limit warning means the remaining timeline is incomplete.
- Trace marker names use `vpp_timeline/<event>` followed by URL-encoded JSON metadata (decode with `urllib.parse.unquote`). This avoids invalid JSON exports on PyTorch versions that do not escape quotes in user annotation names. Native log metadata is plain JSON.
- For this capture, use S0's chunk bounds as authoritative and correlate later stages by batch sequence. Later-stage markers read mutable `Req.extend_range.start`; continuation can advance it before S2 runs (batch 30 demonstrated this). Stage/kernel timing and batch identity remain valid.
- No scheduling policy, communication ordering, CUDA synchronization, or collective is added. Diagnostic resource checks are read-only.

## Capture on the GPU deployment

Keep the existing PP2 × TP4 × VPP2, chunk 2048, burst 6 launch arguments and workload unchanged. Add these environment variables to the **prefill node** (all eight ranks run on one host), then restart using the usual deployment command:

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

Stop and export before stopping the workload. If interrupted after start, run `/stop_profile` manually. Avoid `num_steps`, `start_step`, and `profile_by_stage`: VPP advances the forward counter per local stage, so step-based capture can stop ranks at different scheduling points. Start/stop/export overhead is outside the useful interior window; use a few complete chunks well inside it. The stop HTTP response does not establish that all ranks have exported: wait for all eight per-rank `Profiling done` messages.

Collect:

- All `*.trace.json.gz` files under the printed `$VPP_PROFILE_DIR` on the prefill node (8 rank files for PP2 × TP4).
- Prefill and decode server logs covering the same interval, including `[VPP_TIMELINE]` and profiler start/stop messages.
- Exact launch/benchmark commands, input lengths, concurrency, and throughput.

Remote access was supplied through the user's “远端执行” wiki on 2026-09-17. SSH through the documented jump host succeeded for both machines. Each has 8 H20 GPUs; prefill and decode run in separate `deepseek-v41-pd-dspark` containers with code in `/root/sglang`. Both worktrees were clean and fast-forwarded to `4a3af89e7b`. GPU memory was unused before launching the services.

The documented prefill command omits the burst flag (default 1); this capture explicitly uses burst 6 to measure the previously discussed configuration. The documented benchmark uses 8000 input tokens (approximately four chunks), 256 prompts, output length 1, concurrency 64, request rate 4, seed 5. The first capture preserves that workload. Logs and traces are stored under `/tmp/vpp-prefix-20260917` in the prefill container.

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
- All eight CPU/CUDA traces exported on the deployment and were downloaded locally.

## Capture lifecycle and limitation

- Server time (UTC) 08:42:50: profile started during the warmed 8000-token workload.
- 08:43:06: PP0 stopped recording. PP0 exported until 08:43:41, then forwarded stop to PP1.
- PP1 exported until 08:46:36. Its traces include the PP0 export pause, so absolute capture lengths differ substantially.
- After export, native `Segmentation fault` / `malloc(): unaligned fastbin chunk detected` occurred. The Python stack identifies garbage collection at `SchedulerProfilerManager._stop_profile`, line 400; the underlying native defect is not diagnosed. The subsequent Gloo peer-closed error caused server exit.
- The benchmark did not complete and supplies no valid steady-state throughput result. No throughput improvement estimate is inferred from this capture.
- Prefill GPU memory returned to idle after exit. The decode server started by this session was terminated with SIGTERM; all its GPUs returned to 0 MiB used. Containers and captured artifacts were retained.

## Measured findings

All eight traces parsed successfully with matching `baseTimeNanoseconds`. In the common 12.999568-second interior window:

- Prefix pending: 11.132117 seconds.
- Prefix is the sole recorded admission blocker, with pending input and available slots/resources: 8.306227 seconds.
- Within those prefix-only intervals, all four PP0 TP lanes simultaneously have no recorded CUDA kernel, memcpy, or memset: **3.361355 seconds (25.9% of the entire window)**.
- Per-lane CUDA activity coverage: PP0 50.7–53.9%; PP1 40.9–46.1%. These are time unions including NCCL, not SM utilization.

Example, batch 30, same RID continuation `[4096,6144)` to batch 31 `[6144,7971)`, relative to batch 30 S0's first GPU kernel:

| Event | Time (ms) |
|---|---:|
| S0 first-to-last non-NCCL kernel span | 0.00–114.34 |
| S1 first-to-last non-NCCL kernel span | 126.95–239.73 |
| PP0 receives FIRST_PASS_DONE | 214.36 |
| PP0 queues PREFIX_COMMIT | 216.47 |
| PREFIX_COMMIT returns to PP0 | 233.11 |
| Batch 31 ADMIT queued | 236.65 |
| Batch 30 S2 kernel span | 247.69–302.03 |
| Batch 31 ADMIT returns | 304.90 |
| Batch 31 S0 kernel span | 310.80–424.53 |
| Batch 30 S3 kernel span | 317.46–368.81 |

FIRST_PASS_DONE precedes the end of S1 GPU work, confirming that it must not be interpreted as device completion. The measured overlap supports investigating finer-grained prefix admission dependencies. It does not establish safe same-RID six-S0 execution or a recoverable 25.9% throughput gain; ACK, launch, communication, and profiler overhead remain contributors.

Local artifacts: `.dbg/vpp-prefix-20260917/report.md`, `timeline-batch30.svg`, `analysis-8k.json`, eight raw traces under `trace-8k/`, and server logs. Reproducible analysis scripts: `.dbg/analyze_vpp_timeline.py` and `.dbg/render_vpp_timeline.py`. Raw traces remain on the prefill container under `/tmp/vpp-prefix-20260917/trace-8k/`.
