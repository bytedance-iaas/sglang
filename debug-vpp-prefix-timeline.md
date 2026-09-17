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

## Barrier-removal experiment

User authorized implementation and remote A/B measurement. Candidate is opt-in with
`SGLANG_VPP_OVERLAP_CHUNKS=1`; global `chunked_req` ownership is unchanged.

- Advance allocation mapping to the planned frontier before preparing continuation;
  old first-pass commits cannot regress it.
- Order each RID's stage launches by its immutable batch prefix length on the
  existing forward stream; materialized/committed frontiers retain their GPU events.
- Send middle-chunk KV only when processing its completed result.
- Wait for pending admission ACK during a first-pass burst instead of immediately
  falling back to S2.
- Accept overlapping completed replica updates without regressing their frontier.
- Profile labels use batch prefix lengths instead of mutable `Req.extend_range`.

Validation used 14,336 input tokens, 8 greedy output tokens, concurrency 8,
32 requests per round, three rounds, cache flush between rounds. Throughput runs
do not enable the profiler; output tokens/logprobs are retained for comparison.
Scripts: `.dbg/vpp-ab-launch.sh`, `.dbg/vpp-ab-bench.py`.
Remote artifacts: `/tmp/vpp-ab-20260917/` in the prefill container.

Existing focused tests: 48 passed. Two new mapping tests initially failed because
their fixture omitted `mbs`/`last_mbs`; after fixing the fixture, all 37 scheduler
tests passed. Skill report flush was attempted but blocked writing its own global
`.skill_update_*` file by the sandbox; this did not affect test execution.

Baseline (unmodified remote `4a3af89e7b`, burst 6) completed:

| Round | Wall time (s) | Input token/s |
|---|---:|---:|
| 0 | 64.167701 | 7149.266616 |
| 1 | 61.557656 | 7452.395505 |
| 2 | 61.308717 | 7482.655365 |

Median: 7452.395505 token/s; pooled: 7358.316976 token/s. All 96 responses
completed with zero cached tokens. First token and its logprob match exactly
for each input across all three rounds. Later decode tokens already differ
across baseline runs (only 1/32 full outputs match between rounds 0 and 1).
Use prefill first-token/logprob equality as the direct scheduling comparison;
full PD decode correctness has this pre-existing limitation.

Independent baseline profile completed 7 single-request 14k prompts (one warmup
plus six measured), then stopped with no outstanding benchmark requests.
All eight traces exported successfully; unlike the prior run, no native crash
occurred. Export still pauses PP1 until PP0 finishes and must be excluded.
Raw traces stay under `/tmp/vpp-ab-20260917/trace-baseline` (about 640 MiB).
PP0/TP0 trace and analysis are also local under `.dbg/vpp-ab-20260917/`.
Full-rank analysis completed for both baseline and candidate. Test services
were stopped after collection.

Candidate at `mem_fraction_static=0.85` stalled during its first 14k warmup,
before any timed round. PP0 allocator reported a 939,524,096-byte allocation
failure with only 215–752 MB free. Subsequent native py-spy samples on PP0/TP0,
PP0/TP1 and PP1/TP0 all stopped in `cuKernelSetAttribute -> cublasGemmEx ->
einsum -> DSV4.1 indexer scores`. All eight GPUs remained at 100% utilization.
This rules out a pure CPU admission dependency wait; memory pressure and
CUDA/cuBLAS initialization interacting with pending communication remain
candidate causes. Preserve this failed run; next isolate additional activation
headroom at `mem_fraction_static=0.80`, with all other settings unchanged.
If this succeeds, rerun the baseline at 0.80 for a matched A/B comparison.
Final focused test run: 50 passed.

At 0.80 the long warmup completed, then concurrency-8 hit an ADMIT manifest
mismatch: PP0 chose `[12288,14080)`, PP1 `[12288,14336)` for the same RID.
Difference = one 256-token page. `PrefillAdder.add_chunked_req` reserves a page
from SWA availability, whereas the rank-local resource gate only checked the
chunk size. Candidate v2 adds one page to the KV admission requirement when
overlap is enabled. This is a live runtime failure, not a throughput result;
v2 must complete the entire A/B workload before any gain is claimed.

Candidate v2 at 0.80 completed all three throughput rounds: 7996.057442,
8371.888031, and 8423.448779 input token/s (median 8371.888031). This is
12.34% above the old 0.85 baseline median, but is not yet a valid gain because
the memory setting differs. More importantly, candidate first-token output is
not stable across its own three runs: only 27/32 inputs retain the same greedy
first token and 12/32 retain the same first-token logprob. The old barrier-on
baseline was stable for all 32 inputs on both measures. Candidate output is
therefore not correctness-acceptable.

Candidate profile workload completed normally and all eight traces exported.
The profile must still be analyzed for actual same-RID six-S0 runs and GPU idle
time; trace export time is excluded from throughput.

Matched barrier-on 0.80 baseline completed at 7155.509511, 7397.638345, and
7414.463572 token/s (median 7397.638345, pooled 7320.603985). Its first token
and first-token logprob are exactly equal to the old 0.85 baseline for all 96
responses. Candidate's apparent matched gain is +13.17% median / +12.82%
pooled, but remains invalid while candidate correctness differs.

The VPP proxy already snapshots source pages, but exported indexer metadata and
candidate masks were aliases of reusable forward-metadata buffers. Candidate
v3 clones those tensors at the stage boundary to make them batch/content
scoped before deeper same-RID pipelining. A mutation-after-export unit test
passes (5 tests in the replay module). Runtime correctness must confirm or
reject this aliasing hypothesis.

Candidate v3 rejected that hypothesis: its first measured round still differed
from the matched baseline for 5/32 greedy first tokens and 15/32 first-token
logprobs (max absolute logprob difference 0.512). The clone patch and its test
were removed. The next isolation run keeps barrier removal but sets burst size
to 1. This distinguishes a generic overlap bug from corruption that appears
only when one physical rank gets several stages ahead for the same RID.

Candidate v2 profile proves the requested schedule occurred. On PP0/TP0, one
RID ran six consecutive S0 GPU spans for `[0,2048)`, `[2048,4096)`,
`[4096,6144)`, `[6144,8192)`, `[8192,10240)`, and `[10240,12288)`. Baseline
maximum was two consecutive same-RID S0 stages. Across the profile windows,
all-lane idle fell from 45.92% to 39.90% on PP0 and from 62.81% to 38.76% on
PP1. These utilization gains describe the incorrect v2 and are diagnostic,
not shippable performance.

Barrier removal with burst size 1 completed three correct rounds at
8098.063286, 8266.186635, and 8250.866233 token/s (median 8250.866233,
pooled 8204.332107). All 96 first tokens and first-token logprobs exactly match
the 0.80 baseline. Relative to that matched baseline, median throughput improves
11.53%, pooled throughput improves 12.07%, and median wall time falls from
62.013305 s to 55.600465 s (10.34%).

Conclusion: removing the global first-pass commit round trip is valid at a
one-first-pass lead. Allowing six consecutive same-RID S0 stages is not valid
with the current request-scoped runtime state. It needs versioned,
batch/content-scoped state banks for every mutable DSV4 intermediate consumed
by later virtual stages, followed by commit of the accepted/materialized
frontier. The experimental mode now rejects burst sizes other than 1 rather
than exposing the incorrect six-S0 path.
