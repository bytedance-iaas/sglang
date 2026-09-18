# Opt-in SM90 FP4 grouped indexer

Enable before server startup / CUDA Graph capture:

```bash
export SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER=1
```

The default is **off**. This adds a request-grouped logits path to DeepSeek-V4.1
static target verification on Hopper. It does not change ordinary decode,
prefill, SM100+, DSpark draft execution, ragged verification, or late-layer-tail
execution. The backend also retains its existing path for unsupported layouts,
including 32-head indexers. Enabling the flag is not a promise of a speedup on
every Hopper configuration.

## Supported contract

- Contiguous BF16, already fake-FP4 query `[rows, 64, 128]` and BF16 head weights.
- Request-major static verification groups; ratio 1 or 2 and the existing
  packed E2M1 / UE8M0 index-K page layout. The direct operator supports a partial
  final group; the backend requires metadata proving complete static groups.
- Device-side visible lengths and request-to-token mappings can change between
  CUDA Graph replays. No length `.item()` or capture-time host synchronization.
- FP32 logits, `-inf` outside the visible prefix, and unchanged candidate-mask,
  TopK, sorted logical-index and physical-slot contracts.

The CUDA implementation builds on the grouped-indexer approach in
[sgl-project/sglang#40062](https://github.com/sgl-project/sglang/pull/40062), tuned
against its fixed `56ad1a2` snapshot and then integrated into `dsv4.1`.

## Selected implementation

1. Use mapped Triton for short shapes; it avoids Q packing and dense slots.
2. Use an 8-head/CTA Q pack for the CUDA branch, matching fake quant's scale floor.
3. Reuse K across grouped queries, vectorize packed-FP4 conversion, specialize
   aligned page64/ratio2 addressing, and retain a generic unaligned path.
4. Choose 1/2/4/8 tiles per CTA on the host. Within the fixed graph grid,
   redistribute shorter visible prefixes across more CTAs. Store the tile
   schedule in shared memory instead of keeping dynamic shifts live through MMA.
5. Use three warpgroups (384 threads): six-query groups use two fully populated
   rounds. All compiled native variants were spill-free with this layout.
6. Keep the conservative E5M2 safe domain, BF16 fallback, explicit rounding and
   reduction order, `--ftz=false`, and required generic-to-async proxy fences.

The backend maps only selected TopK positions to physical slots on the opt-in
path. For K <= 1024, one Triton epilogue sorts the selected positions, applies
visibility, maps slots and publishes both outputs. TopK selection/tie-breaking
remains in PyTorch; larger K retains the torch epilogue. The opt-in path does not
materialize a dense `[rows, width]` slot matrix.

## H20 A/B

Measured against the **unchanged default path in `dsv4.1` at `4c10906`**, not
against the unmodified CUDA kernel from PR #40062. GPU: H20, 78 SM; CUDA 13.0,
PyTorch 2.13.0+cu130, Triton 3.7.1; clocks not locked. Inputs: synthetic BF16
fake-FP4 Q, mixed-sign weights, production RNE FP4 K store, shuffled physical
pages; 64 heads, head dimension 128, group 6, page 64, ratio 2, TopK 512.

Each point uses 20 calls per CUDA Graph, nine randomized interleaved rounds,
targeting 100 ms per path per round. Values below are medians in microseconds,
seed 11; all 12 points also passed an independent seed-23 repeat with exact
logits and output-index checks (`atol=rtol=0`).

| rows × width | visible fraction | default logits | opt-in logits | default backend | opt-in backend | backend latency reduction |
|---|---:|---:|---:|---:|---:|---:|
| 6 × 512 | 1 | 25.90 | 6.76 | 90.96 | 28.15 | 69.1% |
| 6 × 4096 | 1 | 40.56 | 16.49 | 127.70 | 59.54 | 53.4% |
| 6 × 16384 | 1 | 95.21 | 33.10 | 244.08 | 137.75 | 43.6% |
| 6 × 131072 | 1 | 573.73 | 170.11 | 691.74 | 241.93 | 65.0% |
| 6 × 131072 | 1/8 | 569.22 | 34.21 | 685.20 | 104.75 | 84.7% |
| 6 × 1048896 | 1 | 4505.41 | 1186.99 | 4731.47 | 1355.54 | 71.4% |
| 384 × 512 | 1 | 161.24 | 63.85 | 262.78 | 117.47 | 55.3% |
| 96 × 4096 | 1 | 294.59 | 91.74 | 408.86 | 161.85 | 60.4% |

`width` is the compressed index-K scan bound, not original KV length. Fraction 1
retains small within-group visibility differences. Logits timing includes dense
slot construction on the default path and Q packing on the CUDA path. Backend
timing additionally includes length preparation, TopK, sorting and index
mapping, but uses prepared Q/weights instead of projection and RoPE. Neither
metric includes host dispatch, JIT compilation, attention or serving overhead.
Short shapes use Triton inside the opt-in wrapper; their gains must not be
attributed to CUDA. Zero/partial-prefix gains do not describe full-prefix work.

The length-aware/three-warpgroup candidate was selected for its aggregate shape
coverage, not because it dominates every point. Compared with the previous
four-warpgroup fused/fenced candidate, the ~1M-column case is essentially tied;
the 1/8-prefix case improves from approximately 70.5 to 34.2 us. Dispatch
thresholds are empirical H20 choices. Other groups/layouts have correctness
coverage, not an exhaustive performance sweep. No end-to-end serving speedup is
claimed; a matching 64-head production workload is still required for that.

A separate seed-31 paired run keeps the opt-in logits and PyTorch TopK selection
fixed, replacing only the selected-position epilogue. All seven points pass
exact output checks; the full backend saves 49.7–55.1 us. Examples: 6×512 is
77.76 → 28.08 us, and 6×131072 (near-full prefix) is 291.98 → 241.65 us.
These are full-backend timings, not isolated epilogue-kernel timings.

Final JIT Nsight Compute profiles cover 6×1048896, 6×16384 and 6×131072 with
1/8 visibility. All three have zero dynamic local-load/store instructions.
The long case has 7.62% tensor activity, 1.54% peak DRAM-read throughput, and
barrier/short-scoreboard stalls accounting for 24.90%/22.55% of PC samples.
This points to synchronization and conversion/reduction dependencies rather
than saturated HBM. PC sample fractions are not fractions of elapsed time;
profiler replay durations are not used as A/B latency measurements.

## Reproduction and tests

From the repository root, with Hopper/CUDA dependencies installed:

```bash
PYTHONPATH=python python3 test/registered/kernel/attention/dsv4/test_sm90_fp4_grouped_indexer.py -v

PYTHONPATH=python SM90_INDEXER_BENCH_SEED=11 \
  SM90_INDEXER_BENCH_OUTPUT=/tmp/sm90-indexer-seed11.json \
  python3 test/registered/kernel/attention/dsv4/bench_sm90_fp4_grouped_indexer.py
```

The benchmark toggles the opt-in flag during separate graph captures; it does
not require the environment switch to be enabled outside the script. Use a new
output filename for every run. Seed 23 repeats the same shape matrix. Optional
`SM90_INDEXER_BENCH_CASES` accepts comma-separated `rows:width:visible_fraction`
triples; backend rows must be complete groups of six. The benchmark's docstring
lists timing overrides. CI uses a smaller shape subset.

The registered test file covers both logits branches; multiple page sizes,
ratios, groups and tail groups; scale/weight boundaries; unaligned cache views;
poisoned invisible mappings; varying-length replay; default-off and unsupported
backend dispatch; int32 metadata; candidate source/consumer masks; and TopK /
physical-index equality, strided request metadata, non-power-of-two TopK sizes,
optional raw-index output and the large-K fallback. It contains 18 methods and
uses the existing Triton implementation as its numerical-contract reference,
plus an exact simple-score case. Tuning also used an independent FP64 reference
for ordinary finite inputs.

For instrumentation, run `compute-sanitizer --tool TOOL --error-exitcode 1`
before the test command, with `TOOL` in `memcheck`, `racecheck`, `initcheck`,
`synccheck`. A zero sanitizer error summary is not sufficient if output
assertions fail. Graph replay and numerical contract checks must also pass.

The final candidate passed all 18 ordinary methods. Instrumented validation
covered these eight methods under each tool: `test_multitile_paths`,
`test_length_aware_schedule`, `test_backend_graph_replay`,
`test_backend_long_graph_replay`, `test_backend_topk_sizes_and_optional_raw`,
`test_backend_topk_and_candidate_masks`, `test_backend_strided_request_metadata`
and `test_scale_and_weight_boundaries`. Memcheck, racecheck and synccheck were
clean; initcheck used the narrowly scoped exception below.

### Validation-environment caveat

On the measured PyTorch 2.13.0+cu130 build (`cf30153c`), a standalone
`torch.randn(12, 131072, device="cuda").topk(512)` reports uninitialized
scratch reads in PyTorch's multi-block TopK under both Compute Sanitizer
2025.3.1 and 2026.3.0. The reports are consistent with the initial `desired`
scratch being read before initialization: the first histogram pass masks it
with zero and later passes replace its bits. See the
[pinned PyTorch implementation](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/TensorTopK.cu#L645).

The optional `pytorch-2.13-topk-initcheck.xml` is a **local validation exception**
for those two exact function/PC/4-byte-read signatures. It is not used by normal
tests or runtime dispatch. Re-evaluate it for another PyTorch binary. It does
not disable API checks or exclude any SGLang/Triton kernel; unsuppressed reports
remain a limitation of full-backend initcheck on this environment. Do not report
an exception-assisted run as an unqualified clean initcheck run.

For this exact binary, add
`--suppressions test/registered/kernel/attention/dsv4/pytorch-2.13-topk-initcheck.xml`
to the initcheck command. The signatures are `computeBlockDigitCounts+0x260`
and `computeBlockwiseWithinKCounts+0x380`, each a four-byte global read. An
intentional uninitialized-read Triton negative control still reports an error
with the same file, confirming that it does not suppress arbitrary new kernels.
