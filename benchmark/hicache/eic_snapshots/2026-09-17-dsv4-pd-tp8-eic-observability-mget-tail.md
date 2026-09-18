# EIC observability + mget-tail snapshot: DeepSeek-V4-Flash FP8, PD TP8 DP2, 2×8×H20

2026-09-17. Observed hot-patch run of observability PR #778 (per-rank `EIC_STATS`)
on image base `a054fa3440` (#768), plus #772 (partial-RPC retry). No image rebuild:
the 5 changed files were mounted over the V4 image via ConfigMap `eic-obs-patch`
(subPath) on the existing StormService `d-20260727194216-7jojv`
(roleset `dmgzq`, prefill 192.168.0.54 / decode 192.168.0.53).

Purpose: answer the three tuning questions — which KV writes error, what fraction
of a prefix hits EIC, why the missed part misses — and measure RPC tail under load.

## Workload

`bench_serving --backend sglang --dataset-name random-ids`, input 3500 / output
512 tokens, `--random-range-ratio 1.0`, 128 prompts, max-concurrency 32,
fixed `--seed 424242`, run 3 times back-to-back. Same seed => identical token ids
every round: round 1 cold-writes, rounds 2/3 read the same prefixes.
Driven through the frontend pod (`*-svc:8080`); only the prefill role has
`enable_eic_cache=True` (decode runs with it False by design in this PD layout).

## Results (aggregated over 8 prefill TP ranks, per-round)

| Metric | R1 cold write | R2 EIC read | R3 device hot |
|---|---|---|---|
| Successful requests | 128 | 128 | 128 |
| Write node acks / mset keys | 511 / 8014 | 4 / 8 | 0 |
| Write failure / mset key failure | 0% / 0% | 0% | – |
| mset retries (#772 path) | 0 | 0 | 0 |
| mget keys / key failures | 0 | 5262 / 0 | 1174 / 0 |
| eic_expected / eic_got tokens | 0 | 750848 / 750848 | 150272 / 150272 |
| **EIC hit ratio** | – | **100%** | **100%** |
| miss cold_or_probe_fail | 300 | 270 | 143 |
| miss headroom / below_threshold / dma_incomplete | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Mean TTFT (ms) | 16224 | 25648 | **1712** |
| P99 TTFT (ms) | 37427 | 75562 | **3198** |
| Request throughput (req/s) | 1.18 | 0.92 | 2.74 |
| Output token throughput (tok/s) | 607 | 471 | 1403 |
| Total token throughput (tok/s) | 4753 | 3694 | 10996 |

The `cold_or_probe_fail` counts include startup health probes and first-touch
requests; the four load-bearing miss reasons (no device headroom, sub-threshold,
broken frozen chain, incomplete DMA) were all zero. Nothing a prefix expected was
dropped for a software reason.

## RPC tail (the finding)

- **mset**: p99 steady 500ms–1s across 119 60s windows; one 14.5s spike in 13
  windows. Zero failures, zero retries.
- **mexist**: p99 100–500ms.
- **mget**: **p99 > 5s in 90 of 99 active windows**, global max 14.1s, p50 only
  93ms. Zero key failures.

Hit ratio is 100%, yet R2 (the round that actually reads KV back over EIC) is the
slowest and R3 (device pool already hot, almost no mget) is 15× faster at Mean
TTFT. **The cost is the read tail, not misses or write errors.**

### byterpc/RDMA client-log evidence

Mounted SDK logs at `/sgl-workspace/log` (`eic_client_byterpc.*WARNING`,
`python3.*ERROR`) over the run:

- 801 WARNING/ERROR lines, **all** `slice_task.cc:652 ... EIC_AGENT_CACHE_MISS`
  on `io_type: KvCheck` (the cold mexist probes). Zero timeout, retransmit,
  connection, or RDMA errors.
- Each key fans out to 8 storage agents (two sets of eight RDMA endpoints,
  `26.6.x.130` / `26.6.x.162`, ports 10007–10016); one key's latency is the max
  over its 8 slices.
- Effective flag file (`/sgl-workspace/config/eic_flag_file`):
  `kv_get_gdr=true`, `kv_set_gdr=false`, `multi_nic=true` over rdma0–7,
  `split_kv_slice_size_byte=65536`, `kv_req/kv_rpc_rdma_timeout_in_ms=6000`,
  `use_polling_mode=true`, `dram_mempool_limit_bytes=16GiB`.

So the >5s tail is not a network error or a backend reject: it is normal-read
queueing / slow-slice fan-out / 5s RDMA timeout on the slowest of hundreds of
slices, with no failure surfaced.

## Why mget is slow — causal chain (client-side amplifiers)

One load op issues a single blocking compiled-SDK `connection.mget`
(`eic_memory_pool.py` `batch_get`). At page bytes 9,158,208 (3 × 4MiB chunks per
256-token page) the 1GiB GDR bounce pool holds 256 chunk slots; a 3500-token
prefix is 13 pages ≈ 39 keys in one wave, a 16K prefix about 63 pages. A wave is
a barrier over every slice: any slow agent slice holds the whole mget up to the
5s timeout. On top of that backend slow-slice source (not fixable in this repo):

1. **Single load thread, FIFO, no cross-request coalescing**
   (`eic_cache_controller.py` `load_parallel=1`). One bad 5–14s op blocks every
   queued load on the rank; request N's TTFT ≈ sum of prior op service times.
2. **Serial refetch doubles the budget** on PARTIAL (`_refetch_failed`,
   same `GetOption`, no per-call timeout): up to 5s + 5s ≈ the 14.1s max.
3. **TP8 gloo MIN barrier per op**: admitted length is the min across 8 ranks,
   each a fan-out — group sample ≈ 8 × slices.
4. **Prefix truncation**: only the contiguous successful prefix is copied and
   admitted; one early slow/empty page discards already-arrived later pages.
5. **load_wait_event stalls all MoE dispatch**: while a load (mget + unpack)
   runs, deepep `internode_dispatch` is hooked to yield, so one bad read pauses
   cross-GPU MoE for the whole process, not just the waiting request.
6. **Unpack is on the same load thread and unmeasured**: read data still does
   GPU→pinned-host→GPU with two full-stream syncs in `device_writeback`; this is
   after `record_rpc`, so it is not in the mget number but consumes the FIFO slot.
7. Latent leak: `batch_get` has no try/finally around the registered read-buffer
   free; an exception can leak the whole 256-slot pool into the slow
   unregistered-CPU fallback until restart (grep `can not allocate tensor from
   pool` to check whether it has happened).

## Optimization ranking

The 25.6s has three mutually-exclusive components today — backend slow slice,
FIFO queueing, PCIe unpack — and the current `EIC_STATS` cannot separate them.
Instrument first (~half day, no production behavior change): record load-op
queue wait vs service time, split mget first-attempt vs refetch timing, and time
`device_writeback`; add `load[wait_p50/p99 service_p99 unpack_p99 qsize]` to the
periodic line. Then one same-seed R1–R3 run decides the order:

Config-only (flag file / yaml), cheapest first:
- Enable/verify multi-NIC spread (`eic_multi_nic_default_local_ips_use_all=true`
  or explicit `multi_net_local_ib_device_names=mlx5_1..8`); confirm each rank's
  NIC in startup logs. Helps only if ranks share a NIC today.
- Size the SDK local read cache (`enable_local_cache_on_read_path`,
  `read_local_cache_cap_mb`): for repeated identical content-hash prefixes this
  can serve most reads from node DRAM instead of the RDMA tail. Largest possible
  config win for this repeat-prefix pattern; confirm it applies to the GDR path.
- A/B the RDMA timeout 6000→3000 with mset failure held at 0% (1000ms once made
  TP8 mset all-timeout, so do not jump to 2s).
- Try SDK IO thread count (`default_io_thread_num`, not the dead yaml
  `eic_thread_num`) ≤4 and smaller `eic_max_batch_size` (256→64) — measure both.

Code:
- **B2 try/finally buffer return** — ~30 min correctness fix, do regardless.
- **B3 hedged retry** — the only client lever that caps a *single* mget's tail:
  after ~p90 (300–500ms) issue the same mget on a second connection/registered
  buffer set, take the first success; needs sampling switch, symmetric TP use,
  and never freeing in-flight DMA. ~2–3 days.
- B4 yaml toggle to skip refetch (~15 lines; bounded by measured refetch rate,
  which is currently 0, so no present benefit).
- B5 configurable `load_parallel` — only if wait p99 ≫ service p99; it lowers
  TTFT queueing but cannot lower a single mget RPC p99 and risks agent
  saturation and TP all_reduce op mismatch (needs enqueue-order ticketing).
- B6 direct-GPU unpack — only if unpack time proves large.

Backend (outside this repo): with the instrumentation, take slow-slice
`data_source` (MEMORY vs DISK), per-agent RPC concurrency/p99, and NIC
utilization to the EIC team; agent memory/eviction is the root fix if slow
slices are disk reads.

## Artifacts

On ctrl-1 (`ssh -p 2022 dev@115.190.238.2`): `/tmp/eic_obs_bench_0917/` tarred at
`/tmp/eic_obs_bench_0917.tgz` — full prefill/decode pod logs, per-round bench
output, the streamed `prefill_stats.log` (1920 `EIC_STATS` lines, 8 ranks ×
9 lines/rank plus idle), `remote-eic.yaml`, the effective `eic_flag_file.txt`,
and byterpc error-type counts. Deploy-side: ConfigMap `eic-obs-patch`; manually
created services `d-20260727194216-7jojv-{prefill,decode}-svc-0` (the operator
did not create them for the replacement roleset).
