# EIC load-path parallelization: DSv4 PD TP8 DP2, 320K agent — code landed, A/B blocked

2026-09-21. Branch `cklxx/eic-load-parallel` (off `eic-observability`), single-file
change `python/sglang/srt/mem_cache/eic_memory_pool.py`, commit `de02f13609`.
Hot-patched onto StormService `d-20260727194216-7jojv` via ConfigMap
`eic-obs-patch` (subPath) with no image rebuild; tuning knobs delivered through
`remote-eic.yaml`.

## Bottleneck (confirmed from 2026-09-20 cc12 `polling_false` tgz)

At 320K agent context (12 sessions, +32K new tokens/round, seed 18000):

- Decode is healthy: TPOT 13.4–14.9 ms, 512 output tokens ≈ 7 s (<8% of e2e).
- Prefill TTFT p50 = 65.5 s; DP0 single-rank traces show **53–57% of wall time
  is the scheduler blocked waiting on EIC load-back**, not computing.
- One load worker drains a FIFO; one op issues ~7 sequential 1 GB mget batches.
  `load.service p50 9.0 s / p95 14.3 s / max 17.7 s`, `load.wait max 19.9 s`,
  single `mget.first p50 1.08 s` over ~1.06 GB (~1 GB/s vs an 8×RDMA fabric that
  is nowhere near saturated).
- GDR read already enabled (`enable_kvget_gpu_direct=true`, 1 GB = 256×4 MiB
  CUDA-registered bounce). `eic_direct_writeback` is not wired on the DSv4 path
  and repack is only ~100 ms → <10% upside, not pursued. `eic_thread_num` is a
  dead knob (read but never passed into `eic.InitOption`).

## Change

The TP `all_reduce(MIN)` collective, the single CUDA `load_stream`,
`load_wait_event`, and the keyed `ack_load_queue` ordering are all left
untouched — running N controller workers would initiate the per-op gloo
collective out of order across ranks (rank A op2 vs rank B op1). Parallelism is
instead pushed **inside one op's DSv4 `get_page_data`**:

- N bounded fetcher threads run only the network `mget` into **disjoint** bounce
  chunks (the 9–14 s part); they touch no CUDA kernel, allocator, or tree.
- The single load worker repacks on the single `load_stream` strictly in fetch
  order; `device_writeback`'s terminal `stream.synchronize()` is the fence that
  makes chunks NIC-reusable before release.
- `FlexibleKVCacheMemoryPool` alloc/free locked.
- Multi-permit chunk credits (`threading.Condition`, capacity =
  `G_GDRBounceTensorCount`) so total in-flight GDR chunks never exceeds the
  registered 1 GB pool — no KeyError/leak, no silent fallback to unregistered
  host memory under contention. Concurrent-allocator logic self-checked
  (oversubscribe/deadlock/leak) under 8 workers.
- Knobs in `remote-eic.yaml`: `eic_load_fanout` (default **1** = byte-identical
  serial path), `eic_load_page_batch` (default 0 = auto = 128 pages // fanout).
- Concurrent same-connection `mget` thread-safety is undocumented in the eic
  binding; it is gated empirically by the A/B (0 fail + data match = safe).

Planned arms (320K, ≥2 runs each, same flags/seed): A `fanout=1/1GB` baseline;
B `fanout=8/1GB` credits (16-page/128 MB batches); C `fanout=8/8GB`
(`gdr_bounce_buffer_size=8589934592`, 128-page batches). C needs mem-fraction
headroom because peak usage is 95.2/97.9 GB and bounce lives outside the static
fraction; KV swa/c4/c128 capacity cost and headroom denials must be reported
(fall back to 4 GB if it eats 320K admission).

## Environmental fixes required before measuring

The pod's `eic_flag_file` had regressed to an old template; tuned set restored:
`eic_use_polling_mode=false`, `eic_client_default_io_thread_num=16`,
`eic_client_split_kv_slice_size_byte=524288`,
`eic_client_kv_req_rdma_timeout_in_ms=8000`,
`eic_client_kv_rpc_rdma_timeout_in_ms=6000`. The rpc=1000 default caused mget
~9.9% key failures misclassified as cold, dropping observed hit to ~50% (same
failure mode that made TP8 mset all-fail in earlier sessions; req/rpc both need
6000+).

## Valid data collected (arm A, fanout=1, full tuned flags, seed 18000)

Rounds 1–4 completed cleanly before the node failure; hit ratios match the
2026-09-20 tgz baseline exactly and latency is at/better than it, mset 0 fail
→ **the fans-off code path is non-regressing and the flag fix is correct.**

| round | prompt | hit% | TTFT p50 (s) | this run | tgz p50 (s) | input tput this/tgz |
|---|---|---|---|---|---|---|
| 1 | 65 536 | 0.0 | 37.6 | 37.6 | 44.7 | 11428 / 9087 |
| 2 | 98 304 | 66.7 | 20.7 | 20.7 | 22.8 | 23563 / 26250 |
| 3 | 131 072 | 75.0 | 20.0 | 20.0 | 27.9 | 30460 / 30256 |
| 4 | 163 840 | 80.0 | 22.9 | 22.9 | 34.8 | 40867 / 32912 |

## Blockers

1. **Node 192.168.0.54 (the pinned prefill node) hard-down since 2026-09-20
   16:23Z** — kubelet stopped posting status, Ready=Unknown, unreachable
   NoSchedule+NoExecute, all pods on the node evicted across all namespaces.
   Physical host needs platform restart; prefill is node-pinned and cannot
   reschedule. This stopped arm A at round 4.
2. **Failover node 192.168.0.68 cannot run the EIC client.** Its RDMA fabric is
   healthy (8×mlx5 ports ACTIVE, eth1–8 on the same 26.4.0.0/14 HPC fabric as
   .53, PCI topology identical; the 2026-09-16 cello "rdma-host deviceID empty"
   is resolved) and 192.168.0.53:10000 is TCP-reachable, but EIC client `init`
   requires ZK discovery (`eic_use_zk_instance=true`). On .68
   `eiclt.vestack.com` resolves only to public IPv6 `2001:41d0:301::28`
   (unreachable); production nodes resolve it via node-local DNS 169.254.25.10
   to in-fabric ZK. Forcing the in-fabric ZK VIPs reaches TCP 4595 but returns
   `EIC_ZK_ERR_ZOO_NOT_EXIST /ebs/az1/eicf0ihqes73teo9jtzlbam/eic_masters`
   → client init ret=-1, mset/mget cannot start. Needs platform to give .68 the
   same EIC ZK split-horizon DNS + an EIC agent for the instance.
3. An ad-hoc fresh EIC namespace does not work: namespaces must be provisioned
   agent-side — a self-invented `...-loadpar` namespace made mset fail 100%
   `StatusCode.FAILED`. Only the platform-provided `s-20260727194216-eg6jw`
   namespace is usable.

## Next step (unblocked)

When .54 is back (or .68 gets EIC ZK/agent): rerun arm A twice to round 9, then
arms B and C with in-run EIC_STATS (`load.wait/service`, `mget.first`,
eic_got/expected, headroom) and client TTFT p50/p99/input tput. Expected upside
from removing the ~53% FIFO idle: 320K TTFT p50 ~65 s toward ~30–40 s.
