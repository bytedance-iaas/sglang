# EIC patch ledger

ep_main is periodically rebuilt from upstream and drops downstream merges, so EIC
lives as one patch commit on `cklxx/eic-patch`. After every ep_main refresh:

```
git fetch origin && git rebase --onto origin/ep_main HEAD~1
```
A maintained reference diff also lives on `cklxx/eic-patch-diff` (see its
README): for a large refresh, branch from origin/ep_main and `git apply --3way`
that diff; resolve only the ep_main-owned files below. The EIC-owned new files
apply untouched.

Conflicts only happen in the ep_main-owned files below. For each touch point this
ledger records what it does, why, where it came from, and which test turns red if
it is lost. PR numbers refer to bytedance-iaas/sglang; their descriptions carry
the measurements. The pre-squash history is on `cklxx/eic-main` (PR #737).

Unit tests: `test/registered/unit/mem_cache/test_eic_hicache_regression.py`
(`EICReg` below), `test_registry.py`, `test_forward_pass_metrics.py`.
Serving benchmark snapshots: `benchmark/hicache/eic_snapshots/`.

## Rules for editing this patch

- ep_main's fix is authoritative. When ep_main already solves a problem, or
  already has a field or metric for it, EIC adapts inside its own files and the
  patch does not touch ep_main. See "Replaced by ep_main mechanisms".
- Touch an ep_main file only for EIC. Every hunk is gated on `enable_eic_cache`,
  an EIC-only hook, or an EIC class, so non-EIC behavior matches ep_main.
- Add a row here in the same commit as the code.
- Squash back to one commit before pushing.

## New files (EIC-owned, no conflict surface)

| File | Role |
|---|---|
| `scripts/eic_integration_check.py` | post-deploy EIC integration check (#709); standalone, no runtime import |
| `managers/eic_cache_controller.py` | EIC write/load threads and queues on top of `HiCacheController` |
| `mem_cache/eic_hiradix_cache.py` | `EICHiRadixCache` / `EICPagedHiRadixCache`: remote match, async load-admit gate, PP verdicts |
| `mem_cache/eic_memory_pool.py` | EIC client and host pools (MHA/MLA/NSA/DSv4) |
| `mem_cache/eic_chunk_cache.py` | `EICChunkCache` / `EICSWAChunkCache` for `--disable-radix-cache` |
| `mem_cache/eic_pp_reconcile.py` | Cross-PP load-length reconciler |
| `mem_cache/eic_stats.py` | Dependency-free per-rank `EIC_STATS` line (#778): write ack/mset/mget/mexist errors, eic_hit%, miss reason counts, RPC p50/p95/p99/max; interval `EIC_STATS_INTERVAL_S` (60s) |
| `test/registered/unit/mem_cache/test_eic_slot_ownership.py` | property test: random cache op sequences, each step checked against the legal per-slot transitions and the three pool counters |
| `test/registered/unit/mem_cache/test_eic_pp_protocol.py` | exhaustive interleaving search over the real PP load-back verdict code (stage skew, ack timing/size, release, rid reuse, failed kicks); `EIC_PP_MODEL_FULL=1` adds a competing request under a 1-row verdict cap |

Upstream APIs these subclass or call, the usual source of silent breakage after
a refresh: `HiCacheController.__init__`, `HiRadixCache`,
`RadixCache.cache_finished_req`, `build_deepseek_v4_hicache_stack`,
`CacheInitParams`, `Req.extend_range`. Three contracts broke only on a real
server, not in unit tests:

- `RadixCache.insert` unpacks `(prefix_len, last_node)` from `_insert_helper`
  (#27058); the EIC override must return the same pair.
- `HostPoolGroup.load_to_device_per_layer` does not move indices;
  `HiCacheController.move_indices` does it for the kernel IO backend. The DSv4
  `device_writeback` repeats that move, or the load kernel rejects CPU indices.
- `Req.extend_range` is None until `PrefillAdder.add_one_req` sets it from
  `prefix_indices`; `_finalize_load_admit` must not read or set it.


## Refresh fixes for the pool/ServerArgs refactor (f4c61f324b)

The 2026-09-15 refresh crossed the host-pool and ServerArgs refactors; these were
not text conflicts but API-contract breaks found by review, all in EIC-owned files:

- DSv4 load/backup drives pools per entry. `HostPoolGroup` no longer has
  `load_to_device_per_layer`/`backup_from_device_all_layer`, and the per-pool
  methods take no `pool_transfers=`. `EICDeepSeekV4TokenToKVPoolHost` iterates
  `_build_pool_transfers` and calls each `entry.host_pool.{backup,load}` with the
  entry's device pool and `layer_mapper`, matching `l2_transfer.py`.
- Req fields moved into `ReqKvInfo`: use `req.kv.req_pool_idx` and
  `req.kv.cache_protected_len` (no top-level forwarding properties).
- `req.fill_ids` is now `req.get_fill_ids()` (bounded by `extend_range.end`).
- `last_matched_prefix_len` was removed upstream; the EIC finalize no longer sets it.
- Test imports updated: `unified_cache.component_type.ComponentType`,
  `unified_cache.unified_tree_core.UnifiedTreeNode`.

## Touch points in ep_main files

### Enablement

| Where | Change | Why | From | Guard |
|---|---|---|---|---|
| `server_args.py` fields | `enable_eic_cache`, `disable_eic_shared` | CLI flags (generated from annotated fields) | #481 | `test_registry.py` EIC cases |
| `server_args.py` post-init | `enable_eic_cache` forces `enable_hierarchical_cache` | `--enable-eic-cache` alone must be enough | #481 | none |
| `arg_groups/hicache_hook.py` `handle_hicache` | When `enable_eic_cache`, declare `enable_hierarchical_cache=True` before the hicache skip test | After the ServerArgs resolution pipeline refactor there is no `__post_init__` body to force it in; resolution writes go through `declare_resolution` | #767 (moved on refresh) | registry EIC cases |
| `mem_cache/registry.py` `default_radix_cache_factory` | EIC chunk cache under `disable_radix_cache`; `EICHiRadixCacheBuilder` ahead of the hybrid/DSA arms; sets `dedup_aliased_swa` on an SWA allocator | Nested under the hierarchical arm it was unreachable for hybrid-SWA/DSA models: the flag was ignored and the first request died on `ongoing_load_admit` | #737 | `test_eic_hi_radix_cache_when_hierarchical_and_eic`, `test_eic_chunk_cache_when_chunked_prefill_disable_radix_and_eic`, `test_eic_swa_chunk_cache_when_hybrid_swa_and_eic`, `test_eic_marks_swa_allocator_for_alias_dedup` |
| `managers/scheduler.py` `__init__` | `self.enable_eic_cache` = flag AND tree cache is `EICHiRadixCache` | EIC bookkeeping must follow what was built, not the flag. `EICChunkCache` (PD decode-save path) was admitted too but has no admission hooks, so aborting a waiting req there raised `AttributeError` on `release_load_admit` | #737, #767 | `test_scheduler_eic_gate_admits_only_caches_with_admission_hooks` |

### Scheduling

| Where | Change | Why | From | Guard |
|---|---|---|---|---|
| `scheduler.py` `_get_new_batch_prefill_raw` | `match_from_remote(waiting_queue)` before the PP-divergent early returns | Its PP all_reduce hangs unless every stage calls it in lockstep | #523, #670 | `EICReg.test_match_from_remote_*` |
| `scheduler.py` candidate loop | Skip `init_next_round_input` while a load-admit is in flight; gate candidates on `check_load_back_progress` (every candidate when PP>1) | Load-back is async; re-matching a mutated tree double-counts; PP>1 needs symmetric per-rid reports | #670, #741 | `EICReg.test_pp1_*`, `test_two_stage_*`, `test_gate_refuses_load_back_without_pool_headroom` |
| `scheduler.py` candidate loop | After `init_next_round_input`, skip a req whose matched prefix runs through an in-flight load-back (`prefix_loading`) | `load_back` publishes `node.value` before the DMA acks; adopting those slots reads KV still landing, and a failed load frees the tail under the req, whose insert then re-links freed slots into the tree (a page both free and cached). Same PR: `cache_unfinished_req` keeps a chunked req private while its prefix loads, `_match_prefix_helper` stops at a resident node under an evicted gap, and `batch_get` re-gets only the failed keys of a partial mget once | #768 | `test_failed_load_under_same_prefix_req_keeps_pool_invariant`, `test_chunk_insert_through_inflight_load_keeps_pool_invariant`, `test_partial_mget_refetches_only_failed_keys` |
| `scheduler.py` `_abort_on_queued_limit`, `_abort_on_waiting_timeout`, `abort_request` | `release_load_admit(rid)` | Aborted or preempted requests leak admit locks otherwise | #670 | `EICReg.test_release_tombstones_and_drops_straggler_verdicts` |
| `scheduler.py` idle check | `ongoing_load_admit` must be empty | PP-symmetric in-flight marker; the rank0-only maps would desync idle | #670 | none |
| `scheduler.py` candidate loop else-branch | A PP=1 candidate that skips the load gate is counted once (`_eic_cold_counted`) as `miss.cold_or_probe_fail` with its device-only admit | Cold reqs never enter `_apply_span`, so the miss-reason accounting would miss the dominant class | #778 | none |
| `managers/schedule_policy.py` `PrefillAdder` | `enable_eic_cache`; under EIC do not subtract `host_hit_length` and skip `init_load_back` | ep_main's `host_hit_length` means "host hit not loaded yet". The EIC gate has to fold the load into `prefix_indices` before admission to keep PP stages at one clamped length, so neither step applies. | #670 | none |
| `distributed/communication_tags.py` | `HIRADIX_PP_VERDICT` | Verdict stream must not share a FIFO tag with num_ready | #670 | `EICReg.test_pp_bcast_from_first_is_nonblocking_isend` |

### SWA memory

| Where | Change | Why | From | Guard |
|---|---|---|---|---|
| `schedule_batch.py` `maybe_evict_swa` | Extend-phase eviction also when the cache exposes `eic_swa_extend_eviction()` | EIC holds no SWA in the tree; without it the SWA pool fills during chunked prefill | #661 | none |
| `schedule_batch.py` → `common.free_swa_out_of_window_slots` | `release_cache_protected_prefix` from `swa_evict_release_prefix` | EIC restores prefix SWA from host, so the out-of-window prefix can be released | #655, #710 | `EICReg.test_swa_evict_release_prefix_reaches_free_swa` |
| `common.py` `free_swa_out_of_window_slots` | Free from `req.prefix_indices` for the EIC-loaded span | Loaded indices are not in `req_to_token` (zeros there), so the free was silently dropped | #710 | `EICReg.test_swa_evict_release_prefix_reaches_free_swa` |
| `allocator/swa.py` `free_swa` | Class field `dedup_aliased_swa = False`; when set, drop the reserved page (`< page_size`) and `unique` | EIC aliases FULL spans onto reused SWA slots; double free otherwise | #661 | `EICReg.test_free_swa_skips_reserved_page_and_dedups_aliases` |

### Host pools and L3 storage

| Where | Change | Why | From | Guard |
|---|---|---|---|---|
| `docker/Dockerfile` | install the parameterized EIC Python SDK in `framework`; `framework_final` inherits it and `runtime` copies its site-packages | EIC runtime modules and the post-deploy integration check import `eic`; the SDK hunk was lost while resolving #767 onto refreshed `ep_main` | #767 follow-up | `scripts/ci/verify_private_image_runtime.py` imports `eic` in built private images |
| `hybrid_cache/hybrid_pool_assembler.py` | `device_indexed` clamps DSv4 host pages to device+1 | EIC host pages are device-indexed; `ratio × device` pinned unreachable RSS and OOMed TP8 | #744 | `EICReg.test_device_indexed_host_pages_ignore_hicache_ratio`, `test_eic_calls_the_assembler_with_its_current_signature` |
| `mem_cache/radix_cache.py` `TreeNode` | `content_hash = None` | Every EIC hash path reads it on nodes `RadixCache` creates | #523 | `EICReg.test_real_tree_node_carries_content_hash` |
| `storage/eic/eic_storage.py` | v2 multi-pool API (`register_mem_host_pool_v2`, `batch_{exists,get,set}_v2`), pp-scoped key prefix, logical anchor, mexist cardinality guard | Unified radix EIC L3 for DSv4 pools | #637, #660 | `EICReg.test_batch_exists_impl_failed_batch_keeps_cardinality`, `test_unified_tree_node_exposes_storage_hash_helpers` |

## Replaced by ep_main mechanisms

| Was in the patch | ep_main mechanism now used | EIC side |
|---|---|---|
| `get_num_allocatable_reqs`: PP=1 uses `max_running_requests` (#721) | Scheduler defaults `pp_max_micro_batch_size` to `max_running_requests // pp_size` when unset | None. Deployments must stop passing `--pp-max-micro-batch-size 1` with PP=1; that flag caused prefill batch 1. |
| `hiradix_cache.py`: disable the eic storage tier under PP>1 (#670) | #746 adds `pp_group` to the L3 prefetch sync groups, so PP stages agree on L3 hit length | None |
| `Req.eic_loaded_len`, `PrefillAdder.log_hit_eic_tokens`, `sglang:eic_cache_hit_rate`, EIC fields in `metrics_reporter` (#523, #767) | Per-request `cached_tokens_details` and `sglang:cached_tokens_total{cache_source}` | `_finalize_load_admit` sets `req.host_hit_length = req.storage_hit_length = loaded`, so EIC load-backs count as `storage`; the metric label is `storage_unknown` because EIC registers no `storage_backend` (same path as PD decode). Guard: `EICReg.test_finalize_*` |
| `free_swa` read `get_global_server_args().enable_eic_cache` | `test_legacy_global_ratchet` forbids new `get_global_server_args` call sites, and allocators are built without published config in unit tests | The registry sets `allocator.dedup_aliased_swa` in its EIC branch |
| `eic_memory_pool` used `get_attn_tensor_model_parallel_{rank,world_size}` | `get_parallel().attn_tp_{rank,size}` (`test_parallel_adoption_ratchet`) | Direct swap |

## Deliberately not carried from bytedance/deepseek_v4

| Item | Reason |
|---|---|
| `scripts/eic_gate.py` (#760) | Pre-benchmark gate only; `scripts/eic_integration_check.py` (#709, carried here) is the deployment check and the gate duplicates it |
| Min host pages `device+1` on the non-EIC DSv4 path | Upstream change; alters non-EIC sizing when `hicache_ratio < 1` |
| `free_swa` negative/out-of-range index filter (#573) | Draft-extend padding fix, not EIC |
| `allow_radix_cache_insert_once` | No setter anywhere |
| `last_matched_prefix_len` / `eic_prefix_len` in the non-EIC `init_load_back` branch | ep_main refactored that branch; with EIC on the branch is unreachable |
| `release_kv_cache(is_decode=...)` | No caller passes `is_decode` on either branch |
| `/enable_eic`, `/disable_eic` (#481) | On ep_main they only reported state; the working runtime toggle (flush and rebuild pools) exists only on poc_weizhong. They misreported under `EICChunkCache`. `/server_info` already returns `enable_eic_cache`. A real toggle should follow ep_main's `/hicache/storage-backend` attach/detach. |

## NetworkInterfaceNames environment variable

Set `NetworkInterfaceNames=eth0,eth1` to select EIC client network devices.
The resolver trims surrounding spaces, removes duplicates and preserves order.
It converts commas to semicolons for the SDK. Empty items, malformed names and
names longer than 15 ASCII bytes raise `ValueError` when EIC is enabled.
The client checks that every name exists in its own network namespace.
This check does not prove that the device supports RDMA or can reach EIC.

Both `EICKVClient` and `EICStorage` apply this list before `Client.init`.
The list takes precedence over the GPU NIC mapping without changing CPU affinity.
When the variable is absent, the existing NIC selection remains unchanged.
When EIC is disabled, the resolver ignores this variable.

With no external flag file, the list goes directly into `InitOption` and creates
no file. With an external flag file, a private wrapper includes that file first
and appends only the NIC flag. This prevents the SDK flag parser from replacing
the env selection. The source file remains unchanged. Wrapper permissions are
0600 in a 0700 directory; the owner process removes it at normal exit.
A missing or unreadable source file raises an error. Forced termination can leave
the temporary directory behind. Other flags and transport defaults do not change.

Tests: `TestEicNetworkInterfaces` and
`TestEICHiCacheRegression.test_network_env_reaches_both_real_client_constructors`.
The constructor test stops at a mock `Client.init`; it is not a live SDK or GPU test.

## Centralized defaults and config resolution

`storage/eic/eic_env_config.py` is the single source for built-in fixed
parameters. `FIXED_CONFIG_SPEC` lists each key with its built-in default,
consumer (`both` clients / `pool` / `hiradix` cache tier), bind point and
status (`live-init` / `live-local` / `cache-tier` / `log-only` /
`forced-off` / `unconsumed`); `DEFAULT_FIXED_CONFIG` is derived from it.
Resolution priority stays env > CLI > yaml > built-in. Fixed parameters have
no env channel; only the switch, endpoint, instance id and
`NetworkInterfaceNames` read from the environment.

File-existence semantics:

- A missing `remote-eic.yaml` falls back to the built-in defaults (no error).
- A present-but-broken yaml (corrupt YAML, non-mapping, unreadable
  permission) raises `ValueError`; it never silently falls back.
- When a yaml is present but pyyaml is unavailable, resolution fails instead
  of warning and ignoring the mounted config.
- `eic_flag_file` is validated independently of `NetworkInterfaceNames`: an
  explicit yaml path must exist, be a regular file and be readable, else it
  raises (missing explicit path is an error, not a silent fallback to the
  standard path); the auto-discovered standard path is used only when it
  exists, and a present-but-broken standard path also raises.

Transport provenance: the centralized built-in `eic_trans_type` is 2 (RDMA).
`load_fixed_config` rewrites it to 3 (GDR) only when no flag file is
resolved and yaml did not set it explicitly; both clients receive the final
value through `resolved.fixed` identically and neither overrides it locally.
The rewrite exists because type 2 without the template flag file keeps KvGet
CRC on over device memory and segfaults in `CalculateKvValueCRCsPerSlice` /
`OnKvGetGDR`.

flagfile parsing semantics (upstream gflags, not measured on embedded 1.5.2):
`--flag=value` takes the whole rest of the line after `=`; spaces and `#` are
legal inside a path and are not truncated or blind-quoted; a nested `--flagfile`
recurses and later entries win (last-wins). A `,` in an external flag_file path
is refused fail-closed because gflags may read it as list separation; spaces
and `#` remain allowed.

`SDK_FLAG_AUDIT` records the standard SDK gflags for audit/comparison only;
`SDK_DEFAULT_FLAGS` (versioned by `SDK_DEFAULTS_VERSION`) holds the conservative
emitted subset on the missing-file path. When no external `eic_flag_file` is
mounted, an `EicDefaultFlagFile` writes the emitted safe defaults to a private
temporary file and hands it to `InitOption.flag_file`, so the defaults truly
reach `Client.init` in both EICKVClient and EICStorage. Priority is safe
defaults -> external flags -> explicit NIC env; the URL/instance-id/switch env
priority is unchanged.

- An existing external `eic_flag_file` is passed through unchanged: defaults
  are never appended to or mixed into it, so an old deployed file's effect does
  not silently change. External file + NIC env only adds the NIC override.
- The default profile writes no hardcoded `eth1;eth2;eth3;eth4` list; without
  `NetworkInterfaceNames` the existing NIC selection / GPU affinity is kept.
- Completion boundary (this round): only the 9-item conservative subset is
  emitted — 4 timeouts (`rpc/kv_req/kv_req_rdma/kv_rpc_rdma`),
  `slice_task_copy_bypass=false`, `slice_task_copy_thread=false`,
  `kv_set_crc=false`, `kv_get_check_crc_type=0`, and `kv_set_enable_gdr=false`.
  This is a safe subset, NOT the full standard flag set; no claim of standard
  full-set completion.
- Deferred (audit-only, not emitted this round): `kv_get_enable_gdr`,
  `eic_use_polling_mode`, `eic_use_byte_express`, `dram_mempool_limit_bytes`,
  `split_kv_slice_size_byte`; they lack live evidence on 1.5.2.
- `--eic_enable_multi_nic`: no target support found in current static evidence;
  this profile does not generate it. This is a per-profile choice, not a
  permanent SDK conclusion; revisit if later evidence shows support.
  `default_io_thread_num` is never mapped (no consumption evidence).

Transport safety: generating the private default file never switches
`eic_trans_type` back to 2; with no explicit user transport and no external
flags the no-flag fallback stays 3 (GDR), and the generated file sets no
transport. The type-2-without-flag segfault in `CalculateKvValueCRCsPerSlice` /
`OnKvGetGDR` is unchanged by this change.

## Dead or misleading config keys

- `load_remote_threshold`: removed as a separate knob in #763. The servingkit ConfigMap template still sets it (RDMA 100, TCP 400), so `eic_env_config.LEGACY_KEY_ALIASES` redirects it to `load_back_threshold` as a best-effort alias; an explicit `load_back_threshold` wins. The two are NOT proven semantically equivalent: the old key was a remote-probe gate (removed in ad7b8d9), while `load_back_threshold` governs L2/PP load admission. The alias does not restore the old 16384-token probe gate, and full legacy compatibility is not claimed. The remote probe budget is `eic_check_max_num` (default 2048).
- `save_decode_cache` in `EICHiRadixCache`: read and logged, never used. Default is False, matching the template. `EICChunkCache` hardcodes True and is not bound to this key.
- Built-in defaults (`DEFAULT_FIXED_CONFIG`, derived from `FIXED_CONFIG_SPEC`) follow the current safe code, not a blind template alignment, so `--enable-eic-cache` plus `EicRemoteUrl`/`EicInstanceId` start EIC without a yaml. `enable_gpu_nic_affinity` stays False until a node-specific ConfigMap sets it. When no explicit `eic_flag_file` is set and `/sgl-workspace/config/eic_flag_file` exists, that file is used after validation.
- No flag file means no `eic_trans_type: 2` default. RDMA forces the GDR read path on, and without the template flag file the client keeps KvGet CRC on and computes it over device memory: SIGSEGV in `CalculateKvValueCRCsPerSlice` (reproduced 2026-10-19, n54). The env-only path therefore defaults to `eic_trans_type: 3`. An explicit yaml `eic_trans_type` or any flag file keeps 2.
- `eic_thread_num`: read, no effect on the client. Set `--eic_client_default_io_thread_num` in `eic_flag_file`.
- `enable_kvset_gpu_direct`: forced off in `eic_memory_pool.py`, with a warning if a
  ConfigMap still sets it. A CUDA write pool takes the server down: the client
  segfaults on its own IO thread (`be::Engine::Transmit`) and the other ranks then
  fail the write-path `all_reduce`. Measured in the 2026-09-12 snapshot, "Write-path
  config A/B". eic 1.5.2 gives the remote write path no CUDA handling to fall back on:
  `Client::register_memory` (`src/client/api/eic.cc:124`) discards the `MemoryInfo` it
  is given (`cuda_id` included) and registers every region as byterpc
  `HolderType::UserNormal` — byterpc v1.1 has no CUDA holder type — and nothing under
  `src/client/io/remote_engine` reads `memory_info` at all; the client's CUDA staging
  (`CudaDeepCopyIOBuf`) exists only on the local cache path. Reads take a different
  path, one-sided RDMA into the device buffer, which is why `enable_kvget_gpu_direct`
  is safe and this one is not. The exact faulting frame is not pinned down: the
  benchmark's flag file is gone, so whether the write was on the plain `KvSet` or the
  two-phase `KvSetGDR` path at the time is unproven. Re-enable only against a client
  that can register CUDA regions for the remote write path.
- The client's flag file wins over `eic_trans_type`. `InitClientInstance`
  (`src/client/api/eic_inner.cc:350`) forces both `kv_get_enable_gdr` and
  `kv_set_enable_gdr` true for `EIC_TRANSPORT_RDMA` (`eic_trans_type: 2`), then parses
  the flag file a few lines later, so the file's value is the one that takes effect.
  Production runs `--eic_client_kv_set_enable_gdr=false` with
  `--eic_client_kv_get_enable_gdr=true` (confirmed in prefill startup logs), i.e. the
  GDR write path is off and only the GDR read path is live. This repo does not
  set that SDK gflag by default; the Python `enable_kvset_gpu_direct` forced-off
  knob above is a separate setting and must not be equated with
  `kv_set_enable_gdr`.

## Open risks

- ep_main #750 appends MTP draft SWA layers to the DSv4 host pool, and
  `EICDeepSeekV4TokenToKVPoolHost._build_page_specs` reads them. With MTP on,
  EIC page bytes may change, and data written by an older build becomes
  layout-incompatible. Not yet run with MTP.
- gflags are process-global and the EicClient singleton re-runs FLAGS on each
  Init (verified against 1.7.3 source only, not 1.5.2). A private generated
  flag file does NOT mean SDK state isolation: once the SDK parses a flag_file
  it writes into the process-global FLAGS, whether or not Python calls a setter
  directly. A same-process second client / second Init is unverified here;
  isolation across different NIC or default-flag configurations is NOT
  guaranteed, and per-client flag isolation must not be claimed. Application
  construction paths (EICMemoryPool -> EICKVClient; storage backend ->
  EICStorage) were not proven to run two clients concurrently here.
- Flags come from one source: `_SDK_FLAG_REGISTRY` (standard value, 1.5.2
  static status, emitted flag, note). `SDK_DEFAULT_FLAGS` is the emitted view
  (the conservative subset written to the private file); `SDK_FLAG_AUDIT` is a
  specification-reference view carrying only (value, static status, note) and
  NO emitted marker, so the two tables cannot disagree on emission. `unverified`
  means statically unconfirmed, NOT unsupported or inapplicable. Unknown flags
  (e.g. a `prefer_dn_subnets`-style knob not scanned locally) stay unknown until
  the full reviewed flag list and live evidence arrive.
- The `EicNetworkFlagFile` wrapper (`--flagfile=<source>` then NIC override)
  retains pending gflags path-parsing questions (whitespace/`#`) awaiting
  precise gflags evidence; no blind quoting applied. ParseInitFlags runs after
  the InitOption NIC assignment, so the trailing-override order is reasonable;
  1.5.2 has not been executed live.
