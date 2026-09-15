# EIC patch wiring checklist

Single patch commit on `cklxx/eic-patch`, base recorded in
`.eic-patch/base-commit.txt`. The reference diff is
`.eic-patch/eic-ep_main.patch` (`git diff origin/ep_main <patch-commit>`).

## Re-applying after an ep_main refresh

```
git fetch origin
git checkout -b rebuild origin/ep_main
git apply --3way .eic-patch/eic-ep_main.patch
# resolve only the ep_main-owned files (below); the EIC-owned files apply untouched
```

`--3way` reports every conflict as ours/theirs. ours = new ep_main (keep it as
the base), theirs = the patch (replay only its EIC addition). Do not rebase a
1500-commit-older tree: the merge driver emits multi-thousand-line false
alignments when a function moved into a different module.

Last full refresh: 2026-09-15, 8ab9652e7c -> f4c61f324b (1525 commits).
Every merged EIC PR (#481..#768, incl. the #709 deploy check) is present; see EIC_PATCH.md.
19 hunks conflicted; resolution rules are per file below.

## Files

### EIC-owned (apply untouched, never conflict unless upstream creates the path)

| File | Role |
|---|---|
| `managers/eic_cache_controller.py` | write/load thread + queues on top of `HiCacheController` |
| `mem_cache/eic_hiradix_cache.py` | `EICHiRadixCache`/`EICPagedHiRadixCache`: remote match, async load-admit, PP verdicts, `prefix_loading` |
| `mem_cache/eic_memory_pool.py` | EIC client, host pools (MHA/MLA/NSA/DSv4), `FlexibleKVCacheMemoryPool` for the hiradix path |
| `mem_cache/eic_chunk_cache.py` | `EICChunkCache`/`EICSWAChunkCache` for `--disable-radix-cache` (PD decode-save) |
| `mem_cache/eic_pp_reconcile.py` | cross-PP load-length reconciler |
| `test/registered/unit/mem_cache/test_eic_hicache_regression.py` | EIC regression suite |
| `scripts/eic_integration_check.py` | post-deploy EIC integration check (#709); standalone ops script, no runtime import |
| `benchmark/hicache/eic_snapshots/2026-09-12-...md` | serving benchmark record |

### ep_main-owned touch points (these are the only files that conflict)

| File | EIC addition | Refresh rule |
|---|---|---|
| `server_args.py` | fields `enable_eic_cache`, `disable_eic_shared` in the hicache section | add fields next to `hicache_host_memory_mode`; do not re-add `hicache_ratio` (upstream owns it) |
| `arg_groups/hicache_hook.py` | `handle_hicache` declares `enable_hierarchical_cache=True` when `enable_eic_cache`, before the skip test | the old `__post_init__` force-on moved here after the ServerArgs resolution-pipeline refactor; write through `declare_resolution`, never set the field directly |
| `mem_cache/registry.py` | EIC SWA dedup flag; EIC chunk-cache and `EICHiRadixCacheBuilder` dispatch ahead of hybrid/DSA arms | keep both early branches (host-pool retraction and EIC dedup); EIC dispatch stays first |
| `managers/scheduler.py` | imports; `enable_eic_cache` flags narrowed to `EICHiRadixCache`; async gate + `prefix_loading` skip; `release_load_admit` on three abort sites; `enable_eic_cache` into PrefillAdder | read resolved flags via `get_memory()`; keep both the new `buffer_only` staged-splice charge and the EIC async gate; abort helper is `_make_abort_req` (not `AbortReq`) |
| `managers/schedule_policy.py` | `enable_eic_cache` ctor param; under EIC do not subtract `host_hit_length` and skip `init_load_back` | keep both `prefill_tile_block_m` and the EIC param |
| `managers/schedule_batch.py` | pass `release_cache_protected_prefix` (from the `swa_evict_release_prefix` hook) into `free_swa_out_of_window_slots` | keep `retain_floor` too; both kwargs coexist |
| `mem_cache/common.py` | `free_swa_out_of_window_slots(..., release_cache_protected_prefix)`: ignore the protected floor and read slots from `req.prefix_indices` for the EIC span | keep upstream `retain_floor`, `req.kv.*`, and the `UnifiedSWATokenToKVPoolAllocator` `start_pos` branch; slot source is the only EIC divergence |
| `mem_cache/allocator/swa.py` | `dedup_aliased_swa` reserved-page skip + unique in `free_swa` | apply the filter before `clear_full_to_swa_mapping`, preserve the `free_group` deferral branch |
| `mem_cache/hybrid_cache/hybrid_pool_assembler.py` | `device_indexed` param on `build_deepseek_v4_hicache_stack` and `_deepseek_v4_num_host_pages` (cap host pages to device pages) | read `hicache_ratio` via `get_memory()`, not `server_args.*`; internal call forwards `device_indexed` |
| `mem_cache/radix_cache.py` | `TreeNode.content_hash` field, default None | additive only |
| `distributed/communication_tags.py` | `P2PTag.HIRADIX_PP_VERDICT` | additive only |
| `mem_cache/storage/eic/eic_storage.py` | fork of the community L3 backend (+235 lines): PP fields, registered-pool map, logical-anchor zero-copy, refetch | this file exists upstream and is registered in `storage/backend_factory.py` for `--hicache-storage-backend=eic`; the patch extends it. Re-resolve against upstream EICStorage every refresh - it is the most likely file to semantically drift even when it applies |
| `test/registered/unit/mem_cache/test_registry.py` | EIC registry cases publish through `_publish(...)`, not `MagicMock` | add `enable_eic_cache=` to the `_publish` kwargs |

## Two EIC storage implementations (not a duplicate)

- `--enable-eic-cache` (hiradix path): `eic_memory_pool.py::FlexibleKVCacheMemoryPool`
  + `EICKVClient`, used by `eic_cache_controller`/`eic_hiradix_cache`/`eic_chunk_cache`.
- `--hicache-storage-backend=eic` (community L3 path): `storage/eic/eic_storage.py`,
  loaded by `storage/backend_factory.py`. It carries its own
  `FlexibleKVCacheMemoryPool` and `EICStorage`. The patch extends this file.

## Post-rebase verification

1. No conflict markers; every touched file parses (`python3 -m py_compile`).
2. `pre-commit run --files <changed>` is clean.
3. EIC-off invariant: every hunk in an ep_main-owned file is gated on
   `enable_eic_cache`, an EIC-only hook, or is an additive default-None field/new
   enum member. Additive-only files: `radix_cache.py`, `communication_tags.py`.
4. Load contract: `managers/cache_controller.py::HiCacheController.move_indices`
   still moves host indices to GPU for `io_backend == "kernel"`; EIC
   `EICDeepSeekV4TokenToKVPoolHost.device_writeback` mirrors it. If that method
   moves or changes, the EIC copy must change too.
5. In-pod: `test/registered/unit/mem_cache/test_eic_hicache_regression.py` and
   `test_registry.py` EIC cases pass. Local laptops cannot import sglang
   (transformers mismatch), so run these in the validation pod.

## Not verified on this refresh

- Unit/in-pod tests not yet run on f4c61f324b base.
- MTP (#750-equivalent on new base) EIC page layout not run.
- Real-model e2e and throughput not rerun; the 2026-09-12 snapshot is from the
  older base.

## Refactor-contract fixes (2026-09-15 review, f4c61f324b)

The 1525-commit refresh crossed two large refactors; text resolution was not
enough. These runtime crashes were caught by reading the merged code, not the
compiler:

- DSv4 `device_backup`/`device_writeback` must iterate
  `host_pool_group.get_entry(name)` and call per-pool
  `backup_from_device_all_layer`/`load_to_device_per_layer` (new signatures:
  no `pool_transfers=`, load uses `entry.layer_mapper`). The old group-level
  methods no longer exist.
- `req.req_pool_idx`/`req.cache_protected_len` -> `req.kv.*`; `req.fill_ids` ->
  `req.get_fill_ids()`.
- Test imports moved under `unified_cache/`.

These are why in-pod tests are required after every large refresh even when the
diff applies and lints clean.
