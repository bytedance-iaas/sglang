# DSv4 DCP Implementation Plan

This document lists the concrete steps required to bring Decode Context Parallel
(DCP) to the bytedance `deepseek_v4` branch. It is derived from PR #14194
(MLA DCP) and PR #18167 (DSA DCP) but adapted to the DSv4 codebase.

## Why a fresh implementation rather than a cherry-pick

- DSv4 already has its own attention backend (`deepseek_v4_backend.py`) with a
  paged KV layout (`page_size=256`), SWA hybrid pool, EAGLE v2 draft-extend
  CUDA graph runner, and DSv4-specific scheduler hooks (`deepseek_v4_hook.py`).
- Both upstream PRs target `flashinfer_mla_backend.py` / `nsa_backend.py` and
  the unpatched `mem_cache/allocator.py` / `memory_pool.py`. None of these
  files match DSv4 exactly.
- PR #14194 is still WIP (120 commits, contains TODOs and Chinese comments
  inside the model code).

## Phased plan

### Phase 1 — Distributed group + server args (mechanical, no behavior change)

Files to edit:

- `python/sglang/srt/distributed/parallel_state.py`
  - Add `_DCP: Optional[GroupCoordinator]`
  - Add `get_dcp_group()`, `get_dcp_group_no_assert()`,
    `get_dcp_world_size()`, `get_dcp_rank()`
  - Add `decode_context_parallel_size` parameter to
    `initialize_model_parallel` and `ensure_model_parallel_initialized`,
    plumbed from `ServerArgs.dcp_size`
  - Build per-DCP-group ranks (consecutive `dcp_size` ranks form one group)
  - Hook `_DCP` into `graph_capture()` and `destroy_model_parallel()`
  - Add `GroupCoordinator.reduce_scatter_along_dim()` helper (copied from
    PR #14194 verbatim)

- `python/sglang/srt/server_args.py`
  - Add `dcp_size: int = 1` field with `--dcp-size` CLI argument
  - Validation: `tp_size % dcp_size == 0`, `dcp_size >= 1`,
    cannot combine with PP > 1, must equal 1 when EAGLE/Eagle3 enabled if
    we don't yet support speculative + DCP (TBD).
  - Plumb to `ServerArgs.print_args` summary and to `ensure_model_parallel_initialized`.

- `python/sglang/srt/entrypoints/engine.py`
  - When `server_args.dcp_size > 1`, set `NCCL_GRAPH_MIXING_SUPPORT=0` if
    not already set (PR #18167).

- `python/sglang/srt/managers/scheduler.py` and
  `scheduler_runtime_checker_mixin.py`
  - Pass `dcp_size` into the model runner / KV cache allocator.

Acceptance: with `--dcp-size 1` the system should be byte-identical to today;
with `--dcp-size 2` (and TP%2==0) `get_dcp_group()` returns a working NCCL
group, but no attention math changes yet.

### Phase 2 — KV cache layout (`token % dcp_size == dcp_rank`)

Files to edit:

- `python/sglang/srt/mem_cache/allocator.py`
  - For DSv4 paged allocator (`PagedTokenToKVPoolAllocator` and the SWA
    variant), introduce a DCP-aware allocation policy:
    - When `dcp_size > 1`, each rank owns only the slots whose token index
      satisfies `slot % dcp_size == dcp_rank`. The simplest implementation is
      to keep the allocator identical, but track a `dcp_kv_mask` per write so
      that the actual `set_mla_kv_buffer` only writes positions belonging to
      this rank.
  - Match upstream's `DcpTokenToKVPoolAllocator` semantics with paged
    arithmetic: `local_index = token_index // dcp_size` after masking.
  - Update `free()` / `free_swa()` to translate global indices to local before
    actually decrementing the bitmap.

- `python/sglang/srt/mem_cache/memory_pool.py`
  - Teach `MHATokenToKVPool.set_mla_kv_buffer` to accept
    `dcp_kv_mask` + `dcp_size` and only write masked positions, mirroring
    PR #18167.
  - Allocate `kv_buffer` with size `pool_size // dcp_size` when DCP > 1
    (each rank stores a strided slice).

- `python/sglang/srt/mem_cache/utils.py`
  - Add interleaved-index helpers used by the scheduler when computing
    `dcp_kv_mask` and per-rank kv lengths
    (`(seq_len - dcp_rank - 1) // dcp_size + 1`).

- `python/sglang/srt/model_executor/forward_batch_info.py`
  - Add `dcp_kv_mask: Optional[torch.Tensor]` to `ForwardBatch`. Populate it
    in `prepare_for_extend` / `prepare_for_decode` when DCP is on.

- `python/sglang/srt/model_executor/cuda_graph_runner.py`
  - Allocate a static buffer for `dcp_kv_mask` and copy at replay time, just
    like `out_cache_loc`.

Acceptance: with DCP on, total KV memory per GPU drops to roughly `1/dcp_size`
of the previous footprint, and writes only target the local rank's slice.

### Phase 3 — DSv4 attention backend (the real math)

Target file: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`.

For each forward path that runs MLA decode (and DSv4's MTP/spec-decode draft
extend that uses MLA-decode kernels), do:

1. Cache `self.dcp_size = get_dcp_group().world_size`,
   `self.dcp_rank = get_dcp_group().rank_in_group` at backend init.

2. In `__init__`, when `dcp_size > 1` multiply
   `self.num_local_heads *= dcp_size` for the metadata builder so that after
   the all-gather Q has the right head dimension per rank.

3. In `init_forward_metadata*` (decode + draft-extend + replay variants):
   - When `dcp_size > 1`, filter `kv_indices` to only the slots belonging to
     `self.dcp_rank` using the same formula as PR #14194:
     `paged_kernel_lens_split = (paged_kernel_lens - dcp_rank - 1) // dcp_size + 1`
     and recompute `kv_indptr = cumsum(paged_kernel_lens_split)`.
   - Translate the page-table values by `// dcp_size` before passing them into
     the local kernel (matches PR #18167's `transform_index_page_table_*` and
     `dequantize_k_cache_paged` adjustments).

4. In `forward_decode`:
   - `q_all = get_dcp_group().all_gather(q_all, dim=1)` (across heads)
   - Run local FlashMLA / DSv4 MLA kernel with `return_softmax_lse=True`,
     getting `(o, lse)` shaped `[B, H_full, V]` and `[B, H_full]`.
   - Call `cp_lse_ag_out_rs(o, lse, get_dcp_group())` to:
     - all-gather LSEs across DCP ranks
     - correct `o` with `exp(local_lse - global_lse)`
     - reduce-scatter the merged `o` back to local heads
   - Return the merged tensor as today.

5. In `forward_extend` (draft-extend): same pattern, plus call
   `_save_kv_cache(layer, forward_batch, k, k_rope)` that forwards
   `dcp_kv_mask`/`dcp_size` so the local rank only writes its slice.

6. Add the Triton kernel + helpers from PR #14194 to
   `python/sglang/srt/layers/attention/utils.py`:
   - `_correct_attn_cp_out_kernel`
   - `CPTritonContext`
   - `correct_attn_out`
   - `cp_lse_ag_out_rs`

### Phase 4 — EAGLE / draft-extend integration

DSv4 runs EAGLE v2 with its own draft-extend CUDA graph runner. We must:

- Make sure `eagle_draft_extend_cuda_graph_runner.replay()` zeros the
  `dcp_kv_mask` padding region the same way we just did for `input_ids`
  and `req_pool_indices`.
- Pass `dcp_size` / `dcp_rank` through the spec-decode draft path so the
  draft model also splits and merges attention exactly the same way.
- Verify that EAGLE's verify path uses the merged `o` (not the per-rank
  partial) when feeding next-token sampling.

### Phase 5 — Scheduler / DSv4 hook

- `arg_groups/deepseek_v4_hook.py` should refuse incompatible combinations
  (e.g. `dcp_size > 1` + `--swa-full-tokens-ratio < 1` until SWA is verified
  with strided slicing).
- `model_executor/pool_configurator.py` must size pools to
  `total_kv / dcp_size` when DCP is on.
- `model_runner_kv_cache_mixin.py` must propagate `dcp_size` when constructing
  the kv pool allocator.

### Phase 6 — Tests & validation

- Port `test/srt/test_dcp_interleaved_storage.py` to test the DSv4 paged
  allocator behavior with DCP.
- Add a numerical equivalence test:
  - Run a short prompt with `dcp_size=1`
  - Run the same prompt with `dcp_size=2`
  - Logits / generated tokens must match within a small tolerance
- Add a smoke test for `--dcp-size 2` startup that exercises:
  - kv pool sizing
  - cuda graph capture
  - one forward decode step

### Phase 7 — Rollout flags

- Gate behind `SGLANG_DSV4_ENABLE_DCP=1` initially so production DSv4 runs are
  unaffected until the feature is fully validated.
- Add a small section to `docs/dcp_research/README.md` describing the
  recommended startup command, e.g.:

  ```
  --tp-size 8 --dcp-size 2 \
  --enable-mla --attention-backend deepseek-v4 \
  --speculative-algorithm EAGLE3 \
  --swa-full-tokens-ratio 1.0
  ```

## Risk notes

1. **EAGLE + DCP** is not implemented in either upstream PR; we will be
   first. Expect issues around the draft-model attention path that uses MLA
   decode kernels with `num_tokens_per_bs > 1`.
2. **SWA + DCP** also unverified upstream. The SWA radix cache stores tokens
   keyed on `out_cache_loc`; with DCP the local cache only sees a strided
   slice and cache hits across ranks would diverge. Disable SWA caching for
   DCP in v1, or perform an all-gather of the SWA mapping.
3. **CUDA graph capture** must include DCP NCCL ops (PR #14194 hooks
   `get_dcp_group().graph_capture(context)`). Verify the `replay()` paths
   we just fixed continue to work when extra collectives are inside the graph.
4. **Page size 256** in DSv4 means token-level interleaving (`token % dcp_size`)
   would interleave inside a page, which is illegal for paged FlashMLA.
   We will likely need page-level interleaving instead: a page belongs to
   rank `(page_index % dcp_size)`. Adjust filter formulas accordingly.

## Suggested branch / PR strategy

- Each phase = one PR / branch on top of the previous one.
  - `bytedance/deepseek_v4_dcp_phase1_groups`
  - `bytedance/deepseek_v4_dcp_phase2_kv_layout`
  - `bytedance/deepseek_v4_dcp_phase3_attention`
  - `bytedance/deepseek_v4_dcp_phase4_eagle`
  - `bytedance/deepseek_v4_dcp_phase5_scheduler`
  - `bytedance/deepseek_v4_dcp_phase6_tests`
- Final integration branch: `bytedance/deepseek_v4_dcp`.
