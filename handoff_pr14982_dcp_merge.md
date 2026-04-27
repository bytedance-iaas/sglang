# PR 14982 DCP Merge Handoff

## Current Git Context

- Branch: `dcp-test-disagg-bootstrap`
- HEAD: `14102a6fe809efc433ee8773cfc1031866e6daf0`
- Base state note: local branch was reset to this commit before the current merge work started.

## Goal Of This Work

Port the runtime-critical parts of upstream PR 14982 into the current branch without replacing the current branch's newer structure.

Targeted behavior:

- expose DCP as a supported runtime option
- make FlashInfer GQA path DCP-aware
- make KV cache logical page layout DCP-aware
- make KV writes land only on the owning DCP shard
- disable fused KV fast path when DCP is enabled

## Files Changed In This Session

Modified files:

- `python/sglang/srt/server_args.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/models/utils.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/layers/attention/utils.py`
- `python/sglang/srt/layers/attention/flashinfer_backend.py`

New untracked test files:

- `test/registered/unit/model_executor/test_forward_batch_info_dcp.py`
- `test/registered/unit/models/test_utils_dcp.py`

## What Was Implemented

### 1. DCP parameter and model-parallel plumbing

- Added `dcp_size` to `ServerArgs`.
- Added CLI flags:
  - `--dcp-size`
  - `--decode-context-parallel-size`
- Wired `server_args.dcp_size` into `ModelRunner` as:
  - `self.dcp_size`
  - `self.dcp_rank`
- Passed `decode_context_parallel_size=self.dcp_size` into `initialize_model_parallel(...)`.

### 2. Scheduler logical page size

- Updated radix/cache init so logical page size becomes:
  - `page_size * dcp_size`

This follows PR 14982's key semantic requirement: DCP changes logical cache page layout even when physical KV buffers still use base page size.

### 3. ForwardBatch DCP KV layout contract

- Added `ForwardBatch.dcp_kv_mask`.
- In `ForwardBatch.init_new(...)`, when `model_runner.dcp_size > 1`:
  - compute `dcp_kv_mask = out_cache_loc % dcp_size == dcp_rank`
  - rewrite `out_cache_loc = out_cache_loc // dcp_size`

This is the key contract used by KV write path and FlashInfer decode/prefill path.

### 4. Disable fused set_kv_buffer under DCP

- Updated `enable_fused_set_kv_buffer(...)` to require `get_dcp_group().world_size == 1`.

This avoids using the fused KV write fast path in a layout where per-rank masked writes are required.

### 5. DCP-aware KV writes in memory pool

- Added Triton kernel `masked_set_kv_buffer_kernel(...)`.
- Extended `MHATokenToKVPool.set_kv_buffer(...)` with optional `dcp_kv_mask`.
- When `dcp_kv_mask` is provided, only write KV rows for locally-owned DCP positions.

### 6. DCP-aware FlashInfer path

Implemented in `python/sglang/srt/layers/attention/flashinfer_backend.py`:

- store `self.dcp_size` and `self.dcp_rank` from `get_dcp_group()`
- make decode tensor-core heuristic use `num_attention_heads * dcp_size`
- require ragged wrapper for DCP prefill path
- for prefill merged path:
  - all-gather `q` across DCP ranks
  - run paged branch with expanded Q head count
  - aggregate outputs via `cp_lse_ag_out_rs(..., return_lse=True)`
  - write KV using `dcp_kv_mask`
- for decode path:
  - all-gather `q`
  - call `forward_return_lse(...)`
  - aggregate outputs via `cp_lse_ag_out_rs(...)`
  - write KV using `dcp_kv_mask`
- for FlashInfer decode/prefill index generation:
  - DCP-adjust per-request local lengths
  - generate DCP-local kv indices
  - scale paged wrapper head count by `dcp_size`
  - adjust CPU-side override indptr for DCP-local lengths

### 7. Attention utility helpers

- Added `create_flashinfer_kv_indices_for_dcp_triton(...)`.
- Extended `cp_lse_ag_out_rs(...)` with `return_lse` support.

## Validation Performed

### Python compile check

Executed:

```bash
python3 -m py_compile \
  python/sglang/srt/models/utils.py \
  python/sglang/srt/mem_cache/memory_pool.py \
  python/sglang/srt/layers/attention/utils.py \
  python/sglang/srt/layers/attention/flashinfer_backend.py \
  python/sglang/srt/model_executor/forward_batch_info.py \
  python/sglang/srt/model_executor/model_runner.py \
  python/sglang/srt/managers/scheduler.py \
  python/sglang/srt/server_args.py
```

Result: passed.

### Unit tests

Executed:

```bash
python3 -m pytest \
  test/registered/unit/model_executor/test_forward_batch_info_dcp.py \
  test/registered/unit/models/test_utils_dcp.py -q
```

Result:

- `3 passed`
- warnings only, no failures

## Important Limitations / Risks

- This session focused on the FlashInfer GQA DCP path, not every possible backend.
- No real GPU end-to-end server launch or inference smoke test has been run yet in this session.
- The two new test files are still untracked.
- Changes are currently unstaged and uncommitted.

## Recommended Next Step

Run one minimal GPU smoke test for DCP decode before committing.

Suggested direction:

- start a single-node configuration that exercises `--dcp-size 2`
- verify server startup
- run a short decode request
- confirm no KV write mismatch or FlashInfer shape issue occurs

## Latest Follow-Up Status

This section captures follow-up review work done after the original handoff draft.

### Confirmed fixes since the first handoff

- `DecodeInputBuffers.create(...)` now accepts `dcp_size`.
- CUDA-graph decode buffers now allocate/copy `dcp_kv_mask`.
- `test/registered/unit/model_executor/test_forward_batch_info_dcp.py` was fixed to use a non-idle path and now actually exercises the DCP rewrite.
- `python/sglang/srt/layers/attention/utils.py` DCP Triton index remap now uses integer division (`//`) instead of `/`.

### Follow-up validation done

- Narrow unit tests passed:

```bash
python3 -m pytest \
  test/registered/unit/model_executor/test_forward_batch_info_dcp.py \
  test/registered/unit/models/test_utils_dcp.py -q
```

Result: `3 passed`

- Narrow compile checks passed for:

```bash
python3 -m py_compile \
  python/sglang/srt/layers/attention/flashinfer_backend.py \
  python/sglang/srt/layers/attention/utils.py \
  python/sglang/srt/managers/scheduler.py \
  python/sglang/srt/managers/scheduler_runtime_checker_mixin.py \
  python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py
```

### Important scope clarification from the user

- Do not extend DCP support into MTP/speculative paths for now.
- Specifically treat `is_draft_extend` and `is_target_verify` as out of scope for DCP support in this round.
- The target remains: prefill does not need DCP support, decode must work with DCP.

### Important remaining risk

- Normal extend still has one main-path configuration conflict in `python/sglang/srt/layers/attention/flashinfer_backend.py`:
  - the non-MTP extend branch can still pass `dcp_size` / `dcp_rank` into prefill planning while some subcases force `use_ragged=False`
  - `FlashInferIndicesUpdaterPrefill.call_begin_forward(...)` asserts that DCP only supports the ragged wrapper
  - this means `DCP + normal extend + multimodal` or `DCP + normal extend + multi-item scoring` can still fail with the ragged-wrapper assertion

### Scheduler/token-capacity accounting note

- In `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`, DCP pool setup already multiplies `self.max_total_num_tokens` by `get_dcp_world_size()` before constructing `DcpTokenToKVPoolAllocator`.
- `DcpTokenToKVPoolAllocator` also treats its `size`/`available_size()` as logical DCP-expanded token capacity.
- Because of that, scheduler/runtime reporting must not multiply `max_total_num_tokens` by `dcp_size` again.
- The scheduler startup log was corrected accordingly in `python/sglang/srt/managers/scheduler.py`.

## How To Continue In A New Chat

Start the next conversation with something like:

```text
先读 handoff_pr14982_dcp_merge.md，然后继续。
当前分支是 dcp-test-disagg-bootstrap。
如果工作区有未提交改动，不要覆盖，先基于现状继续。
优先做 GPU smoke test；如果失败，直接定位 flashinfer DCP 路径。
```

Or, if you want a slightly fuller prompt:

```text
先读 handoff_pr14982_dcp_merge.md。
当前分支 dcp-test-disagg-bootstrap，HEAD 14102a6fe809efc433ee8773cfc1031866e6daf0。
当前工作区里已经移植了一版 PR 14982 的 DCP/GQA 关键语义，但还没做 GPU 端到端验证。
请在不覆盖现有未提交改动的前提下继续，先跑最小 GPU smoke test，再决定是否要修补 flashinfer DCP 路径。
```

## If You Want A Durable Save Point

The strongest persistence option is:

1. keep this markdown file in the repo
2. commit the current work on your branch
3. next time, ask to read this file first

That combination is more reliable than relying on chat history alone.