# DeepSeek-V4 Decode Context Parallel (DCP) Research

This folder contains the upstream PR patches that we analyzed before
implementing DCP for DSv4. The two PRs do **not** apply cleanly on top of
the bytedance `deepseek_v4` branch (different file layouts, different attention
backends, different memory pool, EAGLE/DSv4 hooks, etc.), so we keep them
here as reference rather than cherry-picking them directly.

## Files

- `pr_14194_mla_dcp.patch`
  Upstream PR #14194 "feature(dcp): use dcp for decode with full kv cache on each
  dcp rank" by augusto.yjh. Adds DCP infrastructure for MLA-style models
  (DeepSeek-V2/V3, R1). 120 commits, WIP.

- `pr_18167_dsa_dcp.patch`
  Upstream PR #18167 "feat: add DCP support for DeepSeek v3.2" by FENP.
  Adds DCP for DSA (DeepSeek Sparse Attention) models built on top of #14194's
  primitives. 5 commits, more polished.

## High-level analysis

### PR #14194 (MLA DCP)

Adds these primitives:

1. **Distributed group**:
   - `_DCP` GroupCoordinator in `parallel_state.py` (init_model_parallel_group
     with `group_name="dcp"`)
   - `get_dcp_group()`, `get_dcp_world_size()`, `get_dcp_rank()` accessors
   - `decode_context_parallel_size` argument to `initialize_model_parallel`
     and `ensure_model_parallel_initialized`
   - `GroupCoordinator.reduce_scatter_along_dim()` helper
   - Hook in `graph_capture()` so DCP NCCL streams are captured

2. **KV cache layout**:
   - Tokens are sharded across DCP ranks by `position % dcp_world_size`
   - New `DcpTokenToKVPoolAllocator` in `mem_cache/allocator.py` to distribute
     allocation
   - Page table indices have to be divided by `dcp_size` before indexing into
     the local KV cache

3. **Attention math (MLA)**:
   - Each DCP rank holds a *strided slice* of the KV cache for the same
     sequence and computes partial attention plus log-sum-exp (LSE)
   - All-gather Q-projection across DCP, run local MLA
     (`flashinfer_mla_backend.forward_decode` returns `(o, lse)`)
   - Use `correct_attn_out` Triton kernel + `cp_lse_ag_out_rs()` to merge
     partial outputs:
     1. all-gather LSEs across DCP ranks (`[N, B, H]`)
     2. correct each rank's `o` with `exp(local_lse - global_lse)`
     3. reduce-scatter the corrected `o` back into per-rank head shards
   - Number of attention heads per rank is multiplied by `get_dcp_world_size()`
     so that after all-gather of Q each rank computes a slice of all heads on
     a slice of the sequence, matching DCP semantics.

4. **Index update**:
   - `FlashInferMLAIndicesUpdaterDecode` filters `kv_indices` for tokens that
     belong to the local DCP rank: keep entries where
     `(token_pos - dcp_rank - 1) // dcp_size + 1` describes the local slice.
   - kv_indptr is recomputed from `cumsum(filtered_paged_kernel_lens)`.

5. **Server args / env**:
   - `SGLANG_DCP_WORLD_SIZE` env var (later versions add a `--dcp-size` CLI arg)

6. **Tests**:
   - `test/srt/test_dcp_interleaved_storage.py` validates allocation/free
     correctness with the interleaved layout.

### PR #18167 (DSA DCP)

Builds on #14194 and adapts to DeepSeek Sparse Attention (DSv3.2 / GLM-5):

1. **Server arg**: `--dcp-size` (CLI), persisted in `ServerArgs.dcp_size`,
   plumbed through `ensure_model_parallel_initialized`.

2. **NCCL tuning**: when `dcp_size > 1`, set `NCCL_GRAPH_MIXING_SUPPORT=0`
   to reduce per-replay event sync bubbles.

3. **NSA backend** (`nsa_backend.py`):
   - `self.dcp_size` / `self.dcp_rank` cached at backend construction
   - `_save_kv_cache()` passes `dcp_kv_mask` + `dcp_size` to `set_mla_kv_buffer`
     so only tokens belonging to the local DCP rank get written
   - All-gather Q (q_nope/q_rope or q_all) across DCP before NSA core kernels
   - `_forward_flashmla_sparse / _forward_flashmla_kv / _forward_fa3` all
     return softmax LSE (`return_softmax_lse=True`); merge with
     `cp_lse_ag_out_rs` when DCP > 1.

4. **NSA helpers**:
   - `transform_index_page_table_*`: when `dcp_size > 1`, divide page-table
     indices by `dcp_size` before sparse-attention gather (because the local
     pool stores only the rank's slice).
   - `dequantize_k_cache_paged`: same `// DCP_SIZE` adjustment in the Triton
     kernel.

5. **ForwardBatch**:
   - New field `dcp_kv_mask: Optional[torch.Tensor]` describing which tokens
     in `out_cache_loc` are owned by the local rank (used to filter writes).

6. **Memory pool / scheduler** integration:
   - `MHATokenToKVPool.set_mla_kv_buffer` learns `dcp_kv_mask` / `dcp_size`
   - Scheduler (`scheduler.py`, `scheduler_runtime_checker_mixin.py`) refuses
     conflicting configurations and shards kv allocation accordingly.
   - `mem_cache/utils.py` adds round-robin / interleaved index helpers.
   - `model_runner.py` propagates `dcp_size` to the model.

## DSv4 implementation plan

See `dsv4_implementation_plan.md` in this directory.
