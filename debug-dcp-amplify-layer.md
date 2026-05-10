[OPEN] dcp-amplify-layer

## Goal
- Use the least additional instrumentation to locate the first decoder stage where the small `DCP on block0` drift becomes materially larger than `DCP off`.
- Keep the current decode path and remap behavior unchanged while collecting evidence.

## Scope
- Decode only
- Shared-history steps `seq_len=[2..5]`
- Compare `DCP off` vs `DCP on block0`

## Falsifiable Hypotheses
1. Layer-0 `post_moe` is already the first clear amplification point, and later divergence is just a continuation.
2. Layer-0 stays close, while layer-1 `post_moe` is the first place where the hidden-state delta becomes obviously larger.
3. Early decoder layers stay close, and the first large drift appears only near the final logits path.
4. The current apparent amplification is mostly caused by an unhelpful diagnostic summary, not a real hidden-state blow-up.

## Plan
1. Reuse existing decode diagnostics where possible.
2. Add only one narrow layer-selection control for `post_moe` logging.
3. Run one `DCP off/on` comparison and identify the earliest clearly amplified layer.
4. Record the key evidence here before considering any logic fix.

## Instrumentation Added
- `python/sglang/srt/models/minimax_m2.py`
  - Added env gate `SGLANG_DCP_DIAG_MM2_POST_MOE_LAYER`
  - Added `_should_log_decoder_post_moe_diag()` so `post_moe` can target one decoder layer without reopening all layer-0 attention logs
  - Upgraded `post_moe` log from only `shape/prefix` to:
    - local `hidden_shape`
    - local `hidden_prefix`
    - ranked fingerprint summary `numel/sum/absmax/l2/prefix`

## Intended Run Pattern
- First run with `SGLANG_DCP_DIAG_MM2_POST_MOE_LAYER=1`
- If `layer1` is still close, move the same env var forward to the next layer instead of widening all layers at once

## 2026-05-09 Dual-Node Run
- Setup:
  - dual-node `TP16`, `DCP off/on`
  - `DISABLE_CUDA_GRAPH=1`
  - `SGLANG_DCP_DIAG_MM2_POST_MOE_LAYER=1`
  - prompt=`Hello`, `max_tokens=4`, `temperature=0`
- Response:
  - `DCP off`: `"\"?\n\nThe answer"`
  - `DCP on`: `"战机\n\n\n\n\n\n"`

## Evidence

### Layer-0 `o_proj` remains relatively close
- `seq_len=[3]`
  - off: `sum=1.848095 absmax=1.679688 l2=1.711530`
  - on: `r0 sum=1.643698 absmax=1.609375 l2=1.642857`
- `seq_len=[4]`
  - off: `sum=1.949422 absmax=1.968750 l2=1.988650`
  - on: `r0 sum=1.565374 absmax=1.593750 l2=1.637548`
- Interpretation:
  - layer-0 projection output is not identical, but the drift is still moderate in magnitude

### Layer-1 `post_moe` is the first clearly amplified point
- `seq_len=[3]`
  - off prefix: `[0.067871, 0.04248, 0.004211, 0.019043, 0.021729, 0.03064, 0.056885, -0.02124]`
  - on prefix: `[-0.042969, -0.043945, 0.034668, 0.124023, 0.031738, -0.046387, -0.253906, 0.000122]`
  - norms are similar, but the local coordinates already differ substantially
- `seq_len=[4]`
  - off: `sum=0.188810 absmax=1.156250 l2=1.594210`
  - on: `sum=6.003265 absmax=2.312500 l2=4.147174`
  - this is the first stage where the hidden-state drift becomes obviously much larger in global magnitude
- `seq_len=[5]`
  - off: `sum=2.020905 absmax=0.921875 l2=2.108745`
  - on: `sum=6.163872 absmax=1.976562 l2=3.390564`
  - the amplified drift persists

### Token divergence aligns with the layer-1 finding
- `DCP off`
  - `seq_len=[3]` token `758`
  - `seq_len=[4]` token `4941`
  - `seq_len=[5]` token `355`
- `DCP on`
  - `seq_len=[3]` token `367`
  - `seq_len=[4]` token `367`
  - `seq_len=[5]` token `367`
- Interpretation:
  - by the time decode reaches `seq_len=[4]`, the model has already fallen into a different token attractor

## Conclusion
- Hypothesis 2 is currently strongest:
  - layer-0 remains near-correct
  - layer-1 `post_moe` is the first observed stage with materially amplified hidden-state drift
- The next minimal-debug direction should target the layer-1 input path:
  - compare `layer1 prepare_attn` / `layer1 self_attn output`
  - check whether amplification starts before `block_sparse_moe` or inside the layer-1 residual + MoE path

## 2026-05-09 One-Pass Follow-up

### Additional instrumentation
- `python/sglang/srt/models/minimax_m2.py`
  - Added unified stage logger for the selected decoder layer:
    - `layer_input`
    - `after_prepare_attn`
    - `after_self_attn`
    - `after_prepare_mlp`
    - `post_moe`
    - `postprocess_layer`
  - Each stage now logs:
    - `hidden_shape/prefix`
    - optional `residual_shape/prefix`
    - ranked fingerprint `numel/sum/absmax/l2/prefix`

### New finding: first large amplification happens before `post_moe`
- Same setup as above, still targeting `layer=1`
- For `seq_len=[4]`, `TP0` comparison:
  - `after_prepare_attn`
    - off: `sum=0.333951 absmax=0.136719 l2=0.314934`
    - on: `sum=1.044527 absmax=0.263672 l2=0.575797`
    - conclusion: drift exists but is still moderate
  - `after_self_attn`
    - off: `sum=0.051130 absmax=0.142578 l2=0.216717`
    - on: `r0 sum=1.208519 absmax=0.053955 l2=0.721894`
    - conclusion: attention output is different, but not yet the largest blow-up
  - `after_prepare_mlp`
    - off: `sum=3.775162 absmax=5.156250 l2=6.932152`
    - on: `sum=-0.600604 absmax=5.062500 l2=15.939655`
    - conclusion: this is the first stage where the magnitude gap becomes decisively large
  - `post_moe`
    - off: `sum=0.188810 absmax=1.156250 l2=1.594210`
    - on: `sum=6.003265 absmax=2.312500 l2=4.147174`
    - conclusion: `post_moe` remains wrong, but the big amplification already happened upstream

### Refined localization
- The earliest clearly amplified point is no longer just “`layer1 post_moe`”.
- It is more precisely:
  - after `layer1 self_attn`, before or inside `layer1 prepare_mlp`
- Code inspection aligns with this:
  - `LayerCommunicator.prepare_mlp()` calls `_communicate_with_all_reduce_and_layer_norm_fn(...)`
  - this path applies `post_attention_layernorm` and residual handling before MoE

## Current Best Hypothesis
- The remaining DCP decode drift enters layer 1 as a still-manageable attention delta.
- `layer1 prepare_mlp` then amplifies that delta through residual merge + post-attention norm / communication.
- `block_sparse_moe` is no longer the earliest culprit; it inherits an already amplified input.

## 2026-05-09 `prepare_mlp` Internal Split

### Additional instrumentation
- `python/sglang/srt/layers/communicator.py`
  - Added narrow internal logs for `prepare_mlp` around:
    - `after_attn_allreduce`
    - `before_layernorm`
    - `after_residual_add`
    - `after_layernorm`
  - Passed `layer_id` from `MiniMaxM2DecoderLayer` into `LayerCommunicator` so the internal path can reuse the same layer-1-only gate.

### One-pass result
- Setup:
  - same dual-node `TP16`
  - decode only
  - `DISABLE_CUDA_GRAPH=1`
  - target `layer=1`
  - compare:
    - `DCP off`: `/data01/code/dcp_prepare_mlp_inner_off_rerun/off_node0.log`
    - `DCP on`: `/data01/code/dcp_prepare_mlp_inner/on_node0.log`
- Response sanity:
  - `DCP off`: `"\"?\n\nThe answer"`
  - `DCP on`: `"战机\n\n\n\n\n\n"`

### Key comparison at `seq_len=[4]` (`TP0`)
- `after_attn_allreduce`
  - off hidden: `sum=5.890884 absmax=6.250000 l2=6.426170`
  - on hidden: `sum=14.304817 absmax=12.312500 l2=13.319541`
  - interpretation: drift is present before residual merge, but this is not yet the decisive amplification point
- `after_residual_add`
  - off hidden: `sum=29.431946 absmax=30.250000 l2=30.316614`
  - on hidden: `sum=33.447495 absmax=28.750000 l2=29.271069`
  - interpretation: once the attention output is merged into the residual stream, `off/on` still stay in the same overall magnitude band
- `after_layernorm`
  - off hidden: `sum=3.775162 absmax=5.156250 l2=6.932152`
  - on hidden: `sum=-0.600604 absmax=5.062500 l2=15.939655`
  - interpretation: this is the first place where the norm gap becomes decisively large

### Refined conclusion
- The first decisive amplification inside `layer1 prepare_mlp` does **not** happen at residual addition itself.
- The residual-added tensor is already different, but its global magnitude remains close between `DCP off` and `DCP on`.
- The large blow-up appears at the subsequent `post_attention_layernorm` output.
- `after_prepare_mlp` in the model-level log matches `after_layernorm` exactly, so the previously observed `after_prepare_mlp` amplification is now localized to:
  - `layer1 prepare_mlp -> post_attention_layernorm`

### Implication for the next fix
- The remaining DCP decode error enters `layer1 prepare_mlp` as a moderate attention-side drift.
- Residual merge carries that drift forward but does not yet create a major norm explosion.
- `post_attention_layernorm` is the step that converts the still-manageable drift into a much larger hidden-state divergence that later corrupts MoE and logits.

## 2026-05-09 RMSNorm Preview Follow-up

### Added evidence
- Added `rmsnorm_preview` logging inside `prepare_mlp` to compute the RMSNorm path without changing the real tensor flow:
  - `rms`
  - `residual_add`
  - `norm_no_weight`
  - `norm_weighted`
  - `residual_add_topk`
  - `hidden_topk`
  - `residual_topk`
  - `norm_weighted_topk`
  - `weight`

### Key `DCP on` finding
- At `layer=1`, `seq_len=[4]`, the abnormal dimension is clearly coordinate `1531`:
  - `hidden_topk`: `(1531, 12.3125)`
  - `residual_topk`: `(1531, 16.375)`
  - `residual_add_topk`: `(1531, 28.75)`
  - `norm_weighted_topk`: `(1531, 5.077062)`
- The RMS denominator is not itself anomalous:
  - `rms=0.528115`
  - `norm_no_weight l2=55.425522`, as expected for RMSNorm over 3072 dims
- The manual `norm_weighted` preview matches the actual `after_layernorm` output:
  - preview: `sum=-0.573499 absmax=5.077062 l2=15.913784`
  - actual: `sum=-0.600604 absmax=5.062500 l2=15.939655`

### Sequence trend
- The same coordinate is already dominant from the first decode steps:
  - `seq_len=[2]`: `hidden(1531)=15.0`, `residual(1531)=26.5`, `residual_add(1531)=41.5`
  - `seq_len=[3]`: `hidden(1531)=11.625`, `residual(1531)=17.0`, `residual_add(1531)=28.625`
  - `seq_len=[4]`: `hidden(1531)=12.3125`, `residual(1531)=16.375`, `residual_add(1531)=28.75`
  - `seq_len=[5]`: `hidden(1531)=13.0625`, `residual(1531)=16.125`, `residual_add(1531)=29.25`

### Refined interpretation
- `post_attention_layernorm` is numerically doing what RMSNorm should do.
- The amplification is not caused by an abnormal RMS denominator or layernorm weight corruption.
- The deeper problem is that `DCP on` has already produced a persistent semantic outlier at coordinate `1531` before layernorm:
  - it exists in the residual stream
  - it also appears in the attention output
  - residual addition makes them align and dominate the layernorm input
- Next useful localization is to trace coordinate `1531` upstream:
  - layer1 `after_prepare_attn` residual/input
  - layer1 `after_self_attn`
  - layer0 `postprocess_layer`
  - layer0 `post_moe`
  - determine exactly where coordinate `1531` first becomes the dominant outlier under `DCP on`

## 2026-05-09 Coordinate `1531` Upstream Trace

### Added evidence
- Added model-level stage trace for `layer0..layer1` with:
  - fixed coordinate value at `coord=1531`
  - hidden/residual top-abs coordinates
  - gate controlled by `SGLANG_DCP_DIAG_MM2_TRACE_UP_TO_LAYER=1`
- Run:
  - `DCP on`
  - dual-node `TP16`
  - `max_tokens=4`
  - logs: `/data01/code/dcp_coord_trace/on_node0.log`
- Output still reproduces corruption:
  - `"战机\n\n\n\n\n\n"`

### Key trace
- `DCP on`, `seq_len=[4]`:
  - `L0 layer_input`: `coord1531=-0.005737`, not top-k
  - `L0 after_prepare_attn`: hidden `coord1531=-0.003296`, residual `coord1531=-0.005737`, not top-k
  - `L0 after_self_attn`: hidden `coord1531=1.593750`, top-k
  - `L0 after_prepare_mlp`: hidden `coord1531=14.562500`, residual `coord1531=14.187500`, both top-k
  - `L0 postprocess_layer`: hidden `coord1531=2.156250`, residual `coord1531=14.187500`, both top-k
  - `L1 layer_input`: hidden `coord1531=2.156250`, residual `coord1531=14.187500`, both top-k
  - `L1 after_prepare_attn`: hidden `coord1531=0.135742`, residual `coord1531=16.375000`, both top-k
  - `L1 after_prepare_mlp`: hidden `coord1531=5.062500`, residual `coord1531=28.750000`, both top-k

### Refined localization
- The persistent coordinate-1531 outlier is not born in `layer1 prepare_mlp`.
- The first visible emergence in the `DCP on` decode path is:
  - `layer0 after_self_attn`
  - equivalent to `layer0 self_attn/o_proj` output
- `layer0 prepare_mlp` then RMS-normalizes this outlier into a much larger residual-stream value.
- `layer1 prepare_mlp` only further carries and normalizes an already-polluted residual stream.

### DCP off comparison caveat
- A matching `DCP off` run with `decode_attention_backend=flashinfer` still schedules follow-up tokens through prefill/extend style logs, not the same `forward_mode.is_decode()` path.
- Therefore the coordinate trace table is currently reliable for `DCP on` upstream localization, but not a same-stage `DCP off/on` value comparison.
- This path difference is itself consistent with the original concern that MiniMax off/on take different forward-batch prepare/decode branches.

### `o_proj` contribution caveat
- Added a temporary `o_proj coord_contrib` diagnostic for output coordinate `1531`.
- The first run showed top local input dimensions on `TP0` such as:
  - `head=1, offset=19`
  - `head=1, offset=28`
  - `head=2, offset=19`
  - `head=2, offset=28`
- However, `o_proj.weight` is quantized, so directly multiplying `attn_output * raw_weight` is not a dequantization-aware contribution analysis.
- Treat the contributor indices only as a rough hint; the absolute `contrib` values are not valid proof.

### Next useful fix direction
- Move the root-cause focus back to `layer0 MiniMaxM2Attention.forward_core`:
  - DCP decode attention output before `o_proj`
  - DCP correction / head layout before `o_proj`
  - dequantization-aware or output-level comparison of `o_proj`
- The next high-value diagnostic should compare `DCP on` layer0 attention output against a reliable reference for the same decode step, not continue inside layer1 layernorm.

## 2026-05-09 DCP Decode Deep Check

### Changes tested
- `python/sglang/srt/layers/attention/utils.py`
  - Added `use_log2_lse` to `correct_attn_out()` / `cp_lse_ag_out_rs()`.
  - FlashInfer DCP paths now use natural-log LSE correction (`exp/log`) instead of hard-coded `exp2/log2`.
- `python/sglang/srt/layers/attention/flashinfer_backend.py`
  - Updated Python LSE diagnostics to natural-log merge.
  - Added a dense decode reference diagnostic:
    - gather each DCP rank's actual wrapper KV indices/window
    - reconstruct the full small KV sequence in DCP token order
    - run PyTorch dense GQA attention for the same `wrapper_q`
    - compare with `corrected_post_reduce`

### Run
- Dual-node `TP16`, `DCP2`, decode backend `flashinfer`
- `DISABLE_CUDA_GRAPH=1`
- `DISABLE_PIECEWISE_CUDA_GRAPH=1`
- `SGLANG_DCP_DECODE_REMAP_MODE=local_block`
- Logs:
  - `/data01/code/dcp_decode_natural_lse/on_node0.log`
  - `/data01/code/dcp_decode_natural_lse/on_node1.log`

### Output
- `max_tokens=4` still wrong:
  - `"战机4\\b\\b"`
- `max_tokens=1` is already wrong:
  - `"战机"`

### Decode evidence
- Natural-log LSE correction is active:
  - `python_final_lse` and `kernel_final_lse` match with `kernel_py_delta absmax ~= 0`.
- Dense reference comparison for decode is small:
  - `seq_len=2`: `dense_ref_delta absmax=0.006470 l2=0.054050`
  - `seq_len=3`: `dense_ref_delta absmax=0.005981 l2=0.038626`
  - `seq_len=4`: `dense_ref_delta absmax=0.005127 l2=0.035047`
  - `seq_len=5`: `dense_ref_delta absmax=0.005005 l2=0.032246`
- The DCP decode metadata/window is internally consistent for the observed request:
  - `seq_len=4`: both ranks use local `kv_indices_tail=[7,8]`
  - rank0/rank1 wrapper outputs differ as expected because they own different token shards
  - corrected aggregate closely matches the reconstructed dense attention over the same gathered KV

### Interpretation
- The previous `log2/exp2` diagnostic was not a reliable proof of FlashInfer semantic correctness; natural-log correction is the right direction for FlashInfer LSE.
- However, after switching to natural-log correction, the user-visible output is still wrong.
- The dense reference result strongly suggests `forward_decode` wrapper + LSE correction + all-reduce is not the main remaining corruption source for the current DCP state.
- Since `max_tokens=1` already returns `"战机"`, the bad state is present before the first decode step. Subsequent decode appears to continue from an already wrong first generated token / KV history.

### Next direction
- Shift the root-cause search one step earlier than decode:
  - first-token `forward_extend` / prefill DCP correction
  - DCP head slicing in `cp_lse_ag_out_rs(return_lse=True)`
  - prefill-to-decode KV state written after the wrong first token
- Keep the dense decode reference diagnostic as a guard: if later changes affect decode, it can confirm whether decode remains self-consistent.

## 2026-05-09 Clean FlashInfer Prefill/Decode Result

### Why this rerun was needed
- Earlier runs accidentally still involved FA3 prefill or prefix-cache reuse:
  - Logs showed `DCP-DIAG fa3 first_token`.
  - A manual request could hit a cached prompt and enter decode at `seq_lens=[2]`.
- The clean rerun forced:
  - `--attention-backend flashinfer`
  - `--prefill-attention-backend flashinfer`
  - `--decode-attention-backend flashinfer`
  - `--disable-radix-cache`
  - `--disable-cuda-graph`
  - `--disable-piecewise-cuda-graph`

### Clean baseline
- Prompt: `Hello DCP probe 050905`
- `DCP off`, `max_tokens=1`: `":"`
- `DCP on`, `max_tokens=1`: `":"`
- This indicates the earlier `"战机"` first-token result is strongly tied to the FA3 prefill / prefix-cache-contaminated path, not standalone FlashInfer decode.

### Original prompt comparison
- Prompt: `Hello`, `max_tokens=4`
- `DCP off`, clean FlashInfer prefill/decode:
  - `"\"?\\n\\nThe answer"`
- `DCP on`, clean FlashInfer prefill/decode, default old decode remap `block0`:
  - `"\"?\\n\\nThe first"`
- `DCP on`, clean FlashInfer prefill/decode, `SGLANG_DCP_DECODE_REMAP_MODE=local_block`:
  - `"\"?\\n\\nThe answer"`

### Decode remap conclusion
- With clean FlashInfer prefill, decode is almost correct but diverges at the 4th generated token when every TP rank always takes `block0`.
- `local_block` fixes the original `Hello` output under the clean DCP configuration.
- The corrected 6-head DCP output is ordered by gathered TP q-head blocks. After CP correction/all-reduce, each TP rank must take back the block corresponding to its local attention TP rank:
  - `local_block_idx = get_attention_tp_rank() % dcp_group.world_size`
  - slice `[local_block_idx * local_q_heads : (local_block_idx + 1) * local_q_heads]`
- Code default was updated from `block0` to `local_block` in `FlashInferAttnBackend.forward_decode()`.

### Final default validation
- After changing the code default to `local_block`, reran without setting `SGLANG_DCP_DECODE_REMAP_MODE`.
- Clean DCP-on command still used FlashInfer prefill/decode and disabled radix cache.
- Prompt: `Hello`, `max_tokens=4`
- Result:
  - `"\"?\\n\\nThe answer"`
- This matches the DCP-off clean baseline.

### Remaining caveat
- If prefill is left on FA3, corruption can still appear. For this MiniMax DCP path, the verified clean configuration uses FlashInfer for both prefill and decode.

## Final Root Cause and Fix Plan

### Root cause
- In the DCP decode path, `forward_decode()` all-gathers q-head blocks from the DCP partner ranks before calling FlashInfer.
- After `forward_return_lse()` and CP LSE correction, the corrected attention output has shape equivalent to:
  - `[batch, local_q_heads * dcp_size, head_dim]`
- These heads are still laid out as gathered TP q-head blocks, not as a single local TP rank's output.
- The old code always selected `block0` from this corrected tensor.
- That is only correct for the first TP rank in each DCP pair. The second TP rank should take the second block.
- As a result, some TP ranks fed another TP rank's attention heads into their own `o_proj`, causing decode-token drift and eventually corrupted text.

### Fix
- Keep the CP LSE correction/all-reduce logic.
- Change the final DCP decode remap from fixed `block0` to the local TP rank's DCP block.
- Compute:
  - `local_block_idx = get_attention_tp_rank() % dcp_group.world_size`
- Select:
  - `corrected_out[:, local_block_idx * local_q_heads : (local_block_idx + 1) * local_q_heads, :]`
- The cleanup build removes the temporary `SGLANG_DCP_DECODE_REMAP_MODE` override and candidate-output diagnostics.

### Related LSE fix
- FlashInfer returns natural-log LSE for `forward_return_lse()`.
- The correction helper now supports `use_log2_lse`.
- FlashInfer paths pass `use_log2_lse=False`, so CP correction uses `exp/log` rather than `exp2/log2`.

### Verified clean runtime condition
- MiniMax DCP correctness was verified under:
  - `--attention-backend flashinfer`
  - `--prefill-attention-backend flashinfer`
  - `--decode-attention-backend flashinfer`
  - `--disable-radix-cache`
  - `--disable-cuda-graph`
  - `--disable-piecewise-cuda-graph`
- FA3 prefill and prefix-cache reuse can still reproduce the older severe corruption, so they are outside the final clean DCP decode fix validation.

## 2026-05-09 Broader Validation With Diagnostic Build

### Setup
- Current build still contains diagnostic code, but runtime used `ENABLE_DCP_DIAG=0`.
- Clean command used:
  - `--attention-backend flashinfer`
  - `--prefill-attention-backend flashinfer`
  - `--decode-attention-backend flashinfer`
  - `--disable-radix-cache`
  - `--disable-cuda-graph`
  - `--disable-piecewise-cuda-graph`
- Logs/results:
  - `/data01/code/dcp_final_validation_diag/off_responses.jsonl`
  - `/data01/code/dcp_final_validation_diag/on_responses.jsonl`
  - `/data01/code/dcp_final_validation_diag/off_responses_mt4.jsonl`
  - `/data01/code/dcp_final_validation_diag/on_responses_mt4.jsonl`

### `max_tokens=4`
- 5 prompt batch:
  - `Hello`: exact match, `"\"?\\n\\nThe answer"`
  - `Hello DCP probe 050905`: exact match, `": 0x"`
  - `Write a short answer: 1+1=`: first token `"2"` matches, then text diverges (`"2. But the"` vs `"2, 2"`)
  - `Explain in one sentence what a transformer is.`: exact match, `" Then, in a"`
  - `Give 3 keywords about airplanes.`: exact match, `" (e.g.,"`

### `max_tokens=16`
- Long-form outputs no longer show corrupted/garbled loops.
- DCP on/off are not bitwise text-identical for all prompts after several generated tokens.
- This is consistent with small numeric differences changing later greedy choices on open-ended prompts; the original severe corruption pattern is gone under the clean FlashInfer configuration.

### Validation status
- The DCP decode head-block bug is fixed for the original repro and broader short-output cases.
- Exact long-generation parity with DCP off is not guaranteed by this fix; it would require a separate numerical parity effort beyond the corruption bug.

## 2026-05-09 Cleanup Build Status

### Code cleanup
- Removed the temporary DCP diagnostic helpers, `DCP-DIAG` log paths, debug-event blocks, dense-reference probes, and remap override logic from `flashinfer_backend.py`.
- Restored the temporary model/communicator/logits/model-runner diagnostics to their baseline versions.
- Removed the earlier ad-hoc DCP mock/first-token test files that were only used for diagnosis.
- Retained only the intended source fixes:
  - FlashInfer DCP decode CP correction uses natural-log LSE via `use_log2_lse=False`.
  - DCP decode selects the corrected local q-head block using `get_attention_tp_rank() % dcp_group.world_size`.
  - FlashInfer DCP startup rejects configurations where a DCP group can cross KV-head shards.
  - FlashInfer cuda-graph `fast_decode_plan` remains on the original code path; correctness is handled by the DCP local q-head block remap in `forward_decode()`.

### Local checks
- `python3 -m py_compile python/sglang/srt/layers/attention/flashinfer_backend.py python/sglang/srt/layers/attention/utils.py`: passed.
- VS Code diagnostics for both edited files: no diagnostics.
- `bash -n scripts/playground/dcp_validation_batch.sh`: passed.

### Remote sync
- The cleanup files were synced to both remote containers before validation:
  - `115.191.21.96:minimax_dcp_test:/sgl-workspace/sglang_minimax_new`
  - `115.191.2.23:minimax_dcp_test2:/sgl-workspace/sglang_minimax_new`
- Remote `py_compile` and `flashinfer_backend.py` diagnostic-marker grep passed on both containers.

### Validation blocker
- Starting the validation was initially blocked because the local Kerberos/GSSAPI ticket for `jumpecs-lf.byted.org` expired:
  - `klist` shows `krbtgt/BYTEDANCE.COM` and `host/jumpecs-lf.byted.org` as expired.
  - SSH now fails at the jump host with `Permission denied (gssapi-with-mic)`.
- Resolved with `kinit --keychain zhujunyu.666`; SSH to both remote nodes then passed.

### Final clean validation
- Runtime:
  - `TP16`, dual-node, `DCP off/on`
  - `--attention-backend flashinfer`
  - `--prefill-attention-backend flashinfer`
  - `--decode-attention-backend flashinfer`
  - `--disable-radix-cache`
  - `--disable-cuda-graph`
  - `--disable-piecewise-cuda-graph`
- Logs/results:
  - `/data01/code/dcp_final_validation_clean/off_responses_mt4.jsonl`
  - `/data01/code/dcp_final_validation_clean/on_responses_mt4.jsonl`
  - `/data01/code/dcp_final_validation_clean/off_responses_mt16.jsonl`
  - `/data01/code/dcp_final_validation_clean/on_responses_mt16.jsonl`
  - `/data01/code/dcp_final_validation_clean/off_node0.log`
  - `/data01/code/dcp_final_validation_clean/off_node1.log`
  - `/data01/code/dcp_final_validation_clean/on_node0.log`
  - `/data01/code/dcp_final_validation_clean/on_node1.log`

### Clean `max_tokens=4`
- `Hello`: exact match, `"\"?\\n\\nThe answer"`
- `Hello DCP probe 050905`: exact match, `": 0x"`
- `Write a short answer: 1+1=`: first token `"2"` matches, later text diverges (`"2. But the"` vs `"2, 2"`)
- `Explain in one sentence what a transformer is.`: exact match, `" Then, in a"`
- `Give 3 keywords about airplanes.`: exact match, `" (e.g.,"`

### Clean `max_tokens=16`
- No corrupted/garbled loop appeared.
- DCP on/off still diverge on open-ended long generations after several tokens.
- This matches the diagnostic-build validation and remains a numerical parity caveat, not the original DCP decode corruption.

### Final clean validation status
- The original `Hello,max_tokens=4` corruption repro is fixed in the cleaned code.
- Broader short-output validation remains stable after diagnostic cleanup.
- Long exact text parity is still not guaranteed and is outside this corruption fix.

### Final clean archive
- Local archive path:
  - `.dbg/archives/dcp_decode_fix_clean_20260509/`
- Contents:
  - `code.diff`
  - `source_files.tar.gz`
  - `validation_clean_logs.tar.gz`
  - `README.txt`

## 2026-05-10 Fast Path Production-Like Validation

### Goal
- Re-check whether the DCP decode fix still works when FlashInfer cuda-graph `fast_decode_plan` is enabled.
- Use a runtime closer to production while excluding DeepEP, because DeepEP currently triggers a separate `forward_extend()` ragged prefill padding mismatch (`q.shape[0]` vs `qo_indptr[-1]`).

### Temporary validation change
- On both remote containers, temporarily changed:
  - `self.disable_decode_wrapper_fast_path = False`
- This lets cuda-graph wrapper setup replace `begin_forward` with `fast_decode_plan`.
- After checking git history, the final code was restored to the original fast-path shape:
  - no `disable_decode_wrapper_fast_path` member
  - cuda-graph capture unconditionally replaces `begin_forward` with `fast_decode_plan`

### Smoke validation
- Runtime:
  - `TP16`, dual-node, `DCP on`
  - FlashInfer prefill/decode
  - `--disable-radix-cache`
  - `DISABLE_CUDA_GRAPH=0`
  - `DISABLE_PIECEWISE_CUDA_GRAPH=1`
- Logs confirmed:
  - `disable_cuda_graph=False`
  - `Capture cuda graph begin`
  - decode used `cuda graph: True`
- Result:
  - `Hello,max_tokens=4`: `"\"?\\n\\nThe answer"`
  - 5-prompt `max_tokens=4` batch matched the earlier clean DCP-on results.
  - `max_tokens=16` had no corrupted/garbled loop.

### Production-like concurrency validation
- Runtime matched the production command shape except DeepEP:
  - `--cuda-graph-max-bs 32`
  - `--cuda-graph-bs 1 2 4 8 16 32`
  - `--max-running-requests 64`
  - `--kv-cache-dtype fp8_e4m3`
  - `--enable-single-batch-overlap`
  - `--attention-backend flashinfer`
  - `--prefill-attention-backend flashinfer`
  - `--decode-attention-backend flashinfer`
  - `--enable-symm-mem`
  - `--disable-radix-cache`
  - no `--moe-a2a-backend deepep`
- Process args on both nodes confirmed the intended flags.
- Metrics confirmed decode cuda graph execution:
  - `sglang:cuda_graph_passes_total{mode="decode_cuda_graph", ...} 105.0` after the first concurrent run.
- Concurrent run 1:
  - 32 completion requests + 16 chat requests
  - 16 workers
  - `max_tokens=32`
  - result: `total=48`, `ok=48`, `failed=0`, `suspicious=0`
- Concurrent run 2:
  - 64 completion requests + 32 chat requests
  - 32 workers
  - `max_tokens=64`
  - result: `total=96`, `ok=96`, `failed=0`, `suspicious=0`, `p95_latency≈3.23s`

### Decision
- The previous DCP fast-path bypass is no longer needed for the verified configurations.
- The committed code restores the original FlashInfer `fast_decode_plan` path and relies on the corrected DCP local q-head block selection for correctness.
- DeepEP remains a separate prefill compatibility issue and is not covered by this fast-path decision.

### Restored-original-path validation
- After comparing git history, removed the temporary `disable_decode_wrapper_fast_path` member and restored the original unconditional cuda-graph fast path assignment.
- Validation runtime:
  - same production-like flags as above
  - `--dist-init-addr 192.168.44.93:5040`
  - logs/results under `/data01/code/dcp_fastpath_restore_validation/`
- Results:
  - `Hello,max_tokens=4`: `"\"?\\n\\nThe answer"`
  - 32 completion requests + 16 chat requests, 16 workers, `max_tokens=32`
  - `total=48`, `ok=48`, `failed=0`, `suspicious=0`
- Runtime confirmation:
  - `disable_cuda_graph=False`
  - `enable_single_batch_overlap=True`
  - `kv_cache_dtype='fp8_e4m3'`
  - `max_running_requests=64`
  - metrics showed `sglang:cuda_graph_passes_total{mode="decode_cuda_graph", ...} 109.0`
  - no traceback/runtime/value errors in node0/node1 logs.

## 2026-05-10 DeepEP Prefill Padding Fix

### New failure
- With the production-like DeepEP command:
  - `--moe-a2a-backend deepep`
  - `--deepep-mode auto`
  - `--ep-size 16`
  - FlashInfer prefill/decode
  - cuda graph, single-batch-overlap, fp8 KV cache
- The first request failed in `FlashInferAttnBackend.forward_extend()`:
  - `ValueError: q.shape[0] (16) does not match qo_indptr[-1] (6)`
- The failure happened in `prefill_wrapper_ragged.forward()`, before decode.

### Root cause
- DeepEP/EP MLP sync pads the local token dimension for collective communication in `ForwardBatch.prepare_mlp_sync_batch()`.
- For the observed single request:
  - real extend query tokens: `6`
  - padded local token buffer: `16`
- FlashInfer ragged prefill metadata is still built from real extend sequence lengths:
  - `qo_indptr[-1] = sum(seq_lens - prefix_lens) = 6`
- The model hidden states and q/k/v entering attention had already been padded:
  - `q.shape[0] = 16`
- FlashInfer ragged prefill requires `q.shape[0] == qo_indptr[-1]`, so the mismatch crashed the request.

### Fix
- In `FlashInferAttnBackend.forward_extend()`, when using ragged prefill and MLP sync padding is active:
  - compute the real attention token count from `forward_batch.extend_seq_lens_cpu`
  - slice `q/k/v` to real tokens before calling FlashInfer ragged prefill
  - slice `cache_loc` and `dcp_kv_mask` so KV cache writes only real tokens
  - pad the attention output back to the original padded token count before returning to the model
- This preserves DeepEP's padded tensor shape for later MLP/EP collectives while keeping FlashInfer ragged attention metadata consistent.
- A follow-up hot-path optimization moves the static "can this server have EP-padded extend tokens" check to `FlashInferAttnBackend.__init__()` as `self.may_have_padded_extend_tokens`, using `enable_num_token_non_padded(model_runner.server_args)`.
- Batch-specific checks stay in `forward_extend()` because `use_ragged`, `global_num_tokens_cpu`, `extend_seq_lens_cpu`, and the actual real-token count are runtime metadata.

### Validation
- Runtime:
  - `TP16`, dual-node, `DCP2`
  - `--moe-a2a-backend deepep`
  - `--deepep-mode auto`
  - `--cuda-graph-max-bs 32`
  - `--cuda-graph-bs 1 2 4 8 16 32`
  - `--max-running-requests 64`
  - `--kv-cache-dtype fp8_e4m3`
  - `--enable-single-batch-overlap`
  - FlashInfer prefill/decode
  - `--enable-symm-mem`
  - `--disable-radix-cache`
- Logs/results:
  - `/data01/code/deepep_prefill_fix_validation/hello_chat.json`
  - `/data01/code/deepep_prefill_fix_validation/hello_completion_mt4.json`
  - `/data01/code/deepep_prefill_fix_validation/concurrent_results.jsonl`
  - `/data01/code/deepep_prefill_fix_validation/on_node0.log`
  - `/data01/code/deepep_prefill_fix_validation/on_node1.log`
- Results:
  - chat `"你好"` succeeded and returned a normal Chinese assistant response.
  - `Hello,max_tokens=4`: `"\"?\\n\\nThe answer"`
  - 16 completion requests + 8 chat requests, 8 workers, `max_tokens=32`
  - `total=24`, `ok=24`, `failed=0`, `suspicious=0`
  - no `q.shape[0]` / `qo_indptr[-1]` mismatch lines
  - no traceback/runtime/value errors in node0/node1 logs
  - metrics confirmed decode cuda graph execution: `sglang:cuda_graph_passes_total{mode="decode_cuda_graph", ...} 138.0`
- Post-optimization validation:
  - Re-ran the same DeepEP production-like configuration after adding `self.may_have_padded_extend_tokens`.
  - Logs/results: `/data01/code/deepep_prefill_initopt_validation/`
  - `python -m py_compile` passed on both containers after syncing the optimized file.
  - chat `"你好"` succeeded and returned a normal Chinese assistant response.
  - `Hello,max_tokens=4`: `"\"?\\n\\nThe answer"`
  - 16 completion requests + 8 chat requests, 8 workers, `max_tokens=32`
  - `total=24`, `ok=24`, `failed=0`, `suspicious=0`
  - node0/node1 logs had no `q.shape[0]` / `qo_indptr[-1]` mismatch, traceback, runtime error, or value error.
  - After replacing the temporary `real_extend_num_tokens_cpu` idea with the existing `num_token_non_padded_cpu`, re-ran a quick dual-node DeepEP smoke validation in `/data01/code/dcp_lowrisk_opt_validation2/`:
    - chat `"你好"` succeeded.
    - `Hello,max_tokens=4`: `"\"?\\n\\nThe answer"`
    - node0/node1 logs had no `q.shape[0]` / `qo_indptr[-1]` mismatch, traceback, runtime error, or value error.

## 2026-05-10 Low-Risk Hot-Path Optimization and DCP Decode Profiling

### Low-risk optimization
- Removed the per-layer Python summation of real extend tokens in `FlashInferAttnBackend.forward_extend()`:
  - Reused the existing `ForwardBatch.num_token_non_padded_cpu` instead of adding another cached field.
  - This value is initialized from the original non-padded token count when `ForwardBatch` is built.
  - This avoids confusion with `extend_num_tokens`, which can be overwritten by the DeepEP padded token count in `prepare_mlp_sync_batch()`.
- Reduced padding-output zeroing cost in the DeepEP attention fix:
  - Changed `o.new_zeros((padded_num_tokens, ...))` to `o.new_empty((padded_num_tokens, ...))`.
  - Copied real attention output first, then zeroed only the padded tail.
- Important validation note:
  - An initial version tried to add `real_extend_num_tokens_cpu`, but it was redundant with the existing `num_token_non_padded_cpu`.
  - That initial version also incorrectly overwrote the cached real token count with the DeepEP padded `num_tokens` in `prepare_mlp_sync_batch()`.
  - Remote validation caught the regression as the original `q.shape[0] (16) does not match qo_indptr[-1] (6)` mismatch.
  - The final version uses the existing pre-padding `num_token_non_padded_cpu` and passed validation.

### Validation
- Local:
  - `python3 -m py_compile python/sglang/srt/layers/attention/flashinfer_backend.py python/sglang/srt/model_executor/forward_batch_info.py`
  - `git diff --check`
- Remote:
  - Synced optimized files to both containers.
  - Remote `py_compile` passed on both containers.
  - Runtime: same DeepEP production-like TP16/DCP2/EP16 configuration as the previous validation.
  - Logs/results: `/data01/code/dcp_lowrisk_opt_validation/`
  - chat `"你好"` succeeded and returned a normal Chinese assistant response.
  - `Hello,max_tokens=4`: `"\"?\\n\\nThe answer"`
  - 16 completion requests + 8 chat requests, 8 workers, `max_tokens=32`
  - `total=24`, `ok=24`, `failed=0`, `suspicious=0`
  - node0/node1 logs had no `q.shape[0]` / `qo_indptr[-1]` mismatch, traceback, runtime error, or value error.

### DCP decode profiling
- Method:
  - Used temporary remote-only instrumentation on node0 TP0; it was not kept in the source tree.
  - Timed DCP decode segments with CUDA synchronization around each segment.
  - Restored the container file to the optimized non-profiling version after capture and re-ran `py_compile`.
- Profile logs:
  - Raw logs: `/data01/code/dcp_lowrisk_opt_profile/on_node0.log`
  - Raw summary: `/data01/code/dcp_lowrisk_opt_profile/profile_summary.json`
  - Stable summary excluding the first warmup outlier: `/data01/code/dcp_lowrisk_opt_profile/profile_summary_exclude_first.json`
  - Local copies: `dbg/dcp_lowrisk_opt_profile/`
- Captured shape:
  - `tokens=32`
  - local `heads=3`
  - `n=47` stable samples after excluding the first one-time warmup outlier
- Stable timing summary, average / p50 / p95:
  - total DCP decode correction path: `0.685 / 0.676 / 0.721 ms`
  - q clone + DCP all-gather: `0.165 / 0.162 / 0.192 ms`
  - FlashInfer decode `forward_return_lse`: `0.214 / 0.209 / 0.248 ms`
  - LSE all-gather: `0.130 / 0.127 / 0.148 ms`
  - `correct_attn_out` Triton kernel: `0.068 / 0.067 / 0.072 ms`
  - output all-reduce: `0.047 / 0.046 / 0.050 ms`
  - transpose + contiguous: `0.031 / 0.030 / 0.036 ms`
  - local head-block slice: `0.031 / 0.030 / 0.033 ms`
- Interpretation:
  - The largest DCP-specific costs are q all-gather and LSE all-gather.
  - The correctness remap itself is small: transpose+contiguous plus head-block slice is about `0.062 ms` average in this profile.
  - `correct_attn_out` is also relatively small at about `0.068 ms` average.
  - Larger future gains are more likely from reducing/fusing DCP collectives than from micro-optimizing the final local head remap.

### Follow-up optimization directions
- Highest-value direction: reduce or fuse DCP collectives in decode, especially q all-gather and LSE all-gather.
- Candidate design: replace all-reduce plus local slicing with a layout-aware reduce-scatter if the corrected output can be partitioned by local q-head block.
- Candidate design: make the correction kernel write only the local q-head block to avoid producing and then slicing the full gathered-head output.
- Lower-value direction: micro-optimize the final transpose/slice; profiling shows it is much smaller than the DCP collectives.
- Validation requirement: any collective/layout change must be revalidated with dual-node TP16/DCP2 numerical checks, DeepEP smoke, and concurrent decode traffic.
