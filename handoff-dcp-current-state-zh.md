# DCP/DeepEP 当前工作交接记录

## 当前提交
- 最新提交：`87a4b8f7d Fix FlashInfer DCP decode head remap`，后续会因本交接记录 amend 生成新提交号。
- 主要已合入内容：FlashInfer DCP decode 正确性修复、FlashInfer LSE base 修复、DeepEP prefill padding 修复、低风险热路径优化、验证和 profiling 记录。

## 关键代码状态
- `python/sglang/srt/layers/attention/flashinfer_backend.py`
  - DCP decode 中对 all-gather 后的 q-head block 做本地 remap，避免所有 rank 都取 block0。
  - FlashInfer LSE correction 使用 `use_log2_lse=False`，因为 FlashInfer 返回 natural-log LSE。
  - DeepEP/EP padding 修复：ragged prefill 前只把 `q/k/v/cache_loc/dcp_kv_mask` 裁到真实 token 数，attention 输出再 pad 回 padded token 数。
  - 静态判断已移到初始化：`self.may_have_padded_extend_tokens = enable_num_token_non_padded(model_runner.server_args)`。
  - 低风险优化：使用已有 `forward_batch.num_token_non_padded_cpu` 作为 padding 前真实 token 数，不再新增 `real_extend_num_tokens_cpu`。
  - 输出补齐优化：`new_empty()` 后复制真实输出，只对 padding tail `zero_()`，避免整块清零。
- `python/sglang/srt/model_executor/forward_batch_info.py`
  - 没有新增 `real_extend_num_tokens_cpu`；最终确认它和已有 `num_token_non_padded_cpu` 语义重复。
  - `extend_num_tokens` 在 DeepEP `prepare_mlp_sync_batch()` 后可能被改成 padded token 数，不能作为 attention 真实 token 数使用。
- `python/sglang/srt/layers/attention/utils.py`
  - `correct_attn_out()` 和 `cp_lse_ag_out_rs()` 支持 `use_log2_lse`，FlashInfer 路径传 `False`。

## 已验证配置
- 双节点：`115.191.21.96:minimax_dcp_test` 和 `115.191.2.23:minimax_dcp_test2`
- 模型：`/data00/models/MiniMax-M2.7`
- 典型服务参数：`TP16`、`DCP2`、`EP16`、`--moe-a2a-backend deepep`、`--deepep-mode auto`、FlashInfer prefill/decode、`--kv-cache-dtype fp8_e4m3`、cuda graph、single-batch-overlap、`--disable-radix-cache`。
- 验证结果：chat `你好` 正常；completion `Hello,max_tokens=4` 返回 `"?\n\nThe answer`；24 并发验证通过；日志无 `q.shape[0] / qo_indptr[-1]` mismatch、traceback、runtime/value error。

## Profiling 记录
- 本地可见目录：`dbg/dcp_lowrisk_opt_profile/`
- 文件：
  - `dbg/dcp_lowrisk_opt_profile/on_node0.log`
  - `dbg/dcp_lowrisk_opt_profile/profile_summary.json`
  - `dbg/dcp_lowrisk_opt_profile/profile_summary_exclude_first.json`
- 远端原始目录：`/data01/code/dcp_lowrisk_opt_profile/`
- 推荐看 `profile_summary_exclude_first.json`，因为 `profile_summary.json` 包含第一条 warmup outlier。
- 稳定样本：排除首个 outlier 后 `n=47`，`tokens=32`，local `heads=3`。
- 稳定耗时平均 / P50 / P95：
  - total DCP decode correction path：`0.685 / 0.676 / 0.721 ms`
  - q clone + DCP all-gather：`0.165 / 0.162 / 0.192 ms`
  - FlashInfer decode `forward_return_lse`：`0.214 / 0.209 / 0.248 ms`
  - LSE all-gather：`0.130 / 0.127 / 0.148 ms`
  - `correct_attn_out`：`0.068 / 0.067 / 0.072 ms`
  - output all-reduce：`0.047 / 0.046 / 0.050 ms`
  - transpose + contiguous：`0.031 / 0.030 / 0.036 ms`
  - local head-block slice：`0.031 / 0.030 / 0.033 ms`

## 性能分析结论
- DCP decode 较大的额外成本是 q all-gather 和 LSE all-gather。
- 本次正确性修复新增的最终 head remap 成本较小，transpose+slice 平均约 `0.062 ms`。
- `correct_attn_out` 平均约 `0.068 ms`，不是主要瓶颈。
- 后续更有价值的方向是减少或融合 DCP collective，而不是继续微调最终本地 head remap。

## 后续可做方向
- 评估用 reduce-scatter 替代 all-reduce + local slice。
- 评估让 correction kernel 只写本 rank 需要的 local q-head block。
- 任何 collective/layout 改动都必须重新做双节点 TP16/DCP2 数值验证、DeepEP smoke 和并发 decode 验证。

## 重新开启对话时怎么说
可以直接在新对话里输入：

> 请先阅读 `handoff-dcp-current-state-zh.md`、`debug-dcp-amplify-layer-zh.md`、`debug-dcp-amplify-layer.md`，并查看当前 git 最新提交和 `dbg/dcp_lowrisk_opt_profile/`。这是上个会话的 DCP/DeepEP 修复和 profiling 交接记录，请基于这些继续，不要重复已经完成的验证。

