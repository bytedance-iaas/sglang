# MiniMax DCP Decode 乱码问题排查与修复记录

## 目标
- 定位并修复 MiniMax M2.7 在 `TP16` 双机、`DCP2`、FlashInfer decode 路径下的输出乱码/异常漂移问题。
- 保留关键排查证据、最终根因、修复方案、验证条件和后续 caveat。
- 清理诊断代码后，用 clean build 重新验证并归档。

## 问题现象
- 原始复现条件：MiniMax M2.7、双机 `TP16`、`DCP2`。
- 典型 prompt：`Hello`，`max_tokens=4`，`temperature=0`。
- `DCP off` 输出正常：`"?\n\nThe answer`。
- `DCP on` 在旧路径下会输出异常文本，例如 `战机...` 或后续 token 漂移。
- 错误主要发生在 decode 阶段；采样、logits 行选择、RMSNorm、MoE 都不是最初根因。

## 排查路径摘要
- 首先检查 `logits_processor` 和 sample 前后日志，确认不是取错 logits 行或 token 位置。
- 然后检查 decoder 层内不同阶段，发现异常从早期 attention/o_proj 后开始放大。
- 继续拆分 `prepare_mlp` 内部 residual add 和 layernorm，确认 layernorm 只是放大上游已有异常，不是根因。
- 进一步追踪固定坐标 `1531`，发现异常最早出现在 layer0 self-attention/o_proj 相关路径。
- 最后聚焦 `flashinfer_backend.py::forward_decode()`，比较 q gather、KV metadata、FlashInfer wrapper 输出、LSE correction、CP reduce 后输出。
- dense reference 验证显示 FlashInfer wrapper、KV metadata 和 CP LSE correction 对观察到的 KV 窗口基本自洽，剩余问题集中在 corrected attention output 的 head block 语义映射。

## 根因
- DCP decode 进入 FlashInfer 前会 all-gather q-head block：
  - 本地 q-head 数为 `local_q_heads`。
  - DCP size 为 `dcp_size`。
  - gather 后 FlashInfer 看到的 q-head 数为 `local_q_heads * dcp_size`。
- FlashInfer `forward_return_lse()` 和 CP LSE correction 后，corrected output 的形状等价于：
  - `[batch, local_q_heads * dcp_size, head_dim]`
- 这个 head 维仍然按 DCP group 内 TP rank 的 q-head block 排布：
  - `[rank0 的 local q-head block, rank1 的 local q-head block, ...]`
- 旧逻辑等价于总是取前 `local_q_heads` 个 heads，也就是固定取 `block0`。
- 固定取 `block0` 只对 DCP pair 中第一个 TP rank 正确；第二个 TP rank 应该取 `block1`。
- 因此部分 TP rank 把别的 TP rank 的 attention heads 喂给自己的 `o_proj`，导致 head 语义错位，decode token 逐步漂移，最终表现为乱码。

## 修复方案
- 保留 CP LSE correction 和 DCP rank 间 all-reduce。
- correction 完成后，不再固定取 `block0`。
- 根据当前 attention TP rank 计算本 rank 对应的 DCP q-head block：
  - `local_block_idx = get_attention_tp_rank() % dcp_group.world_size`
- 从 corrected output 中取回当前 TP rank 自己的 q-head block：
  - `corrected_out[:, local_block_idx * local_q_heads : (local_block_idx + 1) * local_q_heads, :]`
- 清理版代码移除了临时 `SGLANG_DCP_DECODE_REMAP_MODE` override 和候选 remap 诊断逻辑。

## 代码位置
- 主修复位置：
  - `python/sglang/srt/layers/attention/flashinfer_backend.py::FlashInferAttnBackend.forward_decode()`
- 关键逻辑：
  - gather LSE：`dcp_group.all_gather(s.clone(), dim=0)`
  - LSE correction：`correct_attn_out(..., use_log2_lse=False)`
  - DCP all-reduce：`dcp_group.all_reduce(corrected_out)`
  - local block 选择：`get_attention_tp_rank() % dcp_group.world_size`
- 相关 LSE 修复：
  - `python/sglang/srt/layers/attention/utils.py::correct_attn_out()`
  - `python/sglang/srt/layers/attention/utils.py::cp_lse_ag_out_rs()`
- DCP/KV shard 保护：
  - `flashinfer_backend.py::__init__()` 中检查 `kv_head_replication >= dcp_size`
- DCP fast decode plan 处理：
  - 经过 fast path 复验和 git 历史对比后，最终代码恢复为原始 FlashInfer cuda graph `fast_decode_plan` 路径。
  - 即不新增 `disable_decode_wrapper_fast_path` 成员，cuda graph capture 中无条件替换 `begin_forward = fast_decode_plan`。
  - DCP 正确性由 `forward_decode()` 中的 local q-head block remap 保证。

## LSE 修复
- FlashInfer `forward_return_lse()` 返回 natural-log LSE。
- 原 correction helper 中曾固定使用 `exp2/log2`，这不适合 FlashInfer LSE。
- 现在 `correct_attn_out()` 和 `cp_lse_ag_out_rs()` 增加 `use_log2_lse` 参数。
- FlashInfer 路径显式传 `use_log2_lse=False`，使用 `exp/log` 做 correction。

## 为什么没有直接复用 `cp_lse_ag_out_rs()`
- `cp_lse_ag_out_rs()` 默认假设 correction 后可以沿 head 维做 `reduce_scatter`，每个 CP rank 拿自己那段 heads。
- MiniMax DCP decode 的 head 维是 q all-gather 后的 DCP peer q-head blocks。
- correction 后需要按 attention TP rank 取回本 rank 的 local block，而不是简单 reduce-scatter。
- 当前修复先使用 `all_reduce + local block slice`，优先保证 correctness。
- 后续可以把这段抽成 helper，并优化成只规约目标 block 的通信路径。

## 验证条件
clean runtime 使用以下条件：
- `TP16` 双机
- `DCP2`
- `--attention-backend flashinfer`
- `--prefill-attention-backend flashinfer`
- `--decode-attention-backend flashinfer`
- `--disable-radix-cache`
- `--disable-cuda-graph`
- `--disable-piecewise-cuda-graph`

## 验证结果
- 原始复现 prompt：`Hello`，`max_tokens=4`
- clean build 下：
  - `DCP off`: `"?\n\nThe answer`
  - `DCP on`: `"?\n\nThe answer`
- 二者完全一致，原始 DCP decode 乱码问题已修复。

## 更复杂验证
- 5 prompt batch，`max_tokens=4`：
  - `Hello`: 完全一致，`"?\n\nThe answer`
  - `Hello DCP probe 050905`: 完全一致，`: 0x`
  - `Write a short answer: 1+1=`: 首 token `2` 一致，后续短文本分叉
  - `Explain in one sentence what a transformer is.`: 完全一致，` Then, in a`
  - `Give 3 keywords about airplanes.`: 完全一致，` (e.g.,`
- `max_tokens=16`：
  - 没有出现乱码或异常循环。
  - DCP on/off 在开放式长生成中仍可能若干 token 后分叉。
  - 这属于数值 parity caveat，不是原始乱码 bug。

## 清理状态
- 已移除临时 DCP 诊断 helper、`DCP-DIAG` 日志、debug event block、dense reference probe 和 remap override。
- 已恢复模型、communicator、logits、model_runner 等文件里的临时诊断埋点。
- 已删除早期 ad-hoc DCP mock/first-token 测试文件。
- clean build 本地检查通过：
  - `python3 -m py_compile python/sglang/srt/layers/attention/flashinfer_backend.py python/sglang/srt/layers/attention/utils.py`
  - VS Code diagnostics 为空
  - `bash -n scripts/playground/dcp_validation_batch.sh`

## 远端同步与验证
- clean 代码已同步到两个容器：
  - `115.191.21.96:minimax_dcp_test:/sgl-workspace/sglang_minimax_new`
  - `115.191.2.23:minimax_dcp_test2:/sgl-workspace/sglang_minimax_new`
- 两个容器上关键源码和 playground 脚本 checksum 与本地一致。
- 远端 `py_compile` 和 `flashinfer_backend.py` 诊断 marker grep 均通过。

## 归档
- 诊断构建归档：
  - `.dbg/archives/dcp_decode_fix_diag_20260509/`
- clean 构建归档：
  - `.dbg/archives/dcp_decode_fix_clean_20260509/`
- clean archive 内容：
  - `code.diff`
  - `source_files.tar.gz`
  - `validation_clean_logs.tar.gz`
  - `README.txt`

## 当前 caveat
- FA3 prefill 尚未作为本修复的一部分适配 DCP；验证使用 FlashInfer prefill/decode。
- prefix/radix cache 可能污染 DCP 首 token 判断，因此 clean 验证禁用了 radix cache。
- 后续发现的 DeepEP prefill 报错是另一个问题：DeepEP padding 后 `q.shape[0]` 与 FlashInfer ragged `qo_indptr[-1]` 不一致，发生在 `forward_extend()`，不是本次 `forward_decode()` head remap bug。

## 2026-05-10 Fast Path 生产近似验证
- 目标：验证打开 FlashInfer cuda graph `fast_decode_plan` 后，DCP decode local q-head block remap 是否仍然正确。
- 临时验证改动：两台远端容器中临时设置 `self.disable_decode_wrapper_fast_path = False`。
- 查看 git 历史后，最终实现按原始 fast path 形态恢复：
  - 不保留 `disable_decode_wrapper_fast_path`
  - cuda graph capture 中无条件替换 `begin_forward = fast_decode_plan`
- smoke 验证配置：
  - `TP16` 双机
  - `DCP2`
  - FlashInfer prefill/decode
  - `--disable-radix-cache`
  - `DISABLE_CUDA_GRAPH=0`
  - `DISABLE_PIECEWISE_CUDA_GRAPH=1`
- 日志确认：
  - `disable_cuda_graph=False`
  - `Capture cuda graph begin`
  - decode 日志出现 `cuda graph: True`
- smoke 结果：
  - `Hello,max_tokens=4`: `"?\n\nThe answer`
  - 5 prompt `max_tokens=4` batch 与 clean DCP-on 结果一致
  - `max_tokens=16` 无乱码或异常循环
- 生产近似并发配置：
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
  - 不包含 DeepEP，因为 DeepEP 当前触发的是另一个 prefill padding/metadata 不一致问题
- 进程参数确认两台节点都使用了上述配置。
- metrics 确认 decode cuda graph 实际执行：
  - `sglang:cuda_graph_passes_total{mode="decode_cuda_graph", ...} 105.0`
- 并发验证 1：
  - 32 个 completion 请求 + 16 个 chat 请求
  - 16 并发 worker
  - `max_tokens=32`
  - 结果：`total=48`，`ok=48`，`failed=0`，`suspicious=0`
- 并发验证 2：
  - 64 个 completion 请求 + 32 个 chat 请求
  - 32 并发 worker
  - `max_tokens=64`
  - 结果：`total=96`，`ok=96`，`failed=0`，`suspicious=0`，`p95_latency≈3.23s`
- 结论：
  - 已验证配置下不需要绕过 `fast_decode_plan`。
  - 最终代码恢复原始 fast path 代码形态，依赖 `forward_decode()` 的 DCP local q-head block remap 保证正确性。
  - DeepEP 仍是单独的 `forward_extend()` prefill 兼容性问题，不属于这次 fast path 决策范围。
- 恢复原始路径后的复验：
  - 删除临时 `disable_decode_wrapper_fast_path` 成员。
  - 恢复 cuda graph capture 中无条件 `begin_forward = fast_decode_plan`。
  - 验证配置仍使用生产近似参数，`--dist-init-addr 192.168.44.93:5040`。
  - 远端日志和结果目录：`/data01/code/dcp_fastpath_restore_validation/`。
  - `Hello,max_tokens=4`: `"?\n\nThe answer`。
  - 32 个 completion 请求 + 16 个 chat 请求，16 并发 worker，`max_tokens=32`。
  - 结果：`total=48`，`ok=48`，`failed=0`，`suspicious=0`。
  - 参数确认：`disable_cuda_graph=False`，`enable_single_batch_overlap=True`，`kv_cache_dtype='fp8_e4m3'`，`max_running_requests=64`。
  - metrics 确认：`sglang:cuda_graph_passes_total{mode="decode_cuda_graph", ...} 109.0`。
  - node0/node1 日志无 traceback/runtime/value error。

## 2026-05-10 DeepEP Prefill Padding 修复
- 新问题现象：
  - 使用接近生产的 DeepEP 命令启动：
    - `--moe-a2a-backend deepep`
    - `--deepep-mode auto`
    - `--ep-size 16`
    - FlashInfer prefill/decode
    - cuda graph、single-batch-overlap、fp8 KV cache
  - 首个请求在 `FlashInferAttnBackend.forward_extend()` 的 ragged prefill 阶段报错：
    - `ValueError: q.shape[0] (16) does not match qo_indptr[-1] (6)`
  - 这个错误发生在 prefill，不是之前的 decode head remap 问题。
- 根因：
  - DeepEP/EP MLP sync 会在 `ForwardBatch.prepare_mlp_sync_batch()` 中把本地 token buffer pad 到 collective 对齐长度。
  - 当前单请求真实 extend query token 数是 `6`。
  - DeepEP padding 后 attention 前的 q/k/v token 维变成 `16`。
  - FlashInfer ragged prefill metadata 仍按真实 extend sequence length 构造：
    - `qo_indptr[-1] = sum(seq_lens - prefix_lens) = 6`
  - FlashInfer ragged prefill 要求 `q.shape[0] == qo_indptr[-1]`，因此 `16 != 6` 直接 crash。
- 修复方案：
  - 在 `FlashInferAttnBackend.forward_extend()` 中，当 ragged prefill 且 MLP sync padding 生效时：
    - 从 `forward_batch.extend_seq_lens_cpu` 计算真实 attention token 数。
    - FlashInfer prefill 前将 `q/k/v` 裁剪到真实 token 数。
    - KV 写入使用同步裁剪后的 `cache_loc` 和 `dcp_kv_mask`，避免 padding token 写入 KV cache。
    - attention 输出再 pad 回原始 padded token 数，保证后续 DeepEP/MLP collective 仍看到对齐后的 tensor shape。
  - 这样同时满足：
    - FlashInfer ragged prefill metadata 与 q/k/v token 数一致。
    - DeepEP 后续 collective 所需 padded shape 不被破坏。
  - 后续性能优化：
    - 将“当前 server 是否可能出现 EP/DeepEP padded extend token”的静态判断移动到 `FlashInferAttnBackend.__init__()`，通过 `enable_num_token_non_padded(model_runner.server_args)` 初始化 `self.may_have_padded_extend_tokens`。
    - `use_ragged`、`global_num_tokens_cpu`、`extend_seq_lens_cpu` 和真实 token 数仍是 batch 级运行时信息，不能提前到初始化阶段。
- 验证配置：
  - `TP16` 双机
  - `DCP2`
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
- 远端日志和结果：
  - `/data01/code/deepep_prefill_fix_validation/hello_chat.json`
  - `/data01/code/deepep_prefill_fix_validation/hello_completion_mt4.json`
  - `/data01/code/deepep_prefill_fix_validation/concurrent_results.jsonl`
  - `/data01/code/deepep_prefill_fix_validation/on_node0.log`
  - `/data01/code/deepep_prefill_fix_validation/on_node1.log`
- 验证结果：
  - chat `"你好"` 请求成功，返回正常中文助手回复。
  - `Hello,max_tokens=4`: `"?\n\nThe answer`。
  - 16 个 completion 请求 + 8 个 chat 请求，8 并发 worker，`max_tokens=32`。
  - 结果：`total=24`，`ok=24`，`failed=0`，`suspicious=0`。
  - node0/node1 日志没有 `q.shape[0] / qo_indptr[-1]` mismatch。
  - node0/node1 日志没有 traceback/runtime/value error。
  - metrics 确认 decode cuda graph 执行：`sglang:cuda_graph_passes_total{mode="decode_cuda_graph", ...} 138.0`。
- 初始化判断优化后的复验：
  - 优化文件同步到两台容器后，远端 `python -m py_compile` 通过。
  - 使用相同 DeepEP 生产近似配置重新验证。
  - 远端日志和结果目录：`/data01/code/deepep_prefill_initopt_validation/`
  - chat `"你好"` 请求成功，返回正常中文助手回复。
  - `Hello,max_tokens=4`: `"?\n\nThe answer`。
  - 16 个 completion 请求 + 8 个 chat 请求，8 并发 worker，`max_tokens=32`。
  - 结果：`total=24`，`ok=24`，`failed=0`，`suspicious=0`。
  - node0/node1 日志没有 `q.shape[0] / qo_indptr[-1]` mismatch，也没有 traceback/runtime/value error。
  - 将临时的 `real_extend_num_tokens_cpu` 思路替换为已有 `num_token_non_padded_cpu` 后，又在 `/data01/code/dcp_lowrisk_opt_validation2/` 做了一轮双节点 DeepEP smoke 复验：
    - chat `"你好"` 请求成功。
    - `Hello,max_tokens=4`: `"?\n\nThe answer`。
    - node0/node1 日志没有 `q.shape[0] / qo_indptr[-1]` mismatch，也没有 traceback/runtime/value error。

## 2026-05-10 低风险热路径优化和 DCP Decode Profiling
- 低风险优化：
  - 去掉 `FlashInferAttnBackend.forward_extend()` 中每层重复计算真实 extend token 数的 Python sum。
  - 复用已有的 `ForwardBatch.num_token_non_padded_cpu`，不再新增额外缓存字段。
  - `num_token_non_padded_cpu` 在 `ForwardBatch` 构造时已经保存 padding 前的真实 token 数。
  - 这样可以避免和 `extend_num_tokens` 混淆；`extend_num_tokens` 在 `prepare_mlp_sync_batch()` 中可能被 DeepEP padding 后的 token 数覆盖。
  - DeepEP attention 输出补齐从整块 `new_zeros()` 改为 `new_empty()`，复制真实输出后只对 padding tail 做 `zero_()`。
- 重要验证插曲：
  - 初版曾尝试新增 `real_extend_num_tokens_cpu`，但它与已有的 `num_token_non_padded_cpu` 语义重复。
  - 初版缓存还曾在 `prepare_mlp_sync_batch()` 中误用 DeepEP padded `num_tokens` 覆盖真实 token 数。
  - 远端验证立即复现原始 mismatch：`q.shape[0] (16) does not match qo_indptr[-1] (6)`。
  - 最终版本改为直接使用已有的 `num_token_non_padded_cpu` 后，复验通过。
- 本地验证：
  - `python3 -m py_compile python/sglang/srt/layers/attention/flashinfer_backend.py python/sglang/srt/model_executor/forward_batch_info.py`
  - `git diff --check`
- 远端复验：
  - 优化文件同步到两台容器后，远端 `py_compile` 通过。
  - 使用相同 DeepEP 生产近似配置重新验证：`TP16 + DCP2 + EP16 + deepep + FlashInfer + fp8 KV + cuda graph + SBO`。
  - 远端日志和结果目录：`/data01/code/dcp_lowrisk_opt_validation/`
  - chat `"你好"` 请求成功，返回正常中文助手回复。
  - `Hello,max_tokens=4`: `"?\n\nThe answer`。
  - 16 个 completion 请求 + 8 个 chat 请求，8 并发 worker，`max_tokens=32`。
  - 结果：`total=24`，`ok=24`，`failed=0`，`suspicious=0`。
  - node0/node1 日志没有 `q.shape[0] / qo_indptr[-1]` mismatch，也没有 traceback/runtime/value error。
- DCP decode profiling 方法：
  - 使用 node0 TP0 的临时远端插桩，未保留到源码提交中。
  - 对 DCP decode 的关键段用 CUDA synchronize 前后计时。
  - profiling 后已将容器文件恢复为正式低风险优化版本，并重新 `py_compile` 通过。
- Profiling 记录：
  - 原始日志：`/data01/code/dcp_lowrisk_opt_profile/on_node0.log`
  - 原始统计：`/data01/code/dcp_lowrisk_opt_profile/profile_summary.json`
  - 排除首个 warmup outlier 后的稳定统计：`/data01/code/dcp_lowrisk_opt_profile/profile_summary_exclude_first.json`
  - 本地副本：`dbg/dcp_lowrisk_opt_profile/`
- Profiling 样本：
  - `tokens=32`
  - 本地 `heads=3`
  - 排除首个一次性 warmup outlier 后，稳定样本数 `n=47`
- 稳定耗时汇总，格式为平均 / P50 / P95：
  - DCP decode correction 总路径：`0.685 / 0.676 / 0.721 ms`
  - q clone + DCP all-gather：`0.165 / 0.162 / 0.192 ms`
  - FlashInfer decode `forward_return_lse`：`0.214 / 0.209 / 0.248 ms`
  - LSE all-gather：`0.130 / 0.127 / 0.148 ms`
  - `correct_attn_out` Triton kernel：`0.068 / 0.067 / 0.072 ms`
  - output all-reduce：`0.047 / 0.046 / 0.050 ms`
  - transpose + contiguous：`0.031 / 0.030 / 0.036 ms`
  - local head-block slice：`0.031 / 0.030 / 0.033 ms`
- 结论：
  - 当前 DCP decode 里较大的 DCP 额外成本是 q all-gather 和 LSE all-gather。
  - 本次正确性修复新增的最终 head remap 本身较小，transpose+contiguous 加 head-block slice 平均约 `0.062 ms`。
  - `correct_attn_out` kernel 平均约 `0.068 ms`，也不是主要瓶颈。
  - 后续更有价值的优化方向是减少或融合 DCP collective，而不是继续微调最终本地 head remap。
- 后续修复方向：
  - 优先研究减少或融合 DCP decode collective，重点是 q all-gather 和 LSE all-gather。
  - 可以评估用按 q-head block 分区的 reduce-scatter 替代 all-reduce 加本地 slice。
  - 可以评估让 correction kernel 只写本 rank 需要的 local q-head block，避免先生成完整 gathered-head 输出再切片。
  - 最终 transpose/slice 的微优化优先级较低，因为 profiling 显示它明显小于 collective 开销。
  - 任何 collective/layout 改动都需要重新跑双节点 TP16/DCP2 数值验证、DeepEP smoke 和并发 decode 流量。
