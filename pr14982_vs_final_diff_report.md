# PR 14982 与最终落地版本差异说明

这份报告比较的是：

- 原始 PR merge 结果：pr-14982-merge
- 最终落地候选提交：12c79fa32c9b5454a210474ee1404bd9de2bf819

注意：

- 这不是“当前分支相对 PR 的全部文本差异”逐行转抄。
- 你的当前分支本身已经比 PR 的基线前进很多，所以直接看原始 git diff 会混入大量当前分支已有演进。
- 这份报告聚焦的是最终落地结果里，相对 PR 14982 的实际落地差异，也就是我在解冲突和适配当前分支时保留、补充或调整的地方。

## 总览

相对 PR 14982 merge 结果，最终版本有差异的文件是：

- [python/sglang/srt/distributed/parallel_state.py](python/sglang/srt/distributed/parallel_state.py)
- [python/sglang/srt/entrypoints/engine.py](python/sglang/srt/entrypoints/engine.py)
- [python/sglang/srt/layers/attention/flashinfer_backend.py](python/sglang/srt/layers/attention/flashinfer_backend.py)
- [python/sglang/srt/layers/attention/flashinfer_mla_backend.py](python/sglang/srt/layers/attention/flashinfer_mla_backend.py)
- [python/sglang/srt/layers/attention/utils.py](python/sglang/srt/layers/attention/utils.py)
- [python/sglang/srt/managers/scheduler.py](python/sglang/srt/managers/scheduler.py)
- [python/sglang/srt/managers/scheduler_runtime_checker_mixin.py](python/sglang/srt/managers/scheduler_runtime_checker_mixin.py)
- [python/sglang/srt/mem_cache/memory_pool.py](python/sglang/srt/mem_cache/memory_pool.py)
- [python/sglang/srt/model_executor/cuda_graph_runner.py](python/sglang/srt/model_executor/cuda_graph_runner.py)
- [python/sglang/srt/model_executor/forward_batch_info.py](python/sglang/srt/model_executor/forward_batch_info.py)
- [python/sglang/srt/model_executor/model_runner.py](python/sglang/srt/model_executor/model_runner.py)
- [python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py](python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py)
- [python/sglang/srt/models/utils.py](python/sglang/srt/models/utils.py)
- [python/sglang/srt/server_args.py](python/sglang/srt/server_args.py)
- [test/srt/run_suite.py](test/srt/run_suite.py)

和 PR 保持一致、没有额外落地差异的文件：

- [test/srt/test_dcp.py](test/srt/test_dcp.py)

## 逐文件差异

### [python/sglang/srt/distributed/parallel_state.py](python/sglang/srt/distributed/parallel_state.py)

最终版本没有退回到 PR 的较简分布式状态实现，而是保留了当前分支已经存在的更大范围并行能力。

相对 PR 的主要差异：

- 保留了当前分支已有的 piecewise cuda graph、global tcp store、musa、npu pg options、mooncake、attention tp 和 moe dp 相关能力。
- 保留了当前分支对 pynccl state、symmetric memory debug、allocation symmetry、piecewise graph 下 pynccl outplace allreduce 的处理。
- 保留了当前分支的 fused allreduce rmsnorm、更多 group graph capture 参与者、更多 distributed backend 选择逻辑。
- 在这个更丰富的骨架上合入了 PR 需要的 DCP 能力，包括 get_dcp_group、decode_context_parallel_size 接线、reduce_scatter_along_dim、graph capture 中包含 DCP、ensure_model_parallel_initialized 的 DCP 校验。

原因：

- 如果按 PR 原样覆盖，会把你当前分支已经存在的一大批并行和通信能力回退掉。
- 所以这里的落地策略是“保留当前分支骨架，只引入 PR 的 DCP 关键语义”。

### [python/sglang/srt/entrypoints/engine.py](python/sglang/srt/entrypoints/engine.py)

这个文件和 PR 的差异很大，但绝大多数差异不是我临时发明的，而是当前分支本来就已经拥有更完整的 Engine 实现。

相对 PR 的实际落地差异：

- 最终版本保留了当前分支已有的 EngineScoreMixin、plugin 加载、bootstrap server、session 接口、watchdog、trace、路由 DP 参数等能力。
- PR 在这里关心的核心其实只有环境变量设置，特别是 DCP 下 NCCL_GRAPH_MIXING_SUPPORT 的处理。
- 最终落地版没有把整个文件退回到 PR 版本，而是在当前分支版本里合入了 DCP 大于 1 时关闭 NCCL_GRAPH_MIXING_SUPPORT 的语义。

原因：

- 如果照搬 PR，这个文件会发生不必要的大回退。
- 这里的策略是“保留当前入口实现，只抽取 PR 真正相关的环境设置变更”。

### [python/sglang/srt/layers/attention/flashinfer_backend.py](python/sglang/srt/layers/attention/flashinfer_backend.py)

这是最关键的手工合并文件之一。

相对 PR 的主要差异：

- 保留了当前分支已有的 debug_kernel_api、piecewise cuda graph、现有 import 组织方式。
- 最终版本把 DCP 的 dcp_size 和 dcp_rank 贯穿到 decode 和 prefill 的 metadata 更新器，而不是回退成 PR 所在基线的更简单调用环境。
- 在 prefill 路径中，最终版本保留并接入了当前分支的 merge_state 结构，同时引入 PR 的 q all_gather、return_lse、cp_lse_ag_out_rs 修正。
- 在 decode 路径中，最终版本使用了 forward_return_lse，再在 DCP 模式下调用 cp_lse_ag_out_rs，而不是简单沿用 PR 基线对应的周边实现。
- KV 写入没有直接写死为 PR 版本，而是通过 kwargs 形式把 dcp_kv_mask 在 DCP 模式下传入 set_kv_buffer。
- 非 ragged wrapper 下显式禁止 DCP，增加了 assert self.dcp_size == 1。

原因：

- 这个文件必须同时兼容你当前分支已有的 backend 结构和 PR 的 DCP 修复语义。
- 我在这里采取的是“保留当前分支调用骨架，吸收 PR 的 DCP 关键逻辑”。

### [python/sglang/srt/layers/attention/flashinfer_mla_backend.py](python/sglang/srt/layers/attention/flashinfer_mla_backend.py)

相对 PR 的主要差异：

- 最终版本没有完全回到 PR 所在基线的 import 结构。
- 这里主要调整为和最终版 [python/sglang/srt/layers/attention/utils.py](python/sglang/srt/layers/attention/utils.py) 的接口保持一致，尤其是 create_flashinfer_kv_indices_triton 和 update_kv_lens_and_indices 的来源。

原因：

- 这个文件本身改动不大，主要是为了跟 attention utils 的最终落地接口对齐。

### [python/sglang/srt/layers/attention/utils.py](python/sglang/srt/layers/attention/utils.py)

这是和 PR 差异最大、也是最重要的手工混合合并文件。

相对 PR 的主要差异：

- 保留了当前分支已有的 GroupCoordinator、is_cuda、现有 Triton helper、update_kv_lens_and_indices 等实现。
- 没有照搬 PR 的整个文件，而是只吸收了 PR 对 cp_lse_ag_out_rs、correct_attn_out 及其输出布局的关键语义。
- 最终版的 cp_lse_ag_out_rs 接口支持 return_lse，供 flashinfer prefill 和 decode 路径共用。
- 最终版保留了当前分支对 DCP KV index 生成和本地 prefix 逻辑的兼容性，不把这些逻辑回退。

原因：

- PR 的 attention 输出修正接口是必须要的。
- 但当前分支在 attention utils 上已经演进了很多，如果整份覆盖会丢掉当前分支现有能力。
- 所以这里采用的是“PR 接口语义 + 当前分支实现骨架”的混合合并方案。

### [python/sglang/srt/managers/scheduler.py](python/sglang/srt/managers/scheduler.py)

相对 PR 的主要差异：

- 最终版本保留了当前分支的 scheduler 主体实现，没有回退到 PR 所在基线版本。
- 这里进入差异列表，主要反映的是当前分支 scheduler 已经比 PR 基线更丰富，最终落地保留了这些现有能力。

原因：

- scheduler 是大文件，强行贴近 PR 版本会产生更大行为风险。
- 因此这里的策略是保留当前分支版本，不为了追求“和 PR 文本接近”而做大回退。

### [python/sglang/srt/managers/scheduler_runtime_checker_mixin.py](python/sglang/srt/managers/scheduler_runtime_checker_mixin.py)

相对 PR 的主要差异：

- 最终版本保留了当前分支已有的 helper 和 runtime stats 流程。
- 在现有逻辑上只引入了 PR 需要的 DCP 语义：allocator available_size 和 tree_cache evictable_size 按 dcp_size 做缩放。

原因：

- 这能保持当前分支 runtime checker 结构不变，只修正 PR 关心的 DCP 统计口径。

### [python/sglang/srt/mem_cache/memory_pool.py](python/sglang/srt/mem_cache/memory_pool.py)

相对 PR 的主要差异：

- 最终版本保留了当前分支已经更复杂的 memory pool 和 allocator 体系，没有退回到 PR 基线版本。
- 该文件的最终落地需要和你当前分支的 token_to_kv_pool、allocator、HiSparse 兼容逻辑保持一致。

原因：

- 这里的差异主要来自保留当前分支实现，而不是对 PR 额外发明新行为。

### [python/sglang/srt/model_executor/cuda_graph_runner.py](python/sglang/srt/model_executor/cuda_graph_runner.py)

这是我明确额外补的适配，不是 PR 在这个文件里的原样改动。

相对 PR 的主要差异：

- 在 DecodeInputBuffers 中新增 dcp_kv_mask 字段。
- 在 DecodeInputBuffers.create 中，当 dcp_size 大于 1 时分配一块 bool 类型的 dcp_kv_mask graph buffer。
- 在 populate_from_forward_batch 中，把 forward_batch.dcp_kv_mask 批量拷贝到 graph buffer。
- 在 capture_one_batch_size 中，从 buffers 取出 dcp_kv_mask 切片，再在构造 ForwardBatch 时传回 dcp_kv_mask。

原因：

- PR 把 dcp_kv_mask 语义引入到了 ForwardBatch 和 attention 写 KV 路径。
- 但你当前分支的 CUDA graph 静态输入缓冲原本没有这条输入链路。
- 如果不补这里，eager 路径和 cuda graph 路径对 dcp_kv_mask 的可见性会不一致。

### [python/sglang/srt/model_executor/forward_batch_info.py](python/sglang/srt/model_executor/forward_batch_info.py)

相对 PR 的主要差异：

- 最终版本没有用 PR 的字段集合覆盖当前分支。
- 而是在当前分支已有的 DCP 相关字段基础上，增加了 dcp_kv_mask。
- 因此最终版同时保留了 dcp_kv_indptr、dcp_kv_buffer、dcp_kv_indices、dcp_local_prefix_kv_indices、dcp_extend_prefix_lens_sum 等字段。

原因：

- 当前分支本身已经有更完整的 DCP batch 数据结构。
- PR 只补了其中一块语义，最终落地需要把它并进去，而不是替掉现有结构。

### [python/sglang/srt/model_executor/model_runner.py](python/sglang/srt/model_executor/model_runner.py)

相对 PR 的主要差异：

- 最终版本保留了当前分支 initialize_model_parallel 调用中 attention_context_model_parallel_size、moe_data_model_parallel_size 等参数。
- 在这个现有调用框架上额外传入了 decode_context_parallel_size 等于 self.dcp_size。

原因：

- 这让 PR 的 DCP 初始化语义进入当前分支，同时不破坏当前分支更丰富的并行参数体系。

### [python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py](python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py)

这是一个非常小但非常明确的额外适配。

相对 PR 的主要差异：

- 在 cp_lse_ag_out_rs 调用之后，最终版删除了额外的一次 attn_output transpose。

原因：

- 最终版 [python/sglang/srt/layers/attention/utils.py](python/sglang/srt/layers/attention/utils.py) 采用了 PR 的 cp_lse_ag_out_rs 输出布局。
- 如果这里还保留旧布局假设，会导致 DeepSeek MLA 路径的输出张量形状约定不一致。

### [python/sglang/srt/models/utils.py](python/sglang/srt/models/utils.py)

这是我手工做的行为合并，而不是简单照抄 PR。

相对 PR 的主要差异：

- PR 的语义是：DCP world size 不为 1 时，不启用 fused set_kv_buffer。
- 当前分支原本已有另一层保护：prefill context parallel 启用时，也不要启用 fused set_kv_buffer。
- 最终版本把这两层条件合并了起来，而不是只保留 PR 的一层。

原因：

- 这样可以避免把当前分支已经存在的保护条件回退掉。
- 最终行为比 PR 更保守，但更符合当前分支已有设计。

### [python/sglang/srt/server_args.py](python/sglang/srt/server_args.py)

相对 PR 的主要差异：

- 最终版本保留了当前分支已经存在的 attention-context-parallel-size、attn-cp-size、moe-data-parallel-size、moe-dp-size 等参数定义。
- 在此基础上加入了 PR 的 decode-context-parallel-size 和 dcp-size。
- 没有把整个 ServerArgs 定义退回到 PR 所在基线版本。

原因：

- 当前分支的参数体系已经比 PR 基线更丰富。
- 最终落地目标是把 DCP 参数接进来，而不是回退当前分支已有参数面。

### [test/srt/run_suite.py](test/srt/run_suite.py)

相对 PR 的主要差异：

- PR 在这里带来了一组 not_in_ci 调整。
- 最终版本没有直接用 PR 覆盖，而是把 PR 的新增项和当前分支已有项做了并集。
- 因此保留了当前分支已有的 ascend/test_embed_interpolate_unittest.py 排除项，同时吸收了 PR 的 test_dcp.py、test_profile_v2.py 等项。

原因：

- 这样不会把当前分支本来就排除的测试配置丢掉。

## 可以这样理解这些差异

最终落地版本相对 PR 14982 的差异，主要分成三类：

- 第一类：必要的当前分支适配。
  代表文件是 [python/sglang/srt/model_executor/cuda_graph_runner.py](python/sglang/srt/model_executor/cuda_graph_runner.py) 和 [python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py](python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py)。

- 第二类：保留当前分支更完整的现有实现，只把 PR 的 DCP 核心语义合进去。
  代表文件是 [python/sglang/srt/distributed/parallel_state.py](python/sglang/srt/distributed/parallel_state.py)、[python/sglang/srt/server_args.py](python/sglang/srt/server_args.py)、[python/sglang/srt/model_executor/model_runner.py](python/sglang/srt/model_executor/model_runner.py)、[python/sglang/srt/managers/scheduler_runtime_checker_mixin.py](python/sglang/srt/managers/scheduler_runtime_checker_mixin.py)。

- 第三类：attention 路径的混合式合并。
  代表文件是 [python/sglang/srt/layers/attention/flashinfer_backend.py](python/sglang/srt/layers/attention/flashinfer_backend.py) 和 [python/sglang/srt/layers/attention/utils.py](python/sglang/srt/layers/attention/utils.py)。

其中最值得重点审查的是：

- [python/sglang/srt/layers/attention/flashinfer_backend.py](python/sglang/srt/layers/attention/flashinfer_backend.py)
 - [python/sglang/srt/layers/attention/utils.py](python/sglang/srt/layers/attention/utils.py)
 - [python/sglang/srt/model_executor/cuda_graph_runner.py](python/sglang/srt/model_executor/cuda_graph_runner.py)
 - [python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py](python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py)

## 关键代码差异片段

下面这些片段不是完整文件 diff 转抄，而是把最有决策意义的差异点摘出来，方便你直接审。

### [python/sglang/srt/model_executor/cuda_graph_runner.py](python/sglang/srt/model_executor/cuda_graph_runner.py)

这个文件的差异基本都是我额外补的，不是 PR 原样内容。

PR 没有在这个文件里补 graph buffer 的 DCP mask 接线；最终版本补了：

```python
@dataclass
class DecodeInputBuffers(ForwardInputBuffers):
  ...
  dcp_kv_mask: Optional[torch.Tensor]

  @classmethod
  def create(..., dcp_size: int, ...):
    ...
    if dcp_size > 1:
      dcp_kv_mask = torch.zeros((max_num_token,), dtype=torch.bool)
    else:
      dcp_kv_mask = None
    ...
```

这一步的意义是把 dcp_kv_mask 变成 CUDA graph 可复用的静态输入。

最终版本还补了从运行时 batch 到 graph buffer 的拷贝：

```python
if self.dcp_kv_mask is not None and forward_batch.dcp_kv_mask is not None:
  dsts.append(self.dcp_kv_mask[:raw_num_token])
  srcs.append(forward_batch.dcp_kv_mask)
```

以及 capture 时再把 graph buffer 回挂到 ForwardBatch：

```python
if self.dcp_size > 1:
  dcp_kv_mask = buffers.dcp_kv_mask[:num_tokens]
else:
  dcp_kv_mask = None

forward_batch = ForwardBatch(
  ...
  dcp_kv_mask=dcp_kv_mask,
)
```

这三段组合起来，确保 eager 路径和 cudagraph 路径都能看到同一条 dcp_kv_mask 语义链。

### [python/sglang/srt/model_executor/forward_batch_info.py](python/sglang/srt/model_executor/forward_batch_info.py)

PR 的重点是引入 dcp_kv_mask；最终版本没有用 PR 的字段集合覆盖当前分支，而是做并集。

最终版本保留并扩展成：

```python
class ForwardBatch(ForwardBatchDeepSeekMHAMixin):
  ...
  dcp_kv_indptr: Optional[torch.Tensor] = None
  dcp_kv_buffer: Optional[torch.Tensor] = None
  dcp_kv_indices: Optional[torch.Tensor] = None
  dcp_local_prefix_kv_indices: Optional[torch.Tensor] = None
  dcp_extend_prefix_lens_sum: Optional[int] = None
  dcp_kv_mask: Optional[torch.Tensor] = None
```

这说明最终版本不是“采用 PR 的更小字段集”，而是“保留当前分支已有 DCP batch 状态，再补上 PR 的 mask”。

### [python/sglang/srt/models/utils.py](python/sglang/srt/models/utils.py)

这里是行为合并，而不是照抄 PR。

PR 的核心语义可以概括成：

```python
return get_dcp_group().world_size == 1
```

最终版本保留了当前分支原有的 prefill context parallel 保护，再叠加 PR 的 DCP 条件：

```python
return (
  not is_prefill_context_parallel_enabled()
  and get_dcp_group().world_size == 1
)
```

所以最终行为比 PR 更保守：prefill CP 和 DCP 两种情况下都禁用 fused set_kv_buffer。

### [python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py](python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py)

这处改动很小，但它是我明确做的额外适配。

PR 对应调用链引入了新的 cp_lse_ag_out_rs 输出布局后，最终版本删除了旧布局假设：

```python
attn_output = cp_lse_ag_out_rs(attn_output, attn_lse, get_dcp_group())
- attn_output = attn_output.transpose(0, 1)
```

如果这里不删，DeepSeek MLA 路径会继续按照旧输出布局做一次额外转置。

### [python/sglang/srt/layers/attention/flashinfer_backend.py](python/sglang/srt/layers/attention/flashinfer_backend.py)

这是 attention 路径里最重要的混合式合并文件。

最终版本在 metadata 更新器里显式传入 dcp_size 和 dcp_rank：

```python
self.indices_updater_decode.update(
  ...
  dcp_size=self.dcp_size,
  dcp_rank=self.dcp_rank,
)
```

prefill 路径增加了 DCP 下 q 的 all-gather 和 return_lse 修正：

```python
if self.dcp_size > 1:
  q = get_dcp_group().all_gather(q, dim=1)

o2, s2 = prefill_wrapper_paged.forward_return_lse(
  q.view(-1, layer.tp_q_head_num * self.dcp_size, layer.head_dim),
  ...
)

if self.dcp_size > 1:
  o2, s2 = cp_lse_ag_out_rs(o2, s2, get_dcp_group(), return_lse=True)
```

decode 路径改成了 forward_return_lse + cp_lse_ag_out_rs：

```python
with use_symmetric_memory(get_dcp_group()):
  o, s = decode_wrapper.forward_return_lse(
    q.view(-1, layer.tp_q_head_num * self.dcp_size, layer.head_dim),
    ...
  )
if self.dcp_size > 1:
  o = cp_lse_ag_out_rs(o, s, get_dcp_group())
```

写 KV 时不是死写 PR 版本，而是做成条件 kwargs：

```python
args = (layer, cache_loc, k, v, layer.k_scale, layer.v_scale)
kwargs = {}
if self.dcp_size > 1:
  kwargs["dcp_kv_mask"] = forward_batch.dcp_kv_mask
forward_batch.token_to_kv_pool.set_kv_buffer(*args, **kwargs)
```

这说明最终版本是在保留当前分支 backend 结构的前提下，把 PR 的 DCP 关键语义嵌进去。

### [python/sglang/srt/layers/attention/utils.py](python/sglang/srt/layers/attention/utils.py)

这个文件很难用几行覆盖全部差异，但最关键的变化是接口和输出布局。

最终版本采用的是 PR 需要的这类接口：

```python
def cp_lse_ag_out_rs(
  out: torch.Tensor,
  lse: torch.Tensor,
  cp_group: GroupCoordinator,
  return_lse: bool = False,
  ctx=None,
):
  ...
```

也就是说，最终版本支持 return_lse，供 prefill merge_state 和 decode 修正共用。

但与此同时，最终版本没有把当前分支已有的 Triton helper 和 DCP KV index 逻辑删掉，而是保留了诸如：

```python
create_flashinfer_kv_indices_for_dcp_triton(...)
update_kv_lens_and_indices(...)
```

所以这不是“PR 文件原样落地”，而是“PR 接口语义 + 当前分支现有实现”的混合方案。

### [python/sglang/srt/distributed/parallel_state.py](python/sglang/srt/distributed/parallel_state.py)

这个文件的最终版本保留了当前分支的大量并行能力，只把 PR 必需的 DCP 语义插入进去。

PR 相关而且最终保留的代表性代码有：

```python
def get_dcp_group() -> GroupCoordinator:
  assert _DCP is not None, "decode context parallel group is not initialized"
  return _DCP
```

```python
def reduce_scatter_along_dim(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
  ...
  self.reduce_scatter_tensor(output_tensor, input_tensor)
  return output_tensor.movedim(0, dim)
```

```python
def ensure_model_parallel_initialized(..., decode_context_parallel_size: int, ...):
  ...
  assert dcp_world_size == decode_context_parallel_size
```

但最终版本同时保留了当前分支额外能力，例如 piecewise graph 下的 pynccl outplace allreduce、更多 graph_capture 参与 group、更多 distributed backend 逻辑。这部分是“保留当前分支”，不是“来自 PR”。

### [python/sglang/srt/model_executor/model_runner.py](python/sglang/srt/model_executor/model_runner.py)

这里的差异集中在 initialize_model_parallel 的参数接线。

PR 想表达的是把 DCP size 传进去；最终版本是在当前分支更完整的调用上补这一项：

```python
initialize_model_parallel(
  tensor_model_parallel_size=self.tp_size,
  expert_model_parallel_size=self.ep_size,
  pipeline_model_parallel_size=self.pp_size,
  attention_context_model_parallel_size=self.attn_cp_size,
  moe_data_model_parallel_size=self.moe_dp_size,
  decode_context_parallel_size=self.dcp_size,
  ...
)
```

### [python/sglang/srt/server_args.py](python/sglang/srt/server_args.py)

这个文件的关键差异不是某一行，而是参数集合的保留策略。

最终版本保留了当前分支已有参数，同时再补上 PR 的 DCP 参数入口：

```python
parser.add_argument(
  "--decode-context-parallel-size",
  "--dcp-size",
  type=int,
  default=1,
)
```

但没有删除当前分支已有的：

```python
--attention-context-parallel-size
--attn-cp-size
--moe-data-parallel-size
--moe-dp-size
```

### [python/sglang/srt/entrypoints/engine.py](python/sglang/srt/entrypoints/engine.py)

这个文件我真正吸收 PR 的地方，集中在 DCP 场景下的 NCCL_GRAPH_MIXING_SUPPORT。

最终版本对应逻辑是：

```python
if (
  ("NCCL_GRAPH_MIXING_SUPPORT" not in os.environ or server_args.enable_symm_mem)
  and server_args.dcp_size > 1
):
  os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"
```

其余大部分和 PR 的巨大文本差异，主要来自保留当前分支已经更完整的 Engine 实现，而不是我额外围绕 PR 发明了新逻辑。

### [python/sglang/srt/managers/scheduler_runtime_checker_mixin.py](python/sglang/srt/managers/scheduler_runtime_checker_mixin.py)

这里的关键变化很聚焦，就是把池子统计按 dcp_size 缩放。

最终版本等价于：

```python
token_kv_available_size = self.token_to_kv_pool_allocator.available_size() / self.dcp_size
tree_cache_evictable_size = self.tree_cache.evictable_size() / self.dcp_size
```

也就是说，这里不是结构性重写，而是在当前分支统计逻辑里插入 PR 需要的 DCP 口径修正。

### [test/srt/run_suite.py](test/srt/run_suite.py)

这里不是直接采用 PR 的 not_in_ci 列表，而是做并集。

最终版本保留了当前分支已有条目，再吸收 PR 的新增项，例如：

```python
__not_in_ci__ = [
  ...
  "ascend/test_embed_interpolate_unittest.py",
  "test_dcp.py",
  "test_profile_v2.py",
  ...
]
```

这样不会把当前分支原来已经排除的测试项覆盖掉。
- [python/sglang/srt/layers/attention/utils.py](python/sglang/srt/layers/attention/utils.py)
- [python/sglang/srt/model_executor/cuda_graph_runner.py](python/sglang/srt/model_executor/cuda_graph_runner.py)
- [python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py](python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py)

因为这几处不只是文本冲突处理，而是我真正做了接口对齐或行为适配。