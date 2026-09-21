# CP8 CuTe DSL：64k C10 有限 prefill 采样

固定generation11、8×B300、源码753bbc090；使用同一65536-token请求集、C10、max_tokens512。10/10请求完成，采集runner exit0。CPU/GPU、num_steps2、详细step标注，关闭stack/shapes；没有改变服务源码或关闭已有CUDA Graph。完整tar SHA256=`0e981db98c6689293e181ed8d7117765df22958ee6907ffffeae995dbd6ba1c5`，8个trace逐文件hash已核对；原始trace保留于profile64k/profiles及远端/work/profiles/cp8-cutedsl-profile64k-20260922。

实际覆盖的是最先执行的一个64k请求的两个32k EXTEND chunk及关联draft_extend；每个rank 8896个kernel，并非10个请求全生命周期。目标prefill GPU step约569/560ms，draft约12ms/步。CPU step时长与GPU时长不同，不能相加；profile有开销，其请求延迟不参与性能门槛。

| Rank | kernel时长总和 ms | GPU忙碌区间并集 ms | GPU首尾跨度 ms | 并集/跨度 |
| --- | --- | --- | --- | --- |
| 0 | 1120.6 | 1118.4 | 1155.5 | 96.8% |
| 1 | 1130.9 | 1128.2 | 1155.2 | 97.7% |
| 2 | 1116.4 | 1113.7 | 1155.5 | 96.4% |
| 3 | 1126.9 | 1124.3 | 1154.9 | 97.3% |
| 4 | 1120.4 | 1117.5 | 1155.1 | 96.8% |
| 5 | 1130.1 | 1127.4 | 1155.4 | 97.6% |
| 6 | 1120.1 | 1117.3 | 1154.9 | 96.7% |
| 7 | 1120.5 | 1118.9 | 1154.6 | 96.9% |

## Kernel与源码归因

| TP0 kernel/类别 | kernel时长 ms / 占总和 | 来源与置信度 |
| --- | --- | --- |
| 两个主要DeepGEMM grouped GEMM | 296.44 / 26.5% | MoE grouped计算，kernel与standard DeepGEMM路径相符；无stack，具体Python调用位置未直接捕获 |
| TRTLLM FP8 sparse FMHA | 124.43 / 11.1% | DSA attention；kernel识别高置信 |
| NCCL AllGather + ReduceScatter | 201.49 / 18.0% | CP相关集合通信；具体调用点需额外映射，时长包含可能的等待 |
| post_reorder_deepgemm | 99.69 / 8.9% | kernels/ops/moe/ep_moe_kernels.py，命名唯一匹配 |
| FP8 Fill + ep_scatter | 144.76 / 12.9% | standard DeepGEMM packed_input清零及分发候选；Fill本身没有Python栈 |

## 重叠机会

| 观察 | 判断 | 后续约束 |
| --- | --- | --- |
| 所有rank GPU忙碌占比96%–98%，rank时长差异小 | 当前两步没有大量GPU空闲或单rank拖尾证据 | 不能外推到全部10请求/更长上下文；集合通信时长不等于可隐藏时间 |
| AllGather/ReduceScatter约18% kernel总和 | 存在通信成本，但尚未证明独立生产者/消费者可以重叠 | 不据此承诺18%加速，不盲开overlap开关 |

## 融合/布局机会

| 路径 | 证据 | 决策 |
| --- | --- | --- |
| MoE清零、scatter、post-reorder | 合计显著；固定source的pre_permute_standard_to_deep_gemm确有packed_input zeros | 先试原生专家分片/布局，避免立即编写融合kernel |
| DSA topk与RoPE/attention | trace已有独立prefill topk、FlashInfer RopeQuantize和attention kernel | 技能自动表把topk_transform_prefill误标为MoE gate，且把FMHA计入RoPE融合家族；这些启发式Confirmed不作为融合证据 |
| EP8 standard dispatcher | source支持local expert映射、non-local -1过滤及DeepGEMM标准分发 | 保留CP8和graph，下一候选仅加ep-size8、moe_a2a_backend保持none；正确性/收益仍需实测 |

本次只有实际配置的单次profile，没有另起eager mapping配置；prefill本来未启用graph。kernel计时和rank覆盖置信度高，Python调用链归因有限。技能原始输出保存在profile64k-skill-analysis.txt，需结合上述纠正读取。

补充停止状态：自动num_steps停止生成全部8个trace后，脚本再次调用stop_profile，HTTP500。固定源码在profile_in_progress=False时返回success=False，tokenizer_control_mixin将其转为RuntimeError；该行为与重复停止一致，但旧Pod已完成替换，未取得当次500的服务日志，保留这一诊断限制。旧Pod已删除，没有遗留profiling进程；本结论不把HTTP500当成功控制响应。
