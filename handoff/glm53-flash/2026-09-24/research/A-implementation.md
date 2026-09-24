# A：固定实现、真实配置边界与近期工程机会

研究日期：2026-09-24。输入材料 E：`eef8d29285ff2493b79bd696caeeb877ba664063`；固定工程源码 S：`57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c`，tree `17ea448109e9ac7b7674b1b13fa64528380632d1`。本记录由主线程独立开展，未调用独立子 agent。没有部署、压测、编译或实施。唯一决策入口是 [DESIGN](../DESIGN.md)，完整候选目录见 [INDEX](INDEX.md)。

证据标记：**E-摘要**为交接包转述的工程结果，未重读其原始测量；**S-源码**为本轮直接检查的固定实现；**U-外部**为原作者资料；**H-推断**为本轮推导，不是实测收益。

## 1. 已恢复的对照，不是已证明最优的方案

[E-摘要：BRIEF 与 launch-e2] 16×H20，P 为 TP4/EP4/PP2，D 为 TP8/DP8/EP8/PP1，Mooncake RDMA；输入 65536、输出 1536，固定 C80。E2 的 D 为 EAGLE 5/1/6、FP8 KV、FP32 KDA state、Replay off；历史约束包括每张 D 卡至少 8 GiB 空闲、真实请求正确、质量不劣、没有异常 retract/restart。旧任务已经结束，这不是当前在线服务验证。

材料给出的镜像摘要为 `sha256:d33d932aee374f3884e7f49c54447c82ce61ac9e58749ebf5aa3a50d25c9cb28`，DeepGEMM 包版本为 0.1.7。模型 inventory 仍是 `ZhipuAI/GLM-5.3-Flash@master`，不能当不可变权重版本。原始 trace、质量逐项结果、实载 config、wheel 源码/build hash 未共享；相应限制列入 REQUESTS，而不是声称已复现。

[S-源码] Git 比较确认 E 直接以 S 为父提交，S→E 只新增六份交接 Markdown，没有改工程源码。

## 2. 按阶段恢复实际计算，而不是只看启动 flag

| 阶段 | 固定实现与对照 | 本轮结论及剩余边界 |
|---|---|---|
| P target extend | GLM hybrid 进入 KDA `forward_extend`，packed qkv conv 已存在，随后 kernel dispatcher 的 extend；P 使用 8192 chunk、PP 24/21、microbatch 1、禁用 overlap | 不再提出“首次合并三个 conv”这种已完成工作。P 的实际 linear prefill backend 未在交接解析摘录中确认；需实载记录，不能从未指定的 CLI 猜测 |
| D target verify | Triton KDA `target_verify` → `fused_sigmoid_gating_delta_rule_update`，`disable_state_update=True`；核心仍沿 T 串行 recurrence，保存每步快照或写 ring，之后由接受阶段提交 | T6 不是六个必然接受的 token。Replay ring 的存在不等于已经实现所有 verify 输出的并行 output-only 算法 |
| D 非 verify decode | packed decode 有专门路径，受 safe-gate/backend 条件限制 | 不能把 T1 microbenchmark 代表 EAGLE 的 target verify 热路 |
| native NextN | `Glm5NextForConditionalGenerationNextN` 继承 DeepSeek NextN；pool 仅给 trunk KDA layers 分配 recurrent state，GLM NextN 是 MLA 路径 | 不能多算一层 draft KDA state，也不能把启动参数中的 speculative MegaMoE 等同于每个 draft 子模块都命中它 |
| 权重与精度 | GLM KDA projection fusion 有 checkpoint unquantized 前缀检查，融合层 `quant_config=None`；NextN 会依据 ignore 列表切换 BF16 modules | “FP8 模型”不是所有 attention/draft projection 都 FP8。改变后端、合并投影必须保留原权重解释 |
| MoE | `mega_moe_sm90.py` 存在 H20/SM90 FP8×FP8 路径，调用 DeepGEMM SM90 pre-dispatch/mega_moe，另有 shared-expert overlap 条件 | 不是直接套用 Blackwell FP8×FP4。权重就绪日志、启动成功和每阶段实际 kernel 命中是三种不同证据 |

依据：[GLM model][s-model]、[NextN][s-nextn]、[KDA backend][s-kda]、[Triton dispatcher][s-triton]、[verify kernel][s-verify]、[SM90 MoE][s-sm90]。

<a id="a1"></a>
## 3. A1：先消除确定存在的重复元数据工作，再决定 Graph 微调

**模型/系统特征 → 成本。** `KDAAttnBackend.forward_extend` 每次执行 `logical_num_tokens = int(query_start_loc[-1])`。若该 tensor 位于 GPU，会触发设备标量读取；同一请求的长度本已存在 host metadata。D 的 DSA 初始化又在 `index_kpool > 1` 时设 `needs_cpu_seq_lens=True`，所以仅看类默认值 false 或开启 KPool fusion 不能宣布所有 D2H 已消除。[s-kda][s-dsa]

**机制与本地增量。** 优先复用有效的 `extend_seq_lens_cpu`，保留缺失时的原 fallback；沿 padded/empty/mixed/track 路径核对逻辑 token 数与物理 tensor 长度，避免跨层反复读取同一个标量。随后仅在已有 trace 显示真实开销时，处理 DSA metadata 的重复构造、固定 buffer 生命周期和 graph bucket padding。不是先大范围开启 Graph，也不是删掉长度检查。

**相近投入备选。** 上游 [SGLang #38431][u-host] 已提出同一 host-length 优化，2026-09-24 查询仍 open，head `eb71d35924a95835ff74bbd7029f1e9cf79d150c`。应复用/对齐该工作而不是宣称新发明。原作者报告的是其他硬件和量化组合的 scalar-read 次数，不是本地独立 TTFT 提升。相比重写 KDA kernel，本项投入小、状态风险低；相比盲调 CUDA Graph max-bs，源码依据更直接。

**下一步与停止条件。** 下一轮 PLAN 可选这一局部改动，先核对 host-length 不变式及 padding/track 测试；运行验证另需授权。如果已有工程叠加已覆盖此变更，或真实路径没有设备读取，记录不适用并停止，不继续扫值。Q3 只限制收益归因，不阻止这一静态结论。

<a id="a2"></a>
## 4. A2：以每个已提交 token 成本选择 native NextN 深度和小 batch kernel

**成本。** C80 在 DP8 下对应每 rank 的运行请求上限/目标 B10；请求还会处于 P、队列和 PD 传输中，因此实测 D batch 未必填满 10。当前 EAGLE 5 steps 的 draft 开销、target T6 verify、接受后的提交，以及暴露的通信/host wait 都可能决定 ITL。只比较接受率，或只比较 verify kernel，都不足以选择深度。

定义同一稳态窗口的 `c = wall_time / committed_target_tokens`；分解 draft、verify、commit、未被隐藏的通信和 host gap 时，重叠部分不能重复相加。更深方案只有当它的 `c` 更低且服务 SLO、质量与状态门均通过才有价值。

**本地增量。** 先对比 E2 原生深度与一个浅层链、无投机对照，使用已有接受长度直方图选择代表点，不铺全参数笛卡尔积。高 batch 时额外 draft/verify token 可能抢占计算和通信资源，应允许浅投机或无投机成为赢家。动态 token-budget 控制是后续集成：必须先审计已有 adaptive 入口，处理 DP 同步、有限 graph tiers、在途状态和切换滞回，不宣称固定 S 完全没有 adaptive 能力。

**投影与融合。** 保留 D 的 `a_only` 为对照。`full`/`b_only` 会改变中间 tensor 与 GEMM shape，不天然更快。P 禁用 fusion 不应被当永远的约束；只有在 P 的实际投影成本值得优化时独立比较。`SGLANG_OPT_KDA_ACCEPTED_STATE=0` 对应的窄 T2 路径不能直接用于 T6。融合 verify 另见 B2，不能在质量门未闭合时强行和 Replay 组合。[s-model][s-kda]

**投入/风险。** 静态浅链研究小到中等；在线 adaptive 调度为中等，涉及全栈组合。相比同等投入重写低占比的单个 norm，native NextN 深度改变的是整条 draft/verify 工作量；但无原始测量，当前不能判定浅链或 E2 谁赢。Q1/Q3 决定分派真实性和排序，不要求 Codex 重新做完整设计。

<a id="a3"></a>
## 5. A3：P/DP/MoE 的阶段成本与并行组织

D 的 SM90 MegaMoE cap 是 **tokens per rank**，不是请求数。T6、graph padding、draft extend/warmup 都可能影响上界；不能因为 B10 就把 1024 改成 10 或 60。`FAIL_CLOSED=1` 应保留；graph capture 路径仍需单独核对 cap、buffer 和 warmup shape。SM90 路径有 FP8 scale/recipe、weight interleave、`fast_math` 与 shared-expert stream 条件，换一个 runner 名不等于同精度同布局。[s-mega][s-sm90]

P 当前为 Triton MoE、PP2 和 microbatch1。其机会有三种，按已有 trace 的主导项择一：

- 若 MoE GEMM/dispatch 是 P 的主要暴露成本，研究 H20 DeepGEMM 的 P 形状、量化布局与 PP 兼容，不借用 D 的 compact/max-token 设置直接上线。
- 若 PP bubble/跨 stage 等待主导，比较 stage 的实际层类型、权重和工作量，而非只按 24/21 层数判断平衡；小幅 microbatch/overlap 改造必须保留 mHC residual、NextN embedding 和在途状态生命周期。
- 若 D 的 DP/EP rank 等待主导，按真实 token/专家负载诊断通信和 straggler。不可采用随机路由、改专家 top-k、丢 token 或只为等长 benchmark 重分配请求。

**拓扑备选。** 固定 8D 卡内，DP8→DP4/attention-TP2 可以分摊 attention 权重，却同时增加每 worker 请求数、attention 通信及可能的 KV 复制；KDA 每请求分片减少不能直接推导总 state 内存减半。D PP2 有 stage bubble，DCP 需要合法 attention 并行组及 KPool/投机/PD 组合，均不是独立开关。旧 PP 负结果只约束原布局/阶段，保留重开条件：新的 profile 证明内存/通信瓶颈，且形状与 PD state 重分片可闭合。

**处置。** 保留为中等投入候选，不与 A1/A2 同时开展全组合搜索。Q3 若显示 P 供给不足，P 路线优先级可高于 D 的容量工作。

<a id="a4"></a>
## 6. A4：容量与 admission 的真实预算

### 更正源码定位

原 `evidence/source-map.md` 中 `pool_configurator.py` 的 `_get_num_req_slots` 指向 DSV4 专用 pool 路径，不能证明 GLM 的 extra-slots 预算。本轮直接检查的正确入口是 `kv_cache_configurator.py::_build_req_to_token_pool`、`_build_hybrid_mamba_decode_req_pool`、`resolve_max_num_reqs` 与 `handle_max_mamba_cache_size`。原证据文件保留不改，纠正在此持久化。[s-pool-wrong][s-pool]

`resolve_max_num_reqs` 把显式 max-running-requests 除以 `attn_dp_size`；token cap 在所读路径不是按 DP 再除一次。GLM target 仅分配实际 trunk recurrent layers，P 不做 D 的 per-draft state scratch 分配，这两项已有修复，不再列作待实现。

### 可复核的数量级，不是测得的分配量

单请求名义 token 数 `65536+1536=67072`。10 个请求为 670720 tokens，距 E2 每 worker 680000 cap 仅 9280 tokens；padding、reserve、在途接收和其它 pool 尚未计算。十个请求都达到 context 69632 时则超过该 cap。extra slots=10 不等于还能额外驻留十个完整长请求；必须有真实 token-credit/admission/backpressure。

若 Q1 确认采用源码默认的每层 64 heads、K=V=128，且 D attention-TP=1，则 34 层 FP32 recurrent state 的逻辑量为 `34×64×128×128×4 = 136 MiB/request`。B10 的一份逻辑 state 约 1.328 GiB，六份快照约 7.969 GiB。这不是完整 GPU allocation，也不是 Replay 的实测节省：实际层配置、slot padding、extra slots、conv、ring、draft KV、graph、MoE workspace 和通信 buffers 都需单独计入。未取得实载 config 前，不把这些默认值冒充已加载模型事实。

**机制。** 用同一个 pool manifest 绑定请求数、token cap、spec scratch/ring、graph tiers、PD 在途 credit 和每卡峰值内存。新增容量优先花在真实能填满且满足 SLO 的 batch，不先把余量全部静态分配。减短投机窗口同样可能释放 scratch，应与 Replay 同尺度比较。

**处置与重开。** 对 C80，若无 retract/等待内存的证据，容量增加不是加速理由；对更大 batch，容量路径必须同时通过质量、实际填充与 goodput 门。历史 96/128/160 档仅是已跑过的组合，不是本次推荐表。Q2/Q3/Q4 分别限制 Replay 采用、容量归因与生产有效性。

## 7. 小改进不因“大优化暂缺”而停止

A1、逻辑/物理 shape 审计、按阶段移除重复 metadata 构造、固定 graph buffer 生命周期、保留低开销 observability 均是独立可推进的小项。它们与非维护性 C1/B1 并行排队，而不是代替结构研究。生产整改分层：身份和数值语义属于研究前置；真实路径 warmup、内存检查及取消/超时闭环属于交付前置；全公司发布/观测平台重构不在本轮范围。

## 检查范围与来源

固定 S 的直接读取范围：`configs/glm5_next.py` 全文；`models/glm5_next.py` 150–625（返回末尾截断部分不计为已读）；NextN 文件全文；KDA backend 1–215、265–415、750–1400；attention hook 330–550；GLM pool 940–1125、2160–2390；MegaMoE 1–300、SM90 wrapper 全文；DSA backend 340–505；Triton KDA 1–250；verify kernel 180–455。没有声称审计完整仓库、所有 kernel、DeepGEMM wheel 或所有生命周期实现。

[s-model]: https://github.com/bytedance-iaas/sglang/blob/57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c/python/sglang/srt/models/glm5_next.py
[s-nextn]: https://github.com/bytedance-iaas/sglang/blob/57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c/python/sglang/srt/models/glm5_next_nextn.py
[s-kda]: https://github.com/bytedance-iaas/sglang/blob/57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c/python/sglang/srt/layers/attention/linear/kda_backend.py
[s-triton]: https://github.com/bytedance-iaas/sglang/blob/57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c/python/sglang/srt/layers/attention/linear/kernels/kda_triton.py
[s-verify]: https://github.com/bytedance-iaas/sglang/blob/57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c/python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py
[s-dsa]: https://github.com/bytedance-iaas/sglang/blob/57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c/python/sglang/srt/layers/attention/dsa_backend.py
[s-mega]: https://github.com/bytedance-iaas/sglang/blob/57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c/python/sglang/srt/layers/moe/mega_moe.py
[s-sm90]: https://github.com/bytedance-iaas/sglang/blob/57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c/python/sglang/srt/layers/moe/mega_moe_sm90.py
[s-pool-wrong]: https://github.com/bytedance-iaas/sglang/blob/57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c/python/sglang/srt/model_executor/pool_configurator.py#L1040
[s-pool]: https://github.com/bytedance-iaas/sglang/blob/57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c/python/sglang/srt/mem_cache/kv_cache_configurator.py
[u-host]: https://github.com/sgl-project/sglang/pull/38431
