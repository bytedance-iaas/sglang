# C：改变工作量、状态物化与串行依赖的结构研究

日期：2026-09-24。材料 E=`eef8d29285ff2493b79bd696caeeb877ba664063`；源码 S=`57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c`。主线程独立分路研究，未调用独立子 agent。本文件是机制设计与条件判断，没有原型、部署、压测或新质量结果。唯一设计为 [DESIGN](../DESIGN.md)。

## 1. 先分清三个不同问题

1. **减少存储的状态份数**：snapshot 改 ring/accepted-prefix fold，首先改变 scratch 与写流量。
2. **减少 verify 的串行 recurrence 依赖**：直接计算一组 causal 输出，改变计算图；不等于只写 ring。
3. **减少生成 proposal 的串行轮数**：native NextN 链改 block-parallel draft，改变 drafter 工作量；不等于更高接受率。

固定 S 的 Triton verify 在 `for ... in range(0,T)` 内做 gate、delta update、输出，然后写 snapshot 或 raw-input ring；所以它已经实现部分第 1 类能力，但不能据此宣布第 2 类已完成。[fixed-verify][fixed-triton] 本轮不把“开 Replay”作为这三类问题的共同答案。

<a id="c1"></a>
## 2. C1：短窗口 KDA output-only verify + 精确接受边界提交

### 2.1 机制来源与本地新问题

[ReplaySSM 原作者文章][replay] 区分输出生成与状态物化，并讨论 speculative output-only 计算。其公开整机结果的模型、NVFP4/B300 与本场景不同，只作为机制来源，不作本地收益依据。成熟 fused+ring 适配归 B2；这里研究的是 **H20、GLM channel-wise gate、FP32 状态、短 T 链的 verify 计算图**，不是给现有开关重新命名。

本地 KDA 与标量衰减 Mamba/GDN 不同：gate 对 K 维各通道生效，delta correction 依赖此前状态。不能把标量衰减公式直接套上去。下面从固定 kernel 的递推形式推导候选等价表达；它是本轮数学推导，尚无浮点等价与性能验证，也不宣称算法原创。

### 2.2 可检查的数学形式

取单 head，使用归一化且已应用原 scale 的 q、归一化 k、conv 后原始 v，state 采用 `H_t ∈ R^(K×V)`；令 `D_t=diag(exp(g_t))`，beta 为原门值。固定递推可写为：

```text
u_t = beta_t * (v_t - (D_t H_(t-1))^T k_t)
H_t = D_t H_(t-1) + k_t u_t^T
o_t = H_t^T q_t
```

令 `P_t = D_t ... D_1`，`D_(j+1:t)=D_t ... D_(j+1)`，空乘积为 I。展开后：

```text
H_t = P_t H_0 + sum_(j<=t) D_(j+1:t) k_j u_j^T
z_t = beta_t * (v_t - H_0^T P_t k_t)
L_(t,j) = beta_t * k_t^T D_(j+1:t) k_j       (j<t；其它为0)
(I + L) U = Z                              (U/Z 按时间堆叠行向量)
o_t = H_0^T P_t q_t + sum_(j<=t) [q_t^T D_(j+1:t) k_j] u_j
```

这允许把大 state 的逐 token 更新转为 checkpoint readout、短窗口系数和三角求解/分块组合；三角系统仍有依赖，不能称完全没有串行工作。价值假设是缩小关键依赖和 state 写出，而非减少所有 FLOPs。短 T、head 数、register/SMEM、并行度、额外 scratch 和 H20 上 GEMM 利用率共同决定胜负。

FP32 累计与 checkpoint 必须保持；不通过 BF16 存放 recurrence state 换速度。对 channel gate 的乘积/指数需处理下溢与动态范围，不能用修改 gate/clamp 模型来掩盖数值问题。代数等价不保证相同舍入、greedy token 或任务质量。

### 2.3 本地需要新增什么

**窗口内**：准确的 raw inputs/gate/beta 与 checkpoint 绑定、causal 输出 kernel、mask/padding、FP32 reference。**接受后**：接受长度直到全模型 verification/sampling 后才已知，不能在 verify 开始时“只写已接受 state”；必须保留 H0 与必要输入，再沿原 recurrence 顺序 fold accepted prefix，或者从合法快照提交。**窗口间**：先以每轮物化 committed state 的有界方案验证；跨窗口 ring 保留是单独选项，不是默认要求。

须处理接受 0/1/部分/全部、bonus token 的引擎计数约定、下一轮 seed、draft extend、请求结束/取消、slot reuse、PD 接收、graph ghost rows。所有状态仅在正确 committed boundary 对外可见；不得发布 speculative state 到 prefix cache。GLM 的 NextN MLA 上下文与 trunk KDA 提交应一致，但不虚构 NextN KDA 层。

### 2.4 同尺度备选、投入与采用门

对照是 S 的 unfused snapshot verify、B2 的 batch-aware fused snapshot/ring，以及 A2 的浅 native NextN；不能只跟一个已知慢的旧 kernel 比。已有 ring/fold 可复用，但接口与数值逐项重新核对，不整体重写 engine。机制原型研究为中等投入，完整生产集成为大投入。

**建议的有界研究题**：先限制 dense chain、实际 GLM head/gate/FP32 state、T2/T6 代表窗口、C80 对应本地 batch 邻域；大 batch 为第二个判断点。T2 是便于对照的研究形状，不宣称现成 accepted-state 窄路径能覆盖 T6。先建立 FP32 数值与状态边界参考；无需先部署完整模型，也不得因为本设计而自动申请 GPU。

**停止/重开条件**：若新的三角系数、状态 readout 和 accepted fold 总成本大于所省的 recurrence/写出，或长期数值漂移未过门，则保留反例，转向 A2/B2/B1。若 Q3 显示 KDA 不在主要关键路径，降低优先级而非证明机制永久无价值；新 B/T/head 或更强的精确 kernel 可重开。

**C80 判断**：以同一 C80 的 wall time/committed tokens 和 ITL 为准，省显存本身不计加速。**大 batch 判断**：同时评估减少 scratch 带来的可填充 B、compute 增量、PD 供给和 SLO goodput，不能只报告理论并发容量。

<a id="c2"></a>
## 3. C2：block-parallel draft 替代原生 NextN 串行链

[DFlash 原始论文][dflash-paper] 与[原作者仓库][dflash-repo]提供用轻量 block diffusion 并行 proposal 的路线。DFlash2 是后续工程演进，不能把原论文实验值当作所有新实现的保证。其机制可能降低多轮 native NextN 的串行成本，但接受率、hidden-state 提取/传输、draft 模型权重/KV、target verify 和 commit 都有代价。

**本地适配范围。** 精确匹配 GLM-5.3-Flash revision、tokenizer、hidden-state 层选择与维度、target/draft 量化解释；保持 target 的分布校正/拒绝采样语义，不把“提高阈值接受更多”当加速。需要 P 侧 draft 上下文产生或合法补算、P/D 非对称 head sharding 与 draft KV sidecar、PP 后 hidden-state 生命周期、KDA rollback 和 finite graph tiers。已有 DSPARK/DFLASH guard/接口不等于这个整套 GLM/H20/PD 组合可直接运行。

**checkpoint 访问结论。** 本轮读取的原作者 README（blob `e84d215cee572291cccb95cfb070c0b169aee909`）未列出经过本场景资格确认的 GLM-5.3-Flash checkpoint；SGLang 的 GLM+DFlash2 issue 是存在使用尝试的线索，不是权重身份或质量认证。GLM DSpark preview 的官方 HF 页面访问失败，不使用镜像站摘要证明已获取/可采用。也不宣布社区不存在兼容权重。

**关键组合反例。** [SGLang #41038][draft-pd] 在 2026-09-24 报告非对称 P/D TP 的 draft-head/stride 传输问题，作者的短回归只证明特定 B300 传输完成率，不是全质量通过。该风险与 target MLA 可传输不矛盾。

**相近投入备选。** A2 优化既有 native NextN 不需新 checkpoint；C1 不需训练新 drafter，但可能只降低 target verify。block draft 的完整适配投入较大；如果兼容权重不可获取，不把训练新 drafter、收集客户数据或新增硬件自动纳入本轮预算。可先形成接口与成本模型，权重/授权/PD 状态契约闭合后再升为近期 B 类工程适配。

**分场景判断。** C80 若 draft 轮次占暴露延迟较高，值得争取；大 B 已算力/通信饱和时，更多并行 proposal 可能反而降低 goodput。采用门是整个 proposal→verify→commit 窗口净收益、真实质量、长上下文与生产状态通过，不承诺接受率，不预设 block8 为最优。

## 4. 没有采用的捷径与其它机制视野

训练阶段的稀疏化/路由重分配不自动成为推理等价优化。低精度 recurrent state、改 KPool/top-k、token dropping、减少生成长度/推理内容、使用 benchmark 特化随机路由都不满足本轮语义，不进入候选默认值。其它硬件的 FP4 MoE、SM100-only kernel 原版不直接采用；可借鉴 IO/tiling 思路，但固定 S 已有 SM90 MegaMoE，应先按 A3 判断局部差异，不能仅因名字不同就宣称缺失完整能力。

将 exact state offload/跨层 pipeline 作为远期资源组织备选：仅在真实大 batch 受内存约束、带宽与重叠窗口可闭合时重开；C80 未见容量阻塞前，不先增加 PD/PCIe 搬运。在本轮范围和证据下，C1 的可检验增量比泛化 offload 平台更明确。

## 5. 研究闭合边界

本轮完成了机制分解、固定实现差异、KDA 形式推导、关键接口、备选与停止条件。尚未完成浮点参考实验、kernel 原型、接受/质量验证或 E2E 收益；这些不是本文隐含授权。C1 的研究推荐不解除历史 Replay 质量失败，C2 的接口研究不意味着兼容 checkpoint 已读到。

来源读取日期均为 2026-09-24。ReplaySSM 使用原作者 HTML 机制说明；DFlash 论文采用公开摘要与原作者 README，未声称读完 PDF。不同论文/PR 不是本地独立测量重复。

[fixed-verify]: https://github.com/bytedance-iaas/sglang/blob/57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c/python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py
[fixed-triton]: https://github.com/bytedance-iaas/sglang/blob/57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c/python/sglang/srt/layers/attention/linear/kernels/kda_triton.py
[replay]: https://tridao.me/blog/2026/replayssm/
[dflash-paper]: https://arxiv.org/abs/2602.06036
[dflash-repo]: https://github.com/z-lab/dflash/blob/main/README.md
[draft-pd]: https://github.com/sgl-project/sglang/issues/41038
