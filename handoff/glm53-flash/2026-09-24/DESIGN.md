# GLM-5.3-Flash：首次完整推理优化设计

状态：**三路研究及条件设计完成；没有采用新的优化组合，没有实施或新增运行结果。**

- 工作流：[`pro-codex-inference-workflow@e08e17bb4e8b13ce4b1bdb3aa6bd5efaa3c08bd4`](https://github.com/HanHan009527/obsidian_remote/blob/e08e17bb4e8b13ce4b1bdb3aa6bd5efaa3c08bd4/codex/skills/pro-codex-inference-workflow/SKILL.md)，已读取 contract、pro、methods。
- 材料基点 **E**：`eef8d29285ff2493b79bd696caeeb877ba664063`；工程源码 **S**：`57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c`，源码 tree：`17ea448109e9ac7b7674b1b13fa64528380632d1`。
- 日期：2026-09-24。发布目录：`handoff/glm53-flash/2026-09-24/`。这是本任务唯一 DESIGN；详细研究与完整候选状态只在 [research/INDEX.md](research/INDEX.md) 维护。
- A/B/C 由主线程分别研究后综合，**没有独立子 agent 委派**。没有部署、编译、压测、新 trace、代码实现或工程 PLAN 修改。下述研究/验证建议不构成运行授权。

## 1. 规划摘要

### 1.1 本次建议：一个可复用对照、两种负载判断、近期小改进与结构研究并行排队

**当前保留的是 E2 的历史已验证启动组合作为工程对照，不是宣布 E2 最优，更不是确认它正在运行。** 暂不采纳 Replay、全局 fused verify、现成 FlashKDA 或新的投机模型作为替代部署。模型质量、FP32 recurrent state、完整状态生命周期与生产语义优先于性能。[BRIEF](BRIEF.md)、[启动及解析证据](evidence/launch-e2.md)

| 投入层次 | 具体工作与价值 | 当前证据、选择理由与限制 |
| --- | --- | --- |
| 下一轮近期工程选题 | **A1：KDA host-length 复用与重复元数据工作消除**；**A2/A4：native NextN 整窗口成本和容量预算联合选择** | A1 有固定源码中的设备标量读取依据，优先对齐已有上游 PR。A2 比只调接受率或单个 kernel 更接近真实成本；具体浅链/无投机/原 T6 谁赢仍需已有测量补证。两项不依赖 Replay 被采用 |
| 下一轮至少保留一项非维护性研究 | **C1：短窗口 KDA output-only verify + FP32 接受边界提交**，先限制真实 GLM 形状和 dense chain | 固定实现仍逐 token 串行 recurrence；ring 只改变存储，并未完成这项计算图改造。建议先做有界数学/接口/原型研究，而非直接生产实现；必须与 A2、B2 的同尺度备选比较 |
| 可替换的主攻方向 | **B1：长 chunk、真正 FP32 状态与 checkpoint 可组合的 KDA prefill**；若 P/TTFT 主导，优先于 C1 | 当前 FlashKDA 既有适用域回退，又有内部 BF16 state，不能直接开。若已有 trace 表明 P 才是瓶颈，应转向此项或 A3 的 P/MoE/PP 工作，不固定押注 Decode |
| 条件后续 | B2 的 batch-aware fused verify/ring、B3 完整混合 prefix cache、B4 DSA 等价后端/并行、C2 block-parallel draft | 分别由质量与形状、真实复用率、关键路径、兼容 checkpoint/PD 接口决定；不是本轮同时启动的参数矩阵，也不是永久排除 |

**本次相对 prepare 的增量**：独立恢复实际执行路径，纠正一处 GLM pool 的源码定位；识别已存在能力而不重复规划；核验 FlashKDA 的实际内部精度；发现固定源码与已合入 fused-ring 方案的差异；给出 output-only KDA 的可检查形式、完整状态接口与同尺度备选；审查启动参数中的生产交付风险。未把未共享的原始质量或性能测量当作已读证据。

### 1.2 两个场景必须分别下结论

| 场景 | 当前判断 | 下一步选择依据 | 不能算作成功的结果 |
| --- | --- | --- | --- |
| **固定 C80**，保持 65536 输入、1536 输出及同一请求/采样语义 | DP8 对应每 rank 的 D 运行目标/上限约 B10，但部分请求可能仍在 P、队列或传输中。先降低每个已提交 token 的实际成本，或消除导致 D 填不满的 P/PD 等待 | 对比完整 proposal→verify→commit 窗口、真实 rank batch、ITL/TTFT/E2E 与质量；A1/A2 优先，C1/B1 依据关键路径择一 | 只把 max-running-requests 提到 128/160；只省显存；只报告高接受率或一个快 kernel；用另一并发结果代替 C80 |
| **真实可以填满更大 Decode batch** | 容量和计算效率要共同优化；浅投机/无投机也可能因释放 scratch、减少 proposal 工作而胜出 | 比较实际填充 B、token-credit、P 供给、每卡峰值内存和满足 SLO 的 goodput；T6+Replay 仅在质量门通过后进入采用比较 | 用理论容量冒充吞吐；忽略 PD 在途内存、长尾和排队；降低质量、缩短输出或改变模型路由换容量 |

历史 Replay 的 96/128/160 档是“曾可运行”的材料，不是本轮推荐配置。C80 尚未证明它改善端到端吞吐，已有质量门也未通过；这限制 **Replay 采用**，不阻塞 A1、其它后端研究、C1 的数学分析或 P 侧优化。[证据入口](evidence/INDEX.md)

### 1.3 保留的硬约束与待决定事项

保持固定模型计算与输出语义、FP32 KDA state、真实请求正确性、质量不劣、无异常 retract/restart，以及材料中的每张 D 卡至少 8 GiB 空闲要求。16×H20 是本轮资源起点；原 TP/DP/EP/PP、微批和后端属于当前选择而非用户永久硬约束，改变它们仍须计入完整成本、组合验证和资源权限。

需要补齐但不阻塞全部设计的事项只有四类：Q1 实载身份/配置/后端，Q2 历史质量失败证据，Q3 现有性能与容量窗口，Q4 生产负载/SLO。详见 [REQUESTS.md](REQUESTS.md)。缺失时保留条件排序，不编造瓶颈比例、收益百分比、质量阈值或可用 checkpoint。

## 2. 方案与技术取舍

### 2.1 成本模型：优化提交给用户的 token，而非内部工作量计数

以同一稳态窗口的 `wall_time / committed_target_tokens` 为核心成本，并同时记录请求级 TTFT、ITL、E2E、失败率与任务质量。窗口拆解包括 draft、target verify、接受后 state commit、暴露的通信和 host gap；重叠部分不能重复相加。T6 是 verify 宽度，不等于每轮接受六个 token。

在固定 C80 下，若没有内存等待/retract，增大静态容量不产生直接速度收益。若 D 未填满，先判断 P 供给、PP bubble、PD 传输/credit 和真实请求生命周期；不能只对 D kernel 做越来越精细的优化。大 batch 下则比较整机 `goodput(B,T)`，而非令 T 和 B 各自越大越好。

容量估算也必须保持量纲：单请求名义 token 数为 `65536+1536=67072`，10 个请求为 670720，距 E2 的 680000 token cap 仅 9280；padding、reserve、其它 pool 和在途接收尚未包含。`extra slots=10` 不代表还可无条件接收十个完整长请求。详细假设、GLM pool 正确入口及条件 state 估算见 [A4](research/A-implementation.md#a4)，不将逻辑估算冒充实测 allocation。

### 2.2 近期路线：先做确定有结构依据的小项，不止停留于维护

**A1 的可交付增量**是沿固定 `KDAAttnBackend.forward_extend` 使用有效 host lengths，避免各层重复 `int(query_start_loc[-1])`；保留 host metadata 缺失时的原 fallback，以及逻辑/物理长度、padding、空 batch、checkpoint 等行为。上游 #38431 已有相同方向，应复用而非重复造轮子。随后只根据实际关键路径处理 DSA metadata、graph buffer 和 padding；不是全面重写 scheduler。[A1 的源码与 PR 核查](research/A-implementation.md#a1)

**A2 的主比较**是原生 NextN 原 T6、一个依据接受长度分布选择的浅链、无投机对照。较深的 proposal 只有在完整窗口的净成本更低时才采用。动态预算控制属于后续：先审计现有 adaptive 入口，限定 graph tiers、DP 同步和状态切换，再考虑在线调整；不宣称固定 S 完全没有 adaptive 能力，也不直接大范围扫描所有 T/B/后端组合。[A2](research/A-implementation.md#a2)

**A3 的 P/MoE/并行机会**同样保留：P 用 Triton MoE、PP2、microbatch1 和禁用 overlap，不一定最适合该形状。若 P 的 MoE/PP 等待主导，应在同卡数内比较阶段形状、量化布局、mHC/NextN 交接和暴露通信；不是借用 D 的 MegaMoE 参数直接上线。DP8→DP4/attention-TP2 或 D PP/DCP 会同时改变请求批量、通信和状态/KV 布局，不自动省下一半整机内存。[A3](research/A-implementation.md#a3)

### 2.3 下一轮非维护性研究：C1 与 B1 的明确选择

**默认有界题 C1**：针对真实 GLM channel-wise gate、FP32 state、dense chain 的短窗口，研究把大 recurrent state 的逐 token 更新，改为 checkpoint readout、短窗口系数/三角求解和 causal 输出；接受长度产生后，再按正确顺序物化 committed state。完整推导见 [C1](research/C-frontier.md#c1)。

固定 kernel 已能写 raw-input ring，但仍沿 T 循环更新 state；因此“省 snapshot”与“改变 verify 串行依赖”是两个不同增量。接受结果在全模型 verification/sampling 后才知道，不能在 verify 开始就只计算已接受 token。须保存 checkpoint 与必要输入，覆盖接受 0/1/部分/全部、bonus token 计数、下一轮 seed、取消和 graph ghost rows。初版每轮物化有界 committed state，不预设跨窗口 persistent Replay 是必要条件。

这条路线的数学等价不保证浮点等价，也不保证加速。短窗口系数构造、三角求解、readout 和 accepted fold 可能抵消所省工作。首先比较 A2 浅链、现有 unfused snapshot 和 B2 的合法 fused 变体；如果净成本不利或质量/状态不成立，应停止或改题，而不是放宽精度。机制原型投入中等，完整生产集成投入较大。

**替代题 B1**：若 Q3 显示长 prompt/P 阶段主导，而 verify 不在关键路径，改做长 chunk + 真正 FP32 recurrence/checkpoint + indexed state 的完整 KDA prefill 能力。固定 S 的 FlashKDA wrapper 会在长序列、checkpoint/spec 等场景回退；本轮进一步读到当前外部 C++ 将 FP32 输入 state 转为 BF16，故不能按 FP32 接口名称直接采用。应比较保持现有数值路径的 Triton 改良与真正 FP32 的 fused/chunk 改造，且把全模型 1×8192 与 4×2048 的额外 pass 成本计入。[B1 的固定源码和外部精度证据](research/B-model-community.md#b1)

下一轮建议只推进这两项结构题中的一项，不默认同时训练 drafter、改 DCP、重做缓存平台或更换硬件。小项 A1 不因结构题未选定而停止。

### 2.4 Replay、融合、缓存与新 drafter 的位置

**Replay / B2**：历史容量收益可以作为研究依据，但质量失败不能被“论文无损”或“上游已合入”覆盖。上游 #36821 已合入 fused verify 写 ring 的组合，固定 S 仍有拒绝该组合的 guard；这是可归属的版本差异。公开 H20 形状还有明显 batch/BV 拐点，故正确方向是少数实际形状的 batch-aware dispatch 与正确 fallback，不是全局强开。snapshot 融合可独立研究，ring 模式采用仍受 Q2 质量门约束。[B2](research/B-model-community.md#b2)

**B3 混合缓存**：S 已有 KDA track/internal snapshot，不能重新包装成新功能。值得补齐的是快 kernel、prefix boundary、DSA/KPool 与外部传输共享的一致状态契约。仅当 Q4 给出真实复用机会时提高优先级；不假定用户场景有 90% 命中。vLLM 的 internal checkpoint/partial prefix/async pin 工作提供接口与生命周期参考，不证明本地存在相同 bug，也不能搬用其性能数字。[B3](research/B-model-community.md#b3)

**B4 DSA 与 C2 并行 drafter**：前者保留 raw/scaled FP8 layout、KPool tail/top-k 和 PD/Graph 语义，后者需精确匹配 GLM checkpoint、hidden-state 层、tokenizer 与 draft KV/head shard。兼容 GLM-5.3-Flash 的 DFlash/DSpark checkpoint 本轮未得到可采用的身份/质量确认；非对称 P/D TP 的 draft 传输已有社区风险线索，target KV 可传不代表 draft sidecar 可传。C2 在 draft 串行开销主导时有价值，但不隐含训练、客户数据或新增资源授权。[B4](research/B-model-community.md#b4)、[C2](research/C-frontier.md#c2)

### 2.5 所有候选共同遵守的状态契约

每个可复用/可传输状态 bundle 必须属于同一 `request epoch + model/tokenizer/template/quant/layout identity + layer/head shard + committed token boundary`。bundle 包含相关 KDA state、conv window、DSA/MLA KV、KPool/index 元数据和 NextN 所需上下文，不能只传一个形状匹配的 tensor。

verify 不得提前污染 committed state；prefix cache 不得发布 speculative state；命中计入的 token 长度须与实际 checkpoint 位置一致，必要的尾段重算不能为了“全命中”被省略。异步传输/cache 写出持有真实对象直到所有 rank 到达终态，取消、错误、去重也要释放引用；晚到回调不得写进被复用的新请求槽。

Graph、overlap、PP 或融合优化须保留 buffer lifetime、padding/mask 和跨 stream/event 依赖。Changing dtype 标签但内部降低 recurrent precision、修改 KPool/专家 top-k、丢 token、放宽接受阈值、缩短 reasoning/输出，以及 benchmark 特化随机路由，均不属于本设计的优化空间。

### 2.6 研究门、采用门与停止条件

| 阶段 | 必需证据 | 失败时如何处理 |
| --- | --- | --- |
| 静态研究门 | 对应 S/权重身份，合法 shape/gate/precision，实际 dispatch/fallback，完整接口和成本模型；引用可以访问的原始依据 | 缺身份只挂起依赖它的数值/后端判断；已有源码可完成的分析继续，不把全部任务转给 Codex |
| 数值/状态验证门 | 与同身份 reference 的输出及 state 检查；长自回归、接受分支、chunk/prefix、PD、取消/重用、graph padding 的正确边界；FP32 recurrence 约束不放宽 | 保存第一处分歧与版本；不得以短 smoke、均值指标或 dtype 名称替代状态闭合 |
| 模型质量门 | 沿既有协议的代表性任务质量及样本数/不确定性；保持 tokenizer/template、采样、工具与 reasoning/stop 行为；包括历史失败案例 | Q2 未共享协议/结果前不宣布质量通过；logit/state 近似只是诊断，不替代任务质量 |
| 性能采用门 | 匹配工作量的成对对照、稳态与冷/热路径分开、真实 B/T 和 committed tokens、每卡峰值/余量、TTFT/ITL/E2E/错误率；收益超过测量不确定性并满足约定 SLO | 单 kernel gain、理论容量或作者异构硬件成绩不构成采用；只保留适用子域，正确 fallback |
| 交付门 | 实载身份/依赖封闭、真实路径 warmup、资源边界与取消/超时闭环、必要 observability；新能力可独立关闭并回到已验证对照 | 拒绝靠关闭生产保护、删语义或无限增加 buffer 实现跑分；完整发布平台建设仍在本轮范围外 |

本轮没有执行以上运行门。下一轮 PLAN 必须给具体命令、改动/验证范围、资源许可、停止与回退条件；本 DESIGN 不把旧 E2 命令复制成新的自动执行计划。

## 3. 参数审查与处置

审查基于 E 的渲染命令、包内最终解析摘录和本轮所读固定源码，**不是声称读取了完整 `server_args` 或全部实际 kernel 命中**。处置用于下一轮选题和交付判断；本轮未修改任何参数。

| 参数/组合 | 当前状态与实际边界 | 处置及依据 |
| --- | --- | --- |
| S、镜像、DeepGEMM 0.1.7、模型/tokenizer/template | S 已固定，模型仍为 master inventory；包版本不等于 wheel/build/ABI 身份 | **保留并补证 Q1，研究前置。** 不用 latest 或未审计依赖替换；启动前恢复 clean-tree/import/model/ABI 检查 |
| 输入/输出长度、context=69632、采样与 parser | C80 材料为 65536/1536；glm45、glm47 与模型 chat template | **保留。** 不缩短输出、移除 reasoning/tools、改采样换速度；多模态是否在交付范围由实际服务合同决定，不自行启停 |
| FP8 weights、KV=`fp8_e4m3`、`SGLANG_DSA_FP8_KV_LAYOUT=raw512` | 不能推导所有 projection/NextN 都 FP8；raw 与 scaled layout 不互换 | **保留。** 后端变化需要权重 ignore、scale、row stride、PD/reader 全链等价检查 |
| `--mamba-ssm-dtype float32` | recurrent state 的精度约束，不只是接口 dtype | **保留硬约束。** 现成 FlashKDA 内部 FP32→BF16 不可作为等价替换 |
| D TP8/DP8/EP8/PP1、P TP4/EP4/PP2 | 当前已验证拓扑，不是永久设计约束 | **保留对照，条件研究 A3。** 先同卡数完整成本模型；不得把 DP/TP/DCP 的内存与通信收益单独相乘 |
| `--dcp-size 1` | DSA/KPool、KDA state、投机和 PD 组合需支持 | **保留。** B4 仅在真实瓶颈与接口闭合时研究，不直接设 N |
| D `--max-running-requests 80` | GLM pool 的显式请求预算按 attention DP 分摊；实际 batch 还受 P/队列/PD 影响 | **C80 保留；高 B 待 Q3/Q4 后修改。** 目标是实际填充，不是参数上限 |
| D `--max-total-tokens 680000` | 名义 B10 token 工作量已接近上限；不能只凭请求槽判断余量 | **与 A4 联合规划。** 不按 DP 再随意除一次，也不单独扩大到侵占 8 GiB 余量 |
| D `--disaggregation-decode-extra-slots 10` | 接收/在途槽不是额外十个完整长请求预算；原 source-map 指向 DSV4 路径有误 | **保留对照，核查 GLM token-credit/admission。** 纠正在 A4，原证据不改写 |
| P max-running=32、mem-fraction=.70；D mem-fraction=.90 | 静态预算不是峰值运行安全证明 | **保留对照。** 根据各 pool/graph/workspace/PD 峰值联合调整，逐卡满足既定空闲门 |
| P chunk/max-prefill=8192；D 命令8192 | 包内 D 最终 chunk=1024，是 DP8 hook 分摊结果 | **记录解析值。** P chunk 变化按完整模型 pass 比较；不能说 D 实际跑8192或为命中 FlashKDA 直接四分 chunk |
| P PP partition=24,21、microbatch=1、disable-overlap | 层数均衡不等于 KDA/DSA/MoE 实际成本均衡 | **条件研究 A3。** 按 stage 关键路径选改动，保留 mHC residual、NextN 与在途 buffer 生命周期 |
| EAGLE：P 1/1/2，D 5/1/6 | 原生 NextN；T6 非接受长度；P/D 所需上下文不同 | **保留对照，优先研究 A2。** 一个浅链和无投机足以先判方向；不任意降低 P 设置破坏 draft 交接 |
| 采样接受阈值/拒绝采样、adaptive 入口 | 材料未给完整生效采样配置；adaptive 的组合能力尚未全审计 | **语义保留、Q1 未决。** 不调松阈值；动态策略先查已有实现，再处理有限 tiers/DP/状态切换 |
| `--linear-replayssm-cache-len 16` | E2 无 `--enable-linear-replayssm-spec`，该长度不表示 Replay 已开启 | **对照可保留声明；生产非 Replay profile 可移除无效声明以免误读。** 打开 spec Replay 需 Q2 与完整采用门 |
| `--enable-linear-replayssm` 与 `--enable-linear-replayssm-spec` | 普通 decode Replay 与 spec Replay 的 PD/状态规则不同 | **禁止混同。** 不从一个 guard 外推另一个不可研究；不自动组合两个模式 |
| `SGLANG_OPT_FUSED_KDA_VERIFY=0` | S 的融合条件受 T/shape/tree/ring 限制；上游已有不同组合 | **保留对照，B2 条件改。** 按真实 batch/shape 分派，不全局强开旧版本 |
| `SGLANG_OPT_KDA_ACCEPTED_STATE=0`、`SGLANG_RAGGED_VERIFY_MODE=static` | 已读 accepted-state 窄路径不覆盖任意 T6；ragged 与 ring/Graph 另有契约 | **保留。** C1 不冒充开启现成 T2 开关；改布局需重审 pool/commit/padding |
| KDA projection fusion：P off，D on/a_only | S 已有融合及 checkpoint unquantized 保护 | **保留 D 对照；P 和其它 mode 条件比较。** 不把 full 名称当更快，不给原 BF16 projection 强加 FP8 |
| linear decode/verify=`triton`；P prefill 未显式指定 | 缺少 P 最终实际 backend 摘录；SM100 kernel 不等于 H20 可用 | **保留已知值、Q1 补实际命中。** B1 是能力研发，不是直接替换 backend 名 |
| D `SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION=1` | 固定实现按 KPool/page/top-k/平台决定支持域，部分路径仍需 host lengths | **保留受支持组合。** 不宣布已消除全部 D2H；不改模型 KPool/top-k/tail 绕过 guard |
| D MoE target/draft=`deep_gemm`、A2A=`megamoe`、layout=compact | S 有 SM90 FP8×FP8 路径；不同阶段是否命中需身份与日志确认 | **保留对照，A3 有据细化。** 不搬用 Blackwell FP8×FP4，也不按 runner 字符串证明所有子模块命中 |
| MegaMoE `NUM_MAX_TOKENS_PER_RANK=1024`、`FAIL_CLOSED=1` | cap 的单位是 token/rank，包含 T 与 padding 等形状，不是 B10 requests | **保留 fail-closed。** cap 只在所有合法 warmup/graph/runtime shape 已核查后调整，不能直接改10或60 |
| P MoE target/draft=`triton`、DeepGEMM layout=auto | 当前 P 与 D 的实现/形状选择不同 | **条件研究 A3。** 后端替换需 PP/量化/工作区等价验证，不因 D 更快而自动迁移 |
| disable-shared-experts-fusion、`SGLANG_OPT_FUSE_MHC_POST_PRE=0` | 禁用项不自动表示浪费；涉及 kernel、stream、mHC tensor lifetime | **保留对照，按关键路径研究。** 不和其它大改动捆绑；融合不得丢 residual/状态依赖 |
| prefill Graph disabled、decode full/max-bs128 | max-bs 是 graph 配置，不证明所有实际请求形状都命中最优 tier | **保留对照，A1/A2 定向审查。** 用实际 B/T/padding 与 graph-memory 选择必要 tiers，生产未覆盖形状须正确 fallback |
| enable-dp-attention、local-control-broadcast、overlap | token ownership、控制同步和数据依赖共同决定行为 | **保留已验证组合。** 只消除可证明冗余的同步，不以省一次 collective 为由破坏 rank 一致性 |
| PD Decode radix 最终 disabled | 来自 PD hook，不能把未显式传 disable 解释为已启用缓存 | **保留当前语义，B3 另行完整设计。** P/外部 cache 也需 checkpoint/index/NextN bundle，不仅比较 hit rate |
| Router P/D round_robin | 当前各一个 P/D worker endpoint；未来多副本时才有实质路由取舍 | **保留对照。** 多副本优先真实 locality/负载/公平性，不为了等长 benchmark 使用随机路由；不能把该现状当生产最优 |
| `--skip-server-warmup` | Ready/health 不证明真实生成及冷路径资源安全 | **仅保留历史测量解释；交付前整改。** 明确真实请求 warmup 与 readiness gate，冷/热性能分别报告 |
| `SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1` | 历史绕过项不能替代逐卡内存安全证据 | **研究时不得借此满足内存门；交付前移除或以经论证的等效检查替代。** 不把不对称 DP 合法差异误判成必须等量内存 |
| Router disable-retries、timeout1800；PD bootstrap600/wait1800 | 重试、取消、超时和请求资源释放涉及生产语义 | **保留可追溯对照，交付前审查。** 不盲目开启重试造成重复生成/计费，也不靠超长超时掩盖挂起 |
| metrics/all-schedulers、JIT/缓存路径、NCCL/RDMA 环境 | 需要角色/构建隔离与真实 fabric；指标保留支撑归因 | **保留低开销观测与身份校验。** 新环境不复用已结束 Pod 地址/许可，不删指标换 benchmark；机密值不写入交接 |

未在上述固定配置/已读源码中闭合的字段记为未决，而不是声称已扫完全部 CLI 默认值或所有实验 env。更多具体实现边界见 [A 路检查范围](research/A-implementation.md)。

## 4. 研究与执行交接

### 4.1 恢复入口与证据等级

先读本 DESIGN 的规划摘要，再从 [唯一研究索引](research/INDEX.md) 进入相关 A/B/C 主题；必要事实回到 E 的 [BRIDGE](BRIDGE.md)、[BRIEF](BRIEF.md)、[evidence/INDEX](evidence/INDEX.md)。完整候选表、七层覆盖、历史排除与重开条件只维护在研究索引，不在此再复制一份。

**E-摘要**：交接包转述的已验证配置/性能/质量边界，本轮未读取包外原始结果。**S-源码**：本轮实际读取的固定实现及上下文。**U-外部**：原作者论文摘要、技术文章、实现、PR/issue 的明确版本与验证范围。**H-推断**：本轮成本/数学/优先级推导，不是测量。

工作流私有固定版本和材料 E 均成功读取，没有该项访问缺口。原始 trace/逐项质量、实际加载 config/权重 revision、DeepGEMM wheel build 等未共享；Z.ai 官方 GLM-5.3-Flash 博客全文、官方 HF config 及 GLM DSpark preview 卡本轮工具访问失败。没有用相近模型报告、镜像站摘要或默认 config 冒充这些原文。PR 描述/状态与完整 patch 审计也已区分；详见各研究文件的来源和读取范围。

### 4.2 最小 collect 与下一轮 PLAN 的范围

先按 [REQUESTS](REQUESTS.md) 从工程侧**已有**记录回答 Q1–Q4；可只回脱敏结构字段、聚合统计和内容 hash。Q2 只阻塞 Replay/相关 state 路径的采用，Q3 影响优化排序与收益归因，Q4 影响高 batch/cache 的生产有效性，均不要求重新跑整套 A/B/C 研究。资料不存在就记录不存在，不擅自重建部署或采集新 trace。

下一轮建议 PLAN 以 **A1 + A2/A4 + 一个 C1/B1 有界结构题**为范围，保留替换理由和停止条件。PLAN 应把具体部署/测试命令、身份与资源检查、成对对照、质量/状态/SLO 门写清楚，供用户 review；本轮设计不自动进入 goal，也不以问题记录代替 GPU/新运行权限。

可交给 Codex 的指令模板（D 由最终提交后的交接消息提供，不在本文件伪造自身 SHA）：

```text
使用 pro-codex-inference-workflow@e08e17bb4e8b13ce4b1bdb3aa6bd5efaa3c08bd4。
以调用者提供的设计提交 D 读取 handoff/glm53-flash/2026-09-24/DESIGN.md、
research/INDEX.md、REQUESTS.md；材料 E=eef8d29285ff2493b79bd696caeeb877ba664063，
工程源码 S=57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c，不静默换 main。
先执行 collect：只读现有工程事实，按 Q1–Q4 回答最小补证，注明版本、时间窗、
证据位置/hash、缺失项及受影响的设计判断；不重新做完整 design。
保持当前工作树与无关修改，只写用户授权的答复/证据文档。
本次 collect 不授权启动服务、构建、压测、新 trace、GPU 原型或实现。
需要新运行时，单列最小工作与资源许可，等待用户给定下一阶段权限。
随后在用户授权的 plan 阶段，围绕 A1、A2/A4 与 C1/B1 中一项形成具体 PLAN，
记录 D/E/S、可执行命令、质量/状态/生产门、停止/回退和小项持续优化路径；不自动 goal。
```

### 4.3 本轮发布边界

只新增本 DESIGN、REQUESTS 与 A/B/C 研究记录，并更新唯一 research/INDEX。原 BRIDGE、BRIEF、证据及所有工程代码、部署、PLAN 均保留。输出 commit 的真实 SHA 在提交后由交接消息给出；发布前核对目标分支相对 E 的变化，使用非强制更新，不覆盖他人提交。
