# GLM-5.3-Flash：唯一研究索引

更新：2026-09-24，首次完整 design。材料 **E=`eef8d29285ff2493b79bd696caeeb877ba664063`**；固定工程源码 **S=`57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c`**，tree=`17ea448109e9ac7b7674b1b13fa64528380632d1`。工作流固定在 `e08e17bb4e8b13ce4b1bdb3aa6bd5efaa3c08bd4`。

## 恢复顺序与本轮完成边界

先读 [唯一 DESIGN 的规划摘要](../DESIGN.md#1-规划摘要)，再进入本索引中的相关主题；补证看 [REQUESTS](../REQUESTS.md)，原始交接入口保留 [BRIDGE](../BRIDGE.md)、[BRIEF](../BRIEF.md) 和 [evidence/INDEX](../evidence/INDEX.md)。不要按新会话另建 DESIGN 或全量研究副本。

A/B/C 是主线程分别研究后综合，**没有独立子 agent 委派**。本轮直接读取了固定工作流、E 的全部六份交接文档、关键 S 源码上下文及相应公开一手资料；完成了源码差异、模型/组合限制、机制/成本推导与条件取舍。没有部署、编译、压测、新 trace、原型、代码实施或 PLAN 修改。研究/设计完成不等于收益测得、质量通过或采用新配置。

证据等级：**E-摘要**=包内转述的历史工程结果，本轮未重读包外原始测量；**S-源码**=本轮直接检查的固定实现；**U-外部**=明确范围的原作者/社区材料；**H-推断**=本轮分析，不是实测。各主题末尾记录文件/版本/读取范围和访问限制。

## 完整候选登记表

本表是全部当前考虑范围的唯一登记；DESIGN 只保留人类 review 所需的选择摘要，不复制另一张全表。“下一轮”表示选题建议，不是已启动工作。

| ID / 主题 | 实质结果与当前状态 | C80 / 更大实际 Decode batch | 下一可判定步骤、阻塞与重开条件 |
| --- | --- | --- | --- |
| [A1：host lengths、重复 metadata 与 Graph](A-implementation.md#a1) | 固定 KDA extend 有设备标量读取；packed conv 已存在。优先复用 #38431 的 host-length 方向，保留 fallback | C80 可改善 P/TTFT 或暴露 host gap；高 B 也可能减少 setup，但不预报收益 | 小项可进入下一轮 PLAN；先闭合 host/padding/track 不变式。Q3 决定收益归因；已有叠加覆盖或真实路径不触发则停止 |
| [A2：native NextN 整窗口、投影与分派](A-implementation.md#a2) | 比较原 T6、一个有据浅链与无投机；不是接受率越高越好。已有 projection/accepted-state 窄路径不重复发明 | C80 优先每 committed token 延迟；大 B 允许浅/无投机因算力或 scratch 成为赢家 | 下一轮重点；Q1 实载/采样、Q3 窗口/接受分布影响选择。动态控制需先审计已有 adaptive 与状态/Graph/DP 组合 |
| [A3：P/MoE/PP/DP/通信](A-implementation.md#a3) | S 已有 SM90 FP8×FP8 MegaMoE；P Triton/PP2/microbatch1 有条件机会。更改拓扑须计算整机复制与通信 | P 供给不足时优先于 D 容量；高 B 关注 EP/DP 暴露等待与实际 P/D 平衡 | Q3 定主导项后择一，不全组合扫参。旧 PP 负结果不永久排除，新的阶段/内存证据可重开 |
| [A4：GLM pool、admission 与容量](A-implementation.md#a4) | 已核对 GLM hybrid pool，纠正 source-map 的 DSV4 定位；extra slots 不能代表完整 token 内存 | C80 无容量阻塞则扩容不是加速；高 B 联合 requests/tokens/scratch/graph/PD credit 与 ≥8 GiB/卡余量 | 下一轮与 A2 联合；Q1/Q3/Q4 约束量纲和真实 fill。逻辑估算不当成 allocation |
| [B1：长 chunk + 真 FP32 + checkpoint prefill](B-model-community.md#b1) | 固定 FlashKDA wrapper 已存在但有长度/track/spec 回退；外部当前 C++ 把 FP32 input state 转 BF16，原版不可直接采用 | P/TTFT 主导时是非维护性主攻备选；不通过缩 chunk 增加全模型 pass 偷换成本 | 中到大投入；比较 Triton 改良与真正 FP32 的完整快路。Q1 配置、Q3 P 成本决定是否优先于 C1 |
| [B2：batch-aware fused verify/ring](B-model-community.md#b2) | 上游 #36821 已合入组合，S 仍有 guard 差异；公开 H20 形状存在 B/BV 拐点 | C80 不预设 fused 赢；高 B 去 snapshot 仍须计总成本，不能搬作者速度比 | 中等投入；先对齐 shape/ring/commit 契约，非支持形状正确 fallback。Replay 采用受 Q2 质量门，snapshot 侧可独立研究 |
| [B3：完整混合 prefix/cache/PD](B-model-community.md#b3) | S 已有 internal track；有价值增量是快 kernel 与 checkpoint/index/NextN bundle 的正确组合。vLLM 提供边界/生命周期参考 | 真正 prefix 复用且 P 成本显著时有价值；不把 hit rate 本身当正确或收益 | 中到大投入；Q4 复用机会成立后先 P-side，外部缓存后续。没有真实复用则保持条件，不假设90%命中 |
| [B4：DSA NoPE/KPool/FP8 与 DCP](B-model-community.md#b4) | E2 已有 raw512+TileLang 工作组合，旧 FP8 issue 不是本地当前故障；替代后端需保留 tail/index/layout/PD 语义 | 仅当 DSA/indexer/内存关键路径支持投入；DCP 不作为直接开关 | Q1/Q3 定实际几何和主导成本；中到大投入。不能改 KPool/top-k 绕 guard；其它硬件能力只作机制参考 |
| [C1：output-only 短窗口 KDA verify](C-frontier.md#c1) | 给出 channel-wise gate 的等价展开与三角系统，区分 ring 存储和 verify 串行依赖；无新原型/浮点验证 | C80 争取窗口净成本；高 B 同时看 scratch 与算力，可被浅投机击败 | **建议下一轮一项有界结构研究**，真实 FP32 GLM/dense chain/T2与T6。Q3 若 P 主导则换 B1；净成本或数值门失败即停止/保留反例 |
| [C2：block-parallel draft](C-frontier.md#c2) | DFlash 机制减少 proposal 串行轮次，但兼容 GLM checkpoint 本轮未资格确认，P/D draft sidecar 有组合风险 | C80 在 draft 暴露成本高时值得；高 B 可能增加无效 verify/通信而退化 | 大投入、条件储备；先权重/hidden-state/PD/采样接口闭合。无 checkpoint 不自动加入训练或客户数据采集 |

## 七层查漏覆盖

| 视野 | 已完成覆盖 | 不是本轮已完成的事情 |
| --- | --- | --- |
| 已有经验/组合 | E2 解析值、Replay/旧融合/PP 负结果的适用域与重开条件 | 未重新运行旧测试，未读包外旧 DESIGN 或原始 trace |
| 成熟库 | 固定 Triton/SM90 DeepGEMM、FlashKDA wrapper 与外部精度源码 | 未审计 DeepGEMM wheel 全部源码/ABI 或编译任何库 |
| 社区新增 | SGLang host lengths、fused ring、GLM/DSA/PD 组合；vLLM checkpoint/partial prefix/async pin | PR 描述/状态不等于完整 patch 审计或本地资格验证 |
| 现有 profile | 使用 E 摘要界定已知/未知，形成 Q2/Q3 最小已有证据请求 | 不虚构 kernel 占比、瓶颈或端到端 speedup，不要求全面新 profile |
| kernel/局部实现 | KDA projection、packed conv、串行 verify、ring、实际 FP32/BF16 边界、host scalar、DSA metadata | 未穷尽所有 env/CLI/分派，不把固定默认值当实载 config |
| 服务/并行/缓存组织 | P/DP/EP/PP、admission/token credits、Graph、混合状态、取消/重用/PD 生命周期 | 不实施完整线上治理、缓存平台或新拓扑，也不从别的引擎 bug 推断本地有同一 bug |
| 前沿结构机制 | C1 状态/输出计算解耦，C2 并行 proposal；与同尺度近期备选比较 | 没有数学浮点证明、GPU 原型、训练或新 checkpoint 资格认证 |

## 保留 prepare 的有效经验与重要解释修订

**E2 是可复用控制，不是结论。** 继续保留精确源代码、镜像/依赖、P/D 角色、已验证启动组合与最终解析边界；没有把旧方案删去，也没有把所有默认参数改成“最好”。原 E 的六份材料仍是历史输入快照。

**Replay 已实现的 GLM 生命周期/容量能力不重新列为待研发。** 曾可运行的 96/128/160 档保留为历史证据；C80 无已证明端到端收益和质量门未通过同样保留。重开条件是明确失败对照/定位、修复与完整质量状态门，或新的可填充负载，而不是换标题后重新声称采用。

**旧 fused-verify 负结果保留但不扩大解释。** 它约束当时 B/T/heads/gate/BV/dispatch 与是否 ring 的组合。#36821 的新组合和不同形状可成为有据重开点；上游 merged 与作者快照数字均不替代本地采用门。

**旧 PP 结果保留为条件排除。** 不在没有新关键路径事实时重复原方案，也不把特定 stage/microbatch/状态问题提升为“PP 永远不适合”。A3 的重开需要新瓶颈/布局/生命周期证据。

**源码定位纠正。** E 的 source-map 用于 extra-slots 的 `pool_configurator.py` 指针落在 DSV4 路径，本轮移到实际 GLM `kv_cache_configurator.py` hybrid pool 分析；原 evidence 不改写。默认 head/state 参数只能用于注明假设的数量级，不能替代实际模型 config。

**FlashKDA 接口精度纠正。** 不只停留在 4 月文章说明：本轮读到外部固定 commit `7afb9f454f160a6c4bbc0999beca0a8c40a38934` 的 C++ `state_acc` 与 FP32→BF16 转换。故现成 FP32 输入/输出接口不足以满足本轮 recurrent state 约束；保留机制借鉴与真正 FP32 改造，不删除保护 fallback。

## 访问缺口、静态/原型/运行边界与增量恢复

工作流的私有固定版本和 E 全部材料成功读取。模型报告/HF config/GLM DSpark preview 官方页面工具访问失败；原始测量、质量逐项结果与实际加载身份未共享。缺口分别映射 Q1–Q4，不用镜像站/相近型号或源码默认值补造证据。当前主张的读取范围见 A/B/C 文末；C1 公式是本轮推导，尚未进行数值实验。

后续 collect 回答 Q1–Q4；静态接口审查、原型、运行与采用各有不同门。只有在用户另行授权的 PLAN 范围内，才开展新运行或实现。Pro 收到新证据后只更新受影响的候选与唯一 DESIGN，保留适用版本、反例和修订理由；无新事实时不全量重做，也不让当前索引冻结后续有据的外部机会发现。
