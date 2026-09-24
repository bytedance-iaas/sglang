# GLM-5.3-Flash：最小工程补证请求

提出日期：2026-09-24。材料 E=`eef8d29285ff2493b79bd696caeeb877ba664063`；工程源码 S=`57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c`。关联唯一 [DESIGN](DESIGN.md) 与 [研究索引](research/INDEX.md)。

**权限边界：本文件只是 collect 问题，不授权任何新运行。** 可在用户已经授权访问的工程环境中读取/解析已有日志、trace、配置和测试产物，回传必要的脱敏字段与聚合量。不启动服务、构建/安装、压测、GPU 原型或新 trace；不读取/上传权重、凭据、内部网络地址和客户原文。证据不存在时记录缺失并停止该条，不以“补证”为名复现任务。需要新运行另列最小动作与许可，进入用户授权的 PLAN。

每条答复保留：问题 ID、答复日期、工程源码/模型/镜像身份、原始产物位置或内容 hash、取值/时间窗口、是否脱敏、适用范围，以及哪些判断仍不能支持。不要替 Pro 再做完整 A/B/C 设计；只刷新依赖该事实的主题。

<a id="q1"></a>
## Q1：E2 的实际加载身份、模型配置与阶段后端是什么？

**为什么改变选择。** 模型 inventory 的 master 不能确定 KDA gate/heads、DSA KPool、NextN 的量化例外；CLI/包名也不能证明实际 kernel 命中。它们决定 B1/C1 的合法形状与精度、B2 的兼容性，以及 A2/A3 的真实成本归属。

**已有事实。** E 固定 S 和源码 tree；包内记录镜像与 DeepGEMM 0.1.7；45 trunk/34 KDA/1 NextN 为交接 inventory。P/D 的命令与部分最终解析值已共享。固定 S 有 KDA/NextN 的 unquantized 处理、后端 fallback 和 SM90 MegaMoE 路径；无需再泛查这些是否存在。

**最小证据。** 从原 E2 产物或已保存的身份记录取一份脱敏 manifest：

- 模型、tokenizer、chat_template 的不可变 revision 或内容 hash；相关 config 字段：trunk/KDA 层索引、KDA head 数/维度/gate lower_bound、mHC 结构、NextN 层、量化 ignore、专家 top-k、DSA index_kpool/top-k/page/layout。只需结构与 hash，不需要权重或完整文件复制。
- 原 run 的镜像/build/DeepGEMM wheel hash 与 ABI、Python 实际 import 路径是否落在 S 的核对结果；原始路径可脱敏。
- 已有日志中 P target、P draft、D target、D draft 各阶段的实际 backend/quantization 解析、分派/回退记录。重点补 P linear-prefill backend、NextN BF16/FP8 解释、D raw512 consumer、MoE 真实命中和采样/拒绝采样的生效配置。日志未记录的 kernel 命中明确写“未知”，不从参数反推。

**停止条件。** 一份与 E2 同 run/构建对应、能够解释配置与已知 fallback 的 manifest 即足够；字段缺失逐项标注。不得下载新模型、重启旧服务或安装包来凑完整。

**影响范围。** 阻塞依赖精确形状/权重身份的采用判断，不阻塞固定 S 的 host-length 静态分析、社区机制对比或条件设计。

**是否需要新运行及许可。** 本请求仅现有记录；不需要也未授权新运行。若身份当时未留存，先反馈缺口，再另提最小核验动作。

**答复状态：未答复。**

<a id="q2"></a>
## Q2：历史 Replay 质量门究竟在哪个对照、哪些样本/状态边界失败？

**为什么改变选择。** E 的结论足以阻止 Replay 采用，但不足以区分权重/构建不一致、采样协议差异、数值漂移与状态生命周期错误；它们对应不同的修复和重开条件。不能以论文声称等价覆盖本地失败。

**已有事实。** Replay spec 已在固定源码集成；历史大 batch 组合曾可运行；固定 C80 未证明端到端提升，已有质量门未通过。原始质量结果没有随 E 共享。本轮没有假称已检查失败样本。

**最小证据。** 选原始记录中一组可比较的 Replay-off/on：双方 run ID、S/模型/构建/配置身份、评测协议/版本、样本数、seed/采样与截断规则、已约定的不劣门；回传分任务聚合分数/差异及现成不确定性指标。若已有失败定位，只需一个最早分歧的 layer/token boundary、tensor 类型、误差摘要/状态 hash 和对应触发条件，不上传客户 prompt/response 或私有 benchmark 原题。若已有等长 C80 的完整时间窗，也标明与该质量对照是否同构建。

**停止条件。** 能明确“哪组对照以什么门失败，以及已知定位到哪里”即停止；尚无定位写未知，不要求先找出根因。没有置信区间就报告没有，不自行用不匹配样本拼接。既有质量协议缺失则只保留未通过结论。

**影响范围。** 直接约束 Replay、相关 ring/fold 的采用与历史负结果解释。A1、native NextN 的独立比较、P 侧优化、C1 的数学/接口研究仍可继续；C1 后续数值/采用也必须过相同质量与状态门。

**是否需要新运行及许可。** 只解析已有产物，不新跑评测、不重新采样。新的最小复现需单独授权。

**答复状态：未答复。**

<a id="q3"></a>
## Q3：现有 C80 与高 batch 记录中，暴露成本和实际容量限制分别在哪里？

**为什么改变选择。** 决定优先 A2/C1 的 Decode 工作、B1/A3 的 Prefill 工作，还是 A4 的 admission/容量；也决定 fused verify 和投机深度应覆盖哪些真实形状。缺少此信号不能给出局部 kernel 的端到端收益预测。

**已有事实。** E2 是 P TP4/EP4/PP2、D TP8/DP8/EP8，D 目标约 B10/T6、Replay off、680000 token cap/extra slots10；包内 D 解析 chunk1024、radix disabled。S 的 verify 仍沿 T 串行 recurrence，GLM pool 正确入口已在 A4 核查，原 source-map 的 DSV4 指针不应继续用于 GLM 预算。

**最小证据。** 从既有日志/trace 选择原 E2 的一个有代表性的稳态窗口，及一组已有高 batch 对照（若存在），回传：

- 窗口起止、输入/输出/到达方式、完成/失败/retract/restart、真实 P/D 排队与 PD 接收状态；逐 D rank 的 actual running B、verify 宽度/graph tier/padding、接受长度直方图、committed tokens 与 wall time。
- 现有 profiler 能支持的 draft、target verify、commit、MoE/DSA/KDA、暴露通信/host gap 与 P stage/PP 等待摘要。重叠计时分开标注，不能直接把各 kernel sum 当墙钟。未采到的类别标未知。
- 已有内存记录中的逐卡 min-free/peak、KV/recurrent state/spec scratch或ring/conv/draft/graph/MoE workspace/PD在途的可得分项；最大请求槽、token credits 与真实阻塞/retract原因。分项缺失时提供已有总量，不能用默认参数推成测量。

**停止条件。** 一个可追溯窗口足以判定主导阶段/真实填充和已知限制即可；不要求全面 profile。不存在匹配窗口则停止并说明，不能拿不同模型/输入/构建的数字混合排序。高 batch 只存在容量日志时就标“只有容量，没有吞吐证据”。

**影响范围。** 决定 A2/A3/A4、B1/B2/B4/C1 的优先级和采用证据；不改变已确定的 FlashKDA 精度边界或 host scalar-read 源码事实。

**是否需要新运行及许可。** 仅现有产物的有界离线解析，不启动 nsys/ncu、新 trace、服务或压测。如果无可用窗口，另列最小新采集及资源许可后再进入 PLAN。

**答复状态：未答复。**

<a id="q4"></a>
## Q4：高 batch 与 prefix 优化应服务于什么真实到达负载和 SLO？

**为什么改变选择。** 决定更多容量能否填满、是否提高满足 SLO 的 goodput、是否值得 B3 缓存集成，以及路由/重试/取消策略的生产边界。等长 closed-loop 的最优设置不自动适合生产。

**已有事实。** 用户要求分别判断固定 C80 与能填满更大 D batch 的场景，生产语义优先。当前材料仅提供固定输入/输出工作负载和总体正确性/内存门，没有数值 TTFT/ITL/E2E SLO，也没有可确认的共享 prefix 复用率。

**最小证据。** 提供现有业务定义或已保存聚合统计：TTFT/ITL/E2E 的目标分位与阈值、质量不劣协议；C80 与高负载的到达模式/持续时间和输入输出长度分布；若做 prefix 路线，给按 token/请求统计的真实可复用 prefix 长度分布与冷/热比例。补充已有的取消/重试/优先级或多副本亲和要求即可，不展开成全套上线规范。只给匿名聚合量，不传客户文本/身份。

**停止条件。** 足以区分“固定 C80 延迟/吞吐”和“高 B 的 SLO goodput”，并判断 prefix 是否值得投入即停止。没有已定义 SLO 时明确由用户/业务待定，不由模型杜撰；没有真实复用数据则 B3 保持条件候选，不假设高命中。

**影响范围。** 直接影响高 B 采用、B3 的优先级、Router/admission 的生产适配；不阻塞 A1 或 C1/B1 的有界结构研究。

**是否需要新运行及许可。** 只读已有业务定义/统计；不新增线上遥测、流量采样或生产配置修改。

**答复状态：未答复。**

## 回传后的增量处理

Codex 在同一问题下保存答复与版本化证据链接，仅补充必要研究事实，不自行重写唯一 DESIGN。Pro 按新增/补强/纠正/排除/仍受阻更新受影响主题；未受影响的结论复用。运行或实施计划须在用户授予的下一阶段权限内另行建立，不能把本文件当自动执行队列。
