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

**答复状态：已按现有记录答复（2026-09-25）；实载细节仍有缺口。** 本轮只摘结构、hash 和聚合字段。下文 `source-workspace:` 均指工程工作区的本地 `artifacts/` 或 `docs/`，**不是本 GitHub 交接包内可供 Pro 直接打开的原始文件**；SHA256 用于将来在工程侧核验。未复制日志、权重、内部地址、凭据、客户文本或评测原题。

- **同一次 E2/off1 的运行身份（观察）。** 2026-09-23 的 P/D 启动日志都打印 S=`57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c`、tree=`17ea448109e9ac7b7674b1b13fa64528380632d1`，`sglang` 从 staged S 源码导入，DeepGEMM 从 0.1.7 依赖目录导入。渲染清单的镜像摘要为 `sha256:d33d932aee374f3884e7f49c54447c82ce61ac9e58749ebf5aa3a50d25c9cb28`，DeepGEMM wheel SHA256=`f4e67086dc685ddcfcbb7833cc9770afd850cab23e173e77b9b18c19de0c2836`；D 启动通过八卡 SM90 FP8 MegaMoE ABI 检查。依据：`source-workspace:artifacts/glm53-replay-capacity-20260923/repeat-claw/off1/{decode-final.log,glm53-rrep-0923-roleset-sl4sq-prefill-57cccb7dd5-0.log}`，SHA256 分别为 `baaba2b2eeec74a71dab016674ed1a1a1055c843448f34d66d76d0c78bd5ca85`、`97910fb0eac017285ab571cda00f8144598d86de3ed395c2cfa713e133d6b5e6`；`off-b10.yaml` SHA256=`89871817eca47a2e6abb310ca04eaad5d37015c8f6d44a6aa6e69ad5b1208a72`。日志时间标签为 2026-09-23 15:34–15:41，未自带时区；后述 HBM CSV 为 UTC。
- **模型内容与结构（校验清单 + 与其 hash 相同的本地 config 静态解析）。** 清单只写可变模型标记 `master`，未留不可变模型仓 revision；inventory SHA256=`69b071f4dd62be45c6e5ef007a41fdf071ea1ff57cc40deb00d3ed3713950da5`，`config.json`=`bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f`，`tokenizer.json`=`19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`，`chat_template.jinja`=`41cff9af7b3a86c96751b107a8444f245fbda0bd5320b636a5bb1f7f4ba1a5c3`。config 为 45 主干层，其中 KDA 索引 `[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38,40,41,42,44]`、其余 11 层 full attention，另有 1 NextN；KDA `num_heads=64, head_dim=128, gate_lower_bound=-5, short_conv_kernel_size=4`，`mhc=true, hc_mult=4`；DSA `index_kpool=4, index_topk=2048, index_n_heads=32, index_head_dim=128, compress=true, always_select_tail=true`，运行解析 page size 64。MoE 为 288 routed experts、top-k 8。顶层权重量化声明是动态 FP8 E4M3、block 128×128；`modules_to_not_convert` 共 1509 项，其中 draft layer 45 匹配 18 项。该 config 内容 hash 与清单相同，**不是逐个权重 shard 的本轮重验**。
- **阶段后端（把声明、解析、命中分开）。** P 的 `server_args` 为 linear backend `triton`、prefill override `None`、DSA prefill/decode `tilelang`、target/draft MoE `triton`，P `mamba_ssm_dtype=None`；D 为 FP32 Mamba state、linear decode/verify `triton`、DSA `tilelang`、target/draft MoE `deep_gemm` + `megamoe`。P/D KDA dispatcher 均报告 decode/verify/extend=`TritonKDAKernel`；两端日志均声明 raw FP8 DSA KV，D raw512 配置来自同一渲染清单。D 八个 rank 均有 `SM90_FP8_MEGAMOE_KERNEL_ACTIVE`（15:40:13，观测形状 `num_tokens=60`），这只证明该路径至少命中一次，不证明所有 MoE 调用均命中。target 与 NextN 都报告 `quant=fp8, fmt=e4m3` 加载；这不排除上述模块级量化例外，也不能据此称 NextN 全部 FP8。D 解析的采样接受阈值 single/acc 均 1.0、`speculative_use_rejection_sampling=false`、EAGLE 5/1/6；P 为 1/1/2。P prefill override `None` 与 KDA dispatcher `TritonKDAKernel` 不等于每个 P linear-prefill kernel 的运行命中。E2 off1 的 Replay 主开关和 spec 开关均为 false，L16 只是已解析的声明。

**仍缺、影响和最小后续动作。** 不可变权重仓 revision、逐模块 target/NextN 实际 dtype、P linear-prefill 与 DSA 每次调用的 kernel 命中率、Graph 中全部合法形状的 MoE 命中、逐调用 fallback 均未保存在上述记录。B1/C1 的精确形状可依据已校验 config 做有界静态分析；需要权重级数值/量化采用结论及 A2/A3 的全路径成本时仍需这些缺口。若未来获新运行权限，最小动作是在相同镜像/模型内容下由 Pod 打印脱敏模型 marker、选定模块 dtype 与限定调用分派计数；本次未运行。

<a id="q2"></a>
## Q2：历史 Replay 质量门究竟在哪个对照、哪些样本/状态边界失败？

**为什么改变选择。** E 的结论足以阻止 Replay 采用，但不足以区分权重/构建不一致、采样协议差异、数值漂移与状态生命周期错误；它们对应不同的修复和重开条件。不能以论文声称等价覆盖本地失败。

**已有事实。** Replay spec 已在固定源码集成；历史大 batch 组合曾可运行；固定 C80 未证明端到端提升，已有质量门未通过。原始质量结果没有随 E 共享。本轮没有假称已检查失败样本。

**最小证据。** 选原始记录中一组可比较的 Replay-off/on：双方 run ID、S/模型/构建/配置身份、评测协议/版本、样本数、seed/采样与截断规则、已约定的不劣门；回传分任务聚合分数/差异及现成不确定性指标。若已有失败定位，只需一个最早分歧的 layer/token boundary、tensor 类型、误差摘要/状态 hash 和对应触发条件，不上传客户 prompt/response 或私有 benchmark 原题。若已有等长 C80 的完整时间窗，也标明与该质量对照是否同构建。

**停止条件。** 能明确“哪组对照以什么门失败，以及已知定位到哪里”即停止；尚无定位写未知，不要求先找出根因。没有置信区间就报告没有，不自行用不匹配样本拼接。既有质量协议缺失则只保留未通过结论。

**影响范围。** 直接约束 Replay、相关 ring/fold 的采用与历史负结果解释。A1、native NextN 的独立比较、P 侧优化、C1 的数学/接口研究仍可继续；C1 后续数值/采用也必须过相同质量与状态门。

**是否需要新运行及许可。** 只解析已有产物，不新跑评测、不重新采样。新的最小复现需单独授权。

**答复状态：已按历史质量记录答复（2026-09-25）；首个状态分歧未知。** 可比较的 off/on 属于较早 SGLang `54b9e1edf7903415e2e4dcad3c5a5a4969c41833` / tree `7f903e00c61b4b1e8120b9965b9a090ee942552f`，而非后来的 S。工程 PLAN E32/E33 与运行态记录表明 on/off 只差 `--enable-linear-replayssm-spec`，模型内容沿同一 E2 inventory，镜像摘要与 Q1 相同；仍只有可变模型 `master` 标记。EvalScope 1.11.1、runner SHA256=`6398459087e9dd42128c45278ec2f911a339d5fa6917e8be6741ce4dacba9f38`；长集 fixture SHA256=`edcf32fc67d794182e5bd84f13030ff77108139c784dcfb9bbbaca4acbf1420b`，两侧输入均恰为 65,536 tokens，`temperature=0, max_tokens=2048, retries=0, stream=false, timeout=600s`。on/off Job 日志分别始于 2026-09-22 18:38:18 / 19:15:53（日志未注明时区）；off 终态记录完成于 `2026-09-22T19:24:34Z`。身份与协议不能自动外推成在 S 上复测。

| 分组 | off 正确 | on 正确 | on−off | 预设单侧 95% 下界 / 非劣门 | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| 短 64 题 | 63/64 | 63/64 | 0 pp | 0 pp / −1 pp | PASS |
| 64K 长 64 题 | 64/64 | 63/64 | −1.5625 pp | −4.6875 pp / −1 pp | **FAIL** |
| 全量 1319 题 | 1261/1319 | 1266/1319 | +0.3791 pp | −0.1516 pp / −1 pp | PASS |

配对记录对齐题号/fixture/generation，双方各 1471 个结果文件通过 SHA256 清单核验；50,000 次 paired bootstrap 的长集双侧 95% 区间为 `[−4.6875, 0]` pp。长集只有一题 off-only（记录索引 12）；这是**任务分数的最早已知分歧范围**，并无 layer/token 首个分歧、tensor 误差、state hash 或根因记录。不能把可读答案差异归因为权重、采样或状态错误。依据：`source-workspace:artifacts/glm53-deepopt-20260922/replay54b9-quality-paired-summary.json` SHA256=`e0b55f0e33ac43d50f42c633335f3b17b07af09c266a9e3b261132551b54e8aa`；`replay54b9-quality-paired.json` SHA256=`dfa013064ecef2b0c9f35bf3019ea9237b9b7b3d30d2f2c67fdbef68dd430459`；长集 fixture identity 文件 SHA256=`85b23fafb1e2cb6cdab40053c4963e92bdc429e1d84f54cc26877c74bc96f482`。本答复只传匿名计数、协议与 hash，**未复制或提交原题、prompt/response、逐题原始结果**。

**局部判断。** 历史 Replay-on 候选的总体质量门仍 FAIL，相关持久 ring/fold 的采用不能凭机制正确或高 B 性能解除；失败定位仅到长集单题，不能定性为某个状态 bug。A1、独立的 native NextN 比较与 B1 的静态研究不受该失败直接阻断；C1 后续数值/采用仍要过同类质量和状态门。若未来获评测/运行权限，先在确定源码和模型内容上对该匿名失败索引采集最小 layer/token state 摘要，再决定是否复跑完整质量；本轮未执行。

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

**答复状态：已按已有窗口答复（2026-09-25）；完整阶段墙钟归因仍缺。** 固定 S 的 E2/off1 是 Replay-off、P TP4/EP4/PP2 + D TP8/DP8/EP8、D EAGLE 5/1/6、D C80/N320；EvalScope 1.11.1 用 65,280-token 共享随机 prefix + 每请求 256-token 后缀，输出固定 1536 tokens、closed-loop C80、seed 2201、temperature 0。独立 C80/N80 暖机后，正式客户端 DB 的 monotonic `start_time` 为 `[386619.134,386691.970]`，`completed_time` 至 `386715.774`，跨 96.6397 秒；单调时钟没有保存绝对墙钟映射。D 日志标签为 2026-09-23 15:59:44–16:01:19（日志自身无时区字段），对应的 HBM CSV 观察窗为 **2026-09-23 15:59:44–16:01:20 UTC**；两者数值时间相合，覆盖该次正式测量的 Decode 活动，但不能把每个日志点严格对齐到单个客户端请求。客户端 320/320 成功、恰为 65536→1536，5086.11 output tok/s，TTFT 5073.39 ms，TPOT 9.77 ms，平均 decoded tokens/iteration 5.4142、spec accept rate 0.8153；这些是端到端均值，不是分阶段 profile。

该 D 窗口的 363 条**周期日志快照**中，八 rank 各曾达 running B10；`#queue-req=0`、`#retracted-req=0`，transfer 队列观察范围 0–5、prealloc 队列 0–2，363 条均报告 `cuda graph: True`。快照 `accept len` 为 1.52–6.00 的聚合值，**不是逐 verify 接受长度直方图**；实际 graph tier、padding、逐轮 committed tokens 和暴露 wall time 没有记录。窗口内逐卡最低 free HBM 为 `[10482,10386,10388,10396,10388,10262,10388,10696]` MiB；KV token cap=680000、Replay-off，窗口日志的 full-token 与 mamba 使用量可观察，但 KV/recurrent/graph/MoE workspace/PD 在途的逐项峰值账本不存在，不能把 cap 推作实测分项。依据：`source-workspace:artifacts/glm53-replay-capacity-20260923/repeat-claw/off1/{benchmark_args.json,summary.json,decode-final.log,hbm.csv}`，SHA256 依次为 `9da263937ca7812f45c41b2873cc16207115efbdc8394a463644f7a04ba6e0f5`、`c6be1804b8eb9771bf4e3326c66c52256dbcbce02a7989e057d7a590fa636b98`、`baaba2b2eeec74a71dab016674ed1a1a1055c843448f34d66d76d0c78bd5ca85`、`2ac4aa83220b631fb99e44001ee72aff4ba2d1f281b23b00be1361fee56cabc7`。

同 S 的六臂交错 C80/N320（off B10、on 容量 B16）显示 off 八 rank 均达到 B10，on 实际只到 B10–B12；各臂 320/320 成功、无 retract/restart，最低 free HBM ≥10262 MiB。off/on 平均 output 吞吐 6106.45/5979.89 tok/s（on −2.07%），配对变化 +12.94%、−8.16%、−7.55%，不支持固定 C80 的稳定吞吐收益。依据：`source-workspace:artifacts/glm53-replay-capacity-20260923/repeat-claw/summary.json` SHA256=`0431d4c4617a1cbd93c6c1e4fd9928c538221a5e9d995a3e7a9c656c01c2bc36`。较早 `54b9e1...` 的性能配对（`source-workspace:artifacts/glm53-replay-perf-20260923/paired-performance.json` SHA256=`d59b120edbdbe2eb66b56bd3a077605f0242bfd2a0654893f4ba7d78ba631065`）不能混入 S 的阶段成本结论。

| 固定 S 的高 B 对照 | C / 请求 | 每 D rank 峰值 B | D KV cap / Replay slots | output tok/s | TTFT / TPOT | 最低 free HBM |
| --- | --- | --- | --- | ---: | ---: | ---: |
| B12 | 96 / 384 | 12 | 814144 / 22 | 7171.60 | 4555 ms / 9.85 ms | 24702 MiB |
| B16 | 128 / 512 | 16 | 1082432 / 26 | 7962.17 | 5869 ms / 11.50 ms | 21458 MiB |
| B20 | 160 / 640 | 20 | 1350720 / 30 | 8835.32 | 6338 ms / 13.19 ms | 18172 MiB |

这三点是**Decode 直连、custom 65,536→1536、closed-loop**，不是上面的共享随机 prefix E2 端到端负载；对应正式测试时长为 82.24/98.77/111.26 秒，HBM 采样总窗分别为 2026-09-23 `12:37:42–12:43:33Z`、`12:53:22–12:57:33Z`、`13:05:50–13:10:06Z`（总窗包含非正式测量段）。每点请求全成功、八 rank 均达目标 B，未见 OOM/显存非法/retract；吞吐随并发提高，TTFT/TPOT 同时升高，不能从跨并发数字归因 A2/C1/B1 的单机制收益。依据：`source-workspace:artifacts/glm53-replay-capacity-20260923/final-comparison.json` SHA256=`d876b44d024c7e994b27fd38acc6584db3851a94c1b7b71a1eb5e0372d1636b4`，各 `b12/b16/b20/summary.json` SHA256=`e226acfd017eb9c53b271468cc950818620ee548a71faf75fd759e718cec8d9a` / `3814737bcb4aca64ec2bde68e3a4ed47ddece11859bf84d35a7a02fa3a3edb7a` / `fe1dadf02325f16db571ffdbe905ba2f9b1b4eca644297d4da263429f5744f16`。

**仍缺、影响和最小后续动作。** 固定 S 的匹配稳态窗口没有 draft、target verify、commit、MoE/DSA/KDA、P/PP、暴露通信和 host gap 的互斥 wall-time 分解，也没有逐 verify 接受直方图、graph tier/padding、P 排队与 PD 接收全链时间戳或显存分项峰值。旧 `54b9e1...` Replay 机制 trace 只证明那版的局部路径；不能把重叠 kernel 累计时长当墙钟。当前事实支持“C80 未填满 on B16；高 C 能填 B12/16/20”，但**不足以给 A2/C1 与 B1 排唯一优先级或预测端到端收益**；A4 只可按已测总余量和 B/credits 做边界研究。若下一阶段获运行/trace 与 Pod 资源权限，最小动作是在同一 S/模型/负载、同一稳态窗采集逐 rank B/T/接受 histogram、P/D 队列与分阶段非重叠墙钟及逐卡内存分项；本轮没有采集。

<a id="q4"></a>
## Q4：高 batch 与 prefix 优化应服务于什么真实到达负载和 SLO？

**为什么改变选择。** 决定更多容量能否填满、是否提高满足 SLO 的 goodput、是否值得 B3 缓存集成，以及路由/重试/取消策略的生产边界。等长 closed-loop 的最优设置不自动适合生产。

**已有事实。** 用户要求分别判断固定 C80 与能填满更大 D batch 的场景，生产语义优先。当前材料仅提供固定输入/输出工作负载和总体正确性/内存门，没有数值 TTFT/ITL/E2E SLO，也没有可确认的共享 prefix 复用率。

**最小证据。** 提供现有业务定义或已保存聚合统计：TTFT/ITL/E2E 的目标分位与阈值、质量不劣协议；C80 与高负载的到达模式/持续时间和输入输出长度分布；若做 prefix 路线，给按 token/请求统计的真实可复用 prefix 长度分布与冷/热比例。补充已有的取消/重试/优先级或多副本亲和要求即可，不展开成全套上线规范。只给匿名聚合量，不传客户文本/身份。

**停止条件。** 足以区分“固定 C80 延迟/吞吐”和“高 B 的 SLO goodput”，并判断 prefix 是否值得投入即停止。没有已定义 SLO 时明确由用户/业务待定，不由模型杜撰；没有真实复用数据则 B3 保持条件候选，不假设高命中。

**影响范围。** 直接影响高 B 采用、B3 的优先级、Router/admission 的生产适配；不阻塞 A1 或 C1/B1 的有界结构研究。

**是否需要新运行及许可。** 只读已有业务定义/统计；不新增线上遥测、流量采样或生产配置修改。

**答复状态：现有记录无法给出业务数值（2026-09-25）。** 固定 S 的 C80 是 65,280 共享**合成随机** prefix + 256 后缀、fixed 1536 输出、closed-loop；高 B 是 Decode 直连 custom 等长请求。它们给出可重复的压力/容量边界，**不是**真实 Agent 到达过程、前缀重访率或满足业务 SLO 的 goodput。当前工程报告明确记录缺少有效的业务 TTFT/TPOT/错误率阈值、真实 Agent 会话 prefix 重访与费用口径：`source-workspace:docs/ai-infra/reports/2026-09-20-glm53-performance-and-optimization-analysis.md` 第 17、25、85、107 行，SHA256=`80f42f17cb7a15b036768cba04091da059dda91d5f8dc0ae1d29a4f824ed82f8`；D/E 的 BRIEF、DESIGN 与本 REQUESTS 也未给数值。现有质量不劣协议见 Q2，但该历史协议不等于业务 SLO。

**缺口与局部判断。** 没有 TTFT/ITL/E2E 的目标分位/阈值、真实 C80/高负载到达率和长度分布、按 token/请求统计的可复用 prefix 长度/冷热点、取消/重试/优先级/多副本亲和的有效业务合同。故高 B 的生产 goodput、B3 的收益与 Router/admission 取舍仍为条件判断；A1 的 host-length 事实、A2/A4 的有界机制和 C1/B1 的结构研究可继续。最小后续动作是由业务方提供现有匿名 SLO/到达/复用聚合定义；若不存在，任何新线上遥测、流量采样或生产配置更改均需另行明确授权，本次未执行。

## 回传后的增量处理

Codex 在同一问题下保存答复与版本化证据链接，仅补充必要研究事实，不自行重写唯一 DESIGN。Pro 按新增/补强/纠正/排除/仍受阻更新受影响主题；未受影响的结论复用。运行或实施计划须在用户授予的下一阶段权限内另行建立，不能把本文件当自动执行队列。
