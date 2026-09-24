# GLM-5.3-Flash：研究输入

## 目标与边界

目标是尽可能优化 GLM-5.3 在当前场景下的推理性能。此次 prepare 仅提供事实；优化方向、投入和唯一 DESIGN 由 Pro 决定。重点场景来自已有工程验证：长上下文 `65,536` 输入、`1,536` 输出，固定 C80 的服务口径，以及可能真正填满更大 Decode batch 的高并发口径。两种口径不能互换或把提高负载造成的总吞吐增长当成单一 kernel 加速。

已知工程约束来自上一轮计划及验证：16×H20，8P+8D 的 PD 分离，FP8 权重/FP8 KV、FP32 KDA 状态、Mooncake RDMA、EAGLE/NextN 投机语义；历史验收要求每张 D 卡至少 8 GiB 空闲、真实请求正确、质量不劣、无异常撤回或重启。这些是先前实验的验收边界；如 Pro 认为新方向需要改拓扑或资源，应写清相应成本和验证，不把当前配置伪装成用户永久硬约束。

## 模型与现有选择

- 本地模型 inventory 断言架构为 `Glm5NextForConditionalGeneration`，45 个主干层、34 个 KDA 层、1 个 NextN draft 层；有 MoE、DSA/MLA 与 KDA 混合路径。inventory 只核对了 `master` 标记及若干配置/文件摘要，**未**提供可复现的上游模型 commit。源码中的[配置和模型实现](evidence/source-map.md)可直接核查机制。
- 当前已验证 E2 组合见[完整启动材料](evidence/launch-e2.md)：P TP4/EP4/PP2、P EAGLE `1/1/2`；D TP8/EP8/DP8、D EAGLE `5/1/6`、MegaMoE、KDA projection `a_only`、full CUDA Graph、raw512 FP8 KV、Replay 关闭；D admission 80、KV token pool 680000、PD extra slots 10。这个组合是固定起始对照，不是本轮设计结论。
- 已有 ReplaySSM GLM KDA 适配在[固定源码](evidence/source-map.md)中。先前实机验证了 Replay ring/状态池生命周期和更大 batch 容量；固定 C80 未证明端到端吞吐改善，历史质量非劣门未通过，因此未默认开启。定量结果和原始日志属于工程侧未共享证据；Pro 如需精确数值、失败题型或运行阶段信号，应提出窄 REQUESTS。
- 旧计划还评估过 fused KDA verify、decode PP 等方向；旧负结论只适用于当时版本、shape、实际 dispatch 与测量口径，不构成 Pro 的研究清单或永久排除。

## 希望 Pro 回答

1. 在该模型与现有源码下，哪些现状组合或参数误用值得修正，哪些瓶颈需要进一步实测才能判断？
2. 模型、其他引擎和上游社区有哪些可复用的完整能力或 GLM 特有缺口？现有 Replay 适配与可重用主线应如何区分？
3. 对 KDA 状态、MoE、DSA、投机验证和 PD 组织，有哪些值得投入的中长期机制？给出近期开工选择与有据暂缓项，同时保留生产语义和质量门。

研究范围先限于可访问的启动组合、公开源码、公开资料与本包说明；新集群运行、trace 或内部仓读取需另列最小工程问题。没有预设 GPU/时间预算，也不以此推导无限搜索。
