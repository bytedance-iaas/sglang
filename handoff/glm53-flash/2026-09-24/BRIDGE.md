# GLM-5.3-Flash 推理性能：Pro–Codex 交接入口

当前交接：Codex prepare 已整理工程事实；下一方由 Pro 独立完成 design。本文不是 DESIGN，也不指定唯一方案。

## 入口与版本

- [研究输入](BRIEF.md) · [证据索引](evidence/INDEX.md) · [研究索引](research/INDEX.md)
- [已验证 E2 启动组合](evidence/launch-e2.md) · [源码定位](evidence/source-map.md)
- DESIGN：尚未生成。新一轮 PLAN：尚未生成；工程侧有上一轮已完成的容量 PLAN，仅作历史来源，不在此仓库复制。
- 工作流：`pro-codex-inference-workflow`，私有仓库 [SKILL.md@e08e17bb4e8b13ce4b1bdb3aa6bd5efaa3c08bd4](https://github.com/HanHan009527/obsidian_remote/blob/e08e17bb4e8b13ce4b1bdb3aa6bd5efaa3c08bd4/codex/skills/pro-codex-inference-workflow/SKILL.md)。Codex 已通过授权 GitHub API 回读该版本，内容与本地安装文件 SHA256 `b741e2da12ff98be5a620b06001f98df06e915de59bbb06708fa44f66537a837` 一致。Pro 需有该私有仓库读取权限；无法读取时应明确说明，不把本包当成完整工作流指令。

## 工程位置与身份

- 交接仓库：`https://github.com/bytedance-iaas/sglang`，分支 `codex/glm53-pro-prepare-20260924`，目录 `handoff/glm53-flash/2026-09-24/`。本次材料提交 SHA 在交接消息中给出；本文不自填自己的 commit。
- SGLang 源码固定为 Byte IaaS fork `poc_glm5.3-flash@57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c`，tree `17ea448109e9ac7b7674b1b13fa64528380632d1`。本交接分支直接从该 commit 建立，只增加材料文件；源码可从同一材料提交阅读。
- 工程工作区外层有既有脏改动；嵌套 SGLang 原 checkout 位于旧容量分支且有无关未跟踪目录。材料在独立 worktree 制作，没有提交或改写上述工作。
- 最近一次验证的交付组合：16×H20，两台各 8 卡；P 为 TP4/EP4/PP2，D 为 TP8/EP8/DP8/PP1，Mooncake RDMA。固定 C80 研究起点是 E2、Decode Replay 关闭、每 rank 目标 B10。此处“已验证”指先前运行，不声称当前仍有服务在线。
- 模型：`ZhipuAI/GLM-5.3-Flash`；本地 inventory 标记 `master`，没有可核验的不可变模型仓提交。镜像 digest、DeepGEMM wheel、实际加载身份在旧工程证据中记录，本包只提供必要的版本摘要，见[证据索引](evidence/INDEX.md)。

## 共享范围与下一步

本轮用户授权把最佳已验证启动组合与对应 SGLang 源码交到上述公开 fork。包中不含内部 ServingKit 代码、原始日志、trace、请求数据、集群地址或模型权重；这些在工程侧保留，Pro 不能声称已经读到。只有必须进入工程环境的问题才写 REQUESTS，由 Codex 后续 collect；问题本身不授权新部署或压测。

既有性能设计和上一轮容量计划只作为带条件的历史信息。Pro 应重新研究模型机制、社区与前沿方向，并分别评估固定 C80 和能填满更大 Decode batch 的场景；不要继承旧设计的唯一候选或把本包的 E2 命令当成硬约束。下一步只写研究记录与唯一 DESIGN，停止于设计交付。
