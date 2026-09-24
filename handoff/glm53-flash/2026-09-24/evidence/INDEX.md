# 工程证据索引

## 版本与可访问性

| 证据 | 来源身份和状态 | 共享入口 | 支持与限制 |
| --- | --- | --- | --- |
| SGLang 源码 | Byte IaaS fork `57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c`，tree `17ea448109e9ac7b7674b1b13fa64528380632d1` | [源码定位](source-map.md)，同仓源码 | 可直接读代码和相应测试；不能单凭代码宣称运行命中或性能。 |
| E2 三角色命令 | 工程侧 2026-09-23 六臂实验的 `off-b10.yaml` 渲染及 P/D 启动日志；同一 SGLang commit | [脱敏命令与状态](launch-e2.md) | 保留影响性能的 args/env 与角色差异；内部挂载、主机、RDMA 设备名、registry、服务 DNS 改为具名变量。不是原始 YAML 的逐字副本。 |
| 参数解析 | D 端日志 `server_args`；静态来源见源码 map | [脱敏命令与状态](launch-e2.md) | 已核对 D `chunked_prefill_size=1024`、`disable_radix_cache=true`、Replay spec `false`、extra slots `10`；P/D 其余解析值应以各次实载日志判定。 |
| 模型结构 | 工程侧模型 inventory/`config.json`，标记 `ZhipuAI/GLM-5.3-Flash@master` | [研究输入](../BRIEF.md) | 45 主干层/34 KDA 层/1 draft 层等为既有 inventory 事实；不可变上游模型 revision 未确证，权重与配置原文未共享。 |
| 旧 benchmark/质量 | 上一轮容量任务的 Pod 内 C80 对照、Replay 容量、质量验证 | [研究输入](../BRIEF.md)仅有定性边界 | 原始 DB、逐请求记录、时间窗、日志未获本包共享授权；不能在此复算分数或独立验证质量失败。需要时由 Pro 写具体 REQUESTS。 |

## 状态区别与缺口

配置声明来自工程侧 ServingKit；本包以**最终渲染**的进程命令为起点。D 解析值另由启动日志核对，实载源码曾由 entrypoint 核对 commit/tree 和 Python import 路径。该旧运行已清理；本次 prepare 没有启动服务，也没有复测。固定镜像摘要在旧工程记录中为 `sha256:d33d932aee374f3884e7f49c54447c82ce61ac9e58749ebf5aa3a50d25c9cb28`，DeepGEMM 为 `sgl-deep-gemm 0.1.7`；公开材料不提供内部 registry 或 wheel 分发地址。

启动材料中的模型目录、源码目录、GPU/RDMA 网络、缓存目录、Router worker 地址必须由下一次目标环境填入。`master` 是可变模型标记，不可当模型 commit。任何新性能主张都需固定模型/镜像/代码、请求窗口和实际 Decode batch 后重新测量。
