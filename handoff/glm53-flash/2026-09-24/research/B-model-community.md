# B：模型、社区实现与关键组合缺口

日期：2026-09-24。材料 E=`eef8d29285ff2493b79bd696caeeb877ba664063`，源码 S=`57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c`。本轮主线程独立研究；未委派独立子 agent，未运行工程验证。研究取舍见 [DESIGN](../DESIGN.md)，全候选状态见 [INDEX](INDEX.md)。

## 1. 模型参考与目标实现：已知什么，未读到什么

材料 inventory 给出 45 trunk layers、34 KDA、1 NextN，模型为混合 MoE/DSA-MLA/KDA。固定 `Glm5NextConfig` 的默认值不能替代实际模型 config，尤其 gate lower_bound、mHC、专家 top-k、DSA KPool 和量化 ignore 列表。`master` 也不能替代 revision。

本轮读取了固定 GLM config/model/NextN 与关键 backend，但 Z.ai 官方 GLM-5.3-Flash 博客全文和 Hugging Face 官方 config 的工具访问失败。没有把搜索摘要、第三方量化模型卡或相近 GLM-5.3 非 Flash 的报告冒充本模型参考。Q1 请求的是本地已加载身份与必要配置，不是要求提供权重或内部地址。

三类缺口分别检查：模型数值/状态定义对目标引擎的约束；其它模型/引擎成熟优化可迁移部分；与 PD、投机、prefix/state cache、Graph、量化、并行的组合。以下每项都把“直接可开”和“值得研发”分开。

<a id="b1"></a>
## 2. B1：H20 KDA 长 chunk + FP32 状态 + checkpoint 的完整 prefill 能力

**特征 → 限制。** 34 个 KDA trunk layers 使长 prompt 的 recurrent prefill 成本值得单列。固定 S 已有 `FlashKDAKernel`，不是缺一个 backend 名字；但 wrapper 对最短序列小于 64、最长大于 2048、非 bounded gate、spec forward，以及需要 intermediate checkpoint 的 batch 会退回 Triton。P 的 8192 chunk 若形成大于 2048 的单序列段，不能靠设置 flashkda 获得该快路。tracked prefix 路径的回退是正确性保护，不应删除。[fixed-wrapper][fixed-kda]

**额外核验：FP32 接口不等于 FP32 递推。** 本轮固定外部 FlashKDA 到 `7afb9f454f160a6c4bbc0999beca0a8c40a38934`。README 允许 FP32 initial/final state，但 `csrc/smxx/fwd_kernel2.cuh` 的 `state_acc` 是 BF16，FP32 输入通过 `smem_cvt_fp32_to_bf16` 转换后进入循环。4 月 v1 深潜也说明内部 BF16 state。因而其原版直接移植不满足本轮保持 FP32 KDA 状态精度的边界；不能只看输出 tensor dtype。[flash-readme][flash-kernel][flash-blog]

**值得研发的增量。** 两个同尺度备选：一是改良现有 Triton chunk KDA 的中间工作区/launch/布局并保持 FP32 数值路径；二是借鉴 FlashKDA 的 token-parallel 预处理与 head-parallel recurrence 分工，实现真正 FP32 checkpoint/递推语义，并扩展长 chunk、indexed-state 和指定边界 checkpoint 输出。后者需要 kernel、dispatcher、pool、Graph lifetime、PD/缓存接口共同完成，不能缩成安装依赖。

**为什么不简单 8192→2048。** 即使小 chunk 命中快 kernel，四次完整模型 pass 的 attention/MoE/collective/PP 开销也可能更大。正确对照是完整模型等工作量的 1×8192 与 4×2048，且处理最后尾段、prefix 命中和 in-flight state，而非拿一个局部 kernel 数字外推。

**组合接口。** 输入/输出必须携带 `request_epoch, layer/head-shard, committed_token_boundary, state_layout`；导出的是准确边界的 conv+KDA state，不能把 final state 伪装为内部 checkpoint。target verify/draft extend 不得通过 inplace prefill kernel 提前提交。超出支持的 gate/shape/track 模式应显式正确 fallback。

**选择与门。** 中到大投入；当 Q3 指向 P/TTFT 瓶颈时，比 D 纯容量工作优先。若 FP32 改造与工作区成本消耗掉收益，选 Triton 改良而非降低精度。研究门：config/gate/shape 闭合、FP32 数值参考和状态接口成立；采用门：真实 P→D 请求、长上下文/缓存/质量与等工作量 E2E 收益。当前结论是有价值的能力补齐，不是推荐开启现成 FlashKDA。

<a id="b2"></a>
## 3. B2：batch-aware fused verify 与 ring 的组合，而非强开旧融合

固定 S 的 `_can_run_fused_chain_verify` 遇到 Replay raw-v ring 或 tree pointers 会拒绝；还要求 T≥3、conv width/精度/stride 等条件。E2 的 `SGLANG_OPT_FUSED_KDA_VERIFY=0`，所以既有历史测量只能约束当时组合，不能证明新的 batch-aware 融合无价值。[fixed-kda]

[SGLang #36821][pr-ring] 在本轮查询为 merged，merge `99060191e7ebfe2a666215658ceb43e9257729e1`，已提供 fused kernel 写 raw-k/raw-v/gate/beta ring 的组合，保留不支持布局的原路径。当前固定 S 仍有上述 guard，因此这是**已存在上游方案的本地差异**，不是从零创新。该 PR 的描述、测试边界及状态已读；未声称对完整 patch 做过逐行审计。

原作者 H20-3e 的 shape 是其它模型、H=HV=8、T4、安全 gate；结果有明显 batch/BV 交叉点，固定 BV4 在一些 batch 反而更慢。不能移用该速度比，也不能把“合入”当生产采用门。

**本地增量。** 先对齐 fork 的 GLM ring slot、raw operands、FP32 fold 和 graph padding 契约；只给已验证形状选择 fused 变体，其余保留 unfused。候选维度是少数真实 B/T/heads 对应的 dispatch，而非任意扫描所有 block size。ring byte-for-byte 对齐与输出/conv/state 不变式优先于性能。

**备选与处置。** 与不改 ring 的 snapshot fused、现有 unfused、浅 native NextN 比较；中等投入，H20 特化不可直接照搬 Blackwell CuTe DSL。Replay 质量门未过时，本项 ring 模式不得采用；snapshot 侧独立研究不必被它阻塞。C80 不预设 fused 赢，高 batch 不预设去掉快照就带来 E2E gain。

<a id="b3"></a>
## 4. B3：完整混合 prefix/state cache，而不是提高一个命中率计数

固定 S 已有 KDA `track_chunk_idx`、FP32 track scratch、内部快照和 packed conv；不能把这些重新列作“新增功能”。真实缺口在于**所选快 kernel 是否完整支持边界输出，以及缓存/外部传输能否维护同一逻辑边界**。E2 的 PD D 解析为 radix disabled，不能简单反转 flag。[fixed-kda][fixed-wrapper]

可借鉴的成熟实现：

- [vLLM #52789][v-checkpoint]，2026-08-22 merged，merge `9eb9d9d3953959695108600c8ed33d36bc6a1e5f`：把 checkpoint 边界处理放到 KDA 内部，避免为尾段再过一次完整模型。其收益不能直接迁移到已经有 track 能力的 S。
- [vLLM #53614][v-partial]，2026-09-06 merged，merge `144e79c8106da23141ac010394b782f730cc7fe8`：partial prefix、Eagle rewind 与外部 connector 共享 checkpoint-validity 规则，按实际导出的边界重建 key。
- [vLLM #51358][v-mooncake]，2026-08-29 merged，merge `6b110badbb22d3f66c7218b71138f13b7a6b3419`：精确 block-ID/boundary handoff 与跨 rank 完成后释放 pin，避免正确 prefix hash 指向 stale/null/speculative state。这是该引擎的已报告问题，不证明本地 S 存在同一 bug。

**本地必须共同维护的 bundle。** trunk KDA state、conv window、DSA/MLA KV、KPool/index 元数据、NextN 所需上下文，以及 model/tokenizer/template/quant/layout 身份。cache 命中的 credited length 必须与实际 state position 相等；不能为“全命中”省略必要的一 token 重算或忽略最后未对齐边界。异步写出必须持有真实对象到所有 rank 终态，错误/取消/去重也须释放，旧 request slot 不得污染新请求。

**收益与投入。** 只有真实 prefix 复用且 P 计算成本/排队显著时才优先；Q4 未提供命中分布，本轮不假设 90% 命中。先 P-side 正确边界复用、再外部 cache/拓扑变化；与同等投入的 B1 kernel 路线比较。中到大投入，不能以局部 hit-rate 代替端到端质量/TTFT。批次拆合、部分命中、chunk boundary、PD、取消和请求槽复用均需覆盖。

<a id="b4"></a>
## 5. B4：DSA NoPE/KPool/FP8 的等价后端与并行组合

S 的 DSA backend 根据实际 `index_kpool`、page size、top-k 决定 metadata fusion 支持域；raw FP8 TileLang 与 scaled MLA layout 的 dequant 契约不同，当前 raw 路径会关闭不兼容的 MHA one-shot。不能通过改 `index_kpool`、少选 tail token 或错用 scaled reader 来换性能。[fixed-dsa]

社区 [#36830][issue-fp8] 的早期 H20 报告与 [DCP roadmap #29736][roadmap-dcp] 提供 KPool tail、FP8 consumer、DSA DCP×MTP/PD 的组合线索；前者不能被当作固定 S 的现存故障，因为 E2 的 raw512+TileLang 已经是已验证起点。roadmap 的其它 GPU/模型已勾选项也不能推出本地组合可用。

**增量方向。** 当 Q3 证明 DSA/indexer 是关键路径，再比较完整的 H20-compatible NoPE/KPool consumer 或现有 TileLang 的 metadata/index gather 与计算布局改良；保留同一 top-k/tail、FP8 values/scales、page strides、verify 长度和 PD relayout。成熟 kernel 不支持该组合时，需要 kernel+adapter+pool/transport 全链完成。DCP 是进一步拓扑研究，需要合法 attention-TP 组、KDA state ownership 和 draft sidecar 映射，不是当前 DCP1→N 的直接参数优化。

**备选/停止。** 与 A2 的整窗口成本降低比较，中到大投入；若 DSA 占比低或主要成本在 MoE/NextN，先不做。不能从 SM100/SM120 原版移植失败推出机制对 H20 无价值，但也不采用换卡、FP4 state 或改变模型索引语义的捷径。

## 6. 跨组合风险：block draft 不能绕过 PD 工程

C2 的并行 drafter 若进入工程阶段，必须补齐自己的 hidden-state、draft KV、TP/head shard 与 PD 边界，而非只复用 target KV。2026-09-24 [SGLang #41038][issue-draft-pd] 报告 GLM+DFlash2 在非 DCP 的非对称 P/D TP 下 draft KV stride/head-slice 不匹配；其 B300 短回归不是 H20 本地故障证明，但足以构成必验边界。该 issue 当前 open。相关退化输出报告也不能直接归因于 drafter 算法，应区分权重身份、采样与状态/传输错误。

## 来源与访问边界

外部 PR 读取的是 API 返回的描述、head/merge 身份及状态，不声称已审计全部代码或复现实验。FlashKDA README、v1 深潜、源码树、`fwd_kernel2.cuh` 1–230/285–410 与固定 S wrapper 已直接读；没有运行其测试或采用公开 benchmark 的速度预测本地结果。模型博客/HF config 与 GLM DSpark preview 卡访问失败，后者也未据镜像站确认为可用 checkpoint。

[fixed-kda]: https://github.com/bytedance-iaas/sglang/blob/57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c/python/sglang/srt/layers/attention/linear/kda_backend.py
[fixed-wrapper]: https://github.com/bytedance-iaas/sglang/blob/57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c/python/sglang/srt/layers/attention/linear/kernels/kda_flashkda.py
[fixed-dsa]: https://github.com/bytedance-iaas/sglang/blob/57d5aa9b45c2c0dd37ce59b9caa1ea30a6df0b5c/python/sglang/srt/layers/attention/dsa_backend.py
[flash-readme]: https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/README.md
[flash-blog]: https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/docs/20260420-flashkda-v1-deep-dive.md
[flash-kernel]: https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/csrc/smxx/fwd_kernel2.cuh
[pr-ring]: https://github.com/sgl-project/sglang/pull/36821
[v-checkpoint]: https://github.com/vllm-project/vllm/pull/52789
[v-partial]: https://github.com/vllm-project/vllm/pull/53614
[v-mooncake]: https://github.com/vllm-project/vllm/pull/51358
[issue-fp8]: https://github.com/sgl-project/sglang/issues/36830
[roadmap-dcp]: https://github.com/sgl-project/sglang/issues/29736
[issue-draft-pd]: https://github.com/sgl-project/sglang/issues/41038
