# DP8 + EP8 DeepEP normal：失败候选

固定镜像及 serving source 753bbc090，generation10；从 DP8 MTP5/1/6 增加 EP8 / DeepEP normal，未改源码。normal 自动关闭 CUDA Graph。

| 输入 / 并发 | 成功 | TTFT avg | TPOT avg | 输出 tok/s |
| --- | --- | --- | --- | --- |
| 16384 / 1 | 3/3 | 1954.7 ms | 105.77 ms | 8.9305 |
| 16384 / 10 | 0/10 | 无有效值 | 无有效值 | 0 |

独立真实 SSE 预热通过。C10 的 EvalScope 子进程退出 0，但全部请求失败，summary -1000 ms 是无数据哨兵，不是负延迟测量。下一点 flush_cache 连接拒绝，runner exit1；64k/100k/224k 没有测量。失败原始归档包含两个 SQLite，已逐成员完整读取。

previous 日志显示多个 rank 在 eagle_worker_v2._draft_extend_for_decode → DSA indexer → deepgemm_paged_mqa_logits_split 触发 attention.hpp:315 `_batch_size == batch_size` 断言。Pod 自动重启 1 次后 Ready；上层批形状不一致根因尚未确认，不能归因于 DeepEP 通信故障，也不能按容器退出0判成功。

CuTe DSL 分支明确排除 draft_extend_v2，不能宣称切换它能修复此错误。该候选既存在并发运行错误，也有明显 graphless 解码代价，暂不继续同配置施压。后续性能候选回到已验证 CP8 chunk32k，仅新增原生 `--dsa-paged-mqa-logits-backend cutedsl`，检验长上下文低批量解码收益。

证据：dp8-ep8-normal-summary.json、dp8-ep8-normal-perf.tar.gz、dp8-ep8-normal-previous.log、dp8-ep8-normal-runtime-identity.json。后续 runner 新增请求成功数门槛，使用这次真实成功/失败 summary 验证通过，同时拒绝样本数不足和空 summary。
