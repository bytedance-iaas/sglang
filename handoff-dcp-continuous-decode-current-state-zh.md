# DCP Continuous Decode / Staging 当前状态

## 当前结论

- 190K generated-shared-prefix 验收已经通过：300 prompts、concurrency 60、request-rate 30、output 1500，最终 300/300 successful。
- DCP staging 的永久卡死已解决：未再出现 `decode_transfer:39` 长时间不动、DCP health timeout、prefill `KVPoll.Bootstrapping` timeout 或 decode Traceback。
- 当前 DCP staging 稳定性已满足手动测试和继续优化的基础要求。
- decode DP2 拓扑在 TPOT/TTFT 上仍可能优于 DCP2，这是架构差异：DP2 是两个完整 decode replica，DCP2 是单请求跨 DCP shard，额外付出 KV 分片、attention 同步、staging gather/scatter 和调度 gate 成本。

## 已保留的代码修复

- `python/sglang/srt/disaggregation/common/staging_handler.py`
  - 增加 final all-prefill-success 的 all-remaining scatter fallback。
  - 在 `advance_scatter()` 中主动 replay pending ready chunks。
  - 等所有 chunk scatter event 完成且没有有效 staging allocation 后才认为 staging done。
- `python/sglang/srt/disaggregation/mooncake/conn.py`
  - `CHUNK_READY` pending state 只在 scatter submit 成功后清理。
  - 清理 `_chunk_writer_counts` 改为幂等 `pop`，避免 replay 并发触发 `KeyError`。
  - final success 阶段 replay ready chunks 并 scatter 所有剩余 allocation。
- `python/sglang/srt/managers/scheduler.py`
  - DCP health check 增加基于 active batch 和 `forward_ct` 的 progress-aware reset。
- `python/sglang/srt/disaggregation/decode.py`
  - 保留 post-transfer queue length 一致性检查，去掉“必须完全 drain 后才能 decode”的过强限制。
- `sgl-model-gateway/src/routers/http/pd_router.rs`
  - HTTP PD router 独立记录 prefill/decode circuit outcome，避免 decode-only failure 误伤 prefill circuit。

## 已清理内容

- 已删除 staging 190K 调试上报代码：`SGLANG_DCP_STAGING_190K_DEBUG`、debug server HTTP reporter、`_debug_report_dcp_staging_190k(...)` 调用点。
- 已删除本地 `.dbg/` debug-server 环境文件。
- 未主动删除历史 `dbg/`、`.trae/`、playground 脚本和其他未跟踪临时内容，因为它们可能属于历史资料或用户本地文件。

## 推荐手动验证

- Backend DCP staging 稳定性验证：router 加 `--disable-circuit-breaker`，避免 router circuit policy 掩盖 backend 行为。
- Gateway attribution 验证：router 不加 `--disable-circuit-breaker`，确认 decode transient failure 不再导致大量 `No available prefill workers` 或 503。
- 成功标准：190K/300 workload 300/300 successful，decode `#transfer-req` 不形成永久积压，prefill/decode 无 DCP health timeout。

## 后续优化方向

- 短期：如果目标是最低 TPOT，且单个 decode replica 能承载目标上下文/并发，优先使用 `decode tp16+dp2+ep16`。
- 短期：如果目标是更大长上下文容量或更高长上下文并发，使用 `decode tp16+dcp2+ep16`，接受其同步和 staging 成本。
- 中期：减少 staging chunk 控制面和 watermark 频率，优化 gather/scatter kernel，降低每 chunk CUDA event 成本。
- 中期：完成 gateway build，并用 circuit breaker enabled 复测 190K，验证 router attribution fix。
- 长期：推进 DCP KV pool 物理布局优化，让 prefill 生成的 KV 更接近 decode DCP 物理布局，从根上减少 staging 重排成本。
