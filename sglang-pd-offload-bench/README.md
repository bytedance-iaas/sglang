# SGLang PD Offload Benchmark (Unified Orchestration Repo)

## 1. 仓库目标
本仓库用于在 **SGLang PD 分离场景** 下统一比较三类 KV cache/offload 模式：`gpu_only`、`eic_offload`、`hicache_mooncake`。目标是输出**可追溯、可复现、可解释**的结论，而非单一平均值排名。

## 2. 三种 backend 定义
- **gpu_only**：仅使用 GPU cache，不启用外部 offload。
- **eic_offload**：GPU + EIC 外部缓存层（host/remote/file/custom backend）。
- **hicache_mooncake**：SGLang HiCache + Mooncake 社区路径，支持在线建缓存与预热远端缓存。

## 3. 主评测线：为何偏真实场景优先
主结论优先来自：
1. shared-context multi-request reuse
2. multi-turn conversation replay
3. real arrival replay

Synthetic workload 仅用于解释机制、扫边界与验证猜想，不作为生产推荐唯一依据。

## 4. synthetic workload 角色
`generated_shared_prefix`、`prefix_repetition`、`long_doc_qa` 等用于：
- 变量控制
- 容量探针与 breakpoint 扫描
- 机制解释

## 5. 如何判断实验是否真的测到 offload
仓库内置 **experiment validity layer**：若实验未触发非 GPU tier，结果标记为无效或弱有效，不可用于最终 winner 结论。

## 6. experiment validity 标签
- `VALID_OFFLOAD_TEST`
- `LIKELY_GPU_FIT`
- `INSUFFICIENT_REUSE_PRESSURE`
- `NO_REMOTE_PARTICIPATION_OBSERVED`

> 最终综合结论仅使用 `VALID_OFFLOAD_TEST`。

## 7. capacity probing 方法
正式压测前先自动 sweep：`context_length`、`active_sessions`、`turn_count`、`concurrency`、`request_rate`、`reuse_distance` 等，定位 GPU fit→offload 触发的拐点。

## 8. 指标定义
- request-level: TTFT/TPOT/ITL/e2e p50/p95/p99, success/timeout
- serving-level: throughput, queue depth, pending
- cache-level: GPU/host/remote hit, eviction, load/writeback
- backend-level: transfer latency/throughput/errors/retries
- resource-level: GPU/CPU/NIC/disk 资源指标

## 9. 报告定义
- 单 workload 报告（Markdown）
- SLO 报告（strict/moderate/relaxed）
- Breakpoint 报告（自动识别 crossover）
- 综合结论报告（三大主线 + 参数区间映射）

## 10. breakpoint 分析方法
基于实验表格进行参数 sweep 曲线求交/区间比较，自动输出：
- context_length crossover
- turn_count crossover
- concurrency crossover
- request_rate crossover
- reuse_distance crossover
- burstiness crossover

## 11. SLO 分析方法
在每组 SLO 约束（例如 TTFT p95、success rate、timeout rate）下，计算每 backend 可持续 request_rate/concurrency 与 goodput，输出 winner-by-SLO。

## 12. 使用方法（CLI）
```bash
python -m bench.run --config configs/pd_compare.yaml
python -m bench.probe --config configs/probe.yaml
python -m bench.analyze --input results/run_xxx
python -m bench.report --input results/run_xxx
python -m bench.aggregate --inputs results/run_a results/run_b results/run_c
python -m bench.backend_bench --backend mooncake --config configs/mooncake_tebench.yaml
python -m bench.backend_bench --backend eic --config configs/eic_microbench.yaml
```

## 13. 如何扩展新的 backend / workload / replay
- 新 backend: 实现 `BackendAdapter` 抽象接口并注册。
- 新 replay: 实现 `ReplayAdapter` 接口并注册。
- 新 workload: 在 `bench/workloads/` 增加 adapter + manifest 输出。

## 14. 如何判断谁更优
从以下视角分别判断：
- strict TTFT SLO
- relaxed TTFT SLO
- throughput-first
- cache-retention-heavy workloads
- short reuse distance
- long reuse distance
- online-built cache
- pre-populated historical cache

关键原则：
- **GPU-only** 可能在缓存装得下且 SLO 极严时占优。
- **Offload** 方案可能在本地 cache 不够且高复用时占优。
- **EIC vs HiCache+Mooncake** 不预设赢家，必须用统一实验自动求 breakpoint。
- 无效实验（未触发 offload）不得用于最终结论。

## 15. 官方 benchmark 复用与等价适配
- SGLang bench_serving / hicache bench：直接调用或通过命令适配。
- Mooncake tebench：通过 backend microbenchmark adapter 对齐。
- vLLM workload 组织：以 dataset/trace 等价 adapter 兼容。
- LMCache Long Doc QA：接入为 long-doc replay 解释性基线。


## 16. Codex 云端仓库最简操作（不会 Git 也能跑）
如果你当前在 Codex 云端环境，不熟悉 Git，本仓库可直接按下面执行：

```bash
cd sglang-pd-offload-bench
bash scripts/cloud_quickstart.sh
```

执行后会自动：
1. 自动设置 `PYTHONPATH`（离线环境友好）
2. 跑一组最小实验（`bench.run`）
3. 生成一份 Markdown 报告（`bench.report`）

你只需要查看输出路径中的：
- `summary.json`
- `request_level.csv`
- `report.md`

如果你后续只想做 capacity probing：
```bash
cd sglang-pd-offload-bench
python -m bench.probe --config configs/probe.yaml
```
