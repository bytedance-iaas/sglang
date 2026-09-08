# 多 cycle：DMA 通信 + 可控 GEMM pipeline benchmark

入口：[bench_dma_compute_pipeline.py](bench_dma_compute_pipeline.py)。它是独立脚本，未修改
[纯通信 benchmark](bench_dma_cycle.py) 的实现、参数或输出。

生产 runtime 已支持 fixed DMA + flag 和可选 slice。本 demo 的无 GUI Nsight Systems/Compute
分析工具位于 [nsight_pipeline](nsight_pipeline)，集中状态见
[SIDP_performance_followup_todos.md](../../../../sidp/document/SIDP_performance_followup_todos.md)。

快速运行一次8卡工具 smoke（本地命令中的 `python` 是远程 wrapper）：

```bash
python sglang/benchmark/kernels/sidp/nsight_pipeline/profile_pipeline.py \
  sglang/benchmark/kernels/sidp/nsight_pipeline/configs/pipeline_smoke.json
```

大体积 `.nsys-rep/.ncu-rep/SQLite` 只写到远端配置的 `artifact_dir`；可回传的
`analysis.json/report.md/manifest.json` 写入 `check_logs`。正式对照配置参考
[pipeline_example.json](nsight_pipeline/configs/pipeline_example.json)，指标解释和排查顺序见
[how_to_analyse.md](nsight_pipeline/how_to_analyse.md)。普通 benchmark 不传
`--nsight-annotations/--nsight-profile-sample` 时不增加 profiler 标记或采样控制。

专门检查SM-copy CTA落在哪些SM，以及GEMM在独立/并行Graph中的指标差异时，改用
[`pipeline_sm_trace_smoke.json`](nsight_pipeline/configs/pipeline_sm_trace_smoke.json)；该配置会同时采集
Systems、isolated node NCU和whole-Graph NCU。

Nsight工具的Systems阶段会额外传`--sm-execution-trace`，让通信/控制kernel的每个CTA记录入口/出口
`%smid`与`%globaltimer`；原始记录在每个rank内聚合后才写入小型JSON。该开关要求单一case和单一
profile sample，普通benchmark与服务runtime不会执行trace分支。DMA memcpy由Copy Engine执行，
本身没有SMID；fixed DMA路径只会记录flag wait/publish等控制kernel。显式选择`compute_sm_flag`或
`dynamic_sm*`时，`copy_selected`的每个CTA也会记录实际SMID。`--sm-copy-ctas 0`表示`4×设备SM数`，
也可以显式设CTA数；`--sm-copy-block`支持128/256/512。

## 要回答的问题

纯通信结果说明：没有计算时，dynamic-owner DMA 在 rank 同相启动时可以避开 incast；相邻 rank
错开 1ms 后，固定 compute-order DMA 反而可能更快。这个 benchmark 在两条通信路径旁加入相同的
BF16 GEMM 链，用来判断：

1. 单 cycle 通信差异进入 rolling pipeline 后，最终落在 RAW wait、compute critical path，还是只留在尾部；
2. dynamic 的乱序 copy 是否延迟了计算顺序靠前的 layer；
3. rank 启动相位变化后，owner claim/WAR 与固定 Event 路径分别花了多少时间；
4. 增加计算量后，dynamic/static 的通信差异能否被覆盖。

## 模拟的 graph

默认 `cycles=6`、cache depth 固定为 2，与当前 Gemma4 SiDP pipeline 对齐：

```text
compute stream                         comm stream
  graph start ───── fork ────────────> prefetch c1
  compute resident c0                  |
    last remote(c0) ─────────────────> prefetch c2 after slot WAR
  compute c1 (per-layer RAW wait)       |
    last remote(c1) ─────────────────> prefetch c3
  compute c2                            ...
  compute c4 ─────────────────────────> refill next-forward c0
  compute c5
  compute end ───── final join <────── comm tail
  graph end
```

| 模式 | 通信/同步 | 用途 |
|---|---|---|
| `compute_only` | 无通信、无等待 | 测完全相同的串行 GEMM 链下限 |
| `compute_dma` | 固定 compute order、DMA、ready/consume Event | 当前默认 SiDP 对照 |
| `compute_dma_slice2` | 固定顺序、DMA、Event；每个 component 切 2 片，按 slice→owner 排序 | S=2 分片交错消融 |
| `compute_dma_slice4` | 固定顺序、DMA、Event；每个 component 切 4 片，按 slice→owner 排序 | S=4 分片交错消融 |
| `compute_dma_slice4_group2` | 固定顺序、DMA、Event；连续 owner 分两组，每组内部 S=4 | S=4、G=2 |
| `compute_dma_slice4_group4` | 固定顺序、DMA、Event；连续 owner 分四组，每组内部 S=4 | S=4、G=4 |
| `compute_dma_flag` | 固定 compute order、DMA、fill/comp generation flag | Event→flag 同步消融 |
| `compute_dma_slice4_flag` | 固定顺序、DMA、generation flag；S=4、G=1 | 分片下 Event→flag 同步消融 |
| `compute_dma_slice4_group2_flag` | 固定顺序、DMA、generation flag；S=4、G=2 | 分片分组下 Event→flag 同步消融 |
| `compute_dma_slice4_group4_flag` | 固定顺序、DMA、generation flag；S=4、G=4 | 分片分组下 Event→flag 同步消融 |
| `dynamic_dma` | rotating owner claim、conditional DMA、fill/comp generation flag | 方向 C 原有动态路径 |
| `dynamic_dma_compute_priority` | 每次从最靠前的未完成layer扫描owner，其余同上 | 动态顺序关键路径消融 |
| `compute_sm_flag` | 固定 compute order、SM copy、generation flag | 显式模式；隔离 DMA→SM-copy 成本 |
| `dynamic_sm` | rotating owner claim、SM copy、generation flag | 显式模式；观测动态仲裁与SM-copy争用 |
| `dynamic_sm_compute_priority` | compute-priority owner claim、SM copy、generation flag | 显式模式；动态SM关键路径消融 |

因此 `compute_dma` vs `compute_dma_flag` 只改变同步协议；`compute_dma_flag` vs
两条 dynamic 路径保持 DMA 和 flag 不变，只增加动态 owner 仲裁、动态 copy 顺序和 conditional
节点；两条 dynamic 之间只改变 claim 扫描优先级。

S=2/4 并没有并行提交 memcpy，而是在同一条 comm stream 上将顺序改成：

```text
slice0(owner0), slice0(owner1), ...,
slice1(owner0), slice1(owner1), ...
```

一个 slot 仅在第一片前执行一次 WAR 等待，并在最后一片结束后记录 ready Event 或发布 fill
generation，因此计算不会读取尚未完整传输的权重。分片边界用整数比例计算，任意字节数（包括不足
整除的尾部）都无空洞、无重叠。

每个模拟 layer 执行 `gemm_repeats` 次相同的 BF16 `torch.mm`。通过 `--gemm-m/n/k` 控制单次
GEMM 形状，通过 `--gemm-repeats-values` 扫描每层计算量。计算 stream 保持串行；通信 stream
按照当前两周期 slot 复用规则推进。

GEMM 有意使用独立的 A/B tensor，不把数百 MiB 通信 payload 解释成矩阵。这使通信字节数与计算量
可以分别调节，也避免 benchmark 伪装成完整 FFN。remote layer 仍会在 GEMM 前等待它的 slot ready，
GEMM 后发布 slot consumed，所以 RAW/WAR 关键路径是真实的；但本 benchmark 不验证模型精度。

## 低扰动主测试

先确认 8 张 GPU 没有其他任务。workspace 本地执行时，`python` wrapper 会同步到远端容器；在远端
容器内可直接执行同一命令。

```bash
# 默认 Gemma4 单层通信量、六个cycle；每个sample独立生成随机rank到达延迟。
# repeats=1/2/4 用于观察计算逐步覆盖通信后的转折。
python sglang/benchmark/kernels/sidp/bench_dma_compute_pipeline.py \
  --k-values 1,4 \
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
  --gemm-repeats-values 1,2,4 \
  --random-delay-max-us 1000,5000,10000 \
  --iterations 20 --warmup 3 \
  --output check_logs/sidp_dma_compute_pipeline_main.json
```

## 分片 + 随机到达 pipeline 测试

`--slice-study` 只 capture/run `compute_only`、整层 S=1、S=2 和 S=4 四组，原有默认五组模式保持
不变。随机场景不是稳定的 rank×offset：每个 sample 都为 8 个 rank 独立抽取 `U[0,max]`，再减掉
该 sample 的最小值。四种模式在同一 sample 内复用完全相同的 offset 向量，下一 sample 重新抽取，
避免模式间输入相位不同。

```bash
python sglang/benchmark/kernels/sidp/bench_dma_compute_pipeline.py \
  --slice-study \
  --k-values 1,4 \
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
  --gemm-repeats-values 1,2,4 \
  --random-delay-max-us 1000,5000,10000 \
  --iterations 20 --warmup 3 \
  --output check_logs/sidp_dma_compute_pipeline_slice_random.json
```

所有模式统一使用随机 delay；省略 `--random-delay-max-us` 时默认采用 `1000,5000,10000`。
固定 `rank×stagger`、显式 rank offsets 和额外 jitter 已移除，避免误将稳定错相当成 serving 到达模型。

### Continuous arrival：连续 decode 稳态对照

默认的 `reset_uniform` 会在每个 sample 前让所有 rank 重新对齐，因此不会保留上一轮执行速度造成的
rank 相位漂移。`--arrival-model continuous` 改为每个 mode/setting 只在 epoch 开头同步一次；之后每个
rank 独立连续 replay，上一轮 Graph 完成时刻加一个 `U[0,max]` scheduler gap 后立即发起下一轮，
epoch 中间没有 barrier 或 all-gather。所有待比较模式复用相同的 per-rank gap 序列。
benchmark 会在 epoch 前显式回收一次 Python 对象，并在 measured epoch 内关闭 cyclic GC；这是为了避免
累积逐 replay 诊断字典触发数百毫秒的 Python GC pause，不能把该观测开销误判为某种 CUDA Graph
模式的 host launch stall。

下面这组正好比较 Event/flag × S=1/S=4；`--continuous-replays 256` 表示每个 rank、每个模式连续
执行 256 次。报告以整个 epoch 的 aggregate rank-replay/s 为主，并直接输出
`event_s4_to_s1`、`flag_s4_to_s1`、`s1_flag_to_event` 和 `s4_flag_to_event` 四个吞吐差值：

```bash
python sglang/benchmark/kernels/sidp/bench_dma_compute_pipeline.py \
  --arrival-model continuous \
  --continuous-replays 256 \
  --modes compute_dma,compute_dma_slice4,compute_dma_flag,compute_dma_slice4_flag \
  --k-values 1,2,4 \
  --cycles 6 \
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
  --gemm-repeats-values 1 \
  --random-delay-max-us 1000 \
  --warmup 3 \
  --output check_logs/sidp_dma_sync_slice_continuous.json
```

这里 `--iterations` 只服务于默认的 `reset_uniform` 路径，在 continuous 模式下不参与测量次数。
continuous arrival 暂不支持 `--nsight-profile-sample`，因为 profiler 的跨 rank capture collectives 会
破坏 epoch 内无同步的语义。

2026-09-06 在 H20×8 上按上述 continuous 语义完成 K=1、M=1024、真实 Gemma4 payload、256
replay/rank 的四模式对照，证据见
[continuous K1/M1024 报告](../../../../check_logs/sidp_dma_sync_slice_continuous_k1_m1024_gcfix_20260906.md)：

- Event：S4/S1 吞吐 `-1.53%`；flag：S4/S1 吞吐 `-1.35%`。continuous rank phase 会在 epoch
  内自然漂移，分片在该场景不再保持旧 reset-per-sample 模型中的收益，方向与 serving 结果一致。
- S1 下 flag/event 为 `+0.35%`，S4 下为 `+0.53%`，都应视为基本持平；demo 仍未复现 serving
  中 Event 约 2% 的优势，因此 arrival 模型只能解释 slice 反转，不能单独解释全部真实模型差异。
- S4 的逐 replay `max-rank graph p50` 虽更低，但跨所有 rank/replay 的平均 graph 时间从 S1
  Event 的约 `71.44ms` 增至 S4 Event 的约 `72.84ms`。continuous workload 应优先看完整 epoch
  吞吐，不能用逐 ordinal 的 max-rank p50 代替稳态吞吐。

### S=4 分组测试

`--group-study` 运行不分片 baseline，以及 S=4 的 G=1/2/4；每种调度同时测试 Event 和
generation flag，共八种模式，不运行 compute-only 或 dynamic 模式。所有模式在每个 sample
复用同一个随机 delay 向量。`G` 表示连续 owner
分组数；以 K=1 的七个 remote layer 为例：

```text
G=1: [1,2,3,4,5,6,7]
G=2: [1,2,3,4] [5,6,7]
G=4: [1,2] [3,4] [5,6] [7]
```

每个组内部按 slice0→slice3 完整搬完，再开始下一组：

```bash
python sglang/benchmark/kernels/sidp/bench_dma_compute_pipeline.py \
  --group-study \
  --k-values 1,4 \
  --cycles 6 \
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
  --gemm-repeats-values 1 \
  --random-delay-max-us 1000,5000,10000 \
  --iterations 20 --warmup 3 \
  --output check_logs/sidp_dma_compute_pipeline_s4_group_random.json
```

若只想快速确认机制，缩小 payload、GEMM 和 cycle 数：

```bash
python sglang/benchmark/kernels/sidp/bench_dma_compute_pipeline.py \
  --k-values 1,4 --cycles 2 \
  --component-bytes 1048576,524288,13 \
  --gemm-m 256 --gemm-n 256 --gemm-k 256 \
  --gemm-repeats-values 1 \
  --random-delay-max-us 100 --iterations 2 --warmup 1 \
  --output check_logs/sidp_dma_compute_pipeline_smoke.json
```

## 诊断 trace

主测试默认只放 graph、compute-cycle、comm-cycle 边界 Event。需要解释差异时另跑 trace，避免把
细粒度 Event 节点的扰动混入主结果：

```bash
python sglang/benchmark/kernels/sidp/bench_dma_compute_pipeline.py \
  --k-values 1 --gemm-repeats-values 1 \
  --random-delay-max-us 1000 --iterations 5 --warmup 2 \
  --trace-layers --trace-steps \
  --output check_logs/sidp_dma_compute_pipeline_trace.json
```

- `--trace-layers`：逐 layer 记录 RAW wait 和 GEMM window。
- `--trace-steps`：逐 copy 记录实际 owner/slot、claim 或 Event-WAR wait、transfer window、碰撞、自旋，
  以及相对当前最靠前未完成layer的`priority_distance`。
- `--random-delay-max-us 1000,5000`：分别测试 U[0,1ms] 和 U[0,5ms]；逐 sample 独立重抽，所有模式使用相同 offsets。

需要直接检查 capture 后的 Graph DAG 时，可导出 rank0 的 slim/verbose DOT 和 Event handle 对照表：

```bash
python sglang/benchmark/kernels/sidp/bench_dma_compute_pipeline.py \
  --k-values 1 --cycles 2 \
  --component-bytes 1048576,524288 \
  --gemm-m 256 --gemm-n 256 --gemm-k 256 \
  --random-delay-max-us 0 --iterations 1 --warmup 0 \
  --dump-graph-dir check_logs/sidp_graphs \
  --dump-graph-modes compute_only,compute_dma,compute_dma_flag \
  --output check_logs/sidp_graph_dump_smoke.json
```

导出直接调用 `cudaGraphDebugDotPrint`：`.dot` 只包含拓扑，`.verbose.dot` 包含节点参数，
`.events.json` 将 graph/cycle/ready/consumed Event 的 CUDA handle 映射回语义名称。该功能默认关闭，
不会改变普通 benchmark 的 capture 路径。

## 如何看报告

输出 `.json`、`.md` 和 `.samples.jsonl` 三个文件，已存在时拒绝覆盖。

- `graph_ms`：本 rank graph start 到 final join 后 graph end；核心端到端指标。
- `compute_path_ms`：start 到最后一个 GEMM 完成，包含所有暴露出来的逐层 RAW wait。
- `tail_join_ms`：最后一个 GEMM 完成后，等待 comm stream/next-c0 refill 的尾巴。
- `compute boundary max-rank`：复用已有 Event，把 compute path 拆成 c0 前空洞、cycle 间空洞、
  最后一个 cycle 后空洞和各 cycle 本体；其中 `c0_start_minus_c1_start_ms` 可判断 Graph fork 后
  comm/compute 哪条分支先被调度。
- `compute_cycles[].elapsed_ms`：每个 cycle 的计算 stream 时间，包含本 cycle 的 RAW wait。
- `comm_cycles[].elapsed_ms`：c1..c5 与 next-c0 的通信 stream 窗口，包含 WAR/claim/控制/DMA。
- S=2/4 的逐 step transfer window 从该 owner 第一片开始、到最后一片结束，中间包含其他 owner 的
  分片；它表示交错完成窗口，不是纯 memcpy duration。
- 报告同时给出跨全部 rank/sample 的 cycle p50 和“每个 sample 先取最慢 rank、再取 p50”的 max-rank 数据；判断关键路径时优先看后者。
- `fixed-flag/event graph ratio`：固定顺序下 Event→flag 的端到端成本。
- `rotating/fixed-flag graph ratio`：相同 DMA+flag 下原有动态 owner/顺序/conditional 的净影响。
- `compute-priority/rotating graph ratio`：只改变动态candidate优先级后的净影响。
- `dynamic/event graph ratio`：完整动态方案相对默认 fixed DMA+Event 的最终结果。
- 各通信模式相对 `compute-only`：相对于纯计算链暴露出的通信与同步成本。

不要跨 GPU 相减 Event timestamp；脚本只汇总各 GPU 内 duration。`all_rank_observed_wall_ms` 使用 host
观测的最早 launch 到最晚 completion，包含随机 launch delay 和 host 误差。

这个 benchmark 刻意不模拟 attention、RMSNorm、真实两段 FFN、不同 layer 的算子形状和 serving
调度漂移。若它复现 dynamic 回退，就适合继续用 Nsight 分析独立 graph；若它不复现，下一步应回到
模型 runtime 检查真实算子序列或调度，而不是继续给 DMA 路径猜原因。

## CPU 回归

```bash
python sglang/test/registered/unit/layers/sidp/test_dma_compute_pipeline_benchmark.py -v
```

## 2026-09-04 首轮验证

- 13 项 CPU 配置/统计测试通过。

## 2026-09-04 M=32768 关键路径定位

针对“计算已经长于通信，为什么带通信后仍慢于 compute-only”增加了两轮诊断：

- [低扰动边界报告](../../../../check_logs/sidp_dma_compute_boundary_m32768_20260904.md)：5 个 sample，
  只复用已有 Graph/cycle Event，不增加 Graph 节点。
- [逐 layer trace 报告](../../../../check_logs/sidp_dma_compute_layer_trace_m32768_20260904.md)：3 个 sample，
  插入 Event 拆分 RAW wait 和 GEMM window，仅用于定性归因。

低扰动测试中，各模式最后一个 GEMM 后的 `tail_join_ms` 都只有约 0.004ms，确认差值不是 forward
末尾残留一个通信 cycle。fixed DMA+Event 的主要额外时间发生在 Graph fork 后、c0 计算开始前：

| K | 模式 | graph max-rank p50(ms) | c0 前空洞(ms) | c1 启动(ms) | c0-c1 启动差(ms) | cycle sum(ms) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | compute-only | 375.62 | 0.004 | - | - | 375.60 |
| 1 | fixed DMA+Event | 397.81 | 20.710 | 0.004 | 20.706 | 378.12 |
| 1 | fixed DMA+flag | 377.81 | 0.185 | 0.013 | 0.177 | 377.54 |
| 4 | compute-only | 375.97 | 0.005 | - | - | 375.95 |
| 4 | fixed DMA+Event | 385.70 | 8.442 | 0.004 | 8.439 | 377.02 |
| 4 | fixed DMA+flag | 382.51 | 0.192 | 0.010 | 0.184 | 382.35 |

因此 fixed DMA+Event 的“尾巴”实际上是 **Graph 开头的调度空洞**：c1 通信几乎立刻启动，c0
计算却分别延迟约 20.7ms/8.4ms，且延迟随 c1 的 copy 工作量变化。逻辑 DAG 并没有要求 c0 等 c1；
实测只能证明当前 CUDA Graph 拓扑的调度结果近似先推进串行 memcpy/Event 分支，再开始 c0。flag
路径改变了 Graph 拓扑，把这个开头空洞压到约 0.2ms。

这一点随后用实际 M=32768、K=1、六 cycle 的
[verbose DOT](../../../../check_logs/sidp_graph_dag_m32768_20260904/rank0_k1_repeat1_compute_dma.verbose.dot)
直接确认：原始图共 207 个节点、277 条边，graph-start node0 同时指向 c1-start node1 和
c0-start node17。可达性检查显示 c1-start/node1、c1-end/node16 到 c0-start/node17 或
c0-end/node50 均不存在路径，c0-start 到 c1-end 也不存在路径。c1 memcpy 到 cycle1 对应 GEMM
的 RAW 边存在且符合预期，但没有连到 cycle0 GEMM。因此这里已经排除“代码里隐藏了一条
c1→c0 依赖”；约 25.7ms 的 c0 启动延迟来自两个 ready 分支的 Graph runtime 调度选择。
事件语义与原始 handle 的映射见同目录
[events.json](../../../../check_logs/sidp_graph_dag_m32768_20260904/rank0_k1_repeat1_compute_dma.events.json)。

逐 layer trace 进一步说明 fixed DMA+flag 在 K=4 仍比 compute-only 慢的主因不是 RAW wait：

| K | 模式 | RAW wait 合计(ms) | GEMM window 合计(ms) | 控制/残差(ms) | 边界空洞(ms) |
|---:|---|---:|---:|---:|---:|
| 1 | compute-only | 0.16 | 376.06 | 0.18 | 0.03 |
| 1 | fixed DMA+flag | 0.75 | 378.50 | 0.44 | 0.21 |
| 4 | compute-only | 0.16 | 376.08 | 0.18 | 0.02 |
| 4 | fixed DMA+flag | 0.34 | 382.54 | 0.30 | 0.21 |

K=4 的约 6.9ms 回退里，约 6.5ms 体现为五个与通信重叠的 cycle 内 GEMM window 变长；最后一个
不再重叠通信的 cycle 基本恢复 compute-only 水平。这确认存在真实的 compute/communication
资源干扰。现有 trace 尚不能在 DMA 对 HBM/L2/互联资源的争用和 comm stream 上常驻的 generation
wait kernel 之间做最终归因；若继续优化，应以 Nsight 的 SM residency、L2/HBM 吞吐为准，或单独
替换 wait kernel 做 A/B，而不是再把这部分解释为未掩盖的 RAW wait。

## 2026-09-04 随机到达主结论

本节取代后文基于固定同相/固定 stagger 的阶段性结论。H20×8、实际 Gemma4 component bytes、六
cycle、每组 20 个 paired sample 的证据为：

- [M=4096 完整报告](../../../../check_logs/sidp_dma_compute_pipeline_20260904_130334.md)
- [M=12288 完整报告](../../../../check_logs/sidp_dma_compute_pipeline_20260904_131728.md)

下表是 max-rank graph p50 的相对变化；负数表示前者更快：

| M | K | random delay | fixed flag vs Event | dynamic vs fixed flag |
|---:|---:|---:|---:|---:|
| 4096 | 1 | U[0,1/5/10ms] | -4.81% / -3.33% / -3.18% | +1.59% / +9.41% / +17.27% |
| 4096 | 4 | U[0,1/5/10ms] | -15.73% / -13.03% / -10.02% | +36.91% / +35.45% / +34.43% |
| 12288 | 1 | U[0,1/5/10ms] | -13.26% / -10.76% / -8.56% | +28.14% / +27.50% / +27.02% |
| 12288 | 4 | U[0,1/5/10ms] | -1.71% / -0.40% / +0.65% | +11.54% / +11.09% / +11.16% |

确定结论：

1. 以当前最优 fixed+flag 为对照，dynamic 在 12 个配置的聚合 p50 中全部回退；除 M=4096、K=1、
   小 delay 的少量单样本外，paired 结果也很稳定。`dynamic_dma_compute_priority` 与 rotating dynamic
   基本相同，改变 candidate 扫描起点没有解决问题。
2. dynamic 对 rank arrival 分布不敏感，M=4096 时 K=1/K=4 分别稳定在约 86.8/70.1ms；这种稳定性
   仍然存在，但稳定性能下界高于随机到达下的 fixed 路径，当前不足以转化为部署优势。
3. fixed+flag 在 12 个聚合结果中有 11 个优于 fixed+Event，唯一例外是 M=12288、K=4、U[0,10ms]
   的约 0.65% 回退。可以把 **fixed DMA + flag 作为当前 benchmark 的最优实现**，但暂时不能外推成
   “flag 原语天然比 Event 快”；两者会形成不同的 CUDA Graph 依赖和调度拓扑。
4. 随机 delay 范围增大时，fixed 路径通常更快，说明自然去相位已能明显缓解固定顺序的 owner 带宽
   争抢；dynamic 为规避冲突付出的 owner claim、串行 SWITCH/conditional 和动态完成顺序成本没有被
   节省的争抢时间覆盖。

critical-path 解释也需要修正：这不是“端到端差值等于纯 DMA 时间”。Graph 中 c1 与 resident c0
并发，next-forward c0 refill 从 c4 尾部开始并与 c5 计算重叠；当计算足够长时，并不存在必然额外增加
一个完整通信 cycle 的尾巴。M=12288、K=1 时，compute-only 每 cycle 约 23.8ms，fixed+flag 约
24.0ms，已经接近完全覆盖；dynamic 却把各 compute-cycle 拉到约 31ms。dynamic 的 tail join 只有
约 1.7–3.5ms，大部分约 41ms 的端到端回退已进入逐 cycle compute path。M=4096 下 dynamic tail
也仅约 0.3ms。因此当前数据衡量的是**通信、同步和动态 Graph 节点共同作用在 pipeline 上的净代价**，
不能解释为独立 memcpy latency；报告中的 comm-cycle 同样包含 WAR、claim/control 及等待计算。

工程决策：当前不继续以 dynamic DMA 作为默认候选。若后续还要挽救该方向，优先用 Nsight 拆分
`claim → set condition → SWITCH DMA → release/publish` 在 GEMM overlap 下的调度与等待，尤其确认
conditional 链、owner 独占及跨 GPU CAS 分别贡献多少；在此之前，不再用固定同相 case 的收益支撑
dynamic 生产价值。

## 2026-09-04 固定 DMA 分片 + 随机到达验证（Event-only阶段结论）

8 卡 smoke 已覆盖两周期 cache、next-forward cycle0 回填、13B 非整除尾部、S=1/2/4 完整 payload
校验，以及“同一 sample 四模式使用同一随机 offset”校验。实际 Gemma4 component bytes、六 cycle、
4096³ BF16 GEMM、每组 10 个 paired sample 的低扰动结果见
[随机到达 pipeline 报告](../../../../check_logs/sidp_dma_compute_pipeline_slice_random_gemma4_20260904.md)：

| K | delay 抽样范围 | S=1 p50(ms) | S=2 p50(ms) | S=4 p50(ms) | S=2/S=1 | S=4/S=1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | U[0,1ms] | 88.1313 | 85.4176 | 98.8742 | 0.9692 | 1.1219 |
| 1 | U[0,5ms] | 80.2464 | 83.7920 | 93.6490 | 1.0442 | 1.1670 |
| 1 | U[0,10ms] | 77.2398 | 81.4223 | 91.7913 | 1.0541 | 1.1884 |
| 4 | U[0,1ms] | 60.4363 | 65.0648 | 69.4483 | 1.0766 | 1.1491 |
| 4 | U[0,5ms] | 59.9280 | 66.1866 | 68.4647 | 1.1044 | 1.1424 |
| 4 | U[0,10ms] | 58.4948 | 65.9007 | 67.7841 | 1.1266 | 1.1588 |

纯通信中分片可以缩短一个 cycle 的总窗口，但 pipeline 还受“第一个可计算 layer 何时完整 ready”
约束。S=2/4 将同一个 owner 的后续 slice 推迟到其他 owner 的前置 slice 之后：某些 comm-cycle
更短，但计算 stream 的早期 RAW wait 反而增长。当前只有 K=1、U[0,1ms] 的 S=2 聚合 p50 改善
约 3.1%，其余配置均回退；因此不能把纯通信 Gate1 的收益直接等价为 pipeline 收益。该阶段尚未加入
fixed DMA + generation flag，关于“是否接入runtime”的判断已被下一节2026-09-05对照更新。10-sample
数据只用于方向判断，若要对边缘差异下结论仍应扩大采样。

## 2026-09-05 fixed DMA Event/flag × S=4 group 收敛结论

`--group-study`已扩为八组：整层S=1与S=4/G=1,2,4分别使用Event和generation flag。H20×8、实际
Gemma4 component bytes、六cycle、K=1/4、20个paired sample的完整控制台汇总见
[test_log.log](../../../../test_log.log)，报告为
`check_logs/sidp_dma_compute_pipeline_s4_group_event_flag_{M}.md`。

机制结论：Event Graph虽然没有`c1 prefetch → resident c0 compute`的DAG依赖，实测仍会先推进初始
memcpy/Event分支，使第一次prefetch没有与c0计算重叠。随后c+2 refill和c0计算几乎同时开始；
S=4/G=1在slice0中依次等待每个复用slot，只完成每层25%，约75%通信要等当前cycle大部分计算完成后
才能继续。整层S=1则能wait一个slot、完整copy并立即publish一个layer，保留逐层wavefront。G增大时
每组更早形成完整layer，因此Event路径逐渐接近S=1。

Flag消除开头调度空洞并保住一轮prefetch领先量后，slice呈现明确的适用区间。下表为K=1下
`S=4/G=1 flag`相对`S=1 flag`的max-rank graph p50；负数表示slice更快：

| GEMM M | U[0,1ms] | U[0,5ms] | U[0,10ms] | 结论 |
|---:|---:|---:|---:|---|
| 16384 | +0.22% | +0.10% | ≈0% | 计算主导，基本持平 |
| 8192 | +0.15% | +0.50% | +0.28% | 计算主导，基本持平 |
| 6144 | -11.59% | -2.62% | +1.55% | 当前配置的转折区间 |
| 4096 | -9.66% | -3.70% | -3.93% | 通信主导，slice有收益 |
| 2048 | -14.12% | -5.93% | -0.38% | 通信主导，收益随arrival skew减小 |
| 1024 | -9.63% | -2.69% | +1.38% | 通信远重于计算，低skew仍可获益 |

M=6144不是硬件无关阈值；它由payload、K、GEMM形状、GPU和rank arrival共同决定。K=4远端copy较少，
同一M下更早进入计算主导或平衡区，没有复现K=1同等稳定的slice收益。随机delay范围增大后fixed
owner通信自然去相位、incast减弱，slice通过细粒度交错获得的优势随之下降。

工程决策：

1. benchmark最优默认候选是fixed compute-order + DMA + generation flag；生产runtime尚未接线。
2. Slice进入runtime TODO，但默认S=1。首版只基于fixed DMA + flag支持S=2/4和可选G，不与dynamic/SM
   同时组合。
3. Slice部署Gate不是单独的通信时间，而是饱和请求下的系统output token/s：SiDP释放HBM后若能扩大
   KV cache和decode batch，则通信主导的S=2/4仍可能优于不开SiDP、KV受限的小batch baseline。
4. 计算主导时slice最多与S=1打平；arrival skew大或K较大时通常没有必要开启。它是场景调优项，
   不是通用默认优化。

## 2026-09-04 旧 fixed-stagger / compute-priority 探索（历史证据）

新增两条完全相同的dynamic DMA路径，仅改变candidate扫描起点：

- `dynamic_dma`：原有`probe_cursor`跨cycle/replay轮转；
- `dynamic_dma_compute_priority`：每次从最靠前的未完成layer开始，若owner繁忙则继续向后扫描。

H20×8、实际Gemma4 component bytes、4096³ BF16 GEMM、20组配对样本结果见
`check_logs/sidp_dma_compute_priority_20x_20260904.md`：

| K | 相邻rank offset | rotating p50(ms) | compute-priority p50(ms) | priority/rotating |
|---:|---:|---:|---:|---:|
| 1 | 0us | 86.9639 | 86.8230 | 0.9984 |
| 1 | 1000us | 86.7453 | 86.8981 | 1.0018 |
| 4 | 0us | 70.2796 | 70.2399 | 0.9994 |
| 4 | 1000us | 70.0124 | 69.9995 | 0.9998 |

四组差异均在±0.2%内，不能认为compute-priority改善端到端性能。K=4同相trace确认实际选择
顺序确有变化、总`priority_distance`下降，但claim collision和累计RAW wait没有同步下降；因此当前
dynamic关键路径并非由持久化`probe_cursor`单独主导。生产默认继续保留`rotating`，新顺序只作为
显式实验项。

### claim-first（先占owner、再等本地slot）消融

还验证了把本地WAR检查从owner CAS之前移到CAS成功之后：这样可以让远端CAS与本地计算尾部重叠，
但会在receive slot尚未可覆盖时提前占住owner。使用与上表相同的H20×8、20组配对样本，结果见
`check_logs/sidp_dma_claim_first_20x_20260904.md`：

| K | 相邻rank offset | rotating变化 | compute-priority变化 |
|---:|---:|---:|---:|
| 1 | 0us | -0.10% | +0.02% |
| 1 | 1000us | +0.45% | +0.40% |
| 4 | 0us | +0.35% | +0.55% |
| 4 | 1000us | +0.53% | +1.00% |

负值表示claim-first更快。它只在一项中出现约0.1%的微小改善，其余场景均回退；可重叠的CAS成本
不足以覆盖提前持有owner带来的排队副作用。因此实现已恢复为**ready-first**：只有
`comp_gen >= required_comp_gen`时才允许CAS owner。该约束既是安全策略，也是当前实测更优的策略。
- 8 卡、两 cycle、含 13B 尾部 component 的 K=1/4 + step/layer trace smoke 通过。
- 8 卡、六 cycle 小 payload smoke 通过；覆盖 c+2 slot 复用、next-c0 回填、最终 generation 和 owner FREE 检查。
- 实际 Gemma4 单层通信字节、K=1、4096³ BF16 GEMM、六 cycle 的 20 次低扰动配对样本：
  - 同相 launch：Event=99.61ms、fixed flag=99.96ms、dynamic=87.05ms；fixed-flag/Event=`1.0035`，dynamic/fixed-flag=`0.8709`；
  - 相邻 rank 错开 1ms：Event=76.60ms、fixed flag=74.84ms、dynamic=86.99ms；fixed-flag/Event=`0.9770`，dynamic/fixed-flag=`1.1624`。

这组旧样本复现了“同相时动态避开 incast 获益、错相后固定顺序获益”的相位反转。固定顺序下
Event→flag 在两个相位分别只改变 `+0.35%/-2.30%`，不足以解释 dynamic 的 `-12.91%/+16.24%`，
它证明了方案对相位敏感，但固定 stagger 不代表 serving 到达分布，**不再作为当前性能结论**。原始证据见
[20次 fixed Event/flag/dynamic 对照](../../../../check_logs/sidp_dma_compute_pipeline_k1_fixed_flag_20x_20260904.md)、
[两 cycle trace smoke](../../../../check_logs/sidp_dma_compute_pipeline_smoke_20260904.md) 和
[六 cycle smoke](../../../../check_logs/sidp_dma_compute_pipeline_6cycle_smoke_20260904.md)。
