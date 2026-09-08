# SiDP pipeline Nsight 分析方法

这套工具只针对 `bench_dma_compute_pipeline.py`。它把原始 `.nsys-rep/.ncu-rep` 留在远端
`artifact_dir`，把小型 `analysis.json`、`report.md`、`manifest.json` 回传到 `check_logs`。
分析过程不依赖 Nsight GUI。

## 1. 一条命令运行

从 `tt_test` 根目录运行：

```bash
python sglang/benchmark/kernels/sidp/nsight_pipeline/profile_pipeline.py \
  sglang/benchmark/kernels/sidp/nsight_pipeline/configs/pipeline_example.json
```

要专门验证SM-copy与GEMM的SM侧争用，可使用包含`compute_only`和`compute_sm_flag`的最小完整配置：

```bash
python sglang/benchmark/kernels/sidp/nsight_pipeline/profile_pipeline.py \
  sglang/benchmark/kernels/sidp/nsight_pipeline/configs/pipeline_sm_trace_smoke.json \
  --force
```

本地 `python` 是远程 wrapper；它会同步 `sglang`，在容器执行并只回传 `check_logs`。重复使用同一目录时，
显式加 `--force`；只做低扰动时间线分析时加 `--no-ncu`。推荐为正式实验修改配置里的
`output_dir/artifact_dir`，每次使用新目录，保留可追溯证据。

## 2. 为什么 Systems 和 Compute 必须分开

- Nsight Systems 保留多 rank、多 stream、CUDA Graph 和 DMA 的真实并行关系，用来回答端到端时间、
  overlap、bubble、等待和跨 rank skew。
- NCU node profiling会单独replay被选中的kernel：适合看GEMM自身的Tensor Core、LSU、cache、stall等
  指标，但不保留它与通信的runtime并行关系。
- NCU whole-Graph profiling将整张CUDA Graph作为一个workload做kernel replay，保留图内stream和节点
  并行；它能观测compute+communication range的总体计数，但指标包含GEMM、wait/control、SM-copy及
  DMA图节点，不能直接归属于GEMM。
- 工具先用Systems选择热点，再分别执行node profile和whole-Graph profile。默认关闭NCU，确认
  Systems case合理后再打开；可用`collection.ncu.node_enabled/workload_enabled`分别控制两类采集。

Systems case还会自动开启profiler-only SM execution trace：被插桩的通信/控制kernel每个CTA记录入口/
出口`%smid + %globaltimer`，随后在rank内聚合。DMA memcpy由Copy Engine执行，没有SMID；SM-copy、
wait、claim、release、publish等kernel有实际SMID。这个trace不插桩黑盒GEMM，因此只能证明通信用了
哪些SM，不能直接给出“同一时刻GEMM也在同一SM”的严格映射。

## 3. 指标阅读顺序

### 3.1 先确认采样可比

先看 `resolved_config.json` 和 `manifest.json`：

1. baseline/variant 的 K、M/N/K、repeat、cycle、component bytes 和随机 delay 是否只有预期差异；
2. GPU、driver、commit、benchmark SHA256 和 Nsight 版本是否一致；
3. 每个命令是否返回 0，报告 `limitations` 是否出现 schema、Graph trace 或 NCU filter 降级；
4. 每次只采一个 case 的一个 measured sample。性能判断要用多次独立 run 看分布，不要把一次 profiler
   样本当成稳定吞吐结果。

### 3.2 再判断关键路径属于哪一类

`report.md` 的核心时间线先回答四个互斥程度不同的问题：

- `compute_union_ms`：该 GPU 上计算 interval 并集；不是所有 GEMM duration 的简单相加。
- `communication_union_ms`：DMA 和 SM copy interval 并集。
- `overlap_ms`：上述两个并集的交集。
- `exposed_communication_ms = communication_union - overlap`：通信存在但没有被计算覆盖的时间。

两个方向的 overlap 比率含义不同：

- `compute_overlap_ratio` 高：大量计算运行时通信也在进行，此时要警惕通信让 GEMM 自身变慢。
- `comm_hidden_ratio` 高：大部分通信被计算覆盖；这不代表零成本，因为 HBM/L2/SM 争用仍可能拉长计算。

`gpu_active_ratio` 高但 `useful_compute_or_comm_ratio` 低，且 `wait_or_claim_union_ms` 高，表示 GPU 并不空闲，
而是控制/自旋 kernel 占住了时间或 SM。`idle_bubble` 才是没有任何相关 CUDA activity 的真空洞。

### 3.3 用 benchmark Event 给 bubble 命名

Systems 的 interval 是事实时间线；benchmark 自身的 Event 则提供 cycle 语义：

- `pre_cycle0_gap` 大：Graph 开始后 c0 计算被无效依赖或调度顺序推迟；历史 fixed Event 路径的
  “白等一轮”会出现在这里。
- `RAW wait` 大：计算已经追上通信，对应 layer 尚未 ready。
- `inter_cycle_gap` 大：cycle 间出现依赖等待或 Graph scheduler 空洞。
- `tail_join` 大：最后计算结束后仍在等待 next-forward c0 refill 或其他 comm 工作。
- `comm_cycle` 变长但 memcpy service time/GB/s不变：优先检查 WAR 等待或 copy 节点之间的控制空洞；
  memcpy 本身变长或带宽下降才指向 P2P/CE/fabric 争抢。

Event span 可能包含 wait；不要把 `comm_cycle_ms` 直接解释成纯 DMA service time。Systems 的 memcpy
聚合才是实际 Copy Engine 活动，二者之差通常是依赖或调度等待。

### 3.4 区分四种常见退化

1. **通信暴露**：variant 的 `exposed_communication_ms` 和 RAW/tail 同时上升，compute kernel mean
   基本不变。优化 copy 顺序、slice、K、通信带宽或提高可覆盖计算量。
2. **GPU 真 bubble**：`idle_bubble` 上升，最大 bubble 的 before/after 指向依赖边界。检查 Event/flag、
   Graph fork/join 和 cycle 发起时机。
3. **wait/claim 占用**：idle 不高但 `wait_or_claim_union_ms` 高。自旋不是有效 overlap；减少控制节点、
   调整 backoff，或回到 fixed plan。
4. **算子自身变慢**：matched operator 的 mean ratio 上升，`count × Δmean` 可以解释大部分 window 差，
   同时 compute overlap 高。此时才进入 NCU，检查 SM/HBM/L2/occupancy/stall。

### 3.5 NCU 与 SMID trace 如何判断 GEMM 变慢原因

NCU 对同一 operator key（kernel name + grid + block + dynamic shared memory）比较：

- `gpu__time_duration`：先确认 NCU 中也能复现 duration 增长；不能复现时，Systems 中的并行资源争用
  可能在 NCU replay 后消失，结论应停在 Systems。
- `sm__throughput`、Tensor pipe/TC 指标、IPC/issue：判断计算管线利用率与发射效率。
- `dram__throughput`、`lts__throughput`、bytes：判断 HBM/L2 压力是否随重叠通信上升。
- achieved/theoretical occupancy、registers/thread、shared memory/block、waves/SM：判断 launch 资源限制；
  同一个 binary 的寄存器数通常不因通信改变，occupancy 下降更可能来自测量/匹配错误，应先核对身份。
- warp stall：只比较同单位同名字的指标；某一 stall 上升要同时结合 duration 和对应吞吐，不能单独当根因。
- roofline/arithmetic intensity：判断算子原本更接近 compute roof 还是 memory roof。通信重叠使 kernel
  沿 roofline 向下移动，且 DRAM/L2 指标升高，才支持 memory contention；SM copy/control kernel 与 GEMM
  重叠、SM issue下降则更支持 SM scheduler/residency contention。

工具输出的是带证据的 `hypotheses`，不会仅凭 duration 自动断言根因。若 NCU 报 counter permission、
Graph/filter 不支持或没有匹配 launch，报告会保留 Systems 结果并把 NCU 标为 limitation。

推荐按三层证据交叉判断：

1. Systems中同一GEMM的mean/p95是否只在pipeline里变长，以及变长期间分别与DMA、SM-copy、wait/
   claim、control重叠多少；
2. node replay中该GEMM是否恢复到compute-only水平。若runtime变慢而node replay基本持平，支持
   concurrency-specific contention，而不是kernel binary/grid发生变化；
3. whole-Graph workload的SM/Tensor/LSU/L2/DRAM指标如何变化，通信kernel实际覆盖多少SMID。SM-copy
   与GEMM重叠、SM覆盖非零、runtime GEMM变慢共同支持SM资源争用；L2/DRAM压力上升支持共享存储层级
   压力。但whole-Graph指标包含通信自身流量，因此结论只标为`plausible`，不能把增量全部归因给GEMM。

报告中的`min/avg/max`是PerfWorks跨hardware-unit instance的rollup离散度，并非逐物理SM原始数组；
精确SMID只来自我们插桩的通信CTA。

## 4. JSON 事实源

`analysis.json` 包含：

- 每 device 的 window、compute/comm/wait/control 并集、overlap、bubble 和相邻节点；
- memcpy count/bytes/service time/effective GB/s；
- kernel identity 与 count/total/mean/median/p95/min/max/std/CV；
- 跨 GPU launch skew、duration min/max/CV 和最慢 device；
- benchmark 的 cycle、layer RAW/GEMM window、tail；
- baseline/variant 的 operator delta、热点选择和 NCU 指标差。
- 通信/控制CTA的实际SMID覆盖、entry→exit SMID变化，以及compute与各通信role的Systems重叠比例；
- NCU node-level isolated compute、whole-Graph runtime workload及跨证据`contention_analyses`。

自动 agent 应引用 `analysis.json` 的具体字段，并用 `manifest.json` 的命令复现。不要解析 Markdown 数字作为
二次分析输入。

## 5. 扰动与边界

- CUDA Graph node-level trace 和 NVTX projection 有观测开销；比较必须让所有 case 使用相同采集选项。
- Nsight 时间线中的绝对跨 GPU 时间只有同一 `.nsys-rep` 内可比；本工具每个 case 单独采集，所以 case
  间只比较 duration，不比较绝对 timestamp。
- `aggregate_payload_gbps_over_global_window` 是所有 GPU payload 除以统一 wall window，不是某条 NVLink
  的物理线速；单 copy `effective_gbps` 是 bytes/service time。
- kernel 参数只有 Nsight 可见的 grid/block/shared-memory；看不到的指针、shape或模板实参不能猜。
- 原始 rep 可能很大并保留在 `/tmp`。需要长期保留时在远端另行归档；不要放入 `check_logs` 触发回传。
