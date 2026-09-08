# 单 cycle：fixed DMA Event/flag vs dynamic-owner DMA Graph

入口：[bench_dma_cycle.py](bench_dma_cycle.py)。用于定位方向 C 的通信成本，不启动模型服务。

## 测什么

每张 GPU 一个独立进程。每个 owner 分配自己的 source 和 owner-control，通过生产
`SidpCudaMemcpy` 的 raw CUDA IPC 导出；两个模式使用完全相同的源地址、目标 buffer、component 大小和 candidate 集合。

| 模式 | 实际调用 | cycle 内操作 |
|---|---|---|
| `compute_dma` | 生产 `SidpCudaMemcpy.async_copy`，由 torch capture 入图 | 按模型层顺序拉取，跳过本地 K 层，逐层 record ready Event |
| `compute_dma_flag` | 与 `compute_dma` 完全相同的固定顺序和 DMA | 每层 copy 后用 production `publish_generation_kernel` 发布 fill，不做 Event record |
| `dynamic_dma` | 生产 `SidpDmaGraphKernels::build<true>` / `append_cycle_to_capture` | reset done → claim owner → setter/SWITCH → 全部 component DMA → release owner → publish fill，再进入下一步 |

不是重新写一套 demo 仲裁 kernel，也不是 SM copy 或静态 peak-shifting。
三种模式都 `CUDAGraph(keep_graph=True)` + 显式 `instantiate()`，在同一个通信 stream 上 replay。
保留实际通信路径的控制成本，包括动态路径的 flag publish；不强求将它伪装成“只换 memcpy API”的消融。

`compute_dma` 与 `compute_dma_flag` 的 copy 地址、顺序和 payload 完全一致；纯通信场景中没有 consumer，
所以这里比较的是 ready Event record 与 fill publish kernel 本身，不包含 Event wait 或
`wait_generation_kernel`。`compute_dma_flag` 与 `dynamic_dma` 都 publish fill，但后者额外包含
claim、SWITCH、release 和运行时动态顺序。
每个 sample 的三种模式使用完全相同的 rank delay，并在六种执行排列间轮转，避免某个模式长期固定
处于第一、第二或第三个测试位置。

为了单独测通信：

- 没有 GEMM、attention 或其他模型计算，也没有 next-forward c0 refill。
- 所有 slot 开始时都可写，动态 `required_comp_gen=0`；不加入计算侧 RAW/WAR wait kernel 或 Event wait。
- 通信结束后才校验结果，不在通信计时窗口内消费目标 buffer。
- 每轮只 replay 一个 cycle，所有 rank 结束后才协调下一轮；不模拟跨 cycle 持续漂移。
- owner control 只在 setup 初始化一次。禁止按轮 reset owner；释放仍由持有者的生产 release kernel 完成。

## 数据量

默认采用当前 Gemma4（hidden=3840、intermediate=15360）的 BF16 FFN component 大小：

| component | bytes | MiB |
|---|---:|---:|
| gate_up | 235929600 | 225 |
| down | 117964800 | 112.5 |
| 一层合计 | 353894400 | 337.5 |

不加载 checkpoint，用不同 owner/component 的固定 byte pattern 校验完整 payload。
K=1 每 rank 拉取 7 层，2,477,260,800 bytes/cycle；K=4 拉取 4 层，1,415,577,600 bytes/cycle。
本地 K 层集合及跳过规则与生产 `remote_positions` 一致，**compute-order 不是 ring permutation**。

`--component-bytes` 可改变大小，也可追加 scale 等 extra 的大小；每层全部 component 复制完才释放 owner。
支持非 16B 对齐的字节尾部。该选项只是模拟编码数据布局，不实现压缩/解压缩。

## 直接执行

先确认目标 GPU 没有其他服务/benchmark 在跑，以免带宽被外部任务占用。本脚本不停止任何已有服务。
以下从 workspace 根目录执行；本地 `python` wrapper 会同步 `sglang`，远端容器内则直接用其 Python。

```bash
# 主要对照：K=1/4，每rank启动时间为 rank × {0,250,1000} us。
# 默认实际Gemma4大小；每配置30对采样、3对warmup；逐对交换A/B执行顺序。
python sglang/benchmark/kernels/sidp/bench_dma_cycle.py \
  --k-values 1,4 --stagger-us 0,250,1000 \
  --iterations 30 --warmup 3 \
  --output check_logs/sidp_dma_cycle_main.json

# 单个慢rank：rank7晚5ms，其余rank同时尝试launch。
python sglang/benchmark/kernels/sidp/bench_dma_cycle.py \
  --k-values 1,4 --rank-offsets-us 0,0,0,0,0,0,0,5000 \
  --iterations 30 --warmup 3 \
  --output check_logs/sidp_dma_cycle_straggler.json

# 在相邻rank固定偏移之外，再添加每rank/每轮随机[0,500]us抖动。
# 同一个sample的三种模式使用相同seed生成相同目标偏移。
python sglang/benchmark/kernels/sidp/bench_dma_cycle.py \
  --k-values 1,4 --stagger-us 0,250 --jitter-us 500 --seed 20260903 \
  --output check_logs/sidp_dma_cycle_jitter.json

# 纯随机到达主对照：没有固定rank顺序，每个rank/每个sample独立抽 U[0,10ms]。
# 三种模式复用完全相同的随机offset；K=1/4各采样30组。
python sglang/benchmark/kernels/sidp/bench_dma_cycle.py \
  --k-values 1,4 --stagger-us 0 \
  --jitter-us 10000 --seed 20260904 \
  --iterations 30 --warmup 3 \
  --output check_logs/sidp_dma_cycle_random_10ms.json

# 独立诊断运行：增加每一步的claim/transfer Event和selected trace。
# 不把这一组的绝对耗时混进默认低扰动结果。
python sglang/benchmark/kernels/sidp/bench_dma_cycle.py \
  --k-values 1,4 --stagger-us 0,1000 --iterations 10 --warmup 3 \
  --trace-steps --output check_logs/sidp_dma_cycle_trace.json

# 小buffer机制检查：包含第三个13B component，校验多component和字节尾部。
python sglang/benchmark/kernels/sidp/bench_dma_cycle.py \
  --k-values 1,4 --component-bytes 1048576,524288,13 \
  --stagger-us 0,500 --iterations 3 --warmup 1 --trace-steps \
  --output check_logs/sidp_dma_cycle_smoke.json
```

输出文件已存在时拒绝覆盖；每次使用新的文件名，或省略 `--output` 自动生成时间戳。
默认 8 卡，也可用 `--num-gpus`；必须 `1 <= K < num-gpus`，所有 peer 支持 P2P/native atomics。
动态路径沿用生产 CUDA 12.8+ SWITCH 门禁；实际验证环境以结果 JSON 的 CUDA/PyTorch 版本为准。

## 如何读结果

输出三种文件：

- `.json`：环境、命令、生产源码 hash、各配置统计、完整 native peer atomic 能力矩阵。
- `.md`：三种模式的 cycle p50/p95、有效带宽、实际 host 启动跨度、all-rank host wall 时间。
- `.samples.jsonl`：每次采样的每 rank 明细；打开 trace 后包含逐 step 的 owner、slot、claim spins/collisions 和时间窗口。

主要比较 **同 K、同启动偏移下 `max_rank_cycle_ms` 的分布**。这是各 rank 从 cycle 第一个计时节点
到最后一个计时节点的耗时，再取 rank 最大值；包含 DMA 路径的控制开销，不含 host sleep。
报告同时给出 `flag/Event`、`dynamic/flag` 和 `dynamic/Event` 三个 ratio；小于 1 表示分子更快。
同时看逐rank分布，避免均值掩盖某个rank的尾延迟。

启动偏移在 host 层实现：所有 rank 先拿到未来的 `monotonic_ns` deadline，分别等待自己的 offset，
再提交 Graph。`--lead-ms` 默认20ms；实际 launch lateness 和 span 会被记录。
这不是设备 barrier，目标零偏移也不保证 GPU 同时开始执行。
**不同 GPU 的 Event 时间不能直接相减**；脚本没有假设跨 GPU 时钟同步。

`all_rank_observed_wall_ms = 最晚host观测到完成 − 最早host开始提交`，包含人为启动差、host提交和完成观测误差。
据此计算的聚合 GB/s 是 host-observed 指标，不冒充精确 NVLink 峰值；尤其有 skew 时，
不能用 `总bytes / max(各rank局部耗时)` 代替 all-rank makespan 带宽。
每 rank 的有效 GB/s 则按本 rank 的 cycle payload / CUDA cycle 时间计算。

### step trace 的限制

- 动态 claim 时间：parent Event before claim → after claim；包含竞争/扫描与调度时间。
- 动态 transfer window：after claim → SWITCH 完成；包含 trace kernel、setter、DMA、release/publish。
- compute-order transfer window：逐层 DMA 与 ready Event 的窗口。
- `claim_spins=0` 仍可能有碰撞：单次扫描跳过忙owner后成功不算完整spin。
- 这些窗口只能解释生产通信流水线；**不能把动态 transfer window 直接当成纯 DMA 时间**。
- 如需纯 Memcpy 条形、kernel调度空档和跨GPU执行起点，应在此小benchmark上进一步抓GPU timeline。

若无计算时动态路径已明显变慢，优先查 conditional/仲裁/节点间空档；若无计算时更快，
则返回模型检查计算并行时的资源争用、slot依赖与Graph尾部。这里不单凭结果宣布任何一种原因成立。

## CPU 回归

```bash
python sglang/test/registered/unit/layers/sidp/test_dma_cycle_benchmark.py -v
```

覆盖默认字节数、K/candidate映射、参数门禁、paired jitter、带skew的makespan计算、分位数和报告生成。

## 2026-09-03 验证记录

8项CPU检查、8卡含13B尾部的小buffer检查、实际Gemma4字节量的K=1/4与三种启动相位均通过。
另有独立逐step trace。结果与解释见
[首轮实测](../../../../check_logs/sidp_dma_cycle_findings_20260903.md)。
目标同相启动时动态DMA更快；K=1相邻rank错开1ms时固定顺序DMA更快。
这些是纯通信结果，不代替serving端到端结论。
