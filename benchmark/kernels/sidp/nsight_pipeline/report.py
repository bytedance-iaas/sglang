"""Render the stable analysis JSON into a concise evidence report."""

from __future__ import annotations

import statistics
from pathlib import Path


def _median(devices, key):
    values = [device[key] for device in devices]
    return statistics.median(values) if values else 0.0


def _short(name, width=90):
    name = str(name).replace("|", "\\|")
    return name if len(name) <= width else name[: width - 1] + "…"


def _neighbor(value):
    if not value:
        return "none"
    return f"{value.get('role', 'unknown')}:{value.get('name', '')}"


def _ratio(value):
    return f"{100 * value:.1f}%"


def _case_overview(case):
    nsys = case["nsys"]
    devices = nsys["devices"]
    slowest_id = nsys["cross_rank"]["slowest_device"]
    slowest = next(
        (item for item in devices if item["device"] == slowest_id),
        devices[0] if devices else {},
    )
    return {
        "slowest_window_ms": slowest.get("window_ms", 0.0),
        "compute_ms": _median(devices, "compute_union_ms"),
        "comm_ms": _median(devices, "communication_union_ms"),
        "overlap_ms": _median(devices, "compute_communication_overlap_ms"),
        "compute_overlap": _median(devices, "compute_overlap_ratio"),
        "comm_hidden": _median(devices, "comm_hidden_ratio"),
        "exposed_comm_ms": _median(devices, "exposed_communication_ms"),
        "idle_ms": _median_nested(devices, "idle_bubbles", "total_ms"),
        "wait_ms": _median(devices, "wait_or_claim_union_ms"),
        "active": _median(devices, "gpu_active_ratio"),
    }


def _median_nested(rows, outer, inner):
    values = [row.get(outer, {}).get(inner, 0.0) for row in rows]
    return statistics.median(values) if values else 0.0


def _find_metric(metrics, token):
    choices = [
        (name, value)
        for name, value in metrics.items()
        if token.lower() in name.lower()
    ]
    return min(choices, key=lambda item: len(item[0])) if choices else None


def _find_metric_delta(rows, token):
    choices = [row for row in rows if token.lower() in row["metric"].lower()]
    return min(choices, key=lambda item: len(item["metric"])) if choices else None


def render_report(analysis, output_path):
    output_path = Path(output_path)
    lines = [
        "# SiDP pipeline 无 GUI Nsight 分析",
        "",
        f"- schema：`{analysis['schema_version']}`；状态：`{analysis['status']}`。",
        f"- Systems：`{analysis['manifest']['tools']['nsys']['version']}`；Compute：`{analysis['manifest']['tools']['ncu']['version']}`。",
        f"- 原始大型 profiler 文件仅保留在远端：`{analysis['artifact_dir']}`。",
        "- Systems 时间来自真实并行执行；NCU node replay 用于隔离单 kernel，NCU graph workload 保留图内并行但指标属于整图。两种 NCU 时间都不替代 Systems 端到端时间。",
        "",
        "## 核心时间线",
        "",
        "| case | mode/K/M/S/G | 最慢GPU窗口(ms) | compute并集(ms) | DMA/SM-copy并集(ms) | overlap(ms) | compute被通信重叠 | 通信被计算隐藏 | 暴露通信(ms) | idle bubble(ms) | wait/claim(ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    overviews = {}
    for name, case in analysis["cases"].items():
        overview = _case_overview(case)
        overviews[name] = overview
        settings = case["settings"]
        lines.append(
            f"| `{name}` | `{settings['mode']}`/{settings['k']}/{settings['gemm_m']}/{settings['dma_slices']}/{settings['dma_slice_groups']} | "
            f"{overview['slowest_window_ms']:.4f} | {overview['compute_ms']:.4f} | "
            f"{overview['comm_ms']:.4f} | {overview['overlap_ms']:.4f} | "
            f"{_ratio(overview['compute_overlap'])} | {_ratio(overview['comm_hidden'])} | "
            f"{overview['exposed_comm_ms']:.4f} | {overview['idle_ms']:.4f} | "
            f"{overview['wait_ms']:.4f} |"
        )
    lines.extend(
        [
            "",
            "> 区间先按每个 GPU 做并集再求交，避免并发 kernel 重复计时。idle bubble 是完全没有相关 GPU activity；wait/claim kernel 占用 GPU 时单列，不能算有效计算，也不能算 idle。",
            "",
            "## Benchmark 语义边界",
            "",
            "CUDA Event 指标用于 cycle/RAW/tail 归因，Systems interval 用于真实 kernel、DMA 和 overlap；两者互相校验，但不混用时间域。",
            "",
            "| case | graph max-rank(ms) | cycle comm mean/p95(ms) | layer GEMM mean/p95(ms) | RAW wait total/p95(ms) | pre-c0 max(ms) | tail join max(ms) | launch skew(ms) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, case in analysis["cases"].items():
        nsys = case["nsys"]
        semantic = nsys.get("benchmark_semantics") or {}
        trial = semantic.get("trial_metrics") or {}
        comm = semantic.get("comm_cycle_ms") or {}
        gemm = semantic.get("gemm_window_ms") or {}
        raw = semantic.get("raw_wait_ms") or {}
        lines.append(
            f"| `{name}` | {trial.get('max_rank_graph_ms', 0):.4f} | "
            f"{comm.get('mean_ms', 0):.4f}/{comm.get('p95_ms', 0):.4f} | "
            f"{gemm.get('mean_ms', 0):.4f}/{gemm.get('p95_ms', 0):.4f} | "
            f"{raw.get('total_ms', 0):.4f}/{raw.get('p95_ms', 0):.4f} | "
            f"{semantic.get('max_rank_pre_cycle0_gap_ms', 0):.4f} | "
            f"{semantic.get('max_rank_tail_join_ms', 0):.4f} | "
            f"{nsys['cross_rank']['launch_skew_ms']:.4f} |"
        )

    for name, case in analysis["cases"].items():
        nsys = case["nsys"]
        lines.extend(
            [
                "",
                f"## `{name}` 细节",
                "",
                f"- CUDA kernels/memcpy：{nsys['kernel_count']}/{nsys['memcpy_count']}；最慢 device：{nsys['cross_rank']['slowest_device']}；device window CV：{nsys['cross_rank']['window_duration_cv']:.4f}。",
                f"- 总 DMA payload：{nsys['dma']['total_gib']:.3f} GiB；按全设备统一窗口折算 aggregate payload：{nsys['dma']['aggregate_payload_gbps_over_global_window']:.2f} GB/s。此值是总 payload rate，不是单链路带宽。",
            ]
        )
        if nsys["dma"]["groups"]:
            lines.extend(
                [
                    "",
                    "| device/copy kind | count | bytes(GiB) | DMA service sum(ms) | mean/p95 copy(ms) | effective GB/s |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for dma in nsys["dma"]["groups"]:
                lines.append(
                    f"| {dma['device']}/{dma['copy_kind']} | {dma['count']} | {dma['gib']:.3f} | "
                    f"{dma['service_time_ms']:.4f} | {dma['mean_ms']:.5f}/{dma['p95_ms']:.5f} | "
                    f"{dma['effective_gbps']:.2f} |"
                )
        devices = nsys.get("devices", [])
        overlap_roles = (
            "communication_dma",
            "communication_sm",
            "wait_or_claim",
            "control",
        )
        role_rows = []
        for role in overlap_roles:
            overlap_ms = [
                device.get("compute_overlap_by_role_ms", {}).get(role, 0.0)
                for device in devices
            ]
            overlap_ratio = [
                device.get("compute_overlap_by_role_ratio", {}).get(role, 0.0)
                for device in devices
            ]
            if any(overlap_ms) or any(overlap_ratio):
                role_rows.append(
                    (
                        role,
                        statistics.median(overlap_ms),
                        statistics.median(overlap_ratio),
                    )
                )
        if role_rows:
            lines.extend(
                [
                    "",
                    "Systems真实时间线中，各通信角色与compute的重叠（各GPU中位数）：",
                    "",
                    "| role | overlap(ms) | compute overlap ratio |",
                    "|---|---:|---:|",
                ]
            )
            for role, overlap_ms, overlap_ratio in role_rows:
                lines.append(
                    f"| `{role}` | {overlap_ms:.4f} | {_ratio(overlap_ratio)} |"
                )
            lines.append(
                "\n- SM-side通信/control区间先取并集后的compute overlap："
                f"{_median(devices, 'compute_sm_side_overlap_ms'):.4f} ms / "
                f"{_ratio(_median(devices, 'compute_sm_side_overlap_ratio'))}。"
            )
        semantic = nsys.get("benchmark_semantics") or {}
        sm_trace = semantic.get("sm_execution_trace") or {}
        if sm_trace.get("enabled"):
            lines.extend(
                [
                    "",
                    "通信/控制 kernel 的 CTA SMID trace（跨 rank 聚合）：",
                    "",
                    "| role | ranks | kernel launches | CTA records | median/max unique SM | median/max device SM coverage | SMID entry→exit changes |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for role in sm_trace.get("roles", []):
                lines.append(
                    f"| `{role['role']}` | {role['rank_count']} | "
                    f"{role['kernel_launches']} | {role['cta_records']} | "
                    f"{role['unique_sm_count_median']:.1f}/{role['unique_sm_count_max']} | "
                    f"{_ratio(role['device_sm_coverage_median'])}/{_ratio(role['device_sm_coverage_max'])} | "
                    f"{role['smid_migrations']} |"
                )
            lines.append(
                "\n> SMID 对被插桩的通信/控制 CTA 是精确值；它不提供黑盒 GEMM 的 SMID。entry/exit 不同用于标记可能的抢占迁移。"
            )
        if semantic.get("cycles"):
            lines.extend(
                [
                    "",
                    "逐 cycle Event 归因（跨 rank 聚合）：",
                    "",
                    "| cycle/comm-op | compute mean/max(ms) | comm mean/max(ms) | RAW total/p95(ms) | WAR/claim total/p95(ms) | transfer mean/p95(ms) |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for cycle in semantic["cycles"]:
                lines.append(
                    f"| {cycle['cycle']}/{cycle['communication_operation']} | "
                    f"{cycle['compute_ms']['mean_ms']:.4f}/{cycle['compute_ms']['max_ms']:.4f} | "
                    f"{cycle['comm_ms']['mean_ms']:.4f}/{cycle['comm_ms']['max_ms']:.4f} | "
                    f"{cycle['raw_wait_ms']['total_ms']:.4f}/{cycle['raw_wait_ms']['p95_ms']:.4f} | "
                    f"{cycle['war_or_claim_wait_ms']['total_ms']:.4f}/{cycle['war_or_claim_wait_ms']['p95_ms']:.4f} | "
                    f"{cycle['transfer_window_ms']['mean_ms']:.4f}/{cycle['transfer_window_ms']['p95_ms']:.4f} |"
                )
        lines.extend(
            [
                "",
                "最热 GPU kernel（相同名字但 grid/block/shared-memory 不同会拆开）：",
                "",
                "| role | kernel | grid/block | count | total(ms) | mean/p95(ms) | CV |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for operator in nsys["operators"][:12]:
            lines.append(
                f"| {operator['role']} | `{_short(operator['name'])}` | "
                f"`{operator['grid']}/{operator['block']}` | {operator['count']} | "
                f"{operator['total_ms']:.4f} | {operator['mean_ms']:.5f}/{operator['p95_ms']:.5f} | {operator['cv']:.3f} |"
            )
        largest = []
        for device in nsys["devices"]:
            for bubble in device["idle_bubbles"]["largest"][:2]:
                largest.append((bubble["duration_ms"], device["device"], bubble))
        if largest:
            lines.extend(["", "最大 idle bubble（含相邻节点）：", ""])
            for duration, device, bubble in sorted(
                largest, key=lambda item: item[0], reverse=True
            )[:5]:
                lines.append(
                    f"- GPU {device}: {duration:.5f} ms；before=`{_short(_neighbor(bubble['before']))}`；after=`{_short(_neighbor(bubble['after']))}`。"
                )

    if analysis.get("comparisons"):
        lines.extend(["", "## Case 对比与差值归因", ""])
    for comparison in analysis.get("comparisons", []):
        baseline_name = comparison["baseline"]
        variant_name = comparison["variant"]
        baseline = overviews[baseline_name]
        variant = overviews[variant_name]
        lines.extend(
            [
                f"### `{variant_name}` vs `{baseline_name}`",
                "",
                f"- 最慢 GPU window 差：{variant['slowest_window_ms'] - baseline['slowest_window_ms']:+.4f} ms。",
                f"- compute/暴露通信/idle/wait 差：{variant['compute_ms'] - baseline['compute_ms']:+.4f} / {variant['exposed_comm_ms'] - baseline['exposed_comm_ms']:+.4f} / {variant['idle_ms'] - baseline['idle_ms']:+.4f} / {variant['wait_ms'] - baseline['wait_ms']:+.4f} ms（各 GPU 中位数）。",
                "- 上述分量可能重叠，不能相加当作严格的 critical-path 分解；operator 的 `count × Δmean` 才用于定位 kernel 自身拉长的贡献。",
                "",
                "| operator | role/status | baseline→variant mean(ms) | ratio | count×Δmean(ms) |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in comparison["operator_deltas"][:12]:
            if row["status"] == "matched":
                lines.append(
                    f"| `{_short(row['name'])}` | {row['role']}/{row['status']} | "
                    f"{row['baseline_mean_ms']:.6f}→{row['variant_mean_ms']:.6f} | "
                    f"{row['mean_ratio']:.3f} | {row['count_times_delta_mean_ms']:+.4f} |"
                )
            else:
                lines.append(
                    f"| `{_short(row['name'])}` | {row['role']}/new | -→{row['variant_mean_ms']:.6f} | - | +{row['variant_total_ms']:.4f} |"
                )
        lines.extend(
            [
                "",
                "Systems 选出的 NCU compute 热点："
                + (
                    ", ".join(f"`{_short(item['name'], 55)}`" for item in comparison["hotspots"])
                    if comparison["hotspots"]
                    else "无"
                )
                + "。",
            ]
        )

    ncu_runs = analysis.get("ncu", {}).get(
        "node_runs", analysis.get("ncu", {}).get("runs", [])
    )
    lines.extend(["", "## Nsight Compute：隔离 kernel 与实际 Graph workload", ""])
    if not ncu_runs:
        lines.append(
            "本次未采集 NCU node profile（或没有可用 compute 热点）。可在配置中启用 `collection.ncu.enabled/node_enabled` 后复跑；Systems 结论仍有效。"
        )
    else:
        for run in ncu_runs:
            lines.extend(
                [
                    f"### `{run['variant']}` vs `{run['baseline']}` — `{_short(run['kernel_name'])}`",
                    "",
                    "该表是 node replay：目标 GEMM 被隔离采集，不保留原 pipeline 的并行资源争用。",
                    "",
                ]
            )
            if run.get("limitations"):
                lines.append("- 限制：" + "; ".join(run["limitations"]))
            lines.append(
                f"- NCU匹配 launch 数 baseline/variant：{run.get('baseline_launch_count', 0)}/{run.get('variant_launch_count', 0)}；Systems mean ratio：{run.get('systems_mean_ratio')}。"
            )
            for observation in run.get("hypotheses", []):
                text = observation["evidence"]
                if observation.get("interpretation"):
                    text += "；" + observation["interpretation"]
                lines.append(f"- `{observation['kind']}`：{text}")
            base_derived = run.get("baseline_derived") or {}
            derived = run.get("variant_derived") or {}
            if derived and base_derived:
                lines.append(
                    "- GEMM派生：algorithmic FLOPs="
                    f"{derived.get('algorithmic_flops', 0):.4g}，achieved="
                    f"{base_derived.get('achieved_tflops') if base_derived.get('achieved_tflops') is not None else '-'}→"
                    f"{derived.get('achieved_tflops') if derived.get('achieved_tflops') is not None else '-'} TFLOP/s，"
                    "DRAM baseline→variant="
                    f"{base_derived.get('ncu_dram_bandwidth_gbps') if base_derived.get('ncu_dram_bandwidth_gbps') is not None else '-'}→"
                    f"{derived.get('ncu_dram_bandwidth_gbps') if derived.get('ncu_dram_bandwidth_gbps') is not None else '-'} GB/s，"
                    "FLOP/NCU-DRAM-byte="
                    f"{base_derived.get('algorithmic_flops_per_ncu_dram_byte') if base_derived.get('algorithmic_flops_per_ncu_dram_byte') is not None else '-'}→"
                    f"{derived.get('algorithmic_flops_per_ncu_dram_byte') if derived.get('algorithmic_flops_per_ncu_dram_byte') is not None else '-'}。"
                )
            metric_deltas = run.get("metric_deltas", [])
            selected = []
            for token in (
                "gpu__time_duration",
                "sm__throughput",
                "dram__throughput",
                "lts__throughput",
                "achieved_occupancy",
                "warps_active.avg.pct",
                "registers_per_thread",
                "waves_per_multiprocessor",
                "roofline",
            ):
                item = _find_metric_delta(metric_deltas, token)
                if item and item["metric"] not in {
                    row["metric"] for row in selected
                }:
                    selected.append(item)
            if selected:
                lines.extend(
                    [
                        "",
                        "| isolated GEMM核心指标 | baseline→variant | ratio | unit |",
                        "|---|---:|---:|---|",
                    ]
                )
                for row in selected:
                    ratio = row.get("ratio")
                    lines.append(
                        f"| `{row['metric']}` | {row['baseline']:.5g}→{row['variant']:.5g} | "
                        f"{ratio:.3f} | {row['unit']} |"
                        if ratio is not None
                        else f"| `{row['metric']}` | {row['baseline']:.5g}→{row['variant']:.5g} | - | {row['unit']} |"
                    )

    workload_runs = analysis.get("ncu", {}).get("workload_runs", [])
    if workload_runs:
        lines.extend(
            [
                "",
                "### Whole-Graph workload counters",
                "",
                "> 这里把整张 CUDA Graph 作为一个 workload replay，保留图内 stream/节点并行；指标包含 GEMM、wait/control、SM-copy 与 DMA 图节点，不能直接归属于某个 GEMM。",
            ]
        )
        for run in workload_runs:
            lines.extend(
                [
                    "",
                    f"#### `{run['variant']}` vs `{run['baseline']}`",
                    "",
                    f"- Graph workload launch 数 baseline/variant：{run['baseline_launch_count']}/{run['variant_launch_count']}。",
                ]
            )
            if run.get("limitations"):
                lines.append("- 限制：" + "; ".join(run["limitations"]))
            interesting = []
            for row in run.get("metric_deltas", []):
                lower = row["metric"].lower()
                if any(
                    token in lower
                    for token in (
                        "gpu__time_duration",
                        "sm__throughput",
                        "dram__throughput",
                        "lts__throughput",
                        "pipe_tensor",
                        "pipe_lsu",
                        "warps_active",
                        "stall",
                    )
                ):
                    interesting.append(row)
            if interesting:
                lines.extend(
                    [
                        "",
                        "| Whole-Graph metric | baseline→variant | ratio | unit |",
                        "|---|---:|---:|---|",
                    ]
                )
                for row in interesting[:16]:
                    ratio = row["ratio"]
                    lines.append(
                        f"| `{row['metric']}` | {row['baseline']:.5g}→{row['variant']:.5g} | "
                        f"{ratio:.3f} | {row['unit']} |"
                        if ratio is not None
                        else f"| `{row['metric']}` | {row['baseline']:.5g}→{row['variant']:.5g} | - | {row['unit']} |"
                    )
            rollups = run.get("variant_unit_rollups", [])
            if rollups:
                lines.extend(
                    [
                        "",
                        "硬件 unit rollup 离散度（不是逐物理 SM 原始数组）：",
                        "",
                        "| metric family | min/avg/max | range/avg | unit |",
                        "|---|---:|---:|---|",
                    ]
                )
                for item in rollups[:10]:
                    spread = item.get("range_over_avg")
                    lines.append(
                        f"| `{item['metric_family']}` | {item['min']:.5g}/{item['avg']:.5g}/{item['max']:.5g} | "
                        f"{spread:.3f} | {item['unit']} |"
                        if spread is not None
                        else f"| `{item['metric_family']}` | {item['min']:.5g}/{item['avg']:.5g}/{item['max']:.5g} | - | {item['unit']} |"
                    )

    contention = analysis.get("ncu", {}).get("contention_analyses", [])
    if contention:
        lines.extend(["", "## 跨证据推断：通信与计算重叠", ""])
        for item in contention:
            lines.extend(
                [
                    f"### `{item['variant']}` vs `{item['baseline']}`",
                    "",
                    f"热点：`{_short(item['kernel_name'])}`。",
                ]
            )
            for finding in item.get("findings", []):
                text = finding["evidence"]
                if finding.get("interpretation"):
                    text += "；" + finding["interpretation"]
                lines.append(
                    f"- `{finding['confidence']}/{finding['kind']}`：{text}"
                )

    limitations = list(analysis.get("limitations", []))
    for case in analysis["cases"].values():
        limitations.extend(case["nsys"].get("limitations", []))
    lines.extend(["", "## 边界与复现", ""])
    if limitations:
        lines.extend(f"- {item}" for item in dict.fromkeys(limitations))
    else:
        lines.append("- 本次所需 CUDA kernel/memcpy schema 均存在，未触发解析降级。")
    lines.extend(
        [
            f"- manifest：`{Path(analysis['output_dir']) / 'manifest.json'}`",
            f"- 机器可读事实源：`{Path(analysis['output_dir']) / 'analysis.json'}`",
            f"- 使用说明：`{Path(analysis['output_dir']) / 'how_to_analyse.md'}`",
            "- 所有实际执行命令都记录在 manifest；报告不要求打开 `.nsys-rep/.ncu-rep` GUI。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
