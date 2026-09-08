"""CPU-only tests for the headless SiDP Nsight pipeline analyzer."""

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SIDP_BENCH = ROOT / "benchmark/kernels/sidp"
sys.path.insert(0, str(SIDP_BENCH))

from nsight_pipeline.analyze_ncu import (
    derive_gemm_metrics,
    overlap_contention_hypotheses,
    parse_ncu_csv,
    unit_rollup_dispersion,
)
from nsight_pipeline.analyze_nsys import (
    _device_metrics,
    analyze_sqlite,
    match_case_operators,
    select_compute_hotspots,
)
from nsight_pipeline.interval_metrics import (
    Interval,
    bubbles_with_neighbors,
    complement_intervals,
    intersect_intervals,
    interval_total,
    merge_intervals,
)
from nsight_pipeline.report import render_report
from nsight_pipeline.profile_pipeline import _validate_run_directories
from nsight_pipeline.schema import benchmark_command, load_config


class IntervalMetricsTest(unittest.TestCase):
    def test_union_intersection_and_bubbles(self):
        left = [Interval(0, 10), Interval(8, 20), Interval(30, 40)]
        right = [Interval(5, 12), Interval(18, 35)]
        self.assertEqual(merge_intervals(left), [(0, 20), (30, 40)])
        self.assertEqual(intersect_intervals(left, right), [(5, 12), (18, 20), (30, 35)])
        self.assertEqual(interval_total(intersect_intervals(left, right)), 14)
        self.assertEqual(complement_intervals(left, 0, 50), [(20, 30), (40, 50)])
        bubbles = bubbles_with_neighbors(left, 0, 50)
        self.assertEqual([item["duration_ms"] for item in bubbles], [1e-5, 1e-5])


class NsightSqliteTest(unittest.TestCase):
    def test_sm_side_overlap_uses_interval_union(self):
        def activity(start, end, role, stream):
            return {
                "start": start,
                "end": end,
                "role": role,
                "name": role,
                "stream": stream,
                "pid": 1,
            }

        metrics = _device_metrics(
            0,
            [
                activity(0, 100, "compute", 1),
                activity(20, 60, "communication_sm", 2),
                activity(40, 80, "wait_or_claim", 3),
                activity(70, 90, "control", 4),
            ],
        )
        # The three per-role overlaps sum to 100ns, but their union is [20, 90].
        self.assertEqual(metrics["compute_sm_side_overlap_ms"], 70e-6)
        self.assertEqual(metrics["compute_sm_side_overlap_ratio"], 0.7)

    def _fixture(self, path, gemm_end=100):
        with sqlite3.connect(path) as connection:
            connection.executescript(
                """
                CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                  start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER,
                  globalPid INTEGER, demangledName INTEGER,
                  gridX INTEGER, gridY INTEGER, gridZ INTEGER,
                  blockX INTEGER, blockY INTEGER, blockZ INTEGER,
                  dynamicSharedMemory INTEGER
                );
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
                  start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER,
                  globalPid INTEGER, bytes INTEGER, copyKind INTEGER,
                  srcKind INTEGER, dstKind INTEGER
                );
                """
            )
            connection.executemany(
                "INSERT INTO StringIds VALUES (?, ?)",
                [(1, "ampere_bf16_gemm"), (2, "wait_generation_kernel"), (3, "publish_generation_kernel")],
            )
            connection.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (0, gemm_end, 0, 1, 10, 1, 10, 1, 1, 256, 1, 1, 0),
                    (160, 180, 0, 1, 10, 2, 1, 1, 1, 1, 1, 1, 0),
                    (185, 190, 0, 1, 10, 3, 1, 1, 1, 1, 1, 1, 0),
                ],
            )
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?,?,?,?)",
                (50, 150, 0, 2, 10, 1000, 8, 1, 1),
            )

    def test_sqlite_overlap_bubble_dma_and_operator_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            self._fixture(path)
            analysis = analyze_sqlite(path)
        device = analysis["devices"][0]
        self.assertAlmostEqual(device["compute_union_ms"], 0.0001)
        self.assertAlmostEqual(device["communication_union_ms"], 0.0001)
        self.assertAlmostEqual(device["compute_communication_overlap_ms"], 0.00005)
        self.assertAlmostEqual(device["idle_bubbles"]["total_ms"], 0.000015)
        self.assertAlmostEqual(device["wait_or_claim_union_ms"], 0.00002)
        self.assertEqual(analysis["dma"]["total_bytes"], 1000)
        self.assertEqual(analysis["operators"][0]["role"], "compute")

    def test_operator_matching_and_hotspot_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "base.sqlite"
            variant_path = Path(directory) / "variant.sqlite"
            self._fixture(baseline_path, 100)
            self._fixture(variant_path, 130)
            baseline = analyze_sqlite(baseline_path)
            variant = analyze_sqlite(variant_path)
        deltas = match_case_operators(baseline, variant)
        gemm = next(item for item in deltas if item["role"] == "compute")
        self.assertAlmostEqual(gemm["mean_ratio"], 1.3)
        hotspots = select_compute_hotspots(variant, 0.99, 1, 1, deltas)
        self.assertEqual(len(hotspots), 1)
        self.assertEqual(hotspots[0]["role"], "compute")

    def test_markdown_report_is_generated_from_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            sqlite_path = directory / "trace.sqlite"
            self._fixture(sqlite_path)
            nsys = analyze_sqlite(sqlite_path)
            analysis = {
                "schema_version": 1,
                "status": "complete",
                "output_dir": str(directory),
                "artifact_dir": "/tmp/raw",
                "manifest": {
                    "tools": {
                        "nsys": {"version": "test"},
                        "ncu": {"version": "test"},
                    }
                },
                "cases": {
                    "case": {
                        "settings": {
                            "mode": "compute_dma_flag",
                            "k": 1,
                            "gemm_m": 1,
                            "dma_slices": 1,
                            "dma_slice_groups": 1,
                        },
                        "nsys": nsys,
                    }
                },
                "comparisons": [],
                "ncu": {"runs": []},
                "limitations": [],
            }
            report = directory / "report.md"
            render_report(analysis, report)
            text = report.read_text(encoding="utf-8")
        self.assertIn("核心时间线", text)
        self.assertIn("compute并集", text)
        self.assertIn("effective GB/s", text)


class NsightConfigAndNcuTest(unittest.TestCase):
    def test_force_cleanup_rejects_broad_directories(self):
        with self.assertRaises(ValueError):
            _validate_run_directories(
                {"working_directory": str(ROOT)}, Path("/"), Path("/tmp/safe-run")
            )

    def test_config_and_benchmark_command(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            benchmark = directory / "bench.py"
            benchmark.write_text("print('x')\n", encoding="utf-8")
            config_path = directory / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "working_directory": str(directory),
                        "benchmark": "bench.py",
                        "output_dir": "check_logs/result",
                        "cases": [{"name": "fixed flag", "mode": "compute_dma_flag", "k": 1}],
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(config_path)
            command = benchmark_command(config, config["cases"][0], directory / "out.json")
        self.assertIn("compute_dma_flag", command)
        self.assertIn("--nsight-profile-sample", command)
        self.assertEqual(config["cases"][0]["name"], "fixed_flag")

    def test_ncu_csv_parser_keeps_core_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ncu.csv"
            path.write_text(
                "noise before csv\n"
                '"ID","Process ID","Kernel Name","Device","Context","Stream","Metric Name","Metric Unit","Metric Value"\n'
                '"0","12","gemm","GPU 0","1","2","gpu__time_duration.sum","nsecond","1000"\n'
                '"0","12","gemm","GPU 0","1","2","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","75"\n',
                encoding="utf-8",
            )
            parsed = parse_ncu_csv([path])
        self.assertEqual(parsed["launch_count"], 1)
        self.assertEqual(parsed["metrics"]["gpu__time_duration.sum"]["mean"], 1000)

    def test_ncu_wide_raw_csv_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ncu_wide.csv"
            path.write_text(
                '"ID","Process ID","Kernel Name","Device","Context","Stream","Block Size","Grid Size","gpu__time_duration.sum","sm__throughput.avg.pct_of_peak_sustained_elapsed"\n'
                '"","","","","","","","","nsecond","%"\n'
                '"0","12","gemm","GPU 0","1","2","(256,1,1)","(10,1,1)","1000","75"\n',
                encoding="utf-8",
            )
            parsed = parse_ncu_csv([path])
        self.assertEqual(parsed["launch_count"], 1)
        self.assertEqual(parsed["metrics"]["gpu__time_duration.sum"]["unit"], "nsecond")
        self.assertEqual(
            parsed["metrics"]["sm__throughput.avg.pct_of_peak_sustained_elapsed"]["mean"],
            75,
        )
        derived = derive_gemm_metrics(
            parsed, {"gemm_m": 2, "gemm_n": 4, "gemm_k": 8}
        )
        self.assertEqual(derived["algorithmic_flops"], 128)
        self.assertAlmostEqual(derived["achieved_tflops"], 0.000128)

    def test_unit_rollups_and_cross_evidence_hypothesis(self):
        profile = {
            "metrics": {
                "sm__cycles_active.avg": {"mean": 100, "unit": "cycle"},
                "sm__cycles_active.min": {"mean": 50, "unit": "cycle"},
                "sm__cycles_active.max": {"mean": 150, "unit": "cycle"},
            }
        }
        rollups = unit_rollup_dispersion(profile)
        self.assertEqual(len(rollups), 1)
        self.assertEqual(rollups[0]["range_over_avg"], 1.0)
        isolated = {
            "metrics": {
                "gpu__time_duration.sum": {"mean": 100, "unit": "ns"}
            }
        }
        variant_node = {
            "metrics": {
                "gpu__time_duration.sum": {"mean": 101, "unit": "ns"}
            }
        }
        workload_base = {
            "metrics": {
                "lts__throughput.avg.pct_of_peak_sustained_elapsed": {
                    "mean": 20,
                    "unit": "%",
                }
            }
        }
        workload_variant = {
            "metrics": {
                "lts__throughput.avg.pct_of_peak_sustained_elapsed": {
                    "mean": 45,
                    "unit": "%",
                }
            }
        }
        findings = overlap_contention_hypotheses(
            systems_mean_ratio=1.2,
            compute_overlap_ratio=0.7,
            overlap_by_role_ratio={"communication_sm": 0.4},
            compute_sm_side_overlap_ratio=0.4,
            isolated_node_baseline=isolated,
            isolated_node_variant=variant_node,
            workload_baseline=workload_base,
            workload_variant=workload_variant,
            sm_trace_roles=[
                {
                    "role": "communication_sm",
                    "device_sm_coverage_median": 0.5,
                }
            ],
        )
        kinds = {item["kind"] for item in findings}
        self.assertIn("concurrency_specific_slowdown", kinds)
        self.assertIn("sm_execution_contention", kinds)
        self.assertIn("workload_memory_pressure", kinds)


if __name__ == "__main__":
    unittest.main()
