"""CPU-only checks for communication-benchmark configuration and accounting."""

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[5]
SPEC = importlib.util.spec_from_file_location(
    "sidp_dma_cycle_benchmark", ROOT / "benchmark/kernels/sidp/bench_dma_cycle.py"
)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class DmaCycleBenchmarkTest(unittest.TestCase):
    def test_defaults_match_gemma4_components_and_k(self):
        args = BENCH.parse_args([])
        self.assertEqual(args.num_gpus, 8)
        self.assertEqual(args.k_values, [1, 4])
        self.assertEqual(args.component_bytes, [3840 * 15360 * 4, 3840 * 15360 * 2])
        self.assertEqual(args.offset_scenarios[0], [0] * 8)
        self.assertEqual(args.offset_scenarios[1][-1], 1750)
        self.assertFalse(args.trace_steps)
        self.assertEqual(
            BENCH.MODES,
            ("compute_dma", "compute_dma_flag", "dynamic_dma"),
        )
        self.assertEqual(
            BENCH.FLAG_MODES,
            ("compute_dma_flag", "dynamic_dma"),
        )
        self.assertEqual(len(BENCH.MODE_ORDERS), 6)
        for position in range(3):
            self.assertEqual(
                sorted(order[position] for order in BENCH.MODE_ORDERS),
                sorted(BENCH.MODES * 2),
            )

    def test_candidate_coverage_and_compute_order(self):
        for k in (1, 2, 4):
            for rank in range(8):
                with self.subTest(k=k, rank=rank):
                    owners = BENCH.candidates(rank, 8, k)
                    self.assertEqual(len(owners), 8 - k)
                    self.assertEqual(owners, sorted(set(owners)))
                    self.assertTrue(
                        set(owners).isdisjoint((rank + i) % 8 for i in range(k))
                    )
        self.assertEqual(BENCH.candidates(7, 8, 4), [3, 4, 5, 6])

    def test_explicit_offsets_and_byte_tails(self):
        args = BENCH.parse_args(
            [
                "--num-gpus",
                "2",
                "--k-values",
                "1",
                "--rank-offsets-us",
                "0,2000",
                "--component-bytes",
                "37,13,5",
                "--trace-steps",
            ]
        )
        self.assertEqual(args.offset_scenarios, [[0, 2000]])
        self.assertEqual(args.component_bytes, [37, 13, 5])
        self.assertTrue(args.trace_steps)

    def test_invalid_configuration_rejected(self):
        for argv in (
            ["--k-values", "8"],
            ["--iterations", "0"],
            ["--component-bytes", "0,16"],
            ["--component-bytes", "1.5"],
            ["--stagger-us=-1"],
            ["--rank-offsets-us", "0,1"],
            ["--jitter-us", "nan"],
            ["--lead-ms", "0"],
            ["--rank-offsets-us", "0,0,0,0,0,0,0,1", "--stagger-us", "0"],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    BENCH.parse_args(argv)

    def test_paired_jitter_is_identical_and_bounded(self):
        base = [0, 250, 500]
        first = BENCH.planned_offsets(base, 100, 11, 5)
        self.assertEqual(first, BENCH.planned_offsets(base, 100, 11, 5))
        self.assertNotEqual(first, BENCH.planned_offsets(base, 100, 11, 6))
        self.assertTrue(all(b <= x <= b + 100 for b, x in zip(base, first)))
        self.assertEqual(base, BENCH.planned_offsets(base, 0, 11, 5))
        self.assertEqual(
            BENCH.delay_scenario_label([0, 0, 0], 1000),
            "independent U[0,1000] us/rank/sample",
        )
        self.assertEqual(
            BENCH.delay_scenario_label([0, 250, 500], 0),
            "fixed offsets [0, 250, 500] us",
        )

    def test_wall_makespan_includes_skew_not_just_max_local_duration(self):
        ranks = [
            {
                "cycle_ms": 1.0,
                "launch_begin_ns": 0,
                "completion_observed_ns": 1_000_000,
                "launch_lateness_us": 0,
            },
            {
                "cycle_ms": 1.5,
                "launch_begin_ns": 2_000_000,
                "completion_observed_ns": 3_500_000,
                "launch_lateness_us": 0,
            },
        ]
        metrics = BENCH.trial_metrics(ranks, 1_000_000)
        self.assertEqual(metrics["max_rank_cycle_ms"], 1.5)
        self.assertEqual(metrics["actual_host_launch_span_us"], 2000)
        self.assertEqual(metrics["all_rank_observed_wall_ms"], 3.5)
        self.assertAlmostEqual(metrics["all_rank_observed_wall_gbps"], 2 / 3.5)

    def test_percentile_and_summary(self):
        records = [
            {
                "metrics": {"time": value},
                "ranks": [{"cycle_ms": value}, {"cycle_ms": value + 1}],
            }
            for value in (1, 2, 3)
        ]
        summary = BENCH.summarize(records, 2)
        self.assertEqual(summary["rank_cycle_ms_p50"], [2, 3])
        self.assertEqual(summary["time"]["p50"], 2)
        self.assertAlmostEqual(summary["time"]["p95"], 2.9)

    def test_report_can_be_written_before_first_completed_case(self):
        args = BENCH.parse_args([])
        payload = {**vars(args), "status": "running", "device": "mock", "cases": []}
        with tempfile.TemporaryDirectory(prefix="sidp_cycle_report_") as directory:
            path = Path(directory) / "report.json"
            BENCH.write_report(path, payload)
            self.assertTrue(path.exists())
            self.assertIn("不是纯DMA耗时", path.with_suffix(".md").read_text())


if __name__ == "__main__":
    unittest.main()
