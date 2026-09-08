"""CPU-only checks for the DMA + controlled-GEMM pipeline benchmark."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SPEC = importlib.util.spec_from_file_location(
    "sidp_dma_compute_pipeline_benchmark",
    ROOT / "benchmark/kernels/sidp/bench_dma_compute_pipeline.py",
)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class DmaComputePipelineBenchmarkTest(unittest.TestCase):
    def test_compute_boundary_metrics_reconstruct_compute_path(self):
        class Event:
            def __init__(self, timestamp):
                self.timestamp = timestamp

            def elapsed_time(self, other):
                return other.timestamp - self.timestamp

        metrics = BENCH.compute_boundary_metrics(
            Event(0),
            Event(30),
            [(Event(2), Event(12)), (Event(15), Event(25))],
            {1: (Event(1), Event(8))},
        )
        self.assertEqual(metrics["pre_cycle0_gap_ms"], 2)
        self.assertEqual(metrics["inter_cycle_gap_ms"], [3])
        self.assertEqual(metrics["post_cycle_gap_ms"], 5)
        self.assertEqual(metrics["boundary_gap_total_ms"], 10)
        self.assertEqual(metrics["cycle_sum_ms"], 20)
        self.assertEqual(metrics["accounted_compute_path_ms"], 30)
        self.assertEqual(metrics["residual_ms"], 0)
        self.assertEqual(metrics["c1_start_from_graph_ms"], 1)
        self.assertEqual(metrics["c0_start_minus_c1_start_ms"], 1)

    def test_defaults_model_six_cycle_gemma4_pipeline(self):
        args = BENCH.parse_args([])
        self.assertEqual(args.num_gpus, 8)
        self.assertEqual(args.k_values, [1, 4])
        self.assertEqual(args.cycles, 6)
        self.assertEqual(args.component_bytes, [3840 * 15360 * 4, 3840 * 15360 * 2])
        self.assertGreater(args.gemm_m, 0)
        self.assertEqual((args.gemm_n, args.gemm_k), (4096, 4096))
        self.assertEqual(args.gemm_repeats_values, [1])
        self.assertEqual(args.arrival_model, "reset_uniform")
        self.assertEqual(args.continuous_replays, 256)
        self.assertEqual(
            [scenario["max_us"] for scenario in args.delay_scenarios],
            [1000, 5000, 10000],
        )
        self.assertEqual(args.modes, list(BENCH.MODES))
        self.assertEqual(
            BENCH.MODES,
            (
                "compute_only",
                "compute_dma",
                "compute_dma_flag",
                "dynamic_dma",
                "dynamic_dma_compute_priority",
            ),
        )
        self.assertEqual(
            BENCH.FLAG_MODES,
            (
                "compute_dma_flag",
                "compute_dma_slice4_flag",
                "compute_dma_slice4_group2_flag",
                "compute_dma_slice4_group4_flag",
                "compute_sm_flag",
                "dynamic_sm",
                "dynamic_sm_compute_priority",
                "dynamic_dma",
                "dynamic_dma_compute_priority",
            ),
        )
        self.assertEqual(
            BENCH.CLAIM_ORDER,
            {
                "dynamic_dma": 0,
                "dynamic_dma_compute_priority": 1,
                "dynamic_sm": 0,
                "dynamic_sm_compute_priority": 1,
            },
        )
        self.assertEqual(
            BENCH.SLICE_MODE_FACTORS,
            {
                "compute_dma_slice2": 2,
                "compute_dma_slice4": 4,
                "compute_dma_slice4_group2": 4,
                "compute_dma_slice4_group4": 4,
                "compute_dma_slice4_flag": 4,
                "compute_dma_slice4_group2_flag": 4,
                "compute_dma_slice4_group4_flag": 4,
            },
        )

    def test_candidate_slot_and_generation_schedule(self):
        self.assertEqual(BENCH.candidates(0, 8, 1), list(range(1, 8)))
        self.assertEqual(BENCH.candidates(7, 8, 4), [3, 4, 5, 6])
        self.assertEqual(BENCH.slot_for(0, 3, 7), 3)
        self.assertEqual(BENCH.slot_for(1, 3, 7), 10)
        self.assertEqual(BENCH.slot_for(2, 3, 7), 3)
        self.assertEqual(
            [BENCH.fill_generation(c) for c in range(6)], [1, 1, 2, 2, 3, 3]
        )
        self.assertEqual(
            [BENCH.reuse_requirement(c) for c in range(6)], [0, 0, 1, 1, 2, 2]
        )
        self.assertEqual(BENCH.next_forward_generations(6), (3, 4))
        self.assertEqual(BENCH.communication_sequence(6), [1, 2, 3, 4, 5, 0])

    def test_custom_compute_and_trace_configuration(self):
        args = BENCH.parse_args(
            [
                "--num-gpus",
                "2",
                "--k-values",
                "1",
                "--cycles",
                "2",
                "--component-bytes",
                "37,13",
                "--gemm-m",
                "64",
                "--gemm-n",
                "128",
                "--gemm-k",
                "32",
                "--gemm-repeats-values",
                "1,2,4",
                "--random-delay-max-us",
                "250",
                "--trace-steps",
                "--trace-layers",
            ]
        )
        self.assertEqual(args.component_bytes, [37, 13])
        self.assertEqual(args.gemm_repeats_values, [1, 2, 4])
        self.assertEqual(args.delay_scenarios[0]["max_us"], 250)
        self.assertTrue(args.trace_steps)
        self.assertTrue(args.trace_layers)

    def test_explicit_single_mode_profiler_configuration(self):
        args = BENCH.parse_args(
            [
                "--k-values",
                "4",
                "--gemm-repeats-values",
                "1",
                "--random-delay-max-us",
                "1000",
                "--modes",
                "compute_dma_flag",
                "--iterations",
                "1",
                "--nsight-annotations",
                "--nsight-profile-sample",
                "0",
            ]
        )
        self.assertEqual(args.modes, ["compute_dma_flag"])
        self.assertEqual(args.nsight_profile_sample, 0)
        self.assertTrue(args.nsight_annotations)

    def test_sm_execution_trace_is_profiler_only_and_compacts_ctas(self):
        args = BENCH.parse_args(
            [
                "--k-values",
                "4",
                "--gemm-repeats-values",
                "1",
                "--random-delay-max-us",
                "1000",
                "--modes",
                "compute_sm_flag",
                "--iterations",
                "1",
                "--nsight-annotations",
                "--nsight-profile-sample",
                "0",
                "--sm-execution-trace",
                "--sm-copy-ctas",
                "10",
            ]
        )
        self.assertTrue(args.sm_execution_trace)
        self.assertEqual(args.sm_copy_ctas, 10)
        summary = BENCH.summarize_sm_execution_trace(
            [
                [0, 0, 3, 3, 100, 140],
                [0, 1, 7, 7, 105, 150],
                [1, 0, 2, 4, 200, 230],
            ],
            3,
            0,
            [
                {"tag": 0, "role": "communication_sm", "kernel": "copy"},
                {"tag": 1, "role": "control", "kernel": "publish"},
            ],
            10,
            8,
        )
        copy = next(
            row for row in summary["roles"] if row["role"] == "communication_sm"
        )
        self.assertEqual(copy["smids"], [3, 7])
        self.assertEqual(copy["unique_sm_count"], 2)
        self.assertEqual(summary["launches"][1]["smid_migrations"], 1)

    def test_profiler_rejects_ambiguous_capture_range(self):
        invalid = (
            ["--nsight-profile-sample", "0"],
            [
                "--modes",
                "compute_only,compute_dma_flag",
                "--nsight-annotations",
                "--nsight-profile-sample",
                "0",
            ],
            ["--modes", "unknown"],
            ["--modes", "compute_only", "--slice-study"],
            ["--sm-execution-trace"],
        )
        for argv in invalid:
            with (
                self.subTest(argv=argv),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                BENCH.parse_args(argv)

    def test_cuda_graph_dump_configuration(self):
        args = BENCH.parse_args(
            [
                "--dump-graph-dir",
                "check_logs/graphs",
                "--dump-graph-modes",
                "compute_only,compute_dma,compute_dma",
            ]
        )
        self.assertEqual(args.dump_graph_dir, "check_logs/graphs")
        self.assertEqual(args.dump_graph_modes, ["compute_only", "compute_dma"])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            BENCH.parse_args(["--dump-graph-modes", "not_a_mode"])

    def test_slice_study_uses_paired_per_sample_random_delays(self):
        args = BENCH.parse_args(
            [
                "--slice-study",
                "--random-delay-max-us",
                "1000,5000",
            ]
        )
        self.assertEqual(args.modes, list(BENCH.SLICE_STUDY_MODES))
        self.assertEqual(
            [scenario["max_us"] for scenario in args.delay_scenarios],
            [1000, 5000],
        )
        scenario = args.delay_scenarios[0]
        first = BENCH.planned_offsets(scenario, args.seed, 3, args.num_gpus)
        paired = BENCH.planned_offsets(scenario, args.seed, 3, args.num_gpus)
        redrawn = BENCH.planned_offsets(scenario, args.seed, 4, args.num_gpus)
        self.assertEqual(first, paired)
        self.assertNotEqual(first, redrawn)
        self.assertEqual(min(first), 0)
        self.assertLessEqual(max(first), 1000)

    def test_continuous_arrival_uses_paired_per_rank_gap_sequences(self):
        args = BENCH.parse_args(
            [
                "--arrival-model",
                "continuous",
                "--continuous-replays",
                "4",
                "--random-delay-max-us",
                "1000",
            ]
        )
        scenario = args.delay_scenarios[0]
        first = BENCH.planned_continuous_gaps(
            scenario, args.seed, args.continuous_replays, args.num_gpus
        )
        paired = BENCH.planned_continuous_gaps(
            scenario, args.seed, args.continuous_replays, args.num_gpus
        )
        self.assertEqual(first, paired)
        self.assertEqual(len(first), args.num_gpus)
        self.assertTrue(all(len(row) == 4 for row in first))
        self.assertEqual(min(row[0] for row in first), 0)
        self.assertTrue(
            all(0 <= gap <= 1000 for row in first for gap in row)
        )
        self.assertIn(
            "continuous per-rank gap", BENCH.delay_label(scenario, "continuous")
        )

    def test_continuous_epoch_reports_drift_and_requested_comparisons(self):
        rank_replays = [
            [
                {
                    "launch_begin_ns": rank * 1_000 + replay * 10_000,
                    "completion_observed_ns": rank * 1_000
                    + replay * 10_000
                    + 5_000,
                    "requested_offset_us": float(replay),
                }
                for replay in range(3)
            ]
            for rank in range(2)
        ]
        metrics = BENCH.continuous_epoch_metrics(rank_replays)
        self.assertEqual(metrics["replays_per_rank"], 3)
        self.assertEqual(metrics["total_rank_replays"], 6)
        self.assertEqual(metrics["launch_span_us"]["initial"], 1)
        self.assertEqual(metrics["launch_span_us"]["final"], 1)
        ratios = BENCH.continuous_throughput_ratios(
            {
                "compute_dma": {"aggregate_rank_replays_per_second": 100},
                "compute_dma_slice4": {
                    "aggregate_rank_replays_per_second": 110
                },
                "compute_dma_flag": {
                    "aggregate_rank_replays_per_second": 120
                },
                "compute_dma_slice4_flag": {
                    "aggregate_rank_replays_per_second": 108
                },
            }
        )
        self.assertAlmostEqual(ratios["event_s4_to_s1"], 1.1)
        self.assertAlmostEqual(ratios["flag_s4_to_s1"], 0.9)
        self.assertAlmostEqual(ratios["s1_flag_to_event"], 1.2)
        self.assertAlmostEqual(ratios["s4_flag_to_event"], 108 / 110)

    def test_slice_bounds_cover_odd_sized_components(self):
        for factor in (1, 2, 4):
            ranges = [BENCH.slice_bounds(13, factor, index) for index in range(factor)]
            self.assertEqual(ranges[0][0], 0)
            self.assertEqual(ranges[-1][1], 13)
            self.assertTrue(
                all(
                    ranges[index][1] == ranges[index + 1][0]
                    for index in range(factor - 1)
                )
            )
            self.assertEqual(sum(end - begin for begin, end in ranges), 13)

    def test_s4_owner_groups_are_contiguous_and_front_heavy(self):
        self.assertEqual(BENCH.owner_group_ranges(7, 1), [(0, 7)])
        self.assertEqual(BENCH.owner_group_ranges(7, 2), [(0, 4), (4, 7)])
        self.assertEqual(
            BENCH.owner_group_ranges(7, 4),
            [(0, 2), (2, 4), (4, 6), (6, 7)],
        )

    def test_group_study_pairs_event_and_flag_for_unsliced_and_s4_groups(self):
        args = BENCH.parse_args(["--group-study", "--random-delay-max-us", "1000"])
        self.assertEqual(args.modes, list(BENCH.GROUP_STUDY_MODES))
        self.assertEqual(args.delay_scenarios[0]["kind"], "random_uniform")
        ratios = BENCH.graph_ratios(
            {
                "compute_dma_slice4": 20,
                "compute_dma_slice4_group2": 18,
                "compute_dma_slice4_group4": 16,
                "compute_dma_slice4_flag": 10,
                "compute_dma_slice4_group2_flag": 9,
                "compute_dma_slice4_group4_flag": 8,
            }
        )
        self.assertEqual(
            ratios["compute_dma_slice4_group2_to_compute_dma_slice4_graph"],
            0.9,
        )
        self.assertEqual(
            ratios["compute_dma_slice4_group4_to_compute_dma_slice4_graph"],
            0.8,
        )
        self.assertEqual(
            ratios[
                "compute_dma_slice4_group2_flag_to_compute_dma_slice4_flag_graph"
            ],
            0.9,
        )
        self.assertEqual(
            ratios[
                "compute_dma_slice4_group4_flag_to_compute_dma_slice4_flag_graph"
            ],
            0.8,
        )
        self.assertEqual(
            ratios["compute_dma_slice4_flag_to_compute_dma_slice4_graph"],
            0.5,
        )

    def test_slice_ratios_are_reported_against_unsliced_dma(self):
        ratios = BENCH.graph_ratios(
            {
                "compute_only": 10,
                "compute_dma": 20,
                "compute_dma_slice2": 15,
                "compute_dma_slice4": 12,
            }
        )
        self.assertEqual(ratios["compute_dma_slice2_to_compute_dma_graph"], 0.75)
        self.assertEqual(ratios["compute_dma_slice4_to_compute_dma_graph"], 0.6)
        self.assertEqual(ratios["event_to_compute_only_graph"], 2)

    def test_invalid_configuration_rejected(self):
        invalid = (
            ["--k-values", "8"],
            ["--cycles", "1"],
            ["--cycles", "5"],
            ["--component-bytes", "0"],
            ["--gemm-m", "0"],
            ["--gemm-repeats-values", "0"],
            ["--iterations", "0"],
            ["--continuous-replays", "0"],
            ["--random-delay-max-us=-1"],
            ["--stagger-us", "1000"],
            ["--rank-offsets-us", "0,1,2,3,4,5,6,7"],
            ["--jitter-us", "100"],
            ["--slice-study", "--group-study"],
            [
                "--arrival-model",
                "continuous",
                "--k-values",
                "1",
                "--random-delay-max-us",
                "1000",
                "--modes",
                "compute_dma",
                "--iterations",
                "1",
                "--nsight-annotations",
                "--nsight-profile-sample",
                "0",
            ],
        )
        for argv in invalid:
            with (
                self.subTest(argv=argv),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                BENCH.parse_args(argv)

    def test_trial_metrics_include_skew_compute_path_and_tail(self):
        ranks = [
            {
                "graph_ms": 5.0,
                "compute_path_ms": 4.0,
                "tail_join_ms": 1.0,
                "launch_begin_ns": 0,
                "completion_observed_ns": 5_000_000,
                "launch_lateness_us": 0,
            },
            {
                "graph_ms": 6.0,
                "compute_path_ms": 5.5,
                "tail_join_ms": 0.5,
                "launch_begin_ns": 2_000_000,
                "completion_observed_ns": 8_000_000,
                "launch_lateness_us": 10,
            },
        ]
        metrics = BENCH.trial_metrics(ranks)
        self.assertEqual(metrics["max_rank_graph_ms"], 6)
        self.assertEqual(metrics["max_rank_compute_path_ms"], 5.5)
        self.assertEqual(metrics["max_rank_tail_join_ms"], 1)
        self.assertEqual(metrics["actual_host_launch_span_us"], 2000)
        self.assertEqual(metrics["all_rank_observed_wall_ms"], 8)

    def test_report_can_be_written_before_first_case(self):
        args = BENCH.parse_args([])
        payload = {**vars(args), "status": "running", "device": "mock", "cases": []}
        with tempfile.TemporaryDirectory(prefix="sidp_compute_pipeline_") as directory:
            path = Path(directory) / "report.json"
            BENCH.write_report(path, payload)
            report = path.with_suffix(".md").read_text(encoding="utf-8")
            self.assertIn("可控 GEMM", report)
            self.assertIn("不验证真实 FFN 数值", report)


if __name__ == "__main__":
    unittest.main()
