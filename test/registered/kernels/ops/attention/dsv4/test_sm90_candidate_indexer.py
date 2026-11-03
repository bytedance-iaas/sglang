"""Bitwise parity for moving the SM90 indexer candidate mask before K reads."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_logits_decode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.sm90_candidate_indexer import (
    make_candidate_case,
    select_candidate_topk,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _dense(case):
    return fp4_index_logits_decode(
        case["q"],
        case["weights"],
        case["slots"],
        case["lens"],
        case["table"],
        case["page_size"],
    ).masked_fill(~case["candidate_mask"], -torch.inf)


def _masked(case):
    return fp4_index_logits_decode(**case)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestSm90CandidateIndexer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if torch.version.hip is not None or torch.cuda.get_device_capability()[0] != 9:
            raise unittest.SkipTest("requires SM90")

    def assert_bitwise_equal(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(
            torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        )

    def check_case(self, case, topk=64):
        expected = _dense(case)
        actual = _masked(case)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assert_bitwise_equal(actual, expected)
        if actual.shape[1] > 0:
            for new, old in zip(
                select_candidate_topk(actual, case["slots"], case["lens"], topk),
                select_candidate_topk(expected, case["slots"], case["lens"], topk),
            ):
                self.assertTrue(torch.equal(new, old))

    def test_scores_and_topk_at_tile_and_page_boundaries(self):
        for heads in (8, 16, 32, 64):
            for width in (1, 7, 8, 9, 63, 64, 65, 127, 129, 513):
                for pattern in ("clustered", "scattered"):
                    with self.subTest(heads=heads, width=width, pattern=pattern):
                        self.check_case(
                            make_candidate_case(
                                4,
                                width,
                                heads,
                                page_size=32 if heads == 16 else 64,
                                mixed_lengths=True,
                                pattern=pattern,
                                seed=width,
                            )
                        )

    def test_strided_masks_empty_tiles_and_ties(self):
        for pattern in ("all", "none", "alternating", "clustered"):
            for ties in (False, True):
                with self.subTest(pattern=pattern, ties=ties):
                    case = make_candidate_case(
                        5, 257, pattern=pattern, mixed_lengths=True
                    )
                    # Both strides differ from those of contiguous [rows, width].
                    storage = torch.empty((10, 520), dtype=torch.bool, device="cuda")
                    mask = storage[::2, 3:517:2]
                    mask.copy_(case["candidate_mask"])
                    case["candidate_mask"] = mask
                    if ties:
                        # All scores tie at zero, including the TopK cutoff.
                        case["weights"].zero_()
                    self.check_case(case)

    def test_rejected_slots_are_not_dereferenced(self):
        case = make_candidate_case(4, 193, mixed_lengths=True, pattern="clustered")
        expected = _dense(case)
        # Invalid addresses in rejected columns detect a load-mask regression.
        poisoned = dict(case)
        poisoned["slots"] = case["slots"].masked_fill(
            ~case["candidate_mask"], case["table"].shape[0] * case["page_size"] + 4096
        )
        self.assert_bitwise_equal(_masked(poisoned), expected)

    def test_empty_shapes(self):
        for rows, width in ((0, 65), (4, 0), (0, 0)):
            with self.subTest(rows=rows, width=width):
                self.check_case(make_candidate_case(rows, width, pattern="none"))

    def test_graph_replay_reads_live_mask_lengths_and_pages(self):
        case = make_candidate_case(4, 513, mixed_lengths=True)
        for _ in range(3):
            _masked(case)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = _masked(case)
        for seed, pattern in enumerate(("none", "all", "scattered", "clustered"), 1):
            update = make_candidate_case(
                4, 513, seed=seed, pattern=pattern, mixed_lengths=True
            )
            # Reuse the captured addresses, including a non-contiguous cache.
            for name in ("q", "weights", "slots", "lens", "table", "candidate_mask"):
                case[name].copy_(update[name])
            case["lens"].copy_(update["lens"].flip(0))
            graph.replay()
            self.assert_bitwise_equal(actual, _dense(case))

    def test_source_and_verify_consumers(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.dsv4.candidate_indexer import (
            sm90_index_logits_with_candidates,
        )

        kernel_module = "sglang.kernels.ops.attention.dsv4.fp4_indexer"
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                source = make_candidate_case(6, 129, seed=10)
                consumer = make_candidate_case(6, 129, seed=11)
                # Flattened verify rows share requests but have distinct causal
                # lengths, including queries across a C2 compression boundary.
                req = torch.tensor([2, 2, 0, 0, 0, 1], device="cuda")
                pos = torch.tensor([0, 1, 62, 63, 64, 128], device="cuda")
                req_to_token = (source["slots"][:3] * ratio).repeat_interleave(
                    ratio, dim=1
                )
                columns = torch.arange(129, device="cuda")
                slots = req_to_token[req[:, None], columns[None, :] * ratio] // ratio
                lens = (pos + 1) // ratio
                indexer = SimpleNamespace(
                    candidate_topk_blocks=4, candidate_block_size=8
                )
                results = []
                for enabled in (False, True):
                    steps = []
                    candidate = None
                    with (
                        envs.SGLANG_DSV41_SM90_CANDIDATE_MASK.override(enabled),
                        envs.SGLANG_DSV41_SM90_CANDIDATE_MASK_MIN_WIDTH.override(0),
                        patch(
                            f"{kernel_module}.fp4_index_logits_decode",
                            wraps=fp4_index_logits_decode,
                        ) as score,
                    ):
                        for is_source, inputs in ((True, source), (False, consumer)):
                            indexer.is_candidate_source = is_source
                            indexer.uses_candidates = not is_source
                            logits, candidate = sm90_index_logits_with_candidates(
                                inputs["q"],
                                inputs["weights"],
                                slots,
                                lens,
                                source["table"],
                                source["page_size"],
                                indexer=indexer,
                                candidate_metadata=candidate,
                            )
                            passed = score.call_args.kwargs["candidate_mask"]
                            self.assertEqual(
                                passed is not None, enabled and not is_source
                            )
                            raw, physical = select_candidate_topk(
                                logits, slots, lens, 64, filter_candidates=not is_source
                            )
                            steps.append((logits.clone(), raw, physical))
                        steps.append((candidate.mask.clone(),))
                    results.append(steps)
                for new_step, old_step in zip(results[1], results[0]):
                    for new, old in zip(new_step, old_step):
                        if new.dtype == torch.float32:
                            self.assert_bitwise_equal(new, old)
                        else:
                            self.assertTrue(torch.equal(new, old))

    def test_plain_indexer_ignores_candidate_state(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.dsv4.candidate_indexer import (
            CandidateMasks,
            sm90_index_logits_with_candidates,
        )

        case = make_candidate_case(4, 129, pattern="none", mixed_lengths=True)
        indexer = SimpleNamespace(is_candidate_source=False, uses_candidates=False)
        candidate = CandidateMasks(mask=case["candidate_mask"])
        with envs.SGLANG_DSV41_SM90_CANDIDATE_MASK.override(True):
            actual, state = sm90_index_logits_with_candidates(
                case["q"],
                case["weights"],
                case["slots"],
                case["lens"],
                case["table"],
                case["page_size"],
                indexer=indexer,
                candidate_metadata=candidate,
            )
        expected = fp4_index_logits_decode(
            **{key: value for key, value in case.items() if key != "candidate_mask"}
        )
        self.assert_bitwise_equal(actual, expected)
        self.assertIs(state, candidate)

    def test_width_policy_keeps_filtering_below_threshold(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.dsv4.candidate_indexer import (
            CandidateMasks,
            sm90_index_logits_with_candidates,
        )

        indexer = SimpleNamespace(is_candidate_source=False, uses_candidates=True)
        kernel_module = "sglang.kernels.ops.attention.dsv4.fp4_indexer"
        for threshold in (0, 64):
            for width in (63, 64, 65):
                for enabled in (False, True):
                    with self.subTest(
                        threshold=threshold, width=width, enabled=enabled
                    ):
                        case = make_candidate_case(
                            4, width, pattern="scattered", mixed_lengths=True
                        )
                        candidate = CandidateMasks(mask=case["candidate_mask"])
                        with (
                            envs.SGLANG_DSV41_SM90_CANDIDATE_MASK.override(enabled),
                            envs.SGLANG_DSV41_SM90_CANDIDATE_MASK_MIN_WIDTH.override(
                                threshold
                            ),
                            patch(
                                f"{kernel_module}.fp4_index_logits_decode",
                                wraps=fp4_index_logits_decode,
                            ) as score,
                        ):
                            actual, state = sm90_index_logits_with_candidates(
                                **{
                                    k: v
                                    for k, v in case.items()
                                    if k != "candidate_mask"
                                },
                                indexer=indexer,
                                candidate_metadata=candidate,
                            )
                        self.assertEqual(
                            score.call_args.kwargs["candidate_mask"] is not None,
                            enabled and width >= threshold,
                        )
                        self.assertIs(state, candidate)
                        expected = _dense(case)
                        self.assert_bitwise_equal(actual, expected)
                        for new, old in zip(
                            select_candidate_topk(
                                actual, case["slots"], case["lens"], 64
                            ),
                            select_candidate_topk(
                                expected, case["slots"], case["lens"], 64
                            ),
                        ):
                            self.assertTrue(torch.equal(new, old))

    def test_width_policy_graph_replay_crosses_live_length_threshold(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.dsv4.candidate_indexer import (
            CandidateMasks,
            sm90_index_logits_with_candidates,
        )

        indexer = SimpleNamespace(is_candidate_source=False, uses_candidates=True)
        # Both capacity branches must read changing lengths AND masks on replay.
        # A long capacity must retain pre-mask even when its live lengths shrink.
        for width in (127, 129):
            with self.subTest(width=width):
                case = make_candidate_case(4, width, mixed_lengths=True)
                candidate = CandidateMasks(mask=case["candidate_mask"])

                def run():
                    logits, _ = sm90_index_logits_with_candidates(
                        **{k: v for k, v in case.items() if k != "candidate_mask"},
                        indexer=indexer,
                        candidate_metadata=candidate,
                    )
                    return logits, select_candidate_topk(
                        logits, case["slots"], case["lens"], 64
                    )

                with (
                    envs.SGLANG_DSV41_SM90_CANDIDATE_MASK.override(True),
                    envs.SGLANG_DSV41_SM90_CANDIDATE_MASK_MIN_WIDTH.override(128),
                ):
                    for _ in range(3):
                        run()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual, selected = run()
                    for seed, pattern in enumerate(("none", "all", "scattered"), 20):
                        update = make_candidate_case(
                            4, width, seed=seed, pattern=pattern
                        )
                        for key in ("q", "weights", "slots", "table", "candidate_mask"):
                            case[key].copy_(update[key])
                        case["lens"].copy_(
                            torch.tensor([0, 63, min(128, width), width], device="cuda")
                        )
                        graph.replay()
                        expected = _dense(case)
                        self.assert_bitwise_equal(actual, expected)
                        for new, old in zip(
                            selected,
                            select_candidate_topk(
                                expected, case["slots"], case["lens"], 64
                            ),
                        ):
                            self.assertTrue(torch.equal(new, old))


if __name__ == "__main__":
    unittest.main()
