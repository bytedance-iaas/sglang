"""Prefix-only score storage and exact-by-value TopK on Hopper."""

import unittest
from unittest.mock import patch

import test_sm90_fp4_grouped_indexer as fixtures
import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsv4.sm90_length_aware_indexer import (
    prefix_logits,
    select_prefix_topk,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    CandidateBlocks,
    published_masks,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
    "requires an SM90 GPU",
)
class TestSm90LengthAwareIndexer(CustomTestCase):
    def setUp(self):
        self.fixture = fixtures.TestSm90Fp4GroupedIndexer("test_matches_triton")

    def run_case(self, case):
        with (
            envs.SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER.override(True),
            envs.SGLANG_OPT_DSV41_SM90_LENGTH_AWARE_INDEXER.override(True),
        ):
            case.run()

    def check_outputs(self, x, case, role="none", input_mask=None, k=512, has_raw=True):
        scores = self.fixture.reference(x)
        if role == "consumer":
            scores = scores.masked_fill(~input_mask, -torch.inf)
        k = min(k, x.width)
        # Recover raw indices from the physical mapping when raw output is absent.
        if has_raw:
            raw = case.raw[:, :k].long()
        else:
            expected_slots = x.slots
            # Small test rows: search the known mapping without assuming order.
            raw = (
                (expected_slots[:, :, None] == case.pages[:, None, :k]).long().argmax(1)
            )
            raw = raw.masked_fill(case.pages[:, :k] < 0, -1)
        valid = raw >= 0
        self.assertTrue(torch.all(raw[valid] < x.lens[:, None].expand_as(raw)[valid]))
        ordered = raw.masked_fill(~valid, x.width)
        self.assertTrue(torch.all(ordered[:, 1:] >= ordered[:, :-1]))
        self.assertTrue(torch.all((ordered[:, 1:] != ordered[:, :-1]) | ~valid[:, 1:]))
        got = scores.gather(1, raw.clamp_min(0)).masked_fill(~valid, -torch.inf)
        expected = scores.topk(k, dim=1).values
        torch.testing.assert_close(
            got.sort(descending=True).values, expected, atol=0, rtol=0
        )
        expected_count = torch.minimum(x.lens, torch.full_like(x.lens, k))
        if role == "consumer":
            expected_count = (scores > -torch.inf).sum(1).clamp_max(k)
        torch.testing.assert_close(valid.sum(1), expected_count, atol=0, rtol=0)
        slots = x.mapping[x.req[:, None], raw.clamp_min(0) * x.ratio].long() // x.ratio
        torch.testing.assert_close(
            case.pages[:, :k].long(), slots.masked_fill(~valid, -1), atol=0, rtol=0
        )
        self.assertTrue(torch.all(case.pages[:, k:] == -1))
        if has_raw:
            self.assertTrue(torch.all(case.raw[:, k:] == -1))
        if role == "source":
            block_size = case.layer.indexer.candidate_block_size
            block_k = case.layer.indexer.candidate_topk_blocks
            block_scores = F.pad(scores, (0, -x.width % block_size), value=-torch.inf)
            block_scores = block_scores.unflatten(1, (-1, block_size)).amax(-1)
            last = (x.lens - 1) // block_size
            block_scores = block_scores.masked_fill(
                torch.arange(block_scores.shape[1], device="cuda")[None, :]
                == last[:, None],
                torch.inf,
            )
            mask = published_masks(case.state.forward_metadata.candidate_metadata).mask
            block_mask = mask[:, ::block_size]
            torch.testing.assert_close(
                mask, block_mask.repeat_interleave(block_size, 1)[:, : x.width]
            )
            for row in range(x.q.shape[0]):
                selected = (
                    block_scores[row][block_mask[row]].sort(descending=True).values
                )
                want = (
                    block_scores[row].topk(min(block_k, block_scores.shape[1])).values
                )
                want = want[want > -torch.inf]
                torch.testing.assert_close(selected, want, atol=0, rtol=0)

    def test_prefix_scores_and_poisoned_tail(self):
        for width, page, ratio in ((257, 16, 1), (65536, 64, 2)):
            x = self.fixture.make_inputs(width=width, page=page, ratio=ratio, heads=32)
            x.lens.copy_(
                torch.tensor(
                    [0, 1, 3, 4, 63, 64, 65, 127, 512, 513, 8191, 8192, width],
                    device="cuda",
                ).clamp_max(width)
            )
            expected = self.fixture.reference(x)
            for row in range(2):
                x.mapping[row, int(x.lens[row * 6 : (row + 1) * 6].max()) * ratio :] = (
                    2**30
                )
            actual = prefix_logits(
                x.q,
                x.weights,
                x.mapping,
                x.req,
                x.lens.int(),
                x.table,
                page,
                ratio,
                width,
            )
            valid = torch.arange(width, device="cuda")[None, :] < x.lens[:, None]
            torch.testing.assert_close(
                actual[:, :width][valid], expected[valid], atol=0, rtol=0
            )

    def test_topk_ignores_tail_and_handles_ties(self):
        for width in (1028, 131072):
            lens = torch.tensor(
                [0, 1, 31, 511, 512, 513, 1025], device="cuda", dtype=torch.int32
            )
            scores = torch.full((7, width), torch.nan, device="cuda")
            pos = torch.arange(width, device="cuda")
            values = (pos % 23).float().expand_as(scores)
            valid = pos[None, :] < lens[:, None]
            scores[valid] = values[valid]
            for k in (1, 31, 512, 1025):
                idx = select_prefix_topk(scores, lens, k).long()
                got = scores.gather(1, idx.clamp_min(0)).masked_fill(
                    idx < 0, -torch.inf
                )
                want = scores.masked_fill(~valid, -torch.inf).topk(k).values
                torch.testing.assert_close(
                    got.sort(descending=True).values, want, atol=0, rtol=0
                )
                self.assertTrue(torch.all((idx < 0) | (idx < lens[:, None])))
                ordered = idx.sort().values
                self.assertTrue(
                    torch.all(
                        (ordered[:, 1:] != ordered[:, :-1]) | (ordered[:, 1:] < 0)
                    )
                )

    def test_backend_roles_and_layouts(self):
        for ratio, role, k, has_raw in (
            (1, "none", 512, True),
            (2, "source", 512, True),
            (1, "consumer", 513, True),
            (2, "consumer", 31, False),
            (2, "none", 2048, True),
        ):
            with self.subTest(ratio=ratio, role=role, k=k, has_raw=has_raw):
                x = self.fixture.make_inputs(rows=12, width=4097, ratio=ratio, heads=32)
                x.lens[:6] = torch.tensor([0, 1, 63, 511, 512, 513], device="cuda")
                case = self.fixture.make_backend_case(x, role, k)
                mask = case.state.forward_metadata.candidate_metadata.mask.clone()
                if not has_raw:
                    case.state.forward_metadata.core_metadata.sparse_raw_indices = (
                        lambda _: None
                    )
                self.run_case(case)
                self.check_outputs(x, case, role, mask, k, has_raw)

    def test_unsupported_layouts_keep_mapped_path(self):
        from sglang.kernels.ops.attention.dsv4 import sm90_length_aware_indexer

        for heads, group, k, block_size in (
            (64, 6, 512, 64),
            (32, 1, 512, 64),
            (32, 6, 2049, 64),
            (32, 6, 512, 3),
        ):
            x = self.fixture.make_inputs(rows=12, width=4096, heads=heads, group=group)
            case = self.fixture.make_backend_case(x, "source", k)
            case.layer.indexer.candidate_block_size = block_size
            with (
                envs.SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER.override(True),
                envs.SGLANG_OPT_DSV41_SM90_LENGTH_AWARE_INDEXER.override(False),
            ):
                case.run()
            pages, raw = case.pages.clone(), case.raw.clone()
            with patch.object(
                sm90_length_aware_indexer,
                "prefix_logits",
                side_effect=AssertionError("unsupported layout entered prefix path"),
            ):
                self.run_case(case)
            torch.testing.assert_close(case.pages, pages, atol=0, rtol=0)
            torch.testing.assert_close(case.raw, raw, atol=0, rtol=0)

    def test_candidate_handoff_to_both_consumers(self):
        x = self.fixture.make_inputs(rows=12, width=32769, heads=32, ratio=1)
        x.lens.copy_((8192 - torch.arange(12, device="cuda") % 6).clamp_min(0))
        source = self.fixture.make_backend_case(x, "source", 512)
        source.layer.indexer.candidate_block_size = 8
        source.layer.indexer.candidate_topk_blocks = 2048
        self.run_case(source)
        mask = source.state.forward_metadata.candidate_metadata.mask
        # At 8K all reachable blocks fit the budget. Full-width mask publication
        # also permits a subsequent layer to fall back to the mapped path.
        self.check_outputs(x, source, "source")
        for optimized in (False, True):
            consumer = self.fixture.make_backend_case(x, "consumer", 512)
            consumer.state.forward_metadata.candidate_metadata.mask = mask
            with (
                envs.SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER.override(True),
                envs.SGLANG_OPT_DSV41_SM90_LENGTH_AWARE_INDEXER.override(optimized),
            ):
                consumer.run()
            self.check_outputs(x, consumer, "consumer", mask)

    def test_graph_replay_grow_shrink_and_candidates(self):
        for rows, role in (
            (12, "none"),
            (48, "none"),
            (12, "source"),
            (12, "consumer"),
        ):
            with self.subTest(rows=rows, role=role):
                x = self.fixture.make_inputs(rows=rows, width=131072, heads=32)
                case = self.fixture.make_backend_case(x, role, 512)
                mask = case.state.forward_metadata.candidate_metadata.mask
                for _ in range(3):
                    self.run_case(case)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.run_case(case)
                for maximum in (8192, 0, 513, 32769, 131072, 1, 16385):
                    x.lens.copy_(
                        (maximum - torch.arange(rows, device="cuda") % 6).clamp_min(0)
                    )
                    case.pos.copy_(x.lens * x.ratio - 1)
                    x.mapping.copy_(x.mapping.roll(x.page * x.ratio, 1))
                    x.slots.copy_(x.mapping[x.req][:, :: x.ratio].long() // x.ratio)
                    if role == "consumer":
                        mask.copy_(mask.roll(1, 1))
                    graph.replay()
                    self.check_outputs(x, case, role, mask)
                graph.reset()

    def test_compact_handoff_and_fallback(self):
        # Include the 6-row / 1M-capacity shape that rejects paged TopK's
        # 16-CTA cluster on H20. Only the visible prefix is initialized.
        for rows, width, ratio in ((6, 1048896, 1), (12, 131073, 2)):
            x = self.fixture.make_inputs(rows=rows, width=width, ratio=ratio, heads=32)
            lengths = torch.tensor([0, 1, 63, 513, 8193, 32769], device="cuda")
            x.lens.copy_(lengths.repeat(rows // 6))
            source = self.fixture.make_backend_case(x, "source", 512)
            source.layer.indexer.candidate_block_size = 8
            source.layer.indexer.candidate_topk_blocks = 2048
            with envs.SGLANG_OPT_DSV41_SM90_COMPACT_CANDIDATES.override(True):
                self.run_case(source)
            candidates = source.state.forward_metadata.candidate_metadata
            self.assertIsInstance(candidates, CandidateBlocks)
            mask = published_masks(candidates).mask
            self.check_outputs(x, source, "source")
            for optimized, compact in ((True, True), (True, False), (False, False)):
                consumer = self.fixture.make_backend_case(x, "consumer", 512)
                consumer.state.forward_metadata.candidate_metadata = candidates
                with (
                    envs.SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER.override(True),
                    envs.SGLANG_OPT_DSV41_SM90_LENGTH_AWARE_INDEXER.override(optimized),
                    envs.SGLANG_OPT_DSV41_SM90_COMPACT_CANDIDATES.override(compact),
                ):
                    consumer.run()
                self.check_outputs(x, consumer, "consumer", mask)

    def test_compact_blocks_partial_and_sparse(self):
        from sglang.kernels.ops.attention.dsv4.sm90_length_aware_indexer import (
            candidate_blocks,
        )

        # Non-power-of-two K, an incomplete final block, finite negative scores,
        # holes, and empty rows must preserve the original full-mask semantics.
        for block_size, block_k in ((1, 3), (8, 3), (128, 3), (1, 2048), (8, 2048)):
            width = 2051
            lens = torch.tensor(
                [0, 1, 65, 129, 1025, width], device="cuda", dtype=torch.int32
            )
            positions = torch.arange(width, device="cuda")
            scores = -((positions % 31) + 1).float().expand(6, -1).clone()
            scores[:, 32:64] = -torch.inf
            scores.masked_fill_(positions[None, :] >= lens[:, None], torch.nan)
            candidates = candidate_blocks(scores, lens, width, block_k, block_size)
            mask = published_masks(candidates).mask
            reference = scores.masked_fill(
                positions[None, :] >= lens[:, None], -torch.inf
            )
            blocks = (
                F.pad(reference, (0, -width % block_size), value=-torch.inf)
                .unflatten(1, (-1, block_size))
                .amax(-1)
            )
            last = (lens - 1) // block_size
            blocks.masked_fill_(
                torch.arange(blocks.shape[1], device="cuda")[None, :] == last[:, None],
                torch.inf,
            )
            for row in range(6):
                got = blocks[row][mask[row, ::block_size]].sort(descending=True).values
                want = blocks[row].topk(min(block_k, blocks.shape[1])).values
                torch.testing.assert_close(got, want[want > -torch.inf], atol=0, rtol=0)

    def test_compact_graph_replay_changes_candidate_layout(self):
        x = self.fixture.make_inputs(rows=6, width=131073, heads=32, ratio=2)
        x.lens.copy_(8192 - torch.arange(6, device="cuda"))
        source = self.fixture.make_backend_case(x, "source", 512)
        consumer = self.fixture.make_backend_case(x, "consumer", 512)
        fallback = self.fixture.make_backend_case(x, "consumer", 512)
        source.layer.indexer.candidate_block_size = 8
        source.layer.indexer.candidate_topk_blocks = 2048
        for case in (source, consumer, fallback):
            case.pos.copy_(x.lens * x.ratio - 1)

        def run():
            with envs.SGLANG_OPT_DSV41_SM90_COMPACT_CANDIDATES.override(True):
                self.run_case(source)
                candidates = source.state.forward_metadata.candidate_metadata
                consumer.state.forward_metadata.candidate_metadata = candidates
                fallback.state.forward_metadata.candidate_metadata = candidates
                self.run_case(consumer)
            with envs.SGLANG_OPT_DSV41_SM90_COMPACT_CANDIDATES.override(False):
                self.run_case(fallback)

        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for maximum in (32769, 0, 1, 8192, 131073, 16385, 513):
            x.lens.copy_((maximum - torch.arange(6, device="cuda")).clamp_min(0))
            for case in (source, consumer, fallback):
                case.pos.copy_(x.lens * x.ratio - 1)
            x.mapping.copy_(x.mapping.roll(x.page * x.ratio, 1))
            x.slots.copy_(x.mapping[x.req][:, :: x.ratio].long() // x.ratio)
            graph.replay()
            mask = published_masks(
                source.state.forward_metadata.candidate_metadata
            ).mask
            self.check_outputs(x, source, "source")
            self.check_outputs(x, consumer, "consumer", mask)
            self.check_outputs(x, fallback, "consumer", mask)
        graph.reset()

    def test_compact_sparse_underfill_without_raw_output(self):
        from sglang.kernels.ops.attention.dsv4.sm90_length_aware_indexer import (
            prepare_candidate_lengths,
        )

        x = self.fixture.make_inputs(rows=6, width=4097, heads=32, ratio=2)
        x.lens.copy_(torch.tensor([0, 1, 9, 255, 4096, 4097], device="cuda"))
        candidates = CandidateBlocks(
            blocks=torch.tensor(
                [[3, 40, 512]], device="cuda", dtype=torch.int32
            ).repeat(6, 1),
            lengths=torch.full((6,), 24, device="cuda", dtype=torch.int32),
            is_prefix=torch.zeros(6, device="cuda", dtype=torch.int32),
            width=4097,
            block_size=8,
        )
        mask = published_masks(candidates).mask
        pos = torch.stack((x.lens * 2 - 1, torch.full_like(x.lens, 999999)), dim=1)
        visible, compact_lengths = prepare_candidate_lengths(
            pos[:, 0], 2, 4097, candidates
        )
        torch.testing.assert_close(visible.long(), x.lens, atol=0, rtol=0)
        torch.testing.assert_close(
            compact_lengths,
            torch.tensor([0, 1, 9, 24, 24, 24], device="cuda", dtype=torch.int32),
            atol=0,
            rtol=0,
        )
        case = self.fixture.make_backend_case(x, "consumer", 31)
        case.state.forward_metadata.candidate_metadata = candidates
        case.state.forward_metadata.core_metadata.sparse_raw_indices = lambda _: None
        with envs.SGLANG_OPT_DSV41_SM90_COMPACT_CANDIDATES.override(True):
            self.run_case(case)
        self.check_outputs(x, case, "consumer", mask, k=31, has_raw=False)


if __name__ == "__main__":
    unittest.main()
