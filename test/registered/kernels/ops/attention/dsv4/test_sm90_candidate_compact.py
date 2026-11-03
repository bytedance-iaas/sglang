"""Exact logits and native TopK parity for SM90 compaction and K reuse."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_logits_decode
from sglang.kernels.ops.attention.dsv4.sm90_candidate_indexer import (
    compact_candidate_logits,
    paired_query_logits,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    select_candidate_blocks,
    sm90_index_logits_with_candidates,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.sm90_candidate_indexer import (
    make_candidate_case,
    select_candidate_topk,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def operands(case):
    return {k: case[k] for k in ("q", "weights", "slots", "lens", "table", "page_size")}


def blocks_from_mask(mask, block_size=8):
    """Test-only packing; production reuses the source's original block TopK."""
    rows, width = mask.shape
    keep = torch.nn.functional.pad(mask, (0, -width % block_size))
    keep = keep.reshape(rows, -1, block_size).any(-1)
    ids = torch.arange(keep.shape[1], device=mask.device, dtype=torch.int32)
    return ids.expand_as(keep).masked_fill(~keep, -1)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestSm90CandidateCompact(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if torch.version.hip is not None or torch.cuda.get_device_capability()[0] != 9:
            raise unittest.SkipTest("requires SM90")

    def assert_scores(self, actual, expected, case):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertTrue(
            torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        )
        if actual.shape[0] and actual.shape[1]:
            for new, old in zip(
                select_candidate_topk(actual, case["slots"], case["lens"], 512),
                select_candidate_topk(expected, case["slots"], case["lens"], 512),
            ):
                self.assertTrue(torch.equal(new, old))

    def test_compact_boundaries_strides_underfill_and_ties(self):
        for width in (1, 7, 8, 9, 63, 64, 65, 129, 513):
            for pattern in ("none", "all", "scattered"):
                for ties in (False, True):
                    with self.subTest(width=width, pattern=pattern, ties=ties):
                        case = make_candidate_case(
                            4, width, pattern=pattern, mixed_lengths=True, seed=width
                        )
                        if ties:
                            case["weights"].zero_()
                        mask = case["candidate_mask"]
                        ids = blocks_from_mask(mask)
                        # The kernel must accept block lists and masks with both strides.
                        ids_storage = torch.empty(
                            (8, ids.shape[1] * 2), device="cuda", dtype=torch.int32
                        )
                        blocks = ids_storage[::2, ::2]
                        blocks.copy_(ids)
                        mask_storage = torch.empty(
                            (8, width * 2), device="cuda", dtype=torch.bool
                        )
                        mask_view = mask_storage[::2, ::2]
                        mask_view.copy_(mask)
                        expected = fp4_index_logits_decode(
                            **operands(case)
                        ).masked_fill(~mask, -torch.inf)
                        actual = compact_candidate_logits(
                            **operands(case), blocks=blocks, candidate_mask=mask_view
                        )
                        self.assert_scores(actual, expected, case)

    def test_compact_poisoned_slots_and_block_ids(self):
        case = make_candidate_case(4, 193, pattern="clustered", mixed_lengths=True)
        mask = case["candidate_mask"]
        expected = fp4_index_logits_decode(**operands(case)).masked_fill(
            ~mask, -torch.inf
        )
        blocks = blocks_from_mask(mask)
        # Invalid/padded block ids must not dereference either slots or index-K.
        blocks[blocks < 0] = 1 << 28
        case["slots"].masked_fill_(
            ~mask, case["table"].shape[0] * case["page_size"] + 4096
        )
        actual = compact_candidate_logits(
            **operands(case), blocks=blocks, candidate_mask=mask
        )
        self.assert_scores(actual, expected, case)

    def test_compact_source_publication_full_sizes(self):
        for rows in (32, 64):
            with self.subTest(rows=rows):
                case = make_candidate_case(rows, 65536, candidate_blocks=2048)
                source = fp4_index_logits_decode(**operands(case))
                original = select_candidate_blocks(
                    source, case["lens"][:, None], 2048, 8
                )
                mask, blocks = select_candidate_blocks(
                    source, case["lens"][:, None], 2048, 8, return_block_indices=True
                )
                self.assertTrue(torch.equal(mask, original))
                actual = compact_candidate_logits(
                    **operands(case), blocks=blocks, candidate_mask=mask
                )
                self.assert_scores(actual, source.masked_fill(~mask, -torch.inf), case)

    def test_paired_request_boundaries_slot_conflicts_and_masks(self):
        for rows in (1, 5, 6, 7):
            for masked in (False, True):
                with self.subTest(rows=rows, masked=masked):
                    case = make_candidate_case(rows, 257, mixed_lengths=True)
                    req = torch.arange(rows, device="cuda", dtype=torch.int64) // 2
                    # Pair 0 shares a map; pair 1 deliberately has conflicting slots.
                    if rows > 1:
                        case["slots"][1].copy_(case["slots"][0])
                    if rows > 5:
                        case["slots"][5].copy_(case["slots"][4])
                        req[5] = req[4] + 1  # same slots, different request ids
                    mask = case["candidate_mask"] if masked else None
                    expected = fp4_index_logits_decode(
                        **operands(case), candidate_mask=mask
                    )
                    actual = paired_query_logits(
                        **operands(case), request_ids=req, candidate_mask=mask
                    )
                    self.assert_scores(actual, expected, case)

    def test_paired_full_sizes_preserve_bf16_rounding(self):
        # Small cases can miss rare BF16 rounding-boundary changes when the MMA
        # or the head reduction is regrouped. Exercise millions of real scores.
        for rows in (32, 64):
            for width in (65536, 131072):
                case = make_candidate_case(rows, width, candidate_blocks=2048)
                case["slots"] = case["slots"][::4].repeat_interleave(4, 0)
                req = torch.arange(rows, device="cuda", dtype=torch.int64) // 4
                for masked in (False, True):
                    with self.subTest(rows=rows, width=width, masked=masked):
                        mask = case["candidate_mask"] if masked else None
                        expected = fp4_index_logits_decode(
                            **operands(case), candidate_mask=mask
                        )
                        actual = paired_query_logits(
                            **operands(case), request_ids=req, candidate_mask=mask
                        )
                        self.assert_scores(actual, expected, case)

    def test_paired_disjoint_validity_and_poisoned_slots(self):
        # A shared K tile can contain columns visible to only one query. Invalid
        # slots must remain unread in the shared path and the odd-row fallback.
        for masked in (False, True):
            with self.subTest(masked=masked):
                case = make_candidate_case(5, 193)
                case["slots"][1::2].copy_(case["slots"][:4:2])
                case["lens"].copy_(torch.tensor([0, 65, 129, 193, 193], device="cuda"))
                cols = torch.arange(193, device="cuda")
                mask = cols[None, :] % 2 == torch.arange(5, device="cuda")[:, None] % 2
                mask = mask if masked else None
                expected = fp4_index_logits_decode(
                    **operands(case), candidate_mask=mask
                )
                valid = cols[None, :] < case["lens"][:, None]
                if masked:
                    valid &= mask
                case["slots"].masked_fill_(
                    ~valid, case["table"].shape[0] * case["page_size"] + 4096
                )
                actual = paired_query_logits(
                    **operands(case),
                    request_ids=torch.arange(5, device="cuda") // 2,
                    candidate_mask=mask,
                )
                self.assert_scores(actual, expected, case)

    def test_source_consumer_pipeline_and_compression(self):
        for ratio in (1, 2):
            for compact, reuse in ((True, False), (False, True), (True, True)):
                with self.subTest(ratio=ratio, compact=compact, reuse=reuse):
                    source = make_candidate_case(6, 513, seed=10)
                    consumer = make_candidate_case(6, 513, seed=11)
                    req = torch.tensor([2, 2, 0, 0, 0, 1], device="cuda")
                    pos = torch.tensor([0, 1, 62, 63, 64, 512], device="cuda")
                    mapping = (source["slots"][:3] * ratio).repeat_interleave(ratio, 1)
                    cols = torch.arange(513, device="cuda") * ratio
                    slots = mapping[req[:, None], cols[None, :]] // ratio
                    lens = (pos + 1) // ratio
                    indexer = SimpleNamespace(
                        candidate_topk_blocks=4, candidate_block_size=8
                    )
                    results = []
                    for enabled in (False, True):
                        steps = []
                        metadata = None
                        with (
                            envs.SGLANG_DSV41_SM90_CANDIDATE_MASK.override(enabled),
                            envs.SGLANG_DSV41_SM90_CANDIDATE_MASK_MIN_WIDTH.override(0),
                            envs.SGLANG_DSV41_SM90_CANDIDATE_COMPACT.override(
                                enabled and compact
                            ),
                            envs.SGLANG_DSV41_SM90_QUERY_K_REUSE.override(
                                enabled and reuse
                            ),
                        ):
                            for is_source, case in ((True, source), (False, consumer)):
                                indexer.is_candidate_source = is_source
                                indexer.uses_candidates = not is_source
                                values = dict(operands(case), slots=slots, lens=lens)
                                scores, metadata = sm90_index_logits_with_candidates(
                                    **values,
                                    indexer=indexer,
                                    candidate_metadata=metadata,
                                    request_ids=req,
                                )
                                raw, physical = select_candidate_topk(
                                    scores,
                                    slots,
                                    lens,
                                    512,
                                    filter_candidates=not is_source,
                                )
                                steps.append((scores, raw, physical))
                            steps.append((metadata.mask,))
                        results.append(steps)
                    for new_step, old_step in zip(results[1], results[0]):
                        for new, old in zip(new_step, old_step):
                            if new.dtype == torch.float32:
                                self.assertTrue(
                                    torch.equal(
                                        new.view(torch.int32), old.view(torch.int32)
                                    )
                                )
                            else:
                                self.assertTrue(torch.equal(new, old))

    def test_graph_replay_refreshes_compact_and_pair_inputs(self):
        case = make_candidate_case(6, 513, mixed_lengths=True)
        req = torch.arange(6, device="cuda", dtype=torch.int64) // 2
        blocks = blocks_from_mask(case["candidate_mask"])

        def run():
            return (
                compact_candidate_logits(
                    **operands(case),
                    blocks=blocks,
                    candidate_mask=case["candidate_mask"],
                ),
                paired_query_logits(
                    **operands(case),
                    request_ids=req,
                    candidate_mask=case["candidate_mask"],
                ),
            )

        for _ in range(3):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        for seed, pattern in enumerate(("none", "all", "scattered", "clustered"), 1):
            updated = make_candidate_case(
                6, 513, seed=seed, pattern=pattern, mixed_lengths=True
            )
            for name in ("q", "weights", "slots", "lens", "table", "candidate_mask"):
                case[name].copy_(updated[name])
            if seed % 2:
                case["slots"][1::2].copy_(case["slots"][::2])
                req.copy_(torch.arange(6, device="cuda") // 2)
            else:
                req.copy_(torch.arange(6, device="cuda"))
            case["lens"].copy_(updated["lens"].flip(0))
            blocks.copy_(blocks_from_mask(case["candidate_mask"]))
            graph.replay()
            expected = fp4_index_logits_decode(
                **operands(case), candidate_mask=case["candidate_mask"]
            )
            for output in actual:
                self.assert_scores(output, expected, case)

    def test_compact_width_policy_dispatch(self):
        from sglang.kernels.ops.attention.dsv4 import sm90_candidate_indexer as kernels

        for width in (127, 128, 129):
            with self.subTest(width=width):
                case = make_candidate_case(4, width, mixed_lengths=True)
                indexer = SimpleNamespace(
                    is_candidate_source=True,
                    uses_candidates=False,
                    candidate_topk_blocks=4,
                    candidate_block_size=8,
                )
                with (
                    envs.SGLANG_DSV41_SM90_CANDIDATE_MASK.override(True),
                    envs.SGLANG_DSV41_SM90_CANDIDATE_MASK_MIN_WIDTH.override(128),
                    envs.SGLANG_DSV41_SM90_CANDIDATE_COMPACT.override(True),
                    envs.SGLANG_DSV41_SM90_QUERY_K_REUSE.override(False),
                ):
                    _, state = sm90_index_logits_with_candidates(
                        **operands(case), indexer=indexer, candidate_metadata=None
                    )
                    indexer.is_candidate_source = False
                    indexer.uses_candidates = True
                    with patch.object(
                        kernels,
                        "compact_candidate_logits",
                        wraps=kernels.compact_candidate_logits,
                    ) as called:
                        actual, _ = sm90_index_logits_with_candidates(
                            **operands(case), indexer=indexer, candidate_metadata=state
                        )
                    self.assertEqual(called.call_count, int(width >= 128))
                    expected = fp4_index_logits_decode(**operands(case)).masked_fill(
                        ~state.mask, -torch.inf
                    )
                    self.assert_scores(actual, expected, case)

    def test_source_publication_refreshes_on_graph_replay(self):
        source = make_candidate_case(6, 513, mixed_lengths=True, seed=30)
        consumer = make_candidate_case(6, 513, mixed_lengths=True, seed=31)
        req = torch.arange(6, device="cuda", dtype=torch.int64) // 2

        def run():
            state = None
            result = []
            for is_source, case in ((True, source), (False, consumer)):
                indexer = SimpleNamespace(
                    is_candidate_source=is_source,
                    uses_candidates=not is_source,
                    candidate_topk_blocks=4,
                    candidate_block_size=8,
                )
                values = dict(
                    operands(case), slots=source["slots"], lens=source["lens"]
                )
                logits, state = sm90_index_logits_with_candidates(
                    **values,
                    indexer=indexer,
                    candidate_metadata=state,
                    request_ids=req,
                )
                result.append(
                    (
                        logits,
                        *select_candidate_topk(
                            logits,
                            source["slots"],
                            source["lens"],
                            512,
                            filter_candidates=not is_source,
                        ),
                    )
                )
            result.append((state.mask,))
            return result

        with (
            envs.SGLANG_DSV41_SM90_CANDIDATE_MASK.override(True),
            envs.SGLANG_DSV41_SM90_CANDIDATE_MASK_MIN_WIDTH.override(0),
            envs.SGLANG_DSV41_SM90_CANDIDATE_COMPACT.override(True),
            envs.SGLANG_DSV41_SM90_QUERY_K_REUSE.override(True),
        ):
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = run()
            for seed in (32, 33, 34):
                for case in (source, consumer):
                    update = make_candidate_case(6, 513, seed=seed, mixed_lengths=True)
                    for name in ("q", "weights", "slots", "lens", "table"):
                        case[name].copy_(update[name])
                source["slots"][1::2].copy_(source["slots"][::2])
                if seed == 33:
                    source["weights"].zero_()  # source block cutoff ties on replay
                graph.replay()
                with (
                    envs.SGLANG_DSV41_SM90_CANDIDATE_COMPACT.override(False),
                    envs.SGLANG_DSV41_SM90_QUERY_K_REUSE.override(False),
                ):
                    expected = run()
                for new_step, old_step in zip(actual, expected):
                    for new, old in zip(new_step, old_step):
                        if new.dtype == torch.float32:
                            self.assertTrue(
                                torch.equal(
                                    new.view(torch.int32), old.view(torch.int32)
                                )
                            )
                        else:
                            self.assertTrue(torch.equal(new, old))

    def test_backend_native_topk_and_physical_outputs(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        for ratio in (1, 2):
            case = make_candidate_case(6, 513, seed=20)
            consumer = make_candidate_case(6, 513, seed=21)
            req = torch.tensor([2, 2, 0, 0, 0, 1], device="cuda", dtype=torch.int64)
            pos = torch.tensor(
                [0, 1, 62, 63, 64, 512], device="cuda", dtype=torch.int64
            )
            mapping = (case["slots"][:3] * ratio).repeat_interleave(ratio, 1)
            pages = torch.empty((6, 512), device="cuda", dtype=torch.int32)
            raw = torch.empty_like(pages)
            core = SimpleNamespace(
                sparse_page_indices=lambda _: pages, sparse_raw_indices=lambda _: raw
            )
            metadata = SimpleNamespace(
                core_metadata=core,
                candidate_metadata=None,
                c1_indexer_metadata=SimpleNamespace(max_compressed_seq_len=513),
                c2_indexer_metadata=SimpleNamespace(max_compressed_seq_len=513),
            )
            backend = SimpleNamespace(
                req_to_token=mapping,
                forward_metadata=metadata,
                token_to_kv_pool=SimpleNamespace(
                    get_index_k_with_scale_buffer=lambda layer: (
                        case["table"] if layer == 0 else consumer["table"]
                    )
                ),
            )
            indexer = SimpleNamespace(
                candidate_topk_blocks=4,
                candidate_block_size=8,
                index_topk=512,
                queries=lambda q, _: q,
                head_weights=lambda x: x,
            )
            layer = SimpleNamespace(
                compress_ratio=ratio,
                indexer=indexer,
                freqs_cis=torch.zeros((513, 1), device="cuda"),
            )
            results = []
            for compact, reuse in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                steps = []
                metadata.candidate_metadata = None
                with (
                    envs.SGLANG_DSV41_SM90_CANDIDATE_MASK.override(True),
                    envs.SGLANG_DSV41_SM90_CANDIDATE_MASK_MIN_WIDTH.override(0),
                    envs.SGLANG_DSV41_SM90_CANDIDATE_COMPACT.override(compact),
                    envs.SGLANG_DSV41_SM90_QUERY_K_REUSE.override(reuse),
                ):
                    for is_source, values in ((True, case), (False, consumer)):
                        indexer.is_candidate_source = is_source
                        indexer.uses_candidates = not is_source
                        layer.layer_id = 0 if is_source else 1
                        DeepseekV4AttnBackend._low_ratio_index_topk_sm90_decode(
                            backend,
                            layer,
                            values["weights"],
                            values["q"],
                            req,
                            pos,
                            request_ids=req,
                        )
                        steps.append(
                            (
                                pages.clone(),
                                raw.clone(),
                                metadata.candidate_metadata.mask.clone(),
                            )
                        )
                results.append(steps)
            for actual in results[1:]:
                for new_step, old_step in zip(actual, results[0]):
                    for new, old in zip(new_step, old_step):
                        self.assertTrue(torch.equal(new, old))

    def test_empty_shapes(self):
        for rows, width in ((0, 65), (4, 0), (0, 0)):
            case = make_candidate_case(rows, width, pattern="none")
            blocks = torch.empty((rows, 0), device="cuda", dtype=torch.int32)
            req = torch.arange(rows, device="cuda", dtype=torch.int32)
            expected = fp4_index_logits_decode(**operands(case))
            self.assert_scores(
                compact_candidate_logits(**operands(case), blocks=blocks),
                expected,
                case,
            )
            self.assert_scores(
                paired_query_logits(**operands(case), request_ids=req), expected, case
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
