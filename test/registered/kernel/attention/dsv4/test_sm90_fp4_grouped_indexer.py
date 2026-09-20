"""Correctness and dispatch tests for the SM90 grouped FP4 indexer."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
    "requires an SM90 GPU",
)
class TestSm90Fp4GroupedIndexer(CustomTestCase):
    def make_inputs(
        self, rows=13, width=131, group=6, page=64, ratio=2, seed=11, heads=64
    ):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            store_fp4_index_k_cache,
        )
        from sglang.kernels.ops.attention.dsv4.torch_quant import fake_quant_fp4

        generator = torch.Generator(device="cuda").manual_seed(seed)
        groups = (rows + group - 1) // group
        pages_per_group = (width + page - 1) // page
        pages = groups * pages_per_group
        q = fake_quant_fp4(
            torch.randn(
                rows,
                heads,
                128,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
        )
        weights = torch.randn(
            rows, heads, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        req = torch.arange(groups, device="cuda", dtype=torch.int64).repeat_interleave(
            group
        )[:rows]
        lens = (width - torch.arange(rows, device="cuda") % group).clamp_min(0)
        physical = torch.stack(
            [
                torch.randperm(pages_per_group, device="cuda", generator=generator)
                + i * pages_per_group
                for i in range(groups)
            ]
        )
        position = torch.arange(width, device="cuda")
        slots_by_request = physical[:, position // page] * page + position % page
        mapping = torch.zeros(groups, width * ratio, device="cuda", dtype=torch.int32)
        mapping[:, ::ratio] = slots_by_request.to(torch.int32) * ratio
        table = torch.empty(pages, page * 68, device="cuda", dtype=torch.uint8)
        k = torch.randn(
            pages * page, 128, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        store_fp4_index_k_cache(
            k,
            table,
            torch.arange(k.shape[0], device="cuda", dtype=torch.int32),
            page_size=page,
            rne=True,
        )
        return SimpleNamespace(
            q=q,
            weights=weights,
            req=req,
            lens=lens,
            mapping=mapping,
            slots=slots_by_request[req].contiguous(),
            table=table,
            k=k,
            page=page,
            ratio=ratio,
            width=width,
            group=group,
        )

    def run_public(self, x):
        from sglang.kernels.ops.attention.dsv4.sm90_fp4_grouped_indexer import (
            fp4_index_logits_grouped_sm90,
        )

        return fp4_index_logits_grouped_sm90(
            x.q,
            x.weights,
            x.mapping,
            x.req,
            x.lens,
            x.table,
            x.page,
            x.ratio,
            x.width,
            x.group,
        )

    def reference(self, x):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            fp4_index_logits_decode,
        )

        return fp4_index_logits_decode(x.q, x.weights, x.slots, x.lens, x.table, x.page)

    def check(self, x):
        from sglang.kernels.ops.attention.dsv4 import (
            sm90_fp4_grouped_indexer as grouped,
        )

        expected = self.reference(x)
        for short in (False, True):
            with (
                self.subTest(short=short),
                patch.object(grouped, "_prefer_triton", return_value=short),
            ):
                actual = self.run_public(x)
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=0,
                    rtol=0,
                    equal_nan=False,
                    msg=lambda detail: f"forced_triton={short}\n{detail}",
                )
                self.assertTrue(
                    torch.equal(
                        torch.topk(actual, min(32, x.width)).indices,
                        torch.topk(expected, min(32, x.width)).indices,
                    )
                )
        torch.testing.assert_close(
            self.run_public(x), expected, atol=0, rtol=0, equal_nan=False
        )

    def test_matches_triton(self):
        for page, ratio, group in ((16, 1, 3), (16, 2, 6), (64, 1, 9), (64, 2, 6)):
            with self.subTest(page=page, ratio=ratio, group=group):
                x = self.make_inputs(
                    rows=2 * group - 1, page=page, ratio=ratio, group=group
                )
                x.lens.copy_(
                    (torch.arange(x.q.shape[0], device="cuda") * 37) % (x.width + 1)
                )
                self.check(x)

    def test_dispatch_boundaries(self):
        from sglang.kernels.ops.attention.dsv4.sm90_fp4_grouped_indexer import (
            _prefer_triton,
        )

        for rows, width in (
            (1, 32768),
            (2, 3072),
            (3, 3584),
            (4, 3072),
            (5, 4096),
            (6, 3584),
        ):
            with self.subTest(rows=rows, width=width):
                self.assertTrue(_prefer_triton(rows, width - 1))
                self.assertFalse(_prefer_triton(rows, width))
        self.assertTrue(_prefer_triton(23, 512))
        self.assertFalse(_prefer_triton(95, 512))

    def test_32_head_mapped_logits(self):
        from sglang.kernels.ops.attention.dsv4 import (
            sm90_fp4_grouped_indexer as grouped,
        )

        # Cover tile boundaries, partial groups and poisoned invisible mappings.
        # Long 32-head rows must never enter the 64-head native kernel.
        for width, page, ratio in ((257, 16, 1), (65536, 64, 2)):
            with self.subTest(width=width, page=page, ratio=ratio):
                x = self.make_inputs(width=width, page=page, ratio=ratio, heads=32)
                x.lens.copy_(
                    torch.tensor(
                        [0, 1, 63, 64, 65, 127, 0, 1, 63, 64, 65, 127, width],
                        device="cuda",
                    )
                )
                x.mapping[:2, 127 * ratio :] = 2**30
                with patch.object(
                    grouped,
                    "_sm90_fp4_grouped_indexer_op",
                    side_effect=AssertionError("32 heads entered native WGMMA"),
                ):
                    self.check(x)

    def test_multitile_paths(self):
        # Exercise TPC=2/4/8 and resident-Q reuse, including under sanitizers.
        # The short layout matrix above only covers the nonresident TPC=1 path.
        for rows, width in ((6, 16384), (23, 4096), (6, 65536)):
            for seed in (17, 23):
                with self.subTest(rows=rows, width=width, seed=seed):
                    self.check(self.make_inputs(rows=rows, width=width, seed=seed))

    def test_length_aware_schedule(self):
        # Each request group selects its schedule independently. Include a tail
        # group and poisoned mappings beyond each group's visible prefix.
        for page, ratio in ((64, 2), (16, 1)):
            with self.subTest(page=page, ratio=ratio):
                x = self.make_inputs(
                    rows=13, width=131072, page=page, ratio=ratio, seed=47
                )
                x.lens[:6] = 16384 - torch.arange(6, device="cuda")
                x.lens[6:12] = 65536 - torch.arange(6, device="cuda")
                x.lens[12] = 1311
                for request, visible in enumerate((16384, 65536, 1311)):
                    x.mapping[request, visible * ratio :] = 2**30
                self.check(x)

    def test_scale_and_weight_boundaries(self):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            store_fp4_index_k_cache,
        )

        for q_exp, k_exp, w_exp, spread in (
            (-16, 0, 0, 0),
            (-12, 0, 0, 0),
            (0, 0, -120, 0),
            (-16, 32, -132, 0),
            (-120, 64, 32, 0),
            (0, 0, 0, 4),
            (0, 0, 0, 16),
            (0, 0, 0, 32),
        ):
            with self.subTest(q_exp=q_exp, k_exp=k_exp, w_exp=w_exp, spread=spread):
                x = self.make_inputs(rows=7, seed=29)
                x.q.mul_(2.0**q_exp)
                x.weights.mul_(2.0**w_exp)
                x.k.mul_(2.0**k_exp)
                if spread:
                    x.q[:, :, :32].mul_(2.0**-spread)
                    x.k[:, 32:64].mul_(2.0**spread)
                store_fp4_index_k_cache(
                    x.k,
                    x.table,
                    torch.arange(x.k.shape[0], device="cuda", dtype=torch.int32),
                    page_size=x.page,
                    rne=True,
                )
                self.check(x)

    def test_strided_unaligned_cache_and_mapping(self):
        x = self.make_inputs()
        cache = torch.full(
            (x.table.shape[0], x.table.shape[1] + 13),
            213,
            dtype=torch.uint8,
            device="cuda",
        )
        view = cache[:, 1 : x.table.shape[1] + 1]
        view.copy_(x.table)
        x.table = view
        before = cache.clone()
        mapping = torch.zeros(
            x.mapping.shape[0], x.mapping.shape[1] + 7, dtype=torch.int32, device="cuda"
        )
        mapping[:, : x.mapping.shape[1]].copy_(x.mapping)
        x.mapping = mapping[:, : x.mapping.shape[1]]
        self.check(x)
        self.assertTrue(torch.equal(cache, before))

    def test_fallback_transition_and_invalid_mapping(self):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            store_fp4_index_k_cache,
        )

        x = self.make_inputs(width=769)
        selected = x.slots[:, 64:128].reshape(-1).unique()
        x.k[selected, :32] *= 2.0**24
        x.k[selected, 32:64] *= 2.0**-8
        store_fp4_index_k_cache(
            x.k,
            x.table,
            torch.arange(x.k.shape[0], device="cuda", dtype=torch.int32),
            page_size=x.page,
            rne=True,
        )
        x.lens.sub_(31)
        for start in range(0, x.q.shape[0], x.group):
            request = x.req[start].item()
            visible = x.lens[start : start + x.group].max().item()
            x.mapping[request, visible * x.ratio :] = 2**30
        self.check(x)

    def test_graph_replay(self):
        from sglang.kernels.ops.attention.dsv4 import (
            sm90_fp4_grouped_indexer as grouped,
        )

        for short in (False, True):
            with (
                self.subTest(short=short),
                patch.object(grouped, "_prefer_triton", return_value=short),
            ):
                x = self.make_inputs(width=1025)
                for _ in range(3):
                    self.run_public(x)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = self.run_public(x)
                for fraction in (1, 0.125, 0, 0.5, 1):
                    x.lens.copy_(
                        (
                            int(x.width * fraction)
                            - torch.arange(x.q.shape[0], device="cuda")
                        ).clamp_min(0)
                    )
                    x.mapping[:, :: x.ratio].copy_(
                        x.mapping[:, :: x.ratio].roll(3, dims=1)
                    )
                    x.slots.copy_(
                        (
                            x.mapping[
                                x.req[:, None],
                                torch.arange(x.width, device="cuda")[None, :] * x.ratio,
                            ]
                            // x.ratio
                        ).long()
                    )
                    expected = self.reference(x)
                    actual.fill_(123)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        actual, expected, atol=0, rtol=0, equal_nan=False
                    )
                graph.reset()

    def test_exact_simple_score_and_ties(self):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            store_fp4_index_k_cache,
        )

        x = self.make_inputs()
        x.q.fill_(1)
        x.weights.fill_(1)
        x.k.fill_(2)
        store_fp4_index_k_cache(
            x.k,
            x.table,
            torch.arange(x.k.shape[0], device="cuda", dtype=torch.int32),
            page_size=x.page,
            rne=True,
        )
        self.check(x)
        expected = torch.full(
            (x.q.shape[0], x.width), 128 * 2 * 64, dtype=torch.float32, device="cuda"
        )
        expected.masked_fill_(
            torch.arange(x.width, device="cuda")[None, :] >= x.lens[:, None], -torch.inf
        )
        torch.testing.assert_close(self.run_public(x), expected, atol=0, rtol=0)

    def test_empty_and_invalid_metadata(self):
        for rows, width in ((0, 65), (7, 0), (0, 0)):
            x = self.make_inputs(rows=7, width=65)
            x.q, x.weights, x.req, x.lens = (
                x.q[:rows],
                x.weights[:rows],
                x.req[:rows],
                x.lens[:rows],
            )
            x.width = width
            actual = self.run_public(x)
            self.assertEqual(actual.shape, (rows, width))
        x = self.make_inputs()
        x.weights = x.weights.cpu()
        with self.assertRaises(ValueError):
            self.run_public(x)

    def test_backend_dispatch_guards(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention import deepseek_v4_backend as backend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        option = envs.SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER", None)
            self.assertFalse(option.get())
        for enabled, mode, ragged, draft, tail, rows, spec_tokens, expected in (
            (False, ForwardMode.TARGET_VERIFY, "static", False, None, 12, 6, 1),
            (True, ForwardMode.TARGET_VERIFY, "static", False, None, 12, 6, 6),
            (True, ForwardMode.DECODE, "static", False, None, 2, 6, 1),
            (True, ForwardMode.TARGET_VERIFY, "compact", False, None, 12, 6, 1),
            (True, ForwardMode.TARGET_VERIFY, "cap-accept", False, None, 12, 6, 1),
            (True, ForwardMode.TARGET_VERIFY, "static", True, None, 12, 6, 1),
            (True, ForwardMode.TARGET_VERIFY, "static", False, object(), 12, 6, 1),
            (True, ForwardMode.TARGET_VERIFY, "static", False, None, 11, 6, 1),
            (True, ForwardMode.TARGET_VERIFY, "static", False, None, 12, 5, 1),
        ):
            with self.subTest(enabled=enabled, mode=mode, ragged=ragged, rows=rows):
                state = SimpleNamespace(
                    is_dspark_draft=draft,
                    speculative_num_draft_tokens=6,
                    forward_metadata=SimpleNamespace(late_layer_tail=tail),
                    _low_ratio_index_topk_sm90_decode=Mock(),
                )
                batch = SimpleNamespace(
                    forward_mode=mode,
                    batch_size=2,
                    spec_info=SimpleNamespace(draft_token_num=spec_tokens),
                )
                tensor = torch.empty(rows)
                with (
                    option.override(enabled),
                    envs.SGLANG_RAGGED_VERIFY_MODE.override(ragged),
                    patch.object(backend, "_is_sm100_or_newer", return_value=False),
                ):
                    backend.DeepseekV4AttnBackend._low_ratio_index_topk(
                        state, None, tensor, tensor, tensor, tensor, batch
                    )
                self.assertEqual(
                    state._low_ratio_index_topk_sm90_decode.call_args.kwargs,
                    {"group_size": expected},
                )

    def make_backend_case(self, x, role="none", topk=32):
        from sglang.srt.layers.attention import deepseek_v4_backend as backend

        rows = x.q.shape[0]
        topk = min(topk, x.width)
        pages = torch.full((rows, topk + 7), -9, dtype=torch.int32, device="cuda")
        raw = torch.empty_like(pages)
        metadata = SimpleNamespace(max_compressed_seq_len=x.width)
        candidate_mask = torch.arange(x.width, device="cuda")[None, :] % 3 != 0
        candidate_mask = candidate_mask.expand(rows, -1).clone()
        candidate_mask[0].fill_(False)
        state = SimpleNamespace(
            req_to_token=x.mapping,
            token_to_kv_pool=SimpleNamespace(
                get_index_k_with_scale_buffer=lambda _: x.table
            ),
            forward_metadata=SimpleNamespace(
                core_metadata=SimpleNamespace(
                    sparse_page_indices=lambda _: pages,
                    sparse_raw_indices=lambda _: raw,
                ),
                c1_indexer_metadata=metadata,
                c2_indexer_metadata=metadata,
                candidate_metadata=backend.CandidateMasks(mask=candidate_mask),
            ),
        )
        layer = SimpleNamespace(
            layer_id=0,
            compress_ratio=x.ratio,
            freqs_cis=torch.zeros(x.width * x.ratio + 1, 1, device="cuda"),
            indexer=SimpleNamespace(
                queries=lambda q_lora, _: q_lora,
                head_weights=lambda weights: weights,
                index_topk=topk,
                is_candidate_source=role == "source",
                uses_candidates=role == "consumer",
                candidate_topk_blocks=2,
                candidate_block_size=64,
            ),
        )
        pos = x.lens * x.ratio - 1

        def run(group_size=x.group):
            backend.DeepseekV4AttnBackend._low_ratio_index_topk_sm90_decode(
                state, layer, x.weights, x.q, x.req, pos, group_size=group_size
            )

        return SimpleNamespace(run=run, state=state, pages=pages, raw=raw, pos=pos)

    def test_backend_topk_and_candidate_masks(self):
        from sglang.kernels.ops.attention.dsv4 import (
            sm90_fp4_grouped_indexer as grouped,
        )
        from sglang.srt.environ import envs

        option = envs.SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER
        for width in (512, 4096):
            for ratio in (1, 2):
                for role in ("none", "source", "consumer"):
                    with self.subTest(width=width, ratio=ratio, role=role):
                        x = self.make_inputs(rows=12, width=width, ratio=ratio)
                        x.lens[0] = 0
                        case = self.make_backend_case(x, role)
                        with option.override(False):
                            case.run()
                        pages, raw = case.pages.clone(), case.raw.clone()
                        mask = (
                            case.state.forward_metadata.candidate_metadata.mask.clone()
                        )
                        with (
                            option.override(True),
                            patch.object(
                                grouped,
                                "fp4_index_logits_grouped_sm90",
                                wraps=grouped.fp4_index_logits_grouped_sm90,
                            ) as score,
                        ):
                            case.run()
                            score.assert_called_once()
                        torch.testing.assert_close(case.pages, pages, atol=0, rtol=0)
                        torch.testing.assert_close(case.raw, raw, atol=0, rtol=0)
                        torch.testing.assert_close(
                            case.state.forward_metadata.candidate_metadata.mask, mask
                        )

    def test_backend_unsupported_layout_keeps_default(self):
        from sglang.kernels.ops.attention.dsv4 import (
            sm90_fp4_grouped_indexer as grouped,
        )
        from sglang.srt.environ import envs

        for heads, group_size in ((16, 6), (32, 1), (64, 1)):
            with self.subTest(heads=heads, group_size=group_size):
                x = self.make_inputs(rows=12, width=512)
                x.q = x.q[:, :heads].contiguous()
                x.weights = x.weights[:, :heads].contiguous()
                case = self.make_backend_case(x)
                with envs.SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER.override(False):
                    case.run(group_size)
                pages, raw = case.pages.clone(), case.raw.clone()
                with (
                    envs.SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER.override(True),
                    patch.object(
                        grouped,
                        "fp4_index_logits_grouped_sm90",
                        side_effect=AssertionError("unsupported layout entered CUDA"),
                    ),
                ):
                    case.run(group_size)
                torch.testing.assert_close(case.pages, pages, atol=0, rtol=0)
                torch.testing.assert_close(case.raw, raw, atol=0, rtol=0)

    def test_backend_32_head_mapped_outputs(self):
        from sglang.kernels.ops.attention.dsv4 import (
            sm90_fp4_grouped_indexer as grouped,
        )
        from sglang.kernels.ops.attention.dsv4 import (
            sm90_fp4_topk,
        )
        from sglang.srt.environ import envs

        option = envs.SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER
        for ratio, role, topk, has_raw in (
            (1, "none", 512, True),
            (2, "source", 512, True),
            (1, "consumer", 513, True),
            (2, "consumer", 31, False),
            (2, "none", 1025, True),
        ):
            with self.subTest(ratio=ratio, role=role, topk=topk, has_raw=has_raw):
                x = self.make_inputs(rows=12, width=4096, ratio=ratio, heads=32)
                x.lens[:6] = torch.arange(6, device="cuda") * 17
                case = self.make_backend_case(x, role, topk)
                if not has_raw:
                    case.state.forward_metadata.core_metadata.sparse_raw_indices = (
                        lambda _: None
                    )
                with option.override(False):
                    case.run()
                pages = case.pages.clone()
                raw = case.raw.clone() if has_raw else None
                mask = case.state.forward_metadata.candidate_metadata.mask.clone()
                with (
                    option.override(True),
                    patch.object(
                        grouped,
                        "_sm90_fp4_grouped_indexer_op",
                        side_effect=AssertionError("32 heads entered native WGMMA"),
                    ),
                    patch.object(
                        grouped,
                        "fp4_index_logits_grouped_sm90",
                        wraps=grouped.fp4_index_logits_grouped_sm90,
                    ) as score,
                    patch.object(
                        sm90_fp4_topk,
                        "sort_map_topk",
                        wraps=sm90_fp4_topk.sort_map_topk,
                    ) as epilogue,
                ):
                    case.run()
                    score.assert_called_once()
                    self.assertEqual(epilogue.call_count, int(topk <= 1024))
                torch.testing.assert_close(case.pages, pages, atol=0, rtol=0)
                if has_raw:
                    torch.testing.assert_close(case.raw, raw, atol=0, rtol=0)
                torch.testing.assert_close(
                    case.state.forward_metadata.candidate_metadata.mask, mask
                )

    def test_backend_int32_metadata(self):
        from sglang.srt.environ import envs

        for width in (512, 4096):
            with self.subTest(width=width):
                x = self.make_inputs(rows=12, width=width)
                x.req = x.req.to(torch.int32)
                x.lens = x.lens.to(torch.int32)
                case = self.make_backend_case(x, topk=512)
                with envs.SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER.override(False):
                    case.run()
                pages, raw = case.pages.clone(), case.raw.clone()
                with envs.SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER.override(True):
                    case.run()
                torch.testing.assert_close(case.pages, pages, atol=0, rtol=0)
                torch.testing.assert_close(case.raw, raw, atol=0, rtol=0)

    def test_backend_strided_request_metadata(self):
        from sglang.srt.environ import envs

        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                x = self.make_inputs(rows=12, width=4096)
                storage = torch.full((24,), -1, dtype=dtype, device="cuda")
                storage[::2].copy_(x.req)
                x.req = storage[::2]
                case = self.make_backend_case(x, topk=513)
                with envs.SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER.override(False):
                    case.run()
                pages, raw = case.pages.clone(), case.raw.clone()
                with envs.SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER.override(True):
                    case.run()
                torch.testing.assert_close(case.pages, pages, atol=0, rtol=0)
                torch.testing.assert_close(case.raw, raw, atol=0, rtol=0)

    def test_backend_graph_replay(self):
        self.check_backend_graph_replay(4096)

    def test_backend_topk_sizes_and_optional_raw(self):
        from sglang.srt.environ import envs

        for topk in (1, 31, 513, 1024, 1025):
            for has_raw in (False, True):
                with self.subTest(topk=topk, has_raw=has_raw):
                    x = self.make_inputs(rows=12, width=4096)
                    x.lens[:6] = torch.arange(6, device="cuda") * 17
                    case = self.make_backend_case(x, topk=topk)
                    if not has_raw:
                        case.state.forward_metadata.core_metadata.sparse_raw_indices = (
                            lambda _: None
                        )
                    with envs.SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER.override(False):
                        case.run()
                    pages = case.pages.clone()
                    raw = case.raw.clone() if has_raw else None
                    with envs.SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER.override(True):
                        case.run()
                    torch.testing.assert_close(case.pages, pages, atol=0, rtol=0)
                    if has_raw:
                        torch.testing.assert_close(case.raw, raw, atol=0, rtol=0)

    def test_backend_long_graph_replay(self):
        self.check_backend_graph_replay(131072)

    def test_backend_32_head_graph_replay(self):
        self.check_backend_graph_replay(131072, heads=32)

    def check_backend_graph_replay(self, width, heads=64):
        from sglang.srt.environ import envs

        option = envs.SGLANG_OPT_DSV41_SM90_GROUPED_INDEXER
        x = self.make_inputs(rows=12, width=width, heads=heads)
        case = self.make_backend_case(x, topk=512)
        with option.override(True):
            for _ in range(3):
                case.run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                case.run()
        try:
            for fraction in (1, 0.125, 0, 0.5, 1):
                x.lens.copy_(
                    (
                        int(x.width * fraction) - torch.arange(12, device="cuda")
                    ).clamp_min(0)
                )
                case.pos.copy_(x.lens * x.ratio - 1)
                x.mapping[:, :: x.ratio].copy_(x.mapping[:, :: x.ratio].roll(3, dims=1))
                with option.override(False):
                    case.run()
                pages, raw = case.pages.clone(), case.raw.clone()
                case.pages.fill_(123)
                case.raw.fill_(123)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(case.pages, pages, atol=0, rtol=0)
                torch.testing.assert_close(case.raw, raw, atol=0, rtol=0)
        finally:
            graph.reset()


if __name__ == "__main__":
    unittest.main()
