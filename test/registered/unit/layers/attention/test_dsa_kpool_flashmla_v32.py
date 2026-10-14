import logging
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_backend_kpool import (
    DeepseekSparseAttnBackendKPoolMixin,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeSchedMeta:
    pass


def _backend(**overrides):
    values = dict(
        dsa_index_kpool=4,
        device_sm_major=9,
        dsa_kv_cache_store_fp8=True,
        qk_rope_head_dim=0,
        flashmla_kv_num_q_heads=2,
        physical_page_size=64,
        kv_cache_dim=528,
        dsa_index_topk=2048,
        flashmla_kv_format="V32_NO_ROPE",
    )
    values.update(overrides)
    backend = SimpleNamespace(**values)
    backend._use_kpool_flashmla_v32 = lambda dsa_impl: (
        DeepseekSparseAttnBackendKPoolMixin._use_kpool_flashmla_v32(backend, dsa_impl)
    )
    return backend


class TestDSAKPoolFlashMLAV32(unittest.TestCase):
    def test_query6_preserves_rows_output_and_scheduler_ownership(self):
        captured = []
        flashmla = ModuleType("sgl_kernel.flash_mla")
        flashmla.FlashMLASchedMeta = _FakeSchedMeta

        def run(**kwargs):
            captured.append(kwargs)
            return kwargs["q"].clone(), None

        flashmla.flash_mla_with_kvcache = run
        sgl_kernel = ModuleType("sgl_kernel")
        sgl_kernel.flash_mla = flashmla
        backend = _backend()
        metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=torch.full((96,), 2048, dtype=torch.int32),
            flashmla_metadata=SimpleNamespace(),
            kpool_flashmla_sched_meta={},
        )
        q = torch.arange(96 * 2 * 512).reshape(96, 2, 512).float()
        kv = torch.empty((64, 64, 1, 528), dtype=torch.float8_e4m3fn)
        indices = torch.arange(96 * 2051, dtype=torch.int32).view(96, 2051)
        with (
            patch.dict(
                "sys.modules",
                {"sgl_kernel": sgl_kernel, "sgl_kernel.flash_mla": flashmla},
            ),
            patch.object(
                envs.SGLANG_EXPERIMENTAL_DSA_KPOOL_FLASHMLA_V32,
                "get",
                return_value=True,
            ),
            patch.object(
                envs.SGLANG_EXPERIMENTAL_DSA_KPOOL_FUSED_SENTINEL_PAD,
                "get",
                return_value=False,
            ),
        ):
            for enabled, stream, query_len in [
                (False, 7, 6),
                (True, 7, 6),
                (True, 8, 6),
                (True, 7, 1),
            ]:
                with (
                    patch.object(
                        envs.SGLANG_EXPERIMENTAL_DSA_KPOOL_MTP_QUERY6,
                        "get",
                        return_value=enabled,
                    ),
                    patch(
                        "torch.cuda.current_stream",
                        return_value=SimpleNamespace(cuda_stream=stream),
                    ),
                ):
                    output = DeepseekSparseAttnBackend._forward_flashmla_kv(
                        backend,
                        q,
                        kv,
                        512,
                        1.0,
                        SimpleNamespace(tp_q_head_num=2, head_dim=512),
                        metadata,
                        indices,
                        query_len=query_len,
                    )
                    self.assertEqual(output.shape, (96, 1, 2, 512))
                    self.assertTrue(torch.equal(output[:, 0], q))
                    self.assertTrue(
                        torch.equal(
                            captured[-1]["indices"].reshape(96, -1)[:, :2051], indices
                        )
                    )
                    self.assertTrue(
                        torch.all(captured[-1]["indices"][..., 2051:] == -1)
                    )
        self.assertEqual(captured[0]["q"].shape, (96, 1, 2, 512))
        self.assertEqual(captured[1]["q"].shape, (16, 6, 2, 512))
        self.assertEqual(len(metadata.kpool_flashmla_sched_meta), 3)
        self.assertIs(
            captured[0]["tile_scheduler_metadata"],
            captured[3]["tile_scheduler_metadata"],
        )

    def test_router_requires_explicit_opt_in(self):
        backend = _backend()
        with patch.object(
            envs.SGLANG_EXPERIMENTAL_DSA_KPOOL_FLASHMLA_V32,
            "get",
            return_value=False,
        ):
            self.assertEqual(
                DeepseekSparseAttnBackendKPoolMixin._resolve_kpool_tail_backend(
                    backend, torch.zeros((1, 2051), dtype=torch.int32), "flashmla_kv"
                ),
                "tilelang",
            )

    def test_router_accepts_only_sm90_fp8_nope(self):
        with patch.object(
            envs.SGLANG_EXPERIMENTAL_DSA_KPOOL_FLASHMLA_V32,
            "get",
            return_value=True,
        ):
            for overrides, expected in (
                ({}, "flashmla_kv"),
                ({"device_sm_major": 10}, "trtllm"),
                ({"dsa_kv_cache_store_fp8": False}, "flashmla_kv"),
                ({"qk_rope_head_dim": 64}, "flashmla_kv"),
            ):
                with self.subTest(overrides=overrides):
                    backend = _backend(**overrides)
                    self.assertEqual(
                        DeepseekSparseAttnBackendKPoolMixin._resolve_kpool_tail_backend(
                            backend,
                            torch.zeros((1, 2051), dtype=torch.int32),
                            "flashmla_kv",
                        ),
                        expected,
                    )
                    if expected == "flashmla_kv":
                        if not overrides:
                            DeepseekSparseAttnBackendKPoolMixin._check_kpool_tail_backend(
                                backend,
                                torch.zeros((1, 2051), dtype=torch.int32),
                                expected,
                                "decode",
                            )
                        else:
                            with self.assertRaises(NotImplementedError):
                                DeepseekSparseAttnBackendKPoolMixin._check_kpool_tail_backend(
                                    backend,
                                    torch.zeros((1, 2051), dtype=torch.int32),
                                    expected,
                                    "decode",
                                )

    def test_native_path_pads_sentinel_and_reuses_scheduler(self):
        captured = []
        flashmla = ModuleType("sgl_kernel.flash_mla")
        flashmla.FlashMLASchedMeta = _FakeSchedMeta

        def fake_flash_mla_with_kvcache(**kwargs):
            captured.append(kwargs)
            rows, _, heads, _ = kwargs["q"].shape
            return torch.ones((rows, 1, heads, 512)), None

        flashmla.flash_mla_with_kvcache = fake_flash_mla_with_kvcache
        sgl_kernel = ModuleType("sgl_kernel")
        sgl_kernel.flash_mla = flashmla
        backend = _backend()
        backend._kpool_flashmla_v32_dispatch_logged = False
        metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=torch.full((96,), 2048, dtype=torch.int32),
            flashmla_metadata=SimpleNamespace(
                flashmla_metadata=torch.empty((1,), dtype=torch.int32),
                num_splits=torch.empty((97,), dtype=torch.int32),
            ),
            kpool_flashmla_sched_meta={},
        )
        q = torch.empty((96, 2, 512))
        kv = torch.empty((64, 64, 1, 528), dtype=torch.float8_e4m3fn)
        indices = torch.arange(96 * 2051, dtype=torch.int32).view(96, 2051)

        with (
            self.assertLogs(
                "sglang.srt.layers.attention.dsa_backend", level=logging.INFO
            ) as logs,
            patch.dict(
                "sys.modules",
                {"sgl_kernel": sgl_kernel, "sgl_kernel.flash_mla": flashmla},
            ),
            patch.object(
                envs.SGLANG_EXPERIMENTAL_DSA_KPOOL_FLASHMLA_V32,
                "get",
                return_value=True,
            ),
            patch(
                "torch.cuda.current_stream", return_value=SimpleNamespace(cuda_stream=7)
            ),
        ):
            for _ in range(2):
                output = DeepseekSparseAttnBackend._forward_flashmla_kv(
                    backend,
                    q_all=q,
                    kv_cache=kv,
                    v_head_dim=512,
                    sm_scale=1.0,
                    layer=SimpleNamespace(tp_q_head_num=2, head_dim=512),
                    metadata=metadata,
                    page_table_1=indices,
                )

        activation_logs = [
            message
            for message in logs.output
            if "Activated experimental DSA KPool native FlashMLA route" in message
        ]
        self.assertEqual(len(activation_logs), 1)
        self.assertIn("query_heads=2 padded_query_heads=2", activation_logs[0])
        self.assertIn("source_width=2051 padded_width=2112", activation_logs[0])
        self.assertEqual(output.shape, (96, 1, 2, 512))
        self.assertEqual(captured[0]["indices"].shape, (96, 1, 2112))
        self.assertTrue(
            torch.equal(captured[0]["indices"][..., :2051], indices[:, None])
        )
        self.assertTrue(torch.all(captured[0]["indices"][..., 2051:] == -1))
        self.assertIsNone(
            captured[0]["topk_length"] if "topk_length" in captured[0] else None
        )
        self.assertIsNone(captured[0]["block_table"])
        self.assertIsNone(captured[0]["cache_seqlens"])
        self.assertIsNone(captured[0]["num_splits"])
        self.assertEqual(captured[0]["kv_format"], "V32_NO_ROPE")
        self.assertIs(
            captured[0]["tile_scheduler_metadata"],
            captured[1]["tile_scheduler_metadata"],
        )
        self.assertEqual(len(metadata.kpool_flashmla_sched_meta), 1)

    def test_non_kpool_path_keeps_legacy_metadata(self):
        captured = {}
        flashmla = ModuleType("sgl_kernel.flash_mla")
        flashmla.FlashMLASchedMeta = _FakeSchedMeta

        def fake_flash_mla_with_kvcache(**kwargs):
            captured.update(kwargs)
            return torch.ones((1, 1, 2, 512)), None

        flashmla.flash_mla_with_kvcache = fake_flash_mla_with_kvcache
        sgl_kernel = ModuleType("sgl_kernel")
        sgl_kernel.flash_mla = flashmla
        backend = _backend(dsa_index_kpool=1)
        legacy_meta = torch.empty((1,), dtype=torch.int32)
        legacy_splits = torch.empty((2,), dtype=torch.int32)
        metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=torch.tensor([2048], dtype=torch.int32),
            flashmla_metadata=SimpleNamespace(
                flashmla_metadata=legacy_meta,
                num_splits=legacy_splits,
            ),
            kpool_flashmla_sched_meta={},
        )

        with (
            patch.dict(
                "sys.modules",
                {"sgl_kernel": sgl_kernel, "sgl_kernel.flash_mla": flashmla},
            ),
            patch.object(
                envs.SGLANG_EXPERIMENTAL_DSA_KPOOL_FLASHMLA_V32,
                "get",
                return_value=True,
            ),
        ):
            DeepseekSparseAttnBackend._forward_flashmla_kv(
                backend,
                q_all=torch.empty((1, 2, 512)),
                kv_cache=torch.empty((64, 64, 1, 528), dtype=torch.float8_e4m3fn),
                v_head_dim=512,
                sm_scale=1.0,
                layer=SimpleNamespace(tp_q_head_num=2, head_dim=512),
                metadata=metadata,
                page_table_1=torch.zeros((1, 2048), dtype=torch.int32),
            )

        self.assertIs(captured["tile_scheduler_metadata"], legacy_meta)
        self.assertIs(captured["num_splits"], legacy_splits)
        self.assertIs(captured["cache_seqlens"], metadata.dsa_cache_seqlens_int32)
        self.assertEqual(captured["block_table"].shape, (1, 0))


if __name__ == "__main__":
    unittest.main()
