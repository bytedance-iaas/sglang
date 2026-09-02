import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
)
from sglang.srt.mem_cache.kvbit_dsv4 import (
    DSV4_BU4_LAYOUT,
    DSV4KVBitPackedSWAPool,
    DSV4KVBitRuntimeCapability,
    decode_dsv4_bu4_reference,
    dsv4_kvbit_enabled_for_worker,
    dsv4_kvbit_flashmla_packed_kwargs,
    dsv4_kvbit_sparse_decode,
    encode_dsv4_bu4_reference,
    fold_dsv4_h256,
    merge_attention_states_natural_log,
    require_dsv4_kvbit_runtime_capability,
    restore_dsv4_h256,
    validate_dsv4_bu4_geometry,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestDSV4KVBitLayout(CustomTestCase):
    """Guard the vendor-defined BU4 byte ABI and its CPU reference codec."""

    def test_layout_is_exactly_380_bytes(self):
        self.assertEqual(
            DSV4_BU4_LAYOUT.offsets(),
            {
                "codes": (0, 224),
                "header": (224, 252),
                "rope": (252, 380),
            },
        )
        self.assertEqual(DSV4_BU4_LAYOUT.row_bytes, 380)

    def test_h256_prefix_round_trip_leaves_tail_unchanged(self):
        kv = torch.randn(2, 512, dtype=torch.bfloat16)

        folded = fold_dsv4_h256(kv)
        restored = restore_dsv4_h256(folded)

        self.assertTrue(torch.equal(folded[..., 256:], kv[..., 256:]))
        torch.testing.assert_close(restored, kv, atol=0.03125, rtol=0)

    def test_reference_codec_handles_constant_quantization_groups(self):
        nope = torch.zeros((2, 448), dtype=torch.bfloat16)
        rope = torch.zeros((2, 64), dtype=torch.bfloat16)
        kv = torch.cat((nope, rope), dim=-1)

        packed = encode_dsv4_bu4_reference(kv)
        decoded = decode_dsv4_bu4_reference(packed)

        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(tuple(packed.shape), (2, 380))
        self.assertTrue(torch.equal(decoded, kv))

    def test_reference_codec_rejects_invalid_shape_or_dtype(self):
        with self.assertRaisesRegex(ValueError, "last dimension must be 512"):
            encode_dsv4_bu4_reference(torch.zeros(1, 511))
        with self.assertRaisesRegex(TypeError, "floating point"):
            encode_dsv4_bu4_reference(torch.zeros(1, 512, dtype=torch.int32))
        with self.assertRaisesRegex(ValueError, "last dimension must be 380"):
            decode_dsv4_bu4_reference(torch.zeros(1, 379, dtype=torch.uint8))

    def test_geometry_rejects_non_dsv4_shape(self):
        validate_dsv4_bu4_geometry(448, 64)
        with self.assertRaisesRegex(ValueError, "448-nope/64-rope"):
            validate_dsv4_bu4_geometry(512, 64)


class TestDSV4KVBitCapability(CustomTestCase):
    """Enabling storage must never silently fall back to native scratch."""

    def test_environment_switch_is_off_by_default_and_restored(self):
        envs.SGLANG_ENABLE_KVBIT.clear()
        self.addCleanup(envs.SGLANG_ENABLE_KVBIT.clear)
        self.assertFalse(envs.SGLANG_ENABLE_KVBIT.get())
        with envs.SGLANG_ENABLE_KVBIT.override(True):
            self.assertTrue(envs.SGLANG_ENABLE_KVBIT.get())
        self.assertFalse(envs.SGLANG_ENABLE_KVBIT.get())

    def test_only_target_worker_is_eligible(self):
        self.assertTrue(
            dsv4_kvbit_enabled_for_worker(enabled=True, is_draft_worker=False)
        )
        self.assertFalse(
            dsv4_kvbit_enabled_for_worker(enabled=True, is_draft_worker=True)
        )
        self.assertFalse(
            dsv4_kvbit_enabled_for_worker(enabled=False, is_draft_worker=False)
        )

    def test_capability_gate_rejects_each_missing_direct_operation(self):
        with self.assertRaisesRegex(RuntimeError, "scratch fallback is disabled"):
            require_dsv4_kvbit_runtime_capability(
                DSV4KVBitRuntimeCapability(
                    direct_packed_write=False,
                    direct_packed_decode=True,
                )
            )

        require_dsv4_kvbit_runtime_capability(
            DSV4KVBitRuntimeCapability(
                direct_packed_write=True,
                direct_packed_decode=True,
            )
        )


class TestDSV4KVBitPackedSWAPool(CustomTestCase):
    """Target persistent SWA/C4/C128 pools use the packed row ABI."""

    def _owner(self, *, enabled):
        owner = object.__new__(DeepSeekV4TokenToKVPool)
        owner.enable_kvbit_swa = enabled
        owner.qk_nope_head_dim = 448
        owner.qk_rope_head_dim = 64
        return owner

    def _pool_kwargs(self, *, page_size=8):
        return {
            "size": 9,
            "page_size": page_size,
            "dtype": torch.uint8,
            "layer_num": 2,
            "device": "cpu",
            "enable_memory_saver": False,
            "global_page_size": 256,
        }

    def test_target_swa_factory_allocates_only_packed_rows(self):
        pool = self._owner(enabled=True)._make_swa_kv_pool(**self._pool_kwargs())

        self.assertIsInstance(pool, DSV4KVBitPackedSWAPool)
        self.assertEqual(pool.get_bytes_per_token(), 380)
        self.assertEqual(tuple(pool.kv_buffer[0].shape), (1, 256 * 380))
        self.assertEqual(pool.bytes_per_page_padded, 256 * 380)
        self.assertNotIn("_flashmla_scratch", vars(pool))
        with self.assertRaisesRegex(RuntimeError, "CUDA and Triton"):
            pool.set_key_buffer_fused(
                0,
                torch.tensor([1], dtype=torch.int32),
                torch.zeros(1, 512, dtype=torch.bfloat16),
            )

    def test_target_compressed_factories_use_packed_rows(self):
        owner = self._owner(enabled=True)
        packed_c4 = owner._make_compressed_kv_pool(**self._pool_kwargs(page_size=64))
        packed_c128 = owner._make_compressed_kv_pool(**self._pool_kwargs(page_size=2))

        self.assertIsInstance(packed_c4, DSV4KVBitPackedSWAPool)
        self.assertEqual(packed_c4.page_size, 64)
        self.assertEqual(packed_c4.bytes_per_page_padded, 64 * 380)
        self.assertIsInstance(packed_c128, DSV4KVBitPackedSWAPool)
        self.assertEqual(packed_c128.page_size, 2)
        self.assertEqual(packed_c128.bytes_per_page_padded, 2 * 380)

    def test_disabled_swa_and_compressed_factories_stay_native(self):
        native_swa = self._owner(enabled=False)._make_swa_kv_pool(**self._pool_kwargs())
        native_c4 = self._owner(enabled=False)._make_compressed_kv_pool(
            **self._pool_kwargs(page_size=2)
        )

        self.assertIsInstance(native_swa, DeepSeekV4SingleKVPool)
        self.assertNotIsInstance(native_swa, DSV4KVBitPackedSWAPool)
        self.assertFalse(native_swa.is_dsv4_kvbit_packed_swa)
        self.assertIsInstance(native_c4, DeepSeekV4SingleKVPool)

    def test_fused_wqkv_tail_is_made_contiguous_before_norm_rope(self):
        owner = object.__new__(DeepSeekV4TokenToKVPool)
        owner._stage_start = 0
        captured = {}

        def set_key_buffer_fused(layer_id, loc, kv):
            captured["layer_id"] = layer_id
            captured["loc"] = loc
            captured["kv"] = kv

        owner.swa_kv_pool = SimpleNamespace(
            is_dsv4_kvbit_packed_swa=True,
            set_key_buffer_fused=set_key_buffer_fused,
        )
        qkv = torch.randn(3, 1536, dtype=torch.bfloat16)
        kv = qkv[:, 1024:]
        loc = torch.arange(3, dtype=torch.int32)
        self.assertFalse(kv.is_contiguous())
        self.assertEqual(kv.stride(0), 1536)

        with patch(
            "sglang.srt.mem_cache.deepseek_v4_memory_pool." "fused_norm_rope_inplace"
        ) as norm_rope:
            owner.set_swa_key_buffer_radix_fused_norm_rope(
                layer_id=0,
                swa_loc=loc,
                kv=kv,
                kv_weight=torch.ones(448),
                eps=1e-6,
                freqs_cis=torch.empty(0),
                positions=torch.arange(3),
            )

        normalized_kv = norm_rope.call_args.args[0]
        self.assertTrue(normalized_kv.is_contiguous())
        self.assertIs(captured["kv"], normalized_kv)
        self.assertEqual(captured["layer_id"], 0)
        self.assertIs(captured["loc"], loc)


class TestDSV4KVBitAttentionMath(CustomTestCase):
    """Guard direct packed Q/K/V=512 semantics and natural-log LSE merging."""

    def test_sparse_decode_reference_uses_page_size_256_and_attention_sink(self):
        torch.manual_seed(7)
        keys = torch.randn(3, 512, dtype=torch.bfloat16) * 0.1
        encoded = encode_dsv4_bu4_reference(keys)
        packed = torch.zeros(2, 256 * 380, dtype=torch.uint8)
        physical_locs = torch.tensor([2, 255, 257], dtype=torch.int64)
        packed.view(-1, 380)[physical_locs] = encoded
        q = torch.randn(1, 1, 2, 512, dtype=torch.bfloat16) * 0.1
        indices = torch.tensor([[[2, 255, 257, -1]]], dtype=torch.int32)
        lengths = torch.tensor([3], dtype=torch.int32)
        sink = torch.tensor([-0.25, 0.5], dtype=torch.float32)

        output, lse = dsv4_kvbit_sparse_decode(
            q=q,
            packed=packed,
            indices=indices,
            lengths=lengths,
            attn_sink=sink,
            page_size=256,
            softmax_scale=512**-0.5,
        )

        decoded = decode_dsv4_bu4_reference(encoded).float()
        scores = torch.einsum("qhd,kd->qhk", q[0].float(), decoded) * (512**-0.5)
        scores = torch.cat((scores, sink.view(1, 2, 1)), dim=-1)
        expected = torch.einsum(
            "qhk,kd->qhd", torch.softmax(scores, dim=-1)[..., :-1], decoded
        )
        torch.testing.assert_close(output[0].float(), expected, atol=0.02, rtol=0.02)
        torch.testing.assert_close(
            lse[0],
            torch.logsumexp(scores, dim=-1).transpose(0, 1),
            atol=2e-5,
            rtol=2e-5,
        )
        with self.assertRaisesRegex(ValueError, r"page_size in \(2, 64, 256\)"):
            dsv4_kvbit_sparse_decode(
                q=q,
                packed=packed,
                indices=indices,
                lengths=lengths,
                attn_sink=sink,
                page_size=128,
                softmax_scale=512**-0.5,
            )

    def test_sparse_extra_decode_supports_compressed_pages_and_empty_state(self):
        q = torch.zeros(1, 1, 2, 512, dtype=torch.bfloat16)
        indices = torch.full((1, 1, 4), -1, dtype=torch.int32)
        lengths = torch.ones(1, dtype=torch.int32)

        for page_size in (64, 2):
            with self.subTest(page_size=page_size):
                packed = torch.zeros(2, page_size * 380, dtype=torch.uint8)
                output, lse = dsv4_kvbit_sparse_decode(
                    q=q,
                    packed=packed,
                    indices=indices,
                    lengths=lengths,
                    attn_sink=None,
                    page_size=page_size,
                    softmax_scale=512**-0.5,
                )
                self.assertTrue(torch.equal(output, torch.zeros_like(output)))
                self.assertTrue(torch.isneginf(lse).all())

    def test_sparse_extra_decode_supports_nonempty_compressed_pages(self):
        torch.manual_seed(9)
        keys = torch.randn(3, 512, dtype=torch.bfloat16) * 0.1
        encoded = encode_dsv4_bu4_reference(keys)
        q = torch.randn(1, 1, 2, 512, dtype=torch.bfloat16) * 0.1

        for page_size, physical_locs in (
            (64, torch.tensor([1, 63, 64], dtype=torch.int64)),
            (2, torch.tensor([0, 1, 2], dtype=torch.int64)),
        ):
            with self.subTest(page_size=page_size):
                packed = torch.zeros(2, page_size * 380, dtype=torch.uint8)
                packed.view(-1, 380)[physical_locs] = encoded
                indices = physical_locs.to(torch.int32).view(1, 1, -1)
                lengths = torch.tensor([3], dtype=torch.int32)

                output, lse = dsv4_kvbit_sparse_decode(
                    q=q,
                    packed=packed,
                    indices=indices,
                    lengths=lengths,
                    attn_sink=None,
                    page_size=page_size,
                    softmax_scale=512**-0.5,
                )

                decoded = decode_dsv4_bu4_reference(encoded).float()
                scores = torch.einsum("qhd,kd->qhk", q[0].float(), decoded) * (
                    512**-0.5
                )
                expected = torch.einsum(
                    "qhk,kd->qhd", torch.softmax(scores, dim=-1), decoded
                )
                torch.testing.assert_close(
                    output[0].float(), expected, atol=0.02, rtol=0.02
                )
                torch.testing.assert_close(
                    lse[0],
                    torch.logsumexp(scores, dim=-1).transpose(0, 1),
                    atol=2e-5,
                    rtol=2e-5,
                )

    def test_natural_log_lse_merge_matches_concatenated_attention(self):
        left_scores = torch.tensor([1.0, -2.0])
        right_scores = torch.tensor([20.0, 19.0])
        left_values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        right_values = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        left_prob = torch.softmax(left_scores, dim=0)
        right_prob = torch.softmax(right_scores, dim=0)
        left_output = (left_prob @ left_values).reshape(1, 1, 1, 2)
        right_output = (right_prob @ right_values).reshape(1, 1, 1, 2)
        left_lse = torch.logsumexp(left_scores, dim=0).reshape(1, 1, 1)
        right_lse = torch.logsumexp(right_scores, dim=0).reshape(1, 1, 1)

        output, lse = merge_attention_states_natural_log(
            left_output, left_lse, right_output, right_lse
        )

        all_scores = torch.cat((left_scores, right_scores))
        all_values = torch.cat((left_values, right_values))
        expected = torch.softmax(all_scores, dim=0) @ all_values
        torch.testing.assert_close(output.reshape(-1), expected)
        torch.testing.assert_close(lse.reshape(()), torch.logsumexp(all_scores, dim=0))

    def test_natural_log_lse_merge_treats_positive_infinity_as_empty(self):
        valid_output = torch.ones(1, 1, 2, 4)
        empty_output = torch.zeros_like(valid_output)
        valid_lse = torch.zeros(1, 2, 1)
        empty_lse = torch.full_like(valid_lse, torch.inf)

        output, lse = merge_attention_states_natural_log(
            valid_output, valid_lse, empty_output, empty_lse
        )

        torch.testing.assert_close(output, valid_output)
        torch.testing.assert_close(lse, valid_lse)

    def test_all_ratios_use_one_flashmla_packed_decode(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        for ratio in (0, 4, 128):
            with self.subTest(compress_ratio=ratio):
                backend = object.__new__(DeepseekV4AttnBackend)
                backend.page_size = 256
                backend.softmax_scale = 512**-0.5
                backend.head_dim_v = 512
                extra_page_size = {0: None, 4: 64, 128: 2}[ratio]
                extra_cache = (
                    None
                    if extra_page_size is None
                    else torch.zeros(1, extra_page_size * 380, dtype=torch.uint8)
                )
                backend.token_to_kv_pool = SimpleNamespace(
                    get_extra_key_buffer=lambda _layer_id, value=extra_cache: value,
                    get_extra_key_page_size=lambda _layer_id, value=extra_page_size: (
                        value
                    ),
                )
                indices = torch.zeros(1, 64, dtype=torch.int32)
                lengths = torch.ones(1, dtype=torch.int32)
                core = SimpleNamespace(
                    swa_page_indices=indices,
                    swa_topk_lengths=lengths,
                    c4_sparse_page_indices=indices,
                    c4_sparse_topk_lengths=lengths,
                    c128_page_indices=indices,
                    c128_topk_lengths_clamp1=lengths,
                    get_flashmla_metadata=lambda _ratio: object(),
                )
                q = torch.zeros(1, 2, 512, dtype=torch.bfloat16)
                expected = torch.randn(1, 1, 2, 512, dtype=torch.bfloat16)
                swa_cache = torch.zeros(1, 256 * 380, dtype=torch.uint8)

                def packed_kwargs(cache, *, page_size):
                    return {
                        "packed_kcache": cache.view(-1, 380),
                        "scale_kcache": torch.ones(448),
                        "R_matrix": torch.eye(448, dtype=torch.bfloat16),
                        "zero_point": torch.zeros(448),
                        "dim_of_bit": torch.arange(448).repeat_interleave(4),
                        "bitpos_in_dim": torch.arange(4).repeat(448),
                        "bit_uniform": 4,
                    }

                with (
                    patch(
                        "sglang.srt.layers.attention.deepseek_v4_backend."
                        "dsv4_kvbit_flashmla_packed_kwargs",
                        side_effect=packed_kwargs,
                    ),
                    patch(
                        "sgl_kernel.flash_mla.flash_mla_with_kvcache",
                        return_value=(expected, torch.zeros(1, 2, 1)),
                    ) as flashmla,
                ):
                    output = backend._forward_kvbit(
                        q=q,
                        layer_id=0,
                        compress_ratio=ratio,
                        packed_swa_cache=swa_cache,
                        core_attn_metadata=core,
                        attn_sink=torch.zeros(2),
                    )

                torch.testing.assert_close(output, expected.squeeze(1))
                flashmla.assert_called_once()
                call = flashmla.call_args.kwargs
                self.assertFalse(call.get("q_nope_is_folded", False))
                self.assertTrue(call["identity_tail_bypass"])
                self.assertEqual(call["bit_uniform"], 4)
                self.assertEqual(tuple(call["k_cache"].shape), (1, 256, 1, 380))
                self.assertIs(call["packed_kcache"]._base, swa_cache)
                if ratio == 0:
                    self.assertIsNone(call["extra_k_cache"])
                    self.assertIsNone(call["extra_packed_kcache"])
                else:
                    self.assertEqual(
                        tuple(call["extra_k_cache"].shape),
                        (1, extra_page_size, 1, 380),
                    )
                    self.assertIs(call["extra_packed_kcache"]._base, extra_cache)


class TestDSV4KVBitPackedCompressor(CustomTestCase):
    def test_packed_online_c128_keeps_mtp_prefix_state_write(self):
        from sglang.srt.layers.attention.dsv4.compressor_v2 import (
            CompressorBackendMixin,
        )

        backend = object.__new__(CompressorBackendMixin)
        packed_pool = SimpleNamespace(is_dsv4_kvbit_packed=True)
        backend.token_to_kv_pool = SimpleNamespace(
            layer_mapping={7: (None, None, packed_pool)}
        )
        backend.enable_deepseek_v4_fp4_indexer = False
        backend.forward_metadata = SimpleNamespace(
            core_metadata=SimpleNamespace(
                c128_out_loc=torch.tensor([3], dtype=torch.int32)
            )
        )
        backend._forward_compress_packed = MagicMock()
        backend._forward_compress_all_in_one = MagicMock()
        backend.online_c128_mtp = MagicMock()
        state_pool = SimpleNamespace(
            kv_score_buffer=SimpleNamespace(kv_score=torch.empty(1, 512))
        )
        compressor = SimpleNamespace(
            ratio=128,
            is_in_indexer=False,
            ape=torch.empty(128, 512),
            compute_kv_score=MagicMock(return_value=torch.empty(1, 512)),
            get_state_pool=MagicMock(return_value=state_pool),
        )
        forward_mode = SimpleNamespace(is_idle=lambda: False)
        forward_batch = SimpleNamespace(forward_mode=forward_mode)

        with patch(
            "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate."
            "is_unified_kv_triton",
            return_value=False,
        ):
            backend.forward_unified(
                x=torch.empty(1, 512),
                forward_batch=forward_batch,
                layer_id=7,
                compressor=compressor,
            )

        backend._forward_compress_packed.assert_called_once()
        backend._forward_compress_all_in_one.assert_not_called()
        backend.online_c128_mtp.write_prefix_states.assert_called_once_with(
            layer_id=7,
            compressor=compressor,
            kv_score_input=compressor.compute_kv_score.return_value,
            logical_forward_mode=forward_mode,
        )

    def test_decode_non_boundary_does_not_write_location_zero(self):
        from sglang.kernels.ops.attention.dsv4 import CompressorDecodePlan
        from sglang.srt.layers.attention.dsv4.compressor_v2 import (
            CompressorBackendMixin,
        )

        backend = object.__new__(CompressorBackendMixin)
        plan_d = torch.zeros((2, 16), dtype=torch.uint8)
        plan_d_i32 = plan_d.view(torch.int32)
        plan_d_i32[:, 0] = torch.tensor([4, 5], dtype=torch.int32)
        backend.forward_metadata = SimpleNamespace(
            c4_compress_metadata=CompressorDecodePlan(4, plan_d)
        )
        captured = {}
        backend.token_to_kv_pool = SimpleNamespace(
            set_extra_key_buffer_fused=lambda **kwargs: captured.update(kwargs)
        )
        compressor = SimpleNamespace(
            ratio=4,
            head_dim=512,
            ape=torch.empty(4, 512),
            norm=SimpleNamespace(weight=torch.ones(512), variance_epsilon=1e-6),
            freqs_cis=torch.empty(8, 32, dtype=torch.complex64),
        )
        compressed = torch.zeros(2, 512, dtype=torch.bfloat16)

        with (
            patch(
                "sglang.srt.layers.attention.dsv4.compressor_v2.compress_forward",
                return_value=compressed,
            ),
            patch(
                "sglang.kernels.ops.attention.deepseek_v4_rope."
                "fused_norm_rope_inplace_triton"
            ),
        ):
            backend._forward_compress_packed(
                kv_score_buffer=torch.empty(1, 4, 2048),
                kv_score_input=torch.empty(2, 512),
                ape=compressor.ape,
                compressor=compressor,
                layer_id=7,
                out_loc=torch.tensor([10, 11], dtype=torch.int32),
            )

        torch.testing.assert_close(
            captured["loc"], torch.tensor([10, -1], dtype=torch.int32)
        )

    def test_prefill_padded_plan_does_not_write_location_zero(self):
        from sglang.kernels.ops.attention.dsv4 import CompressorPrefillPlan
        from sglang.srt.layers.attention.dsv4.compressor_v2 import (
            CompressorBackendMixin,
        )

        backend = object.__new__(CompressorBackendMixin)
        plan_c = torch.zeros((2, 16), dtype=torch.uint8)
        plan_c_i32 = plan_c.view(torch.int32)
        plan_c_i32[0, 0] = 4
        plan_c_i32[0, 1] = 1
        plan_c_i32[1, 0] = -1
        plan_w = torch.empty((0, 8), dtype=torch.uint8)
        backend.forward_metadata = SimpleNamespace(
            c4_compress_metadata=CompressorPrefillPlan(4, plan_c, plan_w)
        )
        captured = {}
        backend.token_to_kv_pool = SimpleNamespace(
            set_extra_key_buffer_fused=lambda **kwargs: captured.update(kwargs)
        )
        compressor = SimpleNamespace(
            ratio=4,
            head_dim=512,
            ape=torch.empty(4, 512),
            norm=SimpleNamespace(weight=torch.ones(512), variance_epsilon=1e-6),
            freqs_cis=torch.empty(8, 32, dtype=torch.complex64),
        )
        compressed = torch.zeros(2, 512, dtype=torch.bfloat16)

        with (
            patch(
                "sglang.srt.layers.attention.dsv4.compressor_v2.compress_forward",
                return_value=compressed,
            ),
            patch(
                "sglang.kernels.ops.attention.deepseek_v4_rope."
                "fused_norm_rope_inplace_triton"
            ),
        ):
            backend._forward_compress_packed(
                kv_score_buffer=torch.empty(1, 4, 2048),
                kv_score_input=torch.empty(1, 512),
                ape=compressor.ape,
                compressor=compressor,
                layer_id=7,
                out_loc=torch.tensor([10, 11, 12], dtype=torch.int32),
            )

        self.assertEqual(captured["layer_id"], 7)
        torch.testing.assert_close(
            captured["loc"], torch.tensor([11, -1], dtype=torch.int32)
        )
        self.assertIs(captured["cache_k"], compressed)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_flashmla_metadata_matches_fixed_bu4_layout(self):
        packed = torch.zeros(2, 64 * 380, dtype=torch.uint8, device="cuda")

        first = dsv4_kvbit_flashmla_packed_kwargs(packed, page_size=64)
        second = dsv4_kvbit_flashmla_packed_kwargs(packed, page_size=64)

        self.assertEqual(tuple(first["packed_kcache"].shape), (128, 380))
        self.assertEqual(first["bit_uniform"], 4)
        self.assertEqual(first["R_matrix"].dtype, torch.bfloat16)
        self.assertEqual(tuple(first["R_matrix"].shape), (448, 448))
        self.assertEqual(tuple(first["dim_of_bit"].shape), (1792,))
        self.assertEqual(tuple(first["bitpos_in_dim"].shape), (1792,))
        self.assertEqual(first["R_matrix"].data_ptr(), second["R_matrix"].data_ptr())
        torch.testing.assert_close(
            first["R_matrix"][256:, 256:].float(),
            torch.eye(192, device="cuda"),
            atol=0,
            rtol=0,
        )
        h256 = first["R_matrix"][:256, :256].float()
        torch.testing.assert_close(
            h256 @ h256.T,
            torch.eye(256, device="cuda"),
            atol=0.02,
            rtol=0,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_flashmla_fused_swa_and_extra_match_cpu_reference(self):
        from sgl_kernel.flash_mla import FlashMLASchedMeta, flash_mla_with_kvcache

        torch.manual_seed(17)
        swa_keys = torch.randn(3, 512, dtype=torch.bfloat16) * 0.1
        extra_keys = torch.randn(3, 512, dtype=torch.bfloat16) * 0.1
        q = torch.randn(1, 1, 2, 512, dtype=torch.bfloat16) * 0.1
        sink = torch.tensor([-0.2, 0.3], dtype=torch.float32)

        for extra_page_size, extra_locs in (
            (64, torch.tensor([1, 63, 64], dtype=torch.int32)),
            (2, torch.tensor([0, 1, 2], dtype=torch.int32)),
        ):
            with self.subTest(extra_page_size=extra_page_size):
                swa_pool = DSV4KVBitPackedSWAPool(
                    size=256,
                    page_size=256,
                    dtype=torch.bfloat16,
                    qk_nope_head_dim=448,
                    qk_rope_head_dim=64,
                    layer_num=1,
                    device="cuda",
                    enable_memory_saver=False,
                )
                extra_pool = DSV4KVBitPackedSWAPool(
                    size=extra_page_size,
                    page_size=extra_page_size,
                    dtype=torch.bfloat16,
                    qk_nope_head_dim=448,
                    qk_rope_head_dim=64,
                    layer_num=1,
                    device="cuda",
                    enable_memory_saver=False,
                )
                swa_locs = torch.tensor([1, 255, 256], dtype=torch.int32)
                swa_pool.set_key_buffer_fused(0, swa_locs.cuda(), swa_keys.cuda())
                extra_pool.set_key_buffer_fused(0, extra_locs.cuda(), extra_keys.cuda())
                swa_cache = swa_pool.get_key_buffer(0)
                extra_cache = extra_pool.get_key_buffer(0)
                swa_kwargs = dsv4_kvbit_flashmla_packed_kwargs(swa_cache, page_size=256)
                extra_rows = dsv4_kvbit_flashmla_packed_kwargs(
                    extra_cache, page_size=extra_page_size
                )["packed_kcache"]
                swa_indices = torch.full(
                    (1, 1, 64), -1, dtype=torch.int32, device="cuda"
                )
                extra_indices = torch.full_like(swa_indices, -1)
                swa_indices[0, 0, :3] = swa_locs.cuda()
                extra_indices[0, 0, :3] = extra_locs.cuda()
                lengths = torch.tensor([3], dtype=torch.int32, device="cuda")

                output, _ = flash_mla_with_kvcache(
                    q=q.cuda(),
                    k_cache=swa_cache.view(swa_cache.shape[0], 256, 1, 380),
                    block_table=None,
                    cache_seqlens=None,
                    head_dim_v=512,
                    tile_scheduler_metadata=FlashMLASchedMeta(),
                    softmax_scale=512**-0.5,
                    is_fp8_kvcache=True,
                    indices=swa_indices,
                    attn_sink=sink.cuda(),
                    extra_k_cache=extra_cache.view(
                        extra_cache.shape[0], extra_page_size, 1, 380
                    ),
                    extra_indices_in_kvcache=extra_indices,
                    topk_length=lengths,
                    extra_topk_length=lengths,
                    identity_tail_bypass=True,
                    extra_packed_kcache=extra_rows,
                    **swa_kwargs,
                )
                output = output.cpu().float()

                all_keys = torch.cat(
                    (
                        decode_dsv4_bu4_reference(encode_dsv4_bu4_reference(swa_keys)),
                        decode_dsv4_bu4_reference(
                            encode_dsv4_bu4_reference(extra_keys)
                        ),
                    ),
                    dim=0,
                ).float()
                scores = torch.einsum("qhd,kd->qhk", q[0].float(), all_keys) * (
                    512**-0.5
                )
                scores = torch.cat((scores, sink.view(1, 2, 1)), dim=-1)
                expected = torch.einsum(
                    "qhk,kd->qhd",
                    torch.softmax(scores, dim=-1)[..., :-1],
                    all_keys,
                )
                torch.testing.assert_close(output[0], expected, atol=0.04, rtol=0.04)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_fused_write_and_direct_decode_match_cpu_reference(self):
        torch.manual_seed(11)
        keys_cpu = torch.randn(4, 512, dtype=torch.bfloat16) * 0.1
        q_cpu = torch.randn(2, 1, 3, 512, dtype=torch.bfloat16) * 0.1
        loc_cpu = torch.tensor([1, 255, 256, 511], dtype=torch.int32)
        indices_cpu = torch.tensor(
            [[[1, 255, -1, -1]], [[256, 511, 1, -1]]], dtype=torch.int32
        )
        lengths_cpu = torch.tensor([2, 3], dtype=torch.int32)
        sink_cpu = torch.tensor([-0.2, 0.0, 0.3], dtype=torch.float32)

        expected_packed = torch.zeros(2, 256 * 380, dtype=torch.uint8)
        expected_packed.view(-1, 380)[loc_cpu.long()] = encode_dsv4_bu4_reference(
            keys_cpu
        )
        expected_output, expected_lse = dsv4_kvbit_sparse_decode(
            q=q_cpu,
            packed=expected_packed,
            indices=indices_cpu,
            lengths=lengths_cpu,
            attn_sink=sink_cpu,
            page_size=256,
            softmax_scale=512**-0.5,
        )

        pool = DSV4KVBitPackedSWAPool(
            size=255,
            page_size=256,
            dtype=torch.bfloat16,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
        )
        pool.set_key_buffer_fused(0, loc_cpu.cuda(), keys_cpu.cuda())
        output, lse = dsv4_kvbit_sparse_decode(
            q=q_cpu.cuda(),
            packed=pool.kv_buffer[0],
            indices=indices_cpu.cuda(),
            lengths=lengths_cpu.cuda(),
            attn_sink=sink_cpu.cuda(),
            page_size=256,
            softmax_scale=512**-0.5,
        )

        actual_rows = pool.kv_buffer[0].cpu().view(-1, 380)[loc_cpu.long()]
        actual_decoded = decode_dsv4_bu4_reference(actual_rows)
        expected_decoded = decode_dsv4_bu4_reference(
            expected_packed.view(-1, 380)[loc_cpu.long()]
        )
        torch.testing.assert_close(
            actual_decoded.float(), expected_decoded.float(), atol=0.03, rtol=0.03
        )
        torch.testing.assert_close(
            output.cpu().float(), expected_output.float(), atol=0.03, rtol=0.03
        )
        torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.03, rtol=0.03)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_compressed_page_write_and_decode_match_cpu_reference(self):
        torch.manual_seed(13)
        keys_cpu = torch.randn(3, 512, dtype=torch.bfloat16) * 0.1
        q_cpu = torch.randn(1, 1, 2, 512, dtype=torch.bfloat16) * 0.1

        for page_size, loc_cpu in (
            (64, torch.tensor([1, 63, 64], dtype=torch.int32)),
            (2, torch.tensor([0, 1, 2], dtype=torch.int32)),
        ):
            with self.subTest(page_size=page_size):
                indices_cpu = loc_cpu.view(1, 1, -1)
                lengths_cpu = torch.tensor([3], dtype=torch.int32)
                expected_packed = torch.zeros(2, page_size * 380, dtype=torch.uint8)
                expected_packed.view(-1, 380)[loc_cpu.long()] = (
                    encode_dsv4_bu4_reference(keys_cpu)
                )
                expected_output, expected_lse = dsv4_kvbit_sparse_decode(
                    q=q_cpu,
                    packed=expected_packed,
                    indices=indices_cpu,
                    lengths=lengths_cpu,
                    attn_sink=None,
                    page_size=page_size,
                    softmax_scale=512**-0.5,
                )

                pool = DSV4KVBitPackedSWAPool(
                    size=page_size,
                    page_size=page_size,
                    dtype=torch.bfloat16,
                    qk_nope_head_dim=448,
                    qk_rope_head_dim=64,
                    layer_num=1,
                    device="cuda",
                    enable_memory_saver=False,
                )
                pool.set_key_buffer_fused(0, loc_cpu.cuda(), keys_cpu.cuda())
                output, lse = dsv4_kvbit_sparse_decode(
                    q=q_cpu.cuda(),
                    packed=pool.kv_buffer[0],
                    indices=indices_cpu.cuda(),
                    lengths=lengths_cpu.cuda(),
                    attn_sink=None,
                    page_size=page_size,
                    softmax_scale=512**-0.5,
                )

                torch.testing.assert_close(
                    output.cpu().float(),
                    expected_output.float(),
                    atol=0.03,
                    rtol=0.03,
                )
                torch.testing.assert_close(
                    lse.cpu(), expected_lse, atol=0.03, rtol=0.03
                )


if __name__ == "__main__":
    unittest.main()
