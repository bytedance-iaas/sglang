import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.fla.kda import chunk_kda, chunk_kda_fwd_intra
from sglang.kernels.ops.attention.linear.kda_nvidia_prefill import (
    chunk_kda_fwd as nvidia_chunk_kda_fwd,
)
from sglang.kernels.ops.attention.linear.kda_ptx_prefill import (
    chunk_kda_fwd as ptx_chunk_kda_fwd,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=80, stage="base-c", runner_config="4-gpu-gb300")


def _inputs(seed, seq_len=128):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    batch_size, num_heads, head_dim = 1, 2, 128
    shape = (batch_size, seq_len, num_heads, head_dim)
    q = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    v = (
        0.1
        * torch.randn(
            shape,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
    ).to(torch.bfloat16)
    gate = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    beta_logits = torch.randn(
        shape[:-1],
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    a_log = torch.randn(
        num_heads, generator=generator, device="cuda", dtype=torch.float32
    )
    dt_bias = torch.randn(
        num_heads * head_dim,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    state = torch.zeros(
        batch_size,
        num_heads,
        head_dim,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    return q, k, v, gate, beta_logits, a_log, dt_bias, state


def _reference(q, k, v, gate, beta, a_log, dt_bias, state, fused_qk_norm):
    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        scale=q.shape[-1] ** -0.5,
        initial_state=state,
        initial_state_indices=torch.arange(
            q.shape[0], device="cuda", dtype=torch.int32
        ),
        use_qk_l2norm_in_kernel=fused_qk_norm,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
    )


class TestKdaPrefill(CustomTestCase):
    @torch.inference_mode()
    def test_nvidia_prefill(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("NVIDIA KDA prefill requires datacenter Blackwell")
        q, k, v, gate, beta_logits, a_log, dt_bias, state = _inputs(0)
        q = F.normalize(q.float(), dim=-1).to(torch.bfloat16)
        k = F.normalize(k.float(), dim=-1).to(torch.bfloat16)
        beta = torch.sigmoid(beta_logits.float()).to(torch.bfloat16)
        actual, actual_state = nvidia_chunk_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=gate,
            beta=beta,
            scale=q.shape[-1] ** -0.5,
            initial_state=state.transpose(-1, -2).contiguous(),
            output_final_state=True,
            safe_gate=True,
            lower_bound=-5.0,
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
        )[:2]
        expected = _reference(
            q, k, v, gate, beta, a_log, dt_bias, state, fused_qk_norm=False
        )
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=2e-2, atol=3e-2
        )
        torch.testing.assert_close(
            actual_state.transpose(-1, -2),
            state,
            rtol=2e-2,
            atol=3e-2,
        )

    @torch.inference_mode()
    def test_ptx_prefill(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            10,
            3,
        ):
            self.skipTest("PTX KDA prefill requires GB300")
        q, k, v, gate, beta_logits, a_log, dt_bias, state = _inputs(1)
        actual, actual_state = ptx_chunk_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=gate,
            beta=beta_logits,
            scale=q.shape[-1] ** -0.5,
            initial_state=state.transpose(-1, -2).contiguous(),
            output_final_state=True,
            safe_gate=True,
            lower_bound=-5.0,
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
        )[:2]
        expected = _reference(
            q,
            k,
            v,
            gate,
            torch.sigmoid(beta_logits.float()).to(torch.bfloat16),
            a_log,
            dt_bias,
            state,
            fused_qk_norm=True,
        )
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=2e-2, atol=3e-2
        )
        torch.testing.assert_close(
            actual_state.transpose(-1, -2),
            state,
            rtol=2e-2,
            atol=3e-2,
        )


class TestKdaTrackState(CustomTestCase):
    @torch.inference_mode()
    def test_fp32_snapshot_matches_truncated_prefix(self):
        for lower_bound in (None, -5.0):
            for num_heads in (2, 16):
                with self.subTest(lower_bound=lower_bound, num_heads=num_heads):
                    self._check_snapshot(lower_bound, num_heads)

    @torch.inference_mode()
    def test_long_chunk_branch_snapshot_preserves_active_state(self):
        def same_intra_schedule(**kwargs):
            # Keep truncation from selecting a different fusion schedule.
            kwargs.update(fuse_diagonal=False, fuse_recompute=False)
            return chunk_kda_fwd_intra(**kwargs)

        with patch(
            "sglang.kernels.ops.attention.fla.kda.chunk_kda_fwd_intra",
            side_effect=same_intra_schedule,
        ):
            for lower_bound in (None, -5.0):
                with self.subTest(lower_bound=lower_bound):
                    self._check_snapshot(lower_bound, 16, seq_len=8192, boundary=576)

    @torch.inference_mode()
    def test_branch_tracking_preserves_backend_continuation(self):
        from sglang.srt.layers.attention.linear.kda_backend import (
            KDAAttnBackend,
            KDAKernelDispatcher,
        )
        from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
        from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        torch.manual_seed(4891)
        heads, dim, width = 16, 128, 3 * 16 * 128
        slots = torch.tensor([5, 1], device="cuda", dtype=torch.int32)
        track_slots = torch.tensor([6, 3], device="cuda", dtype=torch.int32)
        layer = SimpleNamespace(
            layer_id=0,
            q_dim=heads * dim,
            k_dim=heads * dim,
            v_dim=heads * dim,
            head_q_dim=dim,
            head_k_dim=dim,
            head_v_dim=dim,
            conv_weights=(0.1 * torch.randn(width, 4, device="cuda")).bfloat16(),
            bias=None,
            A_log=0.1 * torch.randn(heads, device="cuda"),
            dt_bias=0.1 * torch.randn(heads * dim, device="cuda"),
            lower_bound=-5.0,
        )
        initial_conv = (0.1 * torch.randn(7, 3, width, device="cuda")).bfloat16()
        initial_ssm = 0.01 * torch.randn(7, heads, dim, dim, device="cuda")
        initial_ssm[5].zero_()
        backends = []
        for _ in range(2):
            conv = torch.full(
                (7, 2, 3, width), 123.0, device="cuda", dtype=torch.bfloat16
            )
            ssm = torch.full((7, 2, heads, dim, dim), 123.0, device="cuda")
            conv[:, 0].copy_(initial_conv)
            ssm[:, 0].copy_(initial_ssm)
            cache = SimpleNamespace(conv=[conv[:, 0]], temporal=ssm[:, 0])
            backend = object.__new__(KDAAttnBackend)
            backend.device = torch.device("cuda")
            backend._mamba_chunk_size = 64
            backend.conv_states_shape = (width, 3)
            backend.accepted_state = None
            backend.accept_lens_pool = None
            backend.req_to_token_pool = SimpleNamespace(
                mamba2_layer_cache=lambda _, cache=cache: cache
            )
            kind = LinearAttnKernelBackend.TRITON
            backend.kernel_dispatcher = KDAKernelDispatcher(kind, kind, kind)
            backends.append((backend, cache, conv, ssm))

        for step, (lens, prefixes) in enumerate(
            (([8192, 193], [0, 64]), ([64, 64], [8192, 257]))
        ):
            total = sum(lens)
            mixed = (0.5 * torch.randn(total, width, device="cuda")).bfloat16()
            gate = (0.5 * torch.randn(1, total, heads * dim, device="cuda")).bfloat16()
            beta = torch.randn(1, total, heads, device="cuda").bfloat16()
            outputs = []
            for branch, (backend, cache, conv, ssm) in enumerate(backends):
                cu = torch.tensor([0, lens[0], total], device="cuda", dtype=torch.int32)
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    extend_prefix_lens=torch.tensor(
                        prefixes, device="cuda", dtype=torch.int32
                    ),
                    extend_seq_lens=torch.tensor(
                        lens, device="cuda", dtype=torch.int32
                    ),
                    extend_seq_lens_cpu=lens,
                    mamba_track_mask=torch.tensor([True, False], device="cuda"),
                    mamba_track_indices=track_slots,
                    mamba_track_seqlens=torch.tensor(
                        [577 if branch else 8192, -1], device="cuda", dtype=torch.int32
                    ),
                )
                metadata = ForwardMetadata(
                    query_start_loc=cu, mamba_cache_indices=slots
                )
                if step == 0:
                    metadata.has_mamba_track_mask = True
                    metadata.conv_states_mask_indices = track_slots[:1]
                    with patch(
                        "sglang.srt.layers.attention.hybrid_linear_attn_backend.mamba_cache_chunk_size",
                        return_value=64,
                    ):
                        metadata.track_conv_indices = backend._init_track_conv_indices(
                            cu, batch
                        )
                    (
                        metadata.track_chunk_idx,
                        metadata.track_ssm_h_src,
                        metadata.track_ssm_h_dst,
                        metadata.track_ssm_h_batch_src,
                        metadata.track_ssm_final_src,
                        metadata.track_ssm_final_dst,
                        metadata.track_ssm_seq_idx,
                        metadata.track_ssm_end_locs,
                        metadata.track_ssm_recompute_dst,
                    ) = backend._init_track_ssm_indices(slots, batch)
                backend.forward_metadata = metadata
                outputs.append(
                    backend.forward_extend(
                        layer, batch, mixed.clone(), gate.clone(), beta.clone()
                    )
                )
                self.assertTrue(torch.all(conv[:, 1] == 123.0))
                self.assertTrue(torch.all(ssm[:, 1] == 123.0))
                if step == 0:
                    boundary = 576 if branch else 8192
                    torch.testing.assert_close(
                        cache.conv[0][6], mixed[boundary - 3 : boundary], rtol=0, atol=0
                    )
                    self.assertTrue(torch.isfinite(cache.temporal[6]).all())
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            for name in ("temporal", "conv"):
                left, right = [getattr(item[1], name) for item in backends]
                if name == "conv":
                    left, right = left[0], right[0]
                torch.testing.assert_close(left[:6], right[:6], rtol=0, atol=0)

    def _check_snapshot(self, lower_bound, num_heads, seq_len=100, boundary=64):
        lens = [seq_len, 64, 193]
        boundaries = [boundary, None, 128]
        head_dim = 128
        generator = torch.Generator(device="cuda").manual_seed(0)

        def randn(*shape):
            return torch.randn(
                *shape, generator=generator, device="cuda", dtype=torch.float32
            )

        shape = (1, sum(lens), num_heads, head_dim)
        q, k, gate = [randn(*shape).bfloat16() for _ in range(3)]
        v = (0.1 * randn(*shape)).bfloat16()
        beta = randn(1, sum(lens), num_heads).sigmoid().bfloat16()
        a_log = randn(num_heads)
        dt_bias = randn(num_heads * head_dim)
        initial = 0.01 * randn(7, num_heads, head_dim, head_dim)
        slots = torch.tensor([5, 1, 3], device="cuda", dtype=torch.int32)
        cu_seqlens = torch.tensor(
            [0, seq_len, seq_len + 64, sum(lens)], device="cuda", dtype=torch.int32
        )

        def run(q, k, v, gate, beta, state, indices, cu, **kwargs):
            return chunk_kda(
                q=q.clone(),
                k=k.clone(),
                v=v.clone(),
                g=gate.clone(),
                beta=beta.clone(),
                initial_state=state,
                initial_state_indices=indices,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu,
                A_log=a_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                **kwargs,
            )

        envelope = torch.full(
            (7, 2, num_heads, head_dim, head_dim),
            123.0,
            device="cuda",
            dtype=torch.float32,
        )
        state = envelope[:, 0]
        state.copy_(initial)
        snapshot = torch.full_like(state[:3], float("nan"))
        output, h = run(
            q,
            k,
            v,
            gate,
            beta,
            state,
            slots,
            cu_seqlens,
            output_intermediate_states=True,
            track_state=snapshot,
            track_chunk_idx=torch.tensor(
                [boundary // 64, -1, 2], device="cuda", dtype=torch.int32
            ),
        )
        self.assertTrue(torch.isnan(snapshot[1]).all())
        self.assertTrue(torch.isfinite(snapshot[[0, 2]]).all())
        self.assertTrue(torch.all(envelope[:, 1] == 123.0))
        torch.testing.assert_close(
            state[[0, 2, 4, 6]], initial[[0, 2, 4, 6]], rtol=0, atol=0
        )

        untracked_state = initial.clone()
        untracked_output = run(q, k, v, gate, beta, untracked_state, slots, cu_seqlens)
        torch.testing.assert_close(output, untracked_output, rtol=0, atol=0)
        torch.testing.assert_close(state, untracked_state, rtol=0, atol=0)

        for row, start, boundary, h_row in (
            (0, 0, boundaries[0], boundary // 64),
            (2, seq_len + 64, boundaries[2], (seq_len + 63) // 64 + 3),
        ):
            prefix = slice(start, start + boundary)
            reference = initial[slots[row].item()].unsqueeze(0).clone()
            run(
                q[:, prefix],
                k[:, prefix],
                v[:, prefix],
                gate[:, prefix],
                beta[:, prefix],
                reference,
                torch.tensor([0], device="cuda", dtype=torch.int32),
                torch.tensor([0, boundary], device="cuda", dtype=torch.int32),
            )
            torch.testing.assert_close(
                snapshot[row], reference[0], rtol=1e-5, atol=1e-5
            )
            torch.testing.assert_close(
                h[0, h_row].float(), snapshot[row].bfloat16().float(), rtol=0, atol=0
            )
            self.assertFalse(
                torch.equal(snapshot[row], snapshot[row].bfloat16().float())
            )
            self.assertGreater(
                (h[0, h_row].float() - reference[0]).abs().max().item(), 1e-5
            )

    @torch.inference_mode()
    def test_snapshot_arguments_reject_missing_or_narrow_buffer(self):
        q, k, v, gate, beta, a_log, dt_bias, state = _inputs(2)
        args = dict(
            q=q,
            k=k,
            v=v,
            g=gate,
            beta=beta.sigmoid(),
            initial_state=state,
            initial_state_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
            A_log=a_log,
            dt_bias=dt_bias,
        )
        with self.assertRaisesRegex(AssertionError, "passed together"):
            chunk_kda(**args, track_state=state.clone())
        with self.assertRaisesRegex(AssertionError, "must be fp32"):
            chunk_kda(
                **args,
                track_state=state.bfloat16(),
                track_chunk_idx=torch.tensor([1], device="cuda", dtype=torch.int32),
            )


if __name__ == "__main__":
    unittest.main()
