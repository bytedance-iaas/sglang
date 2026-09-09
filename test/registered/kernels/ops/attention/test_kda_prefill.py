import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.fla.kda import chunk_kda
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

    def _check_snapshot(self, lower_bound, num_heads):
        lens = [100, 64, 193]
        boundaries = [64, None, 128]
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
        cu_seqlens = torch.tensor([0, 100, 164, 357], device="cuda", dtype=torch.int32)

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
            track_chunk_idx=torch.tensor([1, -1, 2], device="cuda", dtype=torch.int32),
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
            (0, 0, boundaries[0], 1),
            (2, 164, boundaries[2], 5),
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
