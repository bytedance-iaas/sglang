"""Fused QK GemmaRMSNorm+RoPE+gate kernel: 1-D and mrope [3, T] positions.

The mrope branch (ported from sgl-project #34446) indexes cos/sin per rotary
lane through the MRotaryEmbedding axis map; before it, every lane silently
read the temporal row, corrupting RoPE on image tokens in every full-attention
layer of Qwen3.5/3.8 hybrids (text was unaffected since all three rows match).
"""

import unittest

import torch

from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
    fused_qk_gemma_rmsnorm_rope_gate,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-large")


def _reference(
    q_gate,
    k,
    q_weight,
    k_weight,
    cos_sin_cache,
    positions,
    eps,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    axis_map,
):
    """Torch mirror of the kernel math, including its bf16 round-trips."""
    out_dtype = q_gate.dtype

    def norm(x, w):
        x = x.to(torch.float32)
        w = w.to(torch.float32)
        var = (x * x).sum(-1, keepdim=True) / x.shape[-1]
        inv_rms = torch.rsqrt(var + eps)
        return (x * inv_rms * (w + 1.0)).to(out_dtype).to(torch.float32)

    T = q_gate.shape[0]
    q = q_gate.view(T, num_q_heads, 2 * head_dim)[..., :head_dim]
    k = k.view(T, num_kv_heads, head_dim)
    qn = norm(q, q_weight)
    kn = norm(k, k_weight)

    half = rotary_dim // 2
    if positions.dim() == 1:
        # One position per token: every lane reads the same cache row.
        pos = positions.to(torch.long).view(T)
        cos = cos_sin_cache[pos, :half].to(torch.float32).unsqueeze(1)
        sin = cos_sin_cache[pos, half : 2 * half].to(torch.float32).unsqueeze(1)
    else:
        # Per-lane positions: lane l of token t reads cache[pos[t, l], l].
        pos = positions.index_select(0, axis_map).t().to(torch.long)
        lanes = torch.arange(half, device=cos_sin_cache.device)
        cos = cos_sin_cache[pos, lanes].to(torch.float32).unsqueeze(1)
        sin = cos_sin_cache[pos, half + lanes].to(torch.float32).unsqueeze(1)

    def rope(xn):
        x1, x2 = xn[..., :half], xn[..., half:rotary_dim]
        return torch.cat(
            [x1 * cos - x2 * sin, x2 * cos + x1 * sin, xn[..., rotary_dim:]],
            dim=-1,
        ).to(out_dtype)

    q_out = rope(qn)
    k_out = rope(kn)
    return q_out, k_out


class TestFusedQKRmsnormRopeGateMrope(unittest.TestCase):
    """The fused kernel must match the torch reference for 1-D and mrope
    positions, and MRotaryEmbedding must build the right axis map per style."""

    HEAD_DIM = 256
    ROTARY_DIM = 64
    NUM_Q_HEADS = 8
    NUM_KV_HEADS = 2
    EPS = 1e-6

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("Test requires CUDA")
        from sglang.srt.runtime_context import publish
        from sglang.srt.server_args import ServerArgs

        # A real local model dir keeps ServerArgs resolution offline and fast;
        # publish() is required because MRotaryEmbedding reads the exec bag.
        publish(ServerArgs(model_path="/data02/models/Qwen3.8-27B"), role="test")
        torch.manual_seed(0)
        self.device = "cuda"

    def _run_case(self, positions, axis_map):
        T = positions.shape[-1]
        q_size = self.NUM_Q_HEADS * self.HEAD_DIM
        kv_size = self.NUM_KV_HEADS * self.HEAD_DIM
        q_gate = torch.randn(
            T, q_size * 2, dtype=torch.bfloat16, device=self.device
        )
        k = torch.randn(T, kv_size, dtype=torch.bfloat16, device=self.device)
        q_weight = torch.randn(self.HEAD_DIM, dtype=torch.bfloat16, device=self.device)
        k_weight = torch.randn(self.HEAD_DIM, dtype=torch.bfloat16, device=self.device)
        cos_sin_cache = torch.randn(
            4096, self.ROTARY_DIM, dtype=torch.float32, device=self.device
        )

        q_out, k_out, gate_out = fused_qk_gemma_rmsnorm_rope_gate(
            q_gate,
            k,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            self.EPS,
            self.NUM_Q_HEADS,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            self.ROTARY_DIM,
            has_gate=True,
            mrope_axis_map=axis_map,
        )

        ref_q, ref_k = _reference(
            q_gate,
            k,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            self.EPS,
            self.NUM_Q_HEADS,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            self.ROTARY_DIM,
            axis_map,
        )
        torch.testing.assert_close(q_out, ref_q.view_as(q_out), rtol=0, atol=0)
        torch.testing.assert_close(k_out, ref_k.view_as(k_out), rtol=0, atol=0)
        # Gate is a pure copy of the interleaved Q+Gate tail.
        gate_ref = q_gate.view(T, self.NUM_Q_HEADS, 2 * self.HEAD_DIM)[
            ..., self.HEAD_DIM :
        ]
        torch.testing.assert_close(gate_out, gate_ref, rtol=0, atol=0)

    def test_1d_positions(self):
        positions = torch.randint(0, 4096, (33,), device=self.device)
        self._run_case(positions, None)

    def test_mrope_positions(self):
        # Contiguous-style map: lanes 0..1..2 own T/H/W in section order.
        half = self.ROTARY_DIM // 2
        axis_map = torch.tensor(
            [0] * 12 + [1] * 10 + [2] * 10, dtype=torch.long, device=self.device
        )
        assert axis_map.numel() == half
        positions = torch.stack(
            [
                torch.randint(0, 4096, (33,), device=self.device),
                torch.randint(0, 512, (33,), device=self.device),
                torch.randint(0, 512, (33,), device=self.device),
            ]
        )
        self._run_case(positions, axis_map)

    def test_axis_map_styles(self):
        from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding

        def build(rotary_dim, **kw):
            return MRotaryEmbedding(
                head_size=64,
                rotary_dim=rotary_dim,
                max_position_embeddings=4096,
                base=1000000,
                is_neox_style=True,
                dtype=torch.bfloat16,
                **kw,
            )

        # Standard contiguous (Qwen3-VL style): sections must sum to
        # rotary_dim // 2 or the constructor rescales them.
        r = build(64, mrope_section=[12, 10, 10])
        self.assertEqual(r.axis_map.tolist(), [0] * 12 + [1] * 10 + [2] * 10)
        self.assertIsNone(r._legacy_axis_map)

        # GLM interleaved: round-robin skipping exhausted axes.
        glm = build(16, mrope_section=[2, 3, 3], mrope_interleaved_glm=True)
        self.assertEqual(glm._legacy_axis_map.tolist(), glm.axis_map.tolist())
        self.assertEqual(len(glm.axis_map), 8)


if __name__ == "__main__":
    unittest.main()
