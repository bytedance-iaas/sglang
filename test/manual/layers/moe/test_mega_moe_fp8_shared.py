"""Target-GPU smoke for native FP8 MegaMoE with an independent shared expert.

This exercises the SGLang-owned path around DeepGEMM rather than calling the
DeepGEMM kernel directly: native routed-weight conversion, shared-weight
conversion, symmetric-buffer sizing, SGLang pre-dispatch (including the shared
activation-scale layout), and shared tuple forwarding into the fused kernel.

The default shape matches one GLM-5.3 EP8 rank (32 routed experts, H=6144,
I=2048, Top8, one shared expert). Run inside the target CUDA image with the
candidate DeepGEMM wheel installed::

    torchrun --standalone --nproc-per-node=1 \
      test/manual/layers/moe/test_mega_moe_fp8_shared.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import mega_moe


class TestNativeFp8SharedMegaMoe(unittest.TestCase):
    NUM_LOCAL_EXPERTS = 32
    NUM_TOKENS = 64
    HIDDEN = 6144
    INTERMEDIATE = 2048
    TOP_K = 8

    @staticmethod
    def _fp8_parameter(shape):
        return torch.nn.Parameter(
            torch.zeros(shape, dtype=torch.float8_e4m3fn, device="cuda"),
            requires_grad=False,
        )

    @staticmethod
    def _scale_parameter(shape):
        return torch.nn.Parameter(
            torch.ones(shape, dtype=torch.float32, device="cuda"),
            requires_grad=False,
        )

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        major, _minor = torch.cuda.get_device_capability()
        if major != 10:
            self.skipTest("DeepGEMM native FP8 MegaMoE requires SM100")

        from deep_gemm.utils.dist import init_dist

        self.rank, self.world_size, self.group = init_dist(
            torch.cuda.current_device(), 1
        )
        self.assertEqual(self.world_size, 1)
        mega_moe._MEGA_MOE_SYMM_BUFFER.clear()

    def tearDown(self):
        import torch.distributed as dist

        for buffer in mega_moe._MEGA_MOE_SYMM_BUFFER.values():
            buffer.destroy()
        mega_moe._MEGA_MOE_SYMM_BUFFER.clear()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def test_glm53_rank_shape(self):
        num_experts = self.NUM_LOCAL_EXPERTS
        hidden = self.HIDDEN
        intermediate = self.INTERMEDIATE

        experts = SimpleNamespace(
            num_experts=num_experts,
            w13_weight=self._fp8_parameter((num_experts, 2 * intermediate, hidden)),
            w13_weight_scale_inv=self._scale_parameter(
                (num_experts, 2 * intermediate // 128, hidden // 128)
            ),
            w2_weight=self._fp8_parameter((num_experts, hidden, intermediate)),
            w2_weight_scale_inv=self._scale_parameter(
                (num_experts, hidden // 128, intermediate // 128)
            ),
            should_fuse_routed_scaling_factor_in_topk=False,
        )
        mega_moe.build_native_fp8_mega_moe_experts_weights(experts)

        gate_up_proj = SimpleNamespace(
            weight=self._fp8_parameter((2 * intermediate, hidden)),
            weight_scale_inv=self._scale_parameter(
                (2 * intermediate // 128, hidden // 128)
            ),
        )
        down_proj = SimpleNamespace(
            weight=self._fp8_parameter((hidden, intermediate)),
            weight_scale_inv=self._scale_parameter(
                (hidden // 128, intermediate // 128)
            ),
        )
        shared_l1, shared_l2 = mega_moe.build_mega_moe_shared_weights(
            SimpleNamespace(gate_up_proj=gate_up_proj, down_proj=down_proj)
        )

        topk_ids = (
            torch.arange(self.TOP_K, dtype=torch.int64, device="cuda")
            .unsqueeze(0)
            .expand(self.NUM_TOKENS, -1)
            .contiguous()
        )
        topk_weights = torch.full(
            (self.NUM_TOKENS, self.TOP_K),
            1.0 / self.TOP_K,
            dtype=torch.float32,
            device="cuda",
        )
        moe = SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=hidden,
                num_experts_per_tok=self.TOP_K,
                moe_intermediate_size=intermediate,
                swiglu_limit=None,
            ),
            experts=experts,
            gate=lambda *_args, **_kwargs: torch.empty(
                (self.NUM_TOKENS, num_experts),
                dtype=torch.float32,
                device="cuda",
            ),
            topk=lambda *_args, **_kwargs: SimpleNamespace(
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            ),
            is_hash=False,
            n_shared_experts=1,
            num_fused_shared_experts=0,
            layer_id=0,
            routed_scaling_factor=1.0,
            shared_experts_is_fp8=True,
            shared_experts_weight_block_size=[128, 128],
            mega_shared_l1_weights=shared_l1,
            mega_shared_l2_weights=shared_l2,
        )
        mega_backend = SimpleNamespace(is_megamoe=lambda: True)

        with (
            mega_moe.envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_FUSE_SHARED_EXPERTS.override(
                True
            ),
            mega_moe.envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.override(
                self.NUM_TOKENS
            ),
            patch.object(mega_moe, "get_moe_a2a_backend", return_value=mega_backend),
            patch.object(
                mega_moe.ExpertLocationDispatchInfo,
                "init_new",
                return_value=object(),
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_moe_ep_group",
                return_value=SimpleNamespace(device_group=self.group),
            ),
        ):
            self.assertTrue(mega_moe.should_fuse_mega_moe_shared_experts(moe))
            output = mega_moe._run_mega_routed(
                moe,
                torch.randn(
                    (self.NUM_TOKENS, hidden),
                    dtype=torch.bfloat16,
                    device="cuda",
                ),
                forward_batch=None,
                input_ids_global=None,
                num_tokens=self.NUM_TOKENS,
            )

        torch.cuda.synchronize()
        self.assertEqual(output.shape, (self.NUM_TOKENS, hidden))
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertEqual(torch.count_nonzero(output).item(), 0)

        self.assertEqual(len(mega_moe._MEGA_MOE_SYMM_BUFFER), 1)
        buffer = next(iter(mega_moe._MEGA_MOE_SYMM_BUFFER.values()))
        self.assertEqual(buffer.num_experts, num_experts)
        self.assertEqual(buffer.num_topk, self.TOP_K)
        self.assertEqual(buffer.mma_type, "fp8xfp8")
        self.assertIsNotNone(buffer.shared_l1_acts_sf)


if __name__ == "__main__":
    unittest.main()
