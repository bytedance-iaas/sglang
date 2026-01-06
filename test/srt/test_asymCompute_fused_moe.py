import unittest

import torch
from tqdm import tqdm

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.asym_comp import AsymCompMoeQuantInfo
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase
from sglang.srt.layers.moe.topk import StandardTopKOutput

set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

class TestFusedMOE(CustomTestCase):
    NUM_EXPERTS = [8, 64]
    TOP_KS = [2, 4]

    @staticmethod
    def create_random_cuda_tensor(shape, dtype, mean=0, std=0.01):
        """Create a random CUDA tensor

        Args:
            shape: Tensor shape
            dtype: Data type
            mean: Mean value
            std: Standard deviation

        Returns:
            torch.Tensor: Randomly initialized CUDA tensor
        """
        return torch.empty(shape, dtype=dtype, device="cuda").normal_(mean, std)

    def get_tolerance(self, dtype):
        """Get tolerance values for different data types

        Args:
            dtype: Data type

        Returns:
            tuple: (relative tolerance, absolute tolerance)
        """
        if dtype == torch.float32:
            return 1e-5, 1e-5
        elif dtype in [torch.float16, torch.bfloat16]:
            return 1e-5, 1e-5
        else:
            return 1e-2, 1e-2  # Default values for other types

    def torch_naive_moe_update(self, a, w1, w2, topk_output, return_per_expert=False):
        B, D = a.shape
        topk_weight, topk_ids, _ = topk_output
        # topk_weight/topk_ids are [B, topk]
        a_rep = a.view(B, 1, D).expand(B, topk_weight.shape[1], D).reshape(-1, D)

        out = torch.zeros(B * topk_weight.shape[1], w2.shape[1], dtype=a.dtype, device=a.device)

        # flatten routing
        topk_weight_f = topk_weight.reshape(-1).to(torch.float32)  # use fp32 weights for ref
        topk_ids_f = topk_ids.reshape(-1).to(torch.int64)

        w1_compute, w2_compute = w1, w2
        for i in range(w1_compute.shape[0]):
            mask = (topk_ids_f == i)
            if mask.any():
                out[mask] = SiluAndMul()(a_rep[mask] @ w1_compute[i].transpose(0, 1)) @ w2_compute[i].transpose(0, 1)

        weighted = out.view(B, -1, w2.shape[1]) * topk_weight_f.view(B, -1, 1).to(out.dtype)
        return weighted if return_per_expert else weighted.sum(dim=1)


    def torch_naive_moe(
        self,
        a,
        w1,
        w2,
        score,
        topk,
        return_per_expert: bool = False,
    ):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

        B, D = a.shape
        a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
        out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)
        topk_weight = topk_weight.view(-1)
        topk_ids = topk_ids.view(-1)

        if w1.dtype == torch.float8_e4m3fn:
            w1_compute = w1.to(a.dtype)
            w2_compute = w2.to(a.dtype)
        else:
            w1_compute = w1
            w2_compute = w2

        for i in range(w1_compute.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                out[mask] = SiluAndMul()(
                    a[mask] @ w1_compute[i].transpose(0, 1)
                ) @ w2_compute[i].transpose(0, 1)

        weighted = out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(
            out.dtype
        )

        if return_per_expert:
            return weighted

        return weighted.sum(dim=1)

    def _test_case(self, m, n, k, e, topk, dtype):
        rtol, atol = self.get_tolerance(dtype)

        a = self.create_random_cuda_tensor((m, k), dtype)
        w1 = self.create_random_cuda_tensor((e, 2 * n, k), dtype)
        w2 = self.create_random_cuda_tensor((e, k, n), dtype)
        w1_tri = w1.clone()
        w2_tri = w2.clone()
        w1_tri = w1_tri.transpose(-2, -1).contiguous()
        w2_tri = w2_tri.transpose(-2, -1).contiguous()
        score = self.create_random_cuda_tensor((m, e), dtype)

        topk_op = TopK(
            top_k=topk,
            renormalize=False,
            use_grouped_topk=False,
        )

        # score: [m, e]
        prob = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(prob, k=topk, dim=-1)
        topk_weights = topk_weights.to(a.dtype).contiguous()  # match compute dtype
        triton_topk_output = StandardTopKOutput(topk_weights, topk_ids.to(torch.int32).contiguous(), None)  # STANDARD format tuple

        # topk_op.topk_config.output_format = TopKOutputFormat.STANDARD
        # triton_topk_output = topk_op.forward_cuda(
        #     hidden_states=a,
        #     router_logits=score,
        # )

        quant_info = AsymCompMoeQuantInfo(w13_weight=w1_tri, w2_weight=w2_tri)
        dispatch_output = StandardDispatchOutput(
            hidden_states=a, hidden_states_scale=None, topk_output=triton_topk_output
        )

        # torch_per_expert = self.torch_naive_moe(
        #     a, w1, w2, score, topk, return_per_expert=True
        # )
        # torch_per_expert = self.torch_naive_moe_update(a, w1, w2, triton_topk_output, return_per_expert=True)
# torch_combined = torch_per_expert.sum(dim=1)
        # torch_combined = torch_per_expert.sum(dim=1)
        # import ipdb
        # ipdb.set_trace()
        # def run_runner(config):
        #     runner = MoeRunner(MoeRunnerBackend.ASYM_COMP, config)
        #     # import ipdb
        #     # ipdb.set_trace()
        #     result = runner.run(dispatch_output, quant_info)
        #     return result.hidden_states

        common = MoeRunnerConfig(
            num_experts=e,
            num_local_experts=e,
            top_k=topk,
            apply_router_weight_on_input=False,  # match your naive path (weights on output)
            routed_scaling_factor=1.0,           # or None, but be explicit
            activation="silu",                   # since your naive uses SiluAndMul
            is_gated=True,                       # SiluAndMul implies gated
        )


        runner_triton = MoeRunner(MoeRunnerBackend.TRITON, common)
        out_triton = runner_triton.run(dispatch_output, quant_info).hidden_states

        runner_custom = MoeRunner(MoeRunnerBackend.ASYM_COMP, common)
        out_custom = runner_custom.run(dispatch_output, quant_info).hidden_states

        torch.testing.assert_close(out_custom, out_triton, rtol=1e-2, atol=5e-5)


        # Combined output (no_combine=False)
        # non_fused_config = MoeRunnerConfig(inplace=False, **common)
        # non_fused_output = run_runner(non_fused_config)
        # torch.testing.assert_close(
        #     non_fused_output, torch_combined, rtol=rtol, atol=atol
        # )

        # # Per-expert output (no_combine=True)
        # non_fused_no_combine_config = MoeRunnerConfig(inplace=False, no_combine=True, **common)
        # non_fused_no_combine_output = run_runner(non_fused_no_combine_config)
        # torch.testing.assert_close(
        #     non_fused_no_combine_output, torch_per_expert, rtol=rtol, atol=atol
        # )

    def test_various_configurations(self):
        # m_values = [1, 32, 64, 256]
        # n_values = [128, 1024]
        # k_values = [128, 512, 1024]

        m_values = [32]
        n_values = [128]
        k_values = [128]

        dtypes = [torch.bfloat16]

        # Calculate total number of tests
        total_tests = (
            len(m_values)
            * len(n_values)
            * len(k_values)
            * len(self.NUM_EXPERTS)
            * len(self.TOP_KS)
            * len(dtypes)
        )

        # Create progress bar
        with tqdm(total=total_tests, desc="Running MoE tests") as pbar:
            for m in m_values:
                for n in n_values:
                    for k in k_values:
                        for e in self.NUM_EXPERTS:
                            for topk in self.TOP_KS:
                                for dtype in dtypes:
                                    with self.subTest(
                                        m=m,
                                        n=n,
                                        k=k,
                                        e=e,
                                        topk=topk,
                                        dtype=dtype,
                                    ):
                                        self._test_case(
                                            m,
                                            n,
                                            k,
                                            e,
                                            topk,
                                            dtype,
                                        )
                                        torch.cuda.empty_cache()
                                    pbar.update(1)


if __name__ == "__main__":
    unittest.main()
