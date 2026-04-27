import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import ForwardBatch


register_cpu_ci(est_time=4, suite="stage-a-test-cpu")

class TestForwardBatchDCP(CustomTestCase):
    def test_init_new_computes_dcp_kv_layout(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            seq_lens=torch.tensor([4], dtype=torch.int32),
            input_ids=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            out_cache_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            mamba_track_indices=None,
            mamba_track_mask=None,
            mamba_track_seqlens=None,
            multimodal_inputs=None,
            encoder_cached=None,
            encoder_lens=None,
            encoder_lens_cpu=None,
            encoder_out_cache_loc=None,
            seq_lens_sum=4,
            seq_lens_cpu=None,
            orig_seq_lens=None,
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            is_extend_in_batch=False,
            all_extend_in_batch=False,
            can_run_dp_cuda_graph=False,
            global_forward_mode=None,
            is_prefill_only=False,
            lora_ids=None,
            sampling_info=None,
            spec_algorithm=None,
            spec_info=None,
            capture_hidden_mode=None,
            input_embeds=None,
            replace_embeds=None,
            replace_positions=None,
            token_type_ids=None,
            tbo_split_seq_index=None,
            dimensions=None,
            return_hidden_states_before_norm=False,
            return_pooled_hidden_states=False,
            reqs=[SimpleNamespace(rid="rid-0")],
            extend_input_logprob_token_ids=None,
            global_num_tokens=None,
            global_num_tokens_for_logprob=None,
            dllm_config=None,
        )
        model_runner = SimpleNamespace(
            device=torch.device("cpu"),
            req_to_token_pool=SimpleNamespace(),
            token_to_kv_pool=SimpleNamespace(),
            attn_backend=SimpleNamespace(),
            server_args=SimpleNamespace(enable_lora=False),
            dcp_size=2,
            dcp_rank=1,
            use_ngram_embedding=False,
            model_is_mrope=False,
            is_hybrid_swa=False,
        )

        with patch(
            "sglang.srt.model_executor.forward_batch_info.enable_num_token_non_padded",
            return_value=False,
        ):
            ret = ForwardBatch.init_new(batch, model_runner)

        self.assertEqual(ret.out_cache_loc.tolist(), [0, 0, 1, 1])
        self.assertEqual(ret.dcp_kv_mask.tolist(), [False, True, False, True])


if __name__ == "__main__":
    unittest.main()