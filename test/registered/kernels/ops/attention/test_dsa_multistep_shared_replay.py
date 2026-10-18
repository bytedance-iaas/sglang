"""Real CUDA capture/replay of shared draft metadata with changed inputs.

This checks metadata values and pointers, not model quality or serving throughput.
The target Docker image must supply the usual DSA/DeepGEMM/FlashInfer kernels.
"""

import unittest
from types import SimpleNamespace

import deep_gemm
import torch

from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DeepseekSparseAttnMultiStepBackend,
)
from sglang.srt.layers.attention.trtllm_mla_backend import (
    make_persistent_multi_ctas_kv_counter_buffer,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "requires target CUDA runtime")
class TestDSAMultiStepSharedReplay(unittest.TestCase):
    def _backend(self, req_to_token, step):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.device = "cuda"
        backend.num_q_heads = 64
        backend.real_page_size = 64
        backend.req_to_token = req_to_token
        backend.speculative_num_draft_tokens = 4
        backend.speculative_step_id = step
        backend.hisparse_coordinator = None
        backend.use_fused_topk = False
        backend.dsa_decode_impl = "trtllm"
        backend.dsa_index_topk = 128
        backend.dsa_topk_backend = SimpleNamespace(should_use_topk_v2=lambda: False)
        backend.set_dsa_prefill_impl = lambda forward_batch: None
        backend._arange_buf = torch.arange(64, dtype=torch.int32, device="cuda")
        backend._multi_ctas_kv_counter_buffer = (
            make_persistent_multi_ctas_kv_counter_buffer(torch.device("cuda"), 64, 8)
        )
        return backend

    def test_changed_inputs_tiers_and_independent_counters(self):
        req_to_token = torch.arange(8 * 256, dtype=torch.int64, device="cuda").view(
            8, 256
        )
        multi = object.__new__(DeepseekSparseAttnMultiStepBackend)
        multi.speculative_num_steps = 3
        multi.attn_backends = [self._backend(req_to_token, step) for step in range(2)]
        multi.init_cuda_graph_state(8, 32)
        first, sibling = multi.attn_backends
        self.assertEqual(sibling.decode_cuda_graph_metadata, {})
        counters = [b._multi_ctas_kv_counter_buffer for b in multi.attn_backends]
        self.assertNotEqual(counters[0].data_ptr(), counters[1].data_ptr())
        graphs = {}

        for bs in (1, 8):
            seq = torch.full((bs,), 65, dtype=torch.int64, device="cuda")
            req = torch.arange(bs, dtype=torch.int64, device="cuda")
            batch = SimpleNamespace(
                batch_size=bs,
                forward_mode=ForwardMode.DECODE,
                req_pool_indices=req,
                seq_lens=seq,
                seq_lens_cpu=None,
                seq_lens_sum=65 * bs,
                spec_info=None,
            )
            multi.init_forward_metadata_out_graph(batch, in_capture=True)
            multi.init_forward_metadata_out_graph(batch)
            metadata = first.forward_metadata
            self.assertIs(metadata, sibling.forward_metadata)
            outputs = [torch.empty_like(seq, dtype=torch.int32) for _ in range(2)]
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for backend, output in zip(multi.attn_backends, outputs):
                    # Each step reads the same metadata but writes its own output
                    # and counter, just as the real draft forward must do.
                    output.copy_(backend.forward_metadata.cache_seqlens_int32)
                    backend._multi_ctas_kv_counter_buffer.fill_(
                        backend.speculative_step_id + 1
                    )
            graphs[bs] = (
                graph,
                batch,
                outputs,
                metadata.cache_seqlens_int32.data_ptr(),
            )

        for bs, length in ((8, 127), (1, 192), (8, 64), (1, 1)):
            graph, batch, outputs, ptr = graphs[bs]
            batch.seq_lens.fill_(length)
            batch.req_pool_indices.copy_(torch.arange(bs - 1, -1, -1, device="cuda"))
            multi.init_forward_metadata_out_graph(batch)
            graph.replay()
            torch.cuda.synchronize()
            metadata = first.forward_metadata
            self.assertEqual(metadata.cache_seqlens_int32.data_ptr(), ptr)
            self.assertIs(metadata, sibling.forward_metadata)
            for output in outputs:
                torch.testing.assert_close(
                    output, batch.seq_lens.to(torch.int32), rtol=0, atol=0
                )
            live = torch.arange(256, device="cuda")[None, :] < batch.seq_lens[:, None]
            expected_pages = req_to_token[batch.req_pool_indices].to(torch.int32)
            torch.testing.assert_close(
                metadata.page_table_1[live], expected_pages[live], rtol=0, atol=0
            )
            torch.testing.assert_close(
                metadata.dsa_cache_seqlens_int32,
                batch.seq_lens.clamp(max=128).to(torch.int32),
                rtol=0,
                atol=0,
            )
            expected_cu = torch.nn.functional.pad(
                batch.seq_lens.clamp(max=128)
                .to(torch.int32)
                .cumsum(0, dtype=torch.int32),
                (1, 0),
            )
            torch.testing.assert_close(
                metadata.dsa_cu_seqlens_k, expected_cu, rtol=0, atol=0
            )
            expected_schedule = deep_gemm.get_paged_mqa_logits_metadata(
                batch.seq_lens.to(torch.int32).view(-1, 1), 64, deep_gemm.get_num_sms()
            )
            torch.testing.assert_close(
                metadata.paged_mqa_schedule_metadata, expected_schedule, rtol=0, atol=0
            )
            for step, (backend, counter) in enumerate(
                zip(multi.attn_backends, counters)
            ):
                self.assertIs(backend._multi_ctas_kv_counter_buffer, counter)
                self.assertTrue(bool(torch.all(counter == step + 1)))


if __name__ == "__main__":
    unittest.main()
