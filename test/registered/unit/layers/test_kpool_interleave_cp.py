"""KPool global compression/local query wiring; CUDA cache parity is separate."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.dsa import dsa_backend_kpool, kpool_plan
from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.layers.cp import interleave_kpool
from sglang.srt.layers.cp.interleave import InterleaveContextParallelMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def batch(extend=(1, 1, 6), seq=(5, 9, 18), slots=(9, 2, 7)):
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        attn_cp_metadata=InterleaveContextParallelMetadata(total_seq_lens=sum(extend)),
        batch_size=len(extend),
        extend_seq_lens_cpu=list(extend),
        seq_lens_cpu=torch.tensor(seq),
        req_pool_indices=torch.tensor(slots),
        req_pool_indices_cpu=torch.tensor(slots),
        positions=torch.arange(sum(extend)),
    )


class TestKPoolInterleaveCP(unittest.TestCase):
    def test_non_cp_keeps_objects_and_does_not_read_parallel(self):
        key, score, positions = object(), object(), object()
        fb = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        with (
            patch.object(interleave_kpool, "get_parallel", side_effect=AssertionError),
            patch.object(
                interleave_kpool, "get_cp_strategy", side_effect=AssertionError
            ),
        ):
            self.assertEqual(interleave_kpool.local_query_inputs(fb, object()), {})
            self.assertEqual(
                interleave_kpool.materialize_compression_inputs(
                    key, score, positions, fb
                ),
                (key, score, positions),
            )

    def test_local_requests_preserve_full_compression_plan(self):
        fb = batch()
        full = kpool_plan._kpool_cpu_plan(fb, 4, 64)
        for rank, selected, lengths in ((0, [0, 2], [1, 3]), (1, [1, 2], [1, 3])):
            with self.subTest(rank=rank):
                metadata = SimpleNamespace(
                    real_page_table=torch.arange(8).reshape(2, 4),
                    dsa_seqlens_expanded=torch.arange(4),
                    dsa_extend_seq_lens_list=lengths,
                    indexer_seq_lens_cpu=fb.seq_lens_cpu[selected],
                )
                with patch.object(
                    interleave_kpool,
                    "get_parallel",
                    return_value=SimpleNamespace(attn_cp_rank=rank, attn_cp_size=2),
                ):
                    inputs = interleave_kpool.local_query_inputs(fb, metadata)
                self.assertIs(inputs["local_real_page_table"], metadata.real_page_table)
                self.assertIs(
                    inputs["local_seqlens_expanded"], metadata.dsa_seqlens_expanded
                )
                self.assertEqual(
                    inputs["local_req_pool_indices"].tolist(),
                    fb.req_pool_indices[selected].tolist(),
                )
                local = kpool_plan._kpool_cpu_plan(
                    fb,
                    4,
                    64,
                    local_extend_seq_lens_cpu=lengths,
                    local_seq_lens_cpu=inputs["local_seq_lens_cpu"],
                )
                for field in (
                    "pool_req",
                    "pool_pool_id",
                    "pool_chunk_src",
                    "pool_n_from_tail",
                    "tail_req",
                    "tail_chunk_src",
                    "tail_dst_logical_start",
                    "tail_n_write",
                ):
                    self.assertEqual(getattr(local, field), getattr(full, field))
                self.assertEqual(sum(local.ragged_q_len), 4)
                self.assertEqual(sum(full.ragged_q_len), 8)

    def test_backend_passes_existing_local_query_contract(self):
        fb = batch(extend=(8192,), seq=(8192,), slots=(3,))
        metadata = SimpleNamespace()
        inputs = dsa_backend_kpool._KPoolForwardInputs(object(), object())
        local = dict(local_seqlens_expanded=object(), local_req_pool_indices=object())
        backend = SimpleNamespace(
            dsa_index_kpool=4,
            physical_page_size=256,
            _kpool_index_page_size=lambda: 64,
            _build_kpool_paged_mqa_schedule_metadata=lambda: False,
        )
        with (
            patch.object(dsa_backend_kpool, "local_query_inputs", return_value=local),
            patch.object(
                dsa_backend_kpool, "init_kpool_extend_metadata", return_value=metadata
            ) as init,
        ):
            self.assertIs(
                dsa_backend_kpool.DeepseekSparseAttnBackendKPoolMixin._init_kpool_metadata(
                    backend, metadata, fb, TopkTransformMethod.PAGED, inputs
                ),
                metadata,
            )
        self.assertIs(
            init.call_args.kwargs["full_seqlens_expanded"], inputs.full_seqlens_expanded
        )
        for key, value in local.items():
            self.assertIs(init.call_args.kwargs[key], value)

    def test_compression_materializes_adjacent_tokens_without_dtype_promotion(self):
        fb = batch(extend=(7,), seq=(263,), slots=(3,))
        key = torch.arange(28).reshape(7, 4).to(torch.bfloat16)
        for score_dtype in (torch.bfloat16, torch.float32):
            score = (key.float() + 100).to(score_dtype)
            for rank in (0, 1):
                calls = []
                outputs = iter((key, score))

                def gather(local, received):
                    self.assertIs(received, fb)
                    calls.append(local)
                    if key.dtype == score.dtype:
                        torch.testing.assert_close(
                            local,
                            torch.cat((key[rank::2], score[rank::2]), -1),
                            rtol=0,
                            atol=0,
                        )
                        return torch.cat((key, score), -1)
                    expected = next(outputs)
                    torch.testing.assert_close(local, expected[rank::2], rtol=0, atol=0)
                    return expected

                with patch.object(
                    interleave_kpool,
                    "get_cp_strategy",
                    return_value=SimpleNamespace(gather_kv_cache=gather),
                ):
                    actual = interleave_kpool.materialize_compression_inputs(
                        key[rank::2], score[rank::2], fb.positions[rank::2], fb
                    )
                torch.testing.assert_close(actual[0], key, rtol=0, atol=0)
                torch.testing.assert_close(actual[1], score, rtol=0, atol=0)
                self.assertEqual(actual[0].dtype, key.dtype)
                self.assertEqual(actual[1].dtype, score.dtype)
                self.assertIs(actual[2], fb.positions)
                self.assertEqual(len(calls), 1 if key.dtype == score.dtype else 2)

    def test_indexer_cache_write_receives_global_compression_inputs(self):
        fb = batch(extend=(8,), seq=(11,), slots=(3,))
        key, score = torch.zeros(8, 4), torch.ones(8, 4)
        marker = object()
        backend = SimpleNamespace(
            _compress_write_extend=lambda **kwargs: kwargs,
            index_kpool_compress_gate=torch.eye(4),
        )
        with patch(
            "sglang.srt.layers.attention.dsa.dsa_indexer_kpool.materialize_compression_inputs",
            return_value=(key, score, fb.positions),
        ) as materialize:
            result = IndexerKPool._compress_write(
                backend,
                torch.ones(4, 4),
                key[:4],
                fb.positions[:4],
                fb,
                2,
                marker,
                return_compressed=True,
                write_cache=True,
            )
        self.assertEqual(materialize.call_count, 1)
        self.assertIs(result["key"], key)
        self.assertIs(result["gate_score"], score)
        self.assertIs(result["positions"], fb.positions)
        self.assertIs(result["metadata"], marker)
        self.assertTrue(result["return_compressed"] and result["write_cache"])


if __name__ == "__main__":
    unittest.main()
