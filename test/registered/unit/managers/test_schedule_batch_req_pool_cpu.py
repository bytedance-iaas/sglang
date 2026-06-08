import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestScheduleBatchReqPoolIndicesCpu(unittest.TestCase):
    def test_set_req_pool_indices_from_reqs_sets_cpu_and_device_tensors(self):
        req0 = types.SimpleNamespace(rid="r0", req_pool_idx=3)
        req1 = types.SimpleNamespace(rid="r1", req_pool_idx=5)
        batch = ScheduleBatch(reqs=[req0, req1], device="cpu")

        batch.set_req_pool_indices_from_reqs()

        self.assertTrue(torch.equal(batch.req_pool_indices_cpu, torch.tensor([3, 5])))
        self.assertTrue(torch.equal(batch.req_pool_indices, torch.tensor([3, 5])))

    def test_prepare_for_prebuilt_sets_req_pool_indices_cpu(self):
        req0 = types.SimpleNamespace(
            req_pool_idx=3,
            fill_ids=[1, 2, 3],
            prefix_indices=torch.tensor([10], dtype=torch.int64),
            extend_input_len=2,
            origin_input_ids=[1, 2, 3],
            output_ids=[],
            retracted_stain=True,
            cached_tokens=0,
            already_computed=0,
            is_retracted=True,
            multimodal_inputs=None,
        )
        req1 = types.SimpleNamespace(
            req_pool_idx=5,
            fill_ids=[4, 5],
            prefix_indices=torch.empty(0, dtype=torch.int64),
            extend_input_len=2,
            origin_input_ids=[4, 5],
            output_ids=[],
            retracted_stain=True,
            cached_tokens=0,
            already_computed=0,
            is_retracted=True,
            multimodal_inputs=None,
        )
        req_to_token_pool = types.SimpleNamespace(
            req_to_token=torch.arange(64, dtype=torch.int64).reshape(8, 8)
        )
        batch = ScheduleBatch(
            reqs=[req0, req1],
            req_to_token_pool=req_to_token_pool,
            model_config=types.SimpleNamespace(vocab_size=16),
            device="cpu",
        )

        with patch(
            "sglang.srt.disaggregation.decode_schedule_batch_mixin."
            "SamplingBatchInfo.from_schedule_batch",
            return_value=MagicMock(),
        ):
            batch.prepare_for_prebuilt()

        self.assertTrue(torch.equal(batch.req_pool_indices_cpu, torch.tensor([3, 5])))
        self.assertTrue(torch.equal(batch.req_pool_indices, torch.tensor([3, 5])))

    def test_merge_batch_repairs_missing_req_pool_indices_cpu(self):
        self_batch = ScheduleBatch(
            reqs=[object(), object()],
            model_config=types.SimpleNamespace(is_encoder_decoder=False),
            req_pool_indices=torch.tensor([1, 2], dtype=torch.int64),
            req_pool_indices_cpu=None,
            seq_lens=torch.tensor([10, 20], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int64),
            orig_seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            output_ids=torch.tensor([7, 8], dtype=torch.int64),
            seq_lens_sum=30,
            sampling_info=MagicMock(),
            return_logprob=False,
            has_stream=False,
            has_grammar=False,
            return_hidden_states=False,
            is_prefill_only=False,
        )
        other_batch = ScheduleBatch(
            reqs=[object()],
            model_config=types.SimpleNamespace(is_encoder_decoder=False),
            req_pool_indices=torch.tensor([3], dtype=torch.int64),
            req_pool_indices_cpu=None,
            seq_lens=torch.tensor([30], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([30], dtype=torch.int64),
            orig_seq_lens=torch.tensor([30], dtype=torch.int32),
            output_ids=torch.tensor([9], dtype=torch.int64),
            seq_lens_sum=30,
            sampling_info=MagicMock(),
            return_logprob=False,
            has_stream=False,
            has_grammar=False,
            return_hidden_states=False,
            is_prefill_only=False,
        )

        self_batch.merge_batch(other_batch)

        self.assertTrue(
            torch.equal(self_batch.req_pool_indices_cpu, torch.tensor([1, 2, 3]))
        )

    def test_prepare_for_decode_repairs_missing_req_pool_indices_cpu(self):
        req = types.SimpleNamespace(
            decode_batch_idx=0,
            kv_committed_len=10,
            kv_allocated_len=10,
        )
        batch = ScheduleBatch(
            reqs=[req],
            model_config=types.SimpleNamespace(is_encoder_decoder=False),
            req_pool_indices=torch.tensor([4], dtype=torch.int64),
            req_pool_indices_cpu=None,
            seq_lens=torch.tensor([10], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([10], dtype=torch.int64),
            orig_seq_lens=torch.tensor([10], dtype=torch.int32),
            output_ids=torch.tensor([7], dtype=torch.int64),
            seq_lens_sum=10,
            sampling_info=types.SimpleNamespace(
                penalizer_orchestrator=types.SimpleNamespace(is_required=False)
            ),
            spec_algorithm=types.SimpleNamespace(is_none=lambda: True),
            enable_overlap=False,
            device="cpu",
            hisparse_coordinator=MagicMock(),
        )

        with (
            patch(
                "sglang.srt.managers.schedule_batch.alloc_for_decode",
                return_value=torch.tensor([42], dtype=torch.int64),
            ),
            patch(
                "sglang.srt.managers.schedule_batch.get_global_server_args",
                return_value=types.SimpleNamespace(
                    enable_mamba_extra_buffer=lambda: False
                ),
            ),
        ):
            batch.prepare_for_decode()

        self.assertTrue(torch.equal(batch.req_pool_indices_cpu, torch.tensor([4])))
        batch.hisparse_coordinator.map_last_loc_to_buffer.assert_called_once()

    def test_filter_batch_to_empty_clears_req_pool_indices(self):
        req = types.SimpleNamespace(finished=lambda: True)
        batch = ScheduleBatch(
            reqs=[req],
            model_config=types.SimpleNamespace(is_encoder_decoder=False),
            req_pool_indices=torch.tensor([4], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([4], dtype=torch.int64),
            seq_lens=torch.tensor([10], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([10], dtype=torch.int64),
            orig_seq_lens=torch.tensor([10], dtype=torch.int32),
            output_ids=torch.tensor([7], dtype=torch.int64),
            seq_lens_sum=10,
            device="cpu",
        )

        batch.filter_batch()

        self.assertEqual(batch.req_pool_indices_cpu.numel(), 0)
        self.assertEqual(batch.req_pool_indices.numel(), 0)
        self.assertEqual(batch.seq_lens_cpu.numel(), 0)
        self.assertEqual(batch.seq_lens.numel(), 0)
        self.assertEqual(batch.seq_lens_sum, 0)


if __name__ == "__main__":
    unittest.main()
