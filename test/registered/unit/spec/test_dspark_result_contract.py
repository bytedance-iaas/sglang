import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import validate_dflash_request
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _SpecAlgorithm:
    def __init__(self, *, dspark: bool):
        self._dspark = dspark

    def is_none(self):
        return False

    def is_dspark(self):
        return self._dspark


class _Req:
    def __init__(self, committed_len: int):
        self.kv_committed_len = committed_len
        self.is_retracted = False
        self.spec_verify_ct = 0
        self.spec_num_correct_drafts = 0
        self.histogram = []

    def finished(self):
        return False

    def update_spec_correct_drafts_histogram(self, value):
        self.histogram.append(value)


def _draft_input(bs: int):
    return SimpleNamespace(
        bonus_tokens=torch.arange(bs, dtype=torch.int64),
        new_seq_lens=torch.arange(100, 100 + bs, dtype=torch.int64),
    )


def _decode_result(bs: int, stride: int = 3, *, active: bool = True):
    next_draft_input = _draft_input(bs)
    return GenerationBatchResult(
        logits_output=object() if active else None,
        next_token_ids=torch.arange(bs * stride, dtype=torch.int64),
        accept_lens=torch.ones(bs, dtype=torch.int32),
        block_accept_lens=torch.ones(bs, dtype=torch.int32),
        cap_lens=torch.full((bs,), stride, dtype=torch.int32),
        speculative_num_draft_tokens=stride,
        next_draft_input=next_draft_input,
        new_seq_lens=next_draft_input.new_seq_lens,
    )


class TestDSparkResultContract(unittest.TestCase):
    def test_generation_result_accepts_complete_dspark_payload(self):
        result = _decode_result(bs=2)
        self.assertEqual(result.block_accept_lens.tolist(), [1, 1])
        self.assertEqual(result.cap_lens.tolist(), [3, 3])
        self.assertEqual(result.new_seq_lens.tolist(), [100, 101])

    def test_prepare_for_decode_calls_dspark_reservation_without_overlap(self):
        draft_input = SimpleNamespace(prepare_for_decode=Mock())
        batch = SimpleNamespace(
            reqs=[object()],
            input_embeds=object(),
            attn_cp_metadata=None,
            uses_result_based_spec=True,
            spec_info=draft_input,
            spec_algorithm=_SpecAlgorithm(dspark=True),
        )

        ScheduleBatch.prepare_for_decode(batch)

        draft_input.prepare_for_decode.assert_called_once_with(batch)
        self.assertEqual(batch.forward_mode, ForwardMode.DECODE)
        self.assertIsNone(batch.input_embeds)

    def test_scheduler_publishes_next_dspark_state(self):
        result = _decode_result(bs=2)
        batch = SimpleNamespace(
            batch_size=lambda: 2,
            seq_lens_cpu=torch.zeros(2, dtype=torch.int64),
            seq_lens=torch.zeros(2, dtype=torch.int64),
            seq_lens_sum=0,
            spec_info=None,
            input_ids=torch.ones(2, dtype=torch.int64),
        )

        Scheduler._apply_dspark_sync_result(object(), batch, result)

        self.assertIs(batch.spec_info, result.next_draft_input)
        self.assertIs(batch.seq_lens, result.new_seq_lens)
        self.assertEqual(batch.seq_lens_cpu.tolist(), [100, 101])
        self.assertEqual(batch.seq_lens_sum, 201)
        self.assertIsNone(batch.input_ids)

    def test_scheduler_rejects_partial_dspark_payload(self):
        result = _decode_result(bs=1)
        result.accept_lens = torch.empty(0, dtype=torch.int32)
        batch = SimpleNamespace(
            batch_size=lambda: 1,
            seq_lens_cpu=torch.zeros(1, dtype=torch.int64),
        )

        with self.assertRaisesRegex(RuntimeError, "invalid accept_lens"):
            Scheduler._apply_dspark_sync_result(object(), batch, result)

    def test_scheduler_accepts_complete_idle_dspark_payload(self):
        result = _decode_result(bs=0, active=False)
        batch = SimpleNamespace(
            batch_size=lambda: 0,
            seq_lens_cpu=torch.empty(0, dtype=torch.int64),
            seq_lens=torch.empty(0, dtype=torch.int64),
            seq_lens_sum=0,
            spec_info=None,
            input_ids=None,
        )

        Scheduler._apply_dspark_sync_result(object(), batch, result)

        self.assertIs(batch.spec_info, result.next_draft_input)
        self.assertEqual(batch.seq_lens.numel(), 0)

    def test_scheduler_rejects_active_result_without_logits(self):
        result = _decode_result(bs=1, active=False)
        batch = SimpleNamespace(batch_size=lambda: 1)

        with self.assertRaisesRegex(RuntimeError, "missing logits_output"):
            Scheduler._apply_dspark_sync_result(object(), batch, result)

    def test_scheduler_rejects_split_sequence_state(self):
        result = _decode_result(bs=1)
        result.new_seq_lens = result.new_seq_lens.clone()
        batch = SimpleNamespace(batch_size=lambda: 1)

        with self.assertRaisesRegex(RuntimeError, "inconsistent sequence state"):
            Scheduler._apply_dspark_sync_result(object(), batch, result)

    def test_dspark_result_commits_full_accepted_run(self):
        reqs = [_Req(10), _Req(20)]
        result = _decode_result(bs=2, stride=3)
        result.next_token_ids = torch.tensor([11, 12, 0, 21, 0, 0], dtype=torch.int64)
        result.accept_lens = torch.tensor([2, 1], dtype=torch.int32)
        worker = SimpleNamespace(on_verify_complete_cpu=Mock())
        processor = SimpleNamespace(model_worker=worker)
        batch = SimpleNamespace(
            reqs=reqs,
            spec_algorithm=_SpecAlgorithm(dspark=True),
        )

        accepted = SchedulerBatchResultProcessor._resolve_result_based_spec_tokens(
            processor, result, batch
        )

        self.assertEqual(accepted, [[11, 12], [21]])
        self.assertEqual([req.kv_committed_len for req in reqs], [12, 21])
        self.assertEqual([req.spec_verify_ct for req in reqs], [1, 1])
        worker.on_verify_complete_cpu.assert_called_once_with([1, 0])

    def test_legacy_spec_v2_keeps_bonus_preclaim_semantics(self):
        req = _Req(10)
        result = _decode_result(bs=1, stride=3)
        processor = SimpleNamespace(
            model_worker=SimpleNamespace(on_verify_complete_cpu=Mock())
        )
        batch = SimpleNamespace(
            reqs=[req],
            spec_algorithm=_SpecAlgorithm(dspark=False),
        )

        SchedulerBatchResultProcessor._resolve_result_based_spec_tokens(
            processor, result, batch
        )

        self.assertEqual(req.kv_committed_len, 10)

    def test_request_validation_is_algorithm_specific(self):
        req = SimpleNamespace(
            return_logprob=True,
            sampling_params=SimpleNamespace(
                json_schema=None,
                regex=None,
                ebnf=None,
                structural_tag=None,
            ),
        )
        self.assertIn(
            "DSPARK",
            validate_dflash_request(req, algorithm="DSPARK"),
        )

    def test_dspark_request_length_keeps_two_verify_windows(self):
        scheduler = SimpleNamespace(
            page_size=1,
            max_req_len=100,
            max_total_num_tokens=1000,
            spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
            server_args=SimpleNamespace(max_speculative_num_draft_tokens=6),
        )
        req = SimpleNamespace(
            origin_input_ids=list(range(10)),
            sampling_params=SimpleNamespace(max_new_tokens=100),
        )

        Scheduler.init_req_max_new_tokens(scheduler, req)

        self.assertEqual(req.sampling_params.max_new_tokens, 78)

    def test_dflash_reservation_fails_before_allocator_or_mapping_write(self):
        draft_input = DFlashDraftInputV2(
            topk_p=torch.empty((1, 0)),
            topk_index=torch.empty((1, 0), dtype=torch.int64),
            bonus_tokens=torch.zeros((1,), dtype=torch.int64),
            new_seq_lens=torch.tensor([10], dtype=torch.int64),
            hidden_states=torch.empty((1, 0)),
        )
        req = SimpleNamespace(
            rid="overflow",
            kv_committed_len=10,
            kv_allocated_len=10,
            output_ids=[1],
            origin_input_ids=[0],
            sampling_params=SimpleNamespace(top_k=1),
        )
        batch = SimpleNamespace(
            batch_size=lambda: 1,
            reqs=[req],
            sampling_info=SimpleNamespace(
                penalizer_orchestrator=SimpleNamespace(is_required=False)
            ),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros((1, 15), dtype=torch.int32)
            ),
            token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
            device=torch.device("cpu"),
            tree_cache=object(),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
        )

        with (
            patch(
                "sglang.srt.speculative.dflash_info_v2.get_global_server_args",
                return_value=SimpleNamespace(speculative_num_draft_tokens=6),
            ),
            patch(
                "sglang.srt.speculative.dflash_info_v2.alloc_token_slots"
            ) as alloc,
            patch(
                "sglang.srt.speculative.dflash_info_v2."
                "assign_req_to_token_pool_func"
            ) as assign,
        ):
            with self.assertRaisesRegex(RuntimeError, "before KV allocation"):
                draft_input.prepare_for_decode(batch)

        alloc.assert_not_called()
        assign.assert_not_called()
        self.assertEqual(req.kv_allocated_len, 10)

    def test_dspark_prepare_accumulates_penalty_token(self):
        draft_input = DFlashDraftInputV2(
            topk_p=torch.empty((1, 0)),
            topk_index=torch.empty((1, 0), dtype=torch.int64),
            bonus_tokens=torch.zeros((1,), dtype=torch.int64),
            new_seq_lens=torch.tensor([10], dtype=torch.int64),
            hidden_states=torch.empty((1, 0)),
        )
        req = SimpleNamespace(
            rid="penalty",
            kv_committed_len=10,
            kv_allocated_len=22,
            output_ids=[7],
            origin_input_ids=[3],
            sampling_params=SimpleNamespace(top_k=1),
        )
        penalizer = SimpleNamespace(
            is_required=True,
            cumulate_output_tokens=Mock(),
        )
        batch = SimpleNamespace(
            batch_size=lambda: 1,
            reqs=[req],
            sampling_info=SimpleNamespace(penalizer_orchestrator=penalizer),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros((1, 64), dtype=torch.int32)
            ),
            token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
            device=torch.device("cpu"),
            seq_lens_cpu=None,
            seq_lens_sum=0,
        )

        with patch(
            "sglang.srt.speculative.dflash_info_v2.get_global_server_args",
            return_value=SimpleNamespace(speculative_num_draft_tokens=6),
        ):
            draft_input.prepare_for_decode(batch)

        penalty_tokens = penalizer.cumulate_output_tokens.call_args.args[0]
        self.assertEqual(penalty_tokens.tolist(), [7])


if __name__ == "__main__":
    unittest.main()
