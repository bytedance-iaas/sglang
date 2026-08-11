import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    DSparkWorkerV2,
    validate_dspark_decode_input,
)
from sglang.srt.speculative.dspark_disaggregation import (
    build_dspark_disagg_draft_input,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _draft_input(bs: int) -> DFlashDraftInputV2:
    return DFlashDraftInputV2(
        topk_p=torch.empty((bs, 0), dtype=torch.float32),
        topk_index=torch.empty((bs, 0), dtype=torch.int64),
        bonus_tokens=torch.arange(bs, dtype=torch.int64),
        new_seq_lens=torch.arange(10, 10 + bs, dtype=torch.int64),
        hidden_states=torch.empty((bs, 0), dtype=torch.float16),
    )


def _decode_batch(bs: int, *, idle: bool = False):
    return SimpleNamespace(
        forward_mode=ForwardMode.IDLE if idle else ForwardMode.DECODE,
        seq_lens=torch.arange(10, 10 + bs, dtype=torch.int64),
        req_pool_indices=torch.arange(bs, dtype=torch.int64),
        out_cache_loc=None,
        global_num_tokens=[bs, 0],
        global_num_tokens_for_logprob=[0, 0],
    )


def _verify_batch(*, idle: bool, graph_tokens: int, local_bs: int):
    return SimpleNamespace(
        forward_mode=ForwardMode.IDLE if idle else ForwardMode.DECODE,
        seq_lens=torch.arange(20, 20 + local_bs, dtype=torch.int64),
        req_pool_indices=torch.arange(local_bs, dtype=torch.int64),
        out_cache_loc=torch.arange(graph_tokens, dtype=torch.int64),
        global_num_tokens=[0, 1] if idle else [1, 0],
        global_num_tokens_for_logprob=[0, 0],
        capture_hidden_mode=None,
        spec_info=None,
    )


class TestDSparkPDBuilder(unittest.TestCase):
    def test_builds_active_bs1_and_bs4(self):
        for bs in (1, 4):
            batch = SimpleNamespace(
                enable_overlap=False,
                seq_lens=torch.arange(32, 32 + bs, dtype=torch.int64),
            )
            bonus = torch.arange(bs, dtype=torch.int32)
            draft_input = build_dspark_disagg_draft_input(
                batch, None, bonus, None
            )
            self.assertEqual(draft_input.bonus_tokens.shape, (bs,))
            self.assertEqual(draft_input.new_seq_lens.shape, (bs,))
            self.assertEqual(draft_input.bonus_tokens.dtype, torch.int64)
            self.assertEqual(draft_input.new_seq_lens.dtype, torch.int64)
            self.assertEqual(draft_input.bonus_tokens.device, batch.seq_lens.device)

    def test_active_shape_mismatch_fails(self):
        batch = SimpleNamespace(
            enable_overlap=False,
            seq_lens=torch.tensor([32], dtype=torch.int64),
        )
        with self.assertRaisesRegex(RuntimeError, "local_bs=1"):
            build_dspark_disagg_draft_input(
                batch, None, torch.empty((0,), dtype=torch.int64), None
            )

    def test_overlap_fails_fast(self):
        batch = SimpleNamespace(
            enable_overlap=True,
            seq_lens=torch.tensor([32], dtype=torch.int64),
        )
        with self.assertRaisesRegex(RuntimeError, "overlap scheduling"):
            build_dspark_disagg_draft_input(
                batch, None, torch.tensor([1], dtype=torch.int64), None
            )

    def test_decode_prebuilt_dispatches_to_dspark_builder(self):
        spec_algorithm = SimpleNamespace(
            is_eagle=lambda: False,
            is_dspark=lambda: True,
        )
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(output_ids=[11], grammar=None)],
            tree_cache=object(),
            device=torch.device("cpu"),
            seq_lens=torch.tensor([128], dtype=torch.int64),
            enable_overlap=False,
            spec_algorithm=spec_algorithm,
            spec_info=None,
        )
        with patch(
            "sglang.srt.disaggregation.decode_schedule_batch_mixin."
            "maybe_cache_unfinished_req"
        ):
            ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(
                batch, None, None
            )
        self.assertIsInstance(batch.spec_info, DFlashDraftInputV2)
        self.assertEqual(batch.spec_info.bonus_tokens.tolist(), [11])
        self.assertEqual(batch.spec_info.new_seq_lens.tolist(), [128])

    def test_decode_prebuilt_non_spec_behavior_is_unchanged(self):
        spec_algorithm = SimpleNamespace(
            is_eagle=lambda: False,
            is_dspark=lambda: False,
        )
        sentinel = object()
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(output_ids=[11], grammar=None)],
            tree_cache=object(),
            device=torch.device("cpu"),
            seq_lens=torch.tensor([128], dtype=torch.int64),
            enable_overlap=False,
            spec_algorithm=spec_algorithm,
            spec_info=sentinel,
        )
        with patch(
            "sglang.srt.disaggregation.decode_schedule_batch_mixin."
            "maybe_cache_unfinished_req"
        ):
            ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(
                batch, None, None
            )
        self.assertIs(batch.spec_info, sentinel)


class TestDSparkDecodeContract(unittest.TestCase):
    def test_active_and_idle_contracts_allow_unallocated_verify_window(self):
        validate_dspark_decode_input(
            batch=_decode_batch(1),
            draft_input=_draft_input(1),
            dp_rank=0,
            tp_rank=0,
            enable_dp_attention=True,
        )
        validate_dspark_decode_input(
            batch=_decode_batch(0, idle=True),
            draft_input=DFlashDraftInputV2.create_idle_input(torch.device("cpu")),
            dp_rank=1,
            tp_rank=1,
            enable_dp_attention=True,
        )

    def test_prebuilt_contract_allows_unallocated_verify_window(self):
        batch = _decode_batch(8)
        batch.forward_mode = ForwardMode.PREBUILT
        validate_dspark_decode_input(
            batch=batch,
            draft_input=_draft_input(8),
            dp_rank=0,
            tp_rank=0,
            enable_dp_attention=True,
        )

    def test_active_empty_bonus_fails_with_rank_context(self):
        batch = _decode_batch(1)
        draft_input = _draft_input(1)
        draft_input.bonus_tokens = torch.empty((0,), dtype=torch.int64)
        with self.assertRaisesRegex(
            RuntimeError, "dp_rank=7, tp_rank=3.*bonus_tokens_shape=\\(0,\\)"
        ):
            validate_dspark_decode_input(
                batch=batch,
                draft_input=draft_input,
                dp_rank=7,
                tp_rank=3,
                enable_dp_attention=True,
            )

    def test_active_none_spec_info_fails_before_proposal(self):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.device = torch.device("cpu")
        worker.ps = SimpleNamespace(dp_rank=5, tp_rank=2)
        batch = _decode_batch(1)
        batch.has_grammar = False
        batch.spec_info = None
        with self.assertRaisesRegex(
            RuntimeError, "active batch has spec_info=None.*dp_rank=5, tp_rank=2"
        ):
            worker._forward_decode(batch, on_publish=None)


class TestDSparkVerifyAdapter(unittest.TestCase):
    def _target_worker(self, *, can_run_graph: bool):
        graph_runner = MagicMock()
        graph_runner.can_run.return_value = can_run_graph
        model_runner = SimpleNamespace(
            graph_runner=graph_runner,
            attn_backend=MagicMock(),
            server_args=SimpleNamespace(enable_dp_attention=True),
        )
        return SimpleNamespace(dp_rank=1, tp_rank=2, model_runner=model_runner)

    def _run(self, *, idle: bool, can_run_graph: bool):
        local_bs = 0 if idle else 1
        graph_tokens = 0 if idle else 6
        batch = _verify_batch(
            idle=idle, graph_tokens=graph_tokens, local_bs=local_bs
        )
        verify_input = DFlashVerifyInput(
            draft_token=torch.arange(graph_tokens, dtype=torch.int64),
            positions=torch.arange(graph_tokens, dtype=torch.int64),
            draft_token_num=6,
        )
        target_worker = self._target_worker(can_run_graph=can_run_graph)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE if idle else ForwardMode.TARGET_VERIFY
        )
        with patch(
            "sglang.srt.speculative.dflash_info.ForwardBatch.init_new",
            return_value=forward_batch,
        ) as init_new:
            result = verify_input.prepare_for_dspark_verify(batch, target_worker)
        return batch, target_worker, forward_batch, init_new, result

    def test_active_graph_adapter_returns_tuple(self):
        batch, worker, forward_batch, init_new, result = self._run(
            idle=False, can_run_graph=True
        )
        self.assertEqual(result, (forward_batch, True))
        self.assertEqual(batch.forward_mode, ForwardMode.TARGET_VERIFY)
        init_new.assert_called_once_with(batch, worker.model_runner)
        worker.model_runner.graph_runner.replay_prepare.assert_called_once_with(
            forward_batch
        )
        worker.model_runner.attn_backend.init_forward_metadata.assert_not_called()

    def test_active_eager_initializes_attention(self):
        _, worker, forward_batch, _, result = self._run(
            idle=False, can_run_graph=False
        )
        self.assertEqual(result, (forward_batch, False))
        worker.model_runner.attn_backend.init_forward_metadata.assert_called_once_with(
            forward_batch
        )

    def test_idle_graph_adapter_returns_tuple(self):
        batch, worker, forward_batch, _, result = self._run(
            idle=True, can_run_graph=True
        )
        self.assertEqual(result, (forward_batch, True))
        self.assertEqual(batch.forward_mode, ForwardMode.IDLE)
        worker.model_runner.graph_runner.replay_prepare.assert_called_once_with(
            forward_batch
        )


if __name__ == "__main__":
    unittest.main()
