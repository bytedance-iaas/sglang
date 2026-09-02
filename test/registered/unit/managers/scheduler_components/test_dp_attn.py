import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler_components import dp_attn  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDPAttnSchedulerMetadata(CustomTestCase):
    @staticmethod
    def _sync_info(*, can_draft_cuda_graph=True):
        return dp_attn.MLPSyncBatchInfo(
            dp_size=2,
            tp_size=1,
            cp_size=1,
            num_tokens=4,
            num_tokens_for_logprob=4,
            can_run_decode_cuda_graph=True,
            can_run_prefill_cuda_graph=False,
            can_draft_cuda_graph=can_draft_cuda_graph,
            is_extend_in_batch=False,
            local_can_run_tbo=True,
            local_forward_mode=ForwardMode.DECODE.value,
        )

    def test_draft_graph_vote_is_min_reduced_in_existing_metadata_gather(self):
        info = self._sync_info()
        gathered = torch.stack(
            [
                info._get_local_tensor(device="cpu"),
                self._sync_info(can_draft_cuda_graph=False)._get_local_tensor(
                    device="cpu"
                ),
            ]
        )

        def gather_into(output, _local, group):
            output.copy_(gathered.flatten())

        with (
            patch.object(
                torch.distributed,
                "all_gather_into_tensor",
                side_effect=gather_into,
            ),
            patch.object(
                dp_attn,
                "get_tp_group",
                return_value=SimpleNamespace(
                    active_ranks_cpu=torch.ones(2, dtype=torch.int64)
                ),
            ),
        ):
            info.all_gather(device="cpu", group=object())

        self.assertFalse(info.can_draft_cuda_graph)
        self.assertEqual(info.tp0_info_cpu.shape, (2, 8))

    def test_inactive_rank_is_permissive_for_draft_graph_vote(self):
        info = self._sync_info(can_draft_cuda_graph=False)
        fallback = info._get_fallback_tensor(device="cpu")

        self.assertEqual(fallback.numel(), 8)
        self.assertEqual(fallback[-1].item(), 1)

    def test_skip_all_gather_policy(self):
        with envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False):
            self.assertTrue(dp_attn.should_skip_scheduler_all_gather(dp_size=1))
            self.assertFalse(dp_attn.should_skip_scheduler_all_gather(dp_size=2))
        with envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(True):
            self.assertTrue(dp_attn.should_skip_scheduler_all_gather(dp_size=2))

    def test_dp1_skip_preserves_local_tbo_metadata(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=lambda: 4,
        )
        tbo_preparer = Mock()
        tbo_preparer.prepare_all_gather.return_value = (
            True,
            ForwardMode.DECODE.value,
        )
        tbo_preparer.compute_output.return_value = (2, ForwardMode.DECODE)

        with (
            envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False),
            patch.object(dp_attn, "TboDPAttentionPreparer", return_value=tbo_preparer),
            patch.object(dp_attn, "world_dp_gather_enabled", return_value=False),
            patch.object(dp_attn, "check_cuda_graph_backend", return_value=False),
            patch.object(dp_attn.MLPSyncBatchInfo, "all_gather") as all_gather,
        ):
            result = dp_attn.prepare_mlp_sync_batch_raw(
                batch,
                model_runner=SimpleNamespace(
                    prefill_cuda_graph_runner=None,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    model_config=object(),
                ),
                dp_size=1,
                attn_tp_size=4,
                attn_cp_size=1,
                tp_group=SimpleNamespace(
                    device_group=object(), device="cpu", cpu_group=object()
                ),
                get_idle_batch=Mock(
                    side_effect=AssertionError("DP1 must not emit idle batch")
                ),
                disable_cuda_graph=False,
                require_mlp_tp_gather=False,
                disable_overlap_schedule=True,
                offload_tags=set(),
            )

        all_gather.assert_not_called()
        self.assertEqual(result.global_num_tokens, [4])
        self.assertEqual(result.tbo_split_seq_index, 2)
        self.assertEqual(result.global_forward_mode, ForwardMode.DECODE)
        self.assertEqual(result.recv_skipper_forward_mode, ForwardMode.DECODE)
        self.assertTrue(result.can_run_dp_draft_cuda_graph)
        self.assertEqual(
            tbo_preparer.compute_output.call_args.args[0].tolist(),
            [[1, ForwardMode.DECODE.value]],
        )


if __name__ == "__main__":
    unittest.main()
