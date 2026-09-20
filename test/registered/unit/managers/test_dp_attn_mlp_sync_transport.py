import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.dp_attn import (  # noqa: E402
    MLPSyncBatchInfo,
    _use_device_mlp_sync_transport,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDPAttnMLPSyncTransport(unittest.TestCase):
    def test_world_size_one_fallback_does_not_alias_local_tensor(self):
        info = MLPSyncBatchInfo(
            dp_size=1,
            tp_size=1,
            cp_size=1,
            num_tokens=0,
            num_tokens_for_logprob=0,
            can_run_decode_cuda_graph=True,
            can_run_prefill_cuda_graph=False,
            can_draft_cuda_graph=True,
            is_extend_in_batch=False,
            local_can_run_tbo=True,
            local_forward_mode=ForwardMode.IDLE.value,
        )
        tp_group = SimpleNamespace(
            active_ranks_cpu=torch.tensor([0], dtype=torch.int64),
            active_ranks=torch.tensor([0], dtype=torch.int64),
        )
        with (
            patch(
                "sglang.srt.managers.scheduler_components.dp_attn.get_tp_group",
                return_value=tp_group,
            ),
            patch("torch.distributed.all_gather_into_tensor"),
        ):
            info.all_gather(device="cpu", group=object())

        self.assertEqual(info.global_num_tokens, [0])

    def test_pp_pd_prefill_forces_cpu_transport(self):
        with patch(
            "sglang.srt.managers.scheduler_components.dp_attn.envs."
            "SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get",
            return_value=True,
        ):
            self.assertFalse(
                _use_device_mlp_sync_transport(
                    disable_overlap_schedule=True,
                    offload_tags=set(),
                    force_cpu_mlp_sync=True,
                )
            )

    def test_non_overlap_keeps_existing_device_transport(self):
        self.assertTrue(
            _use_device_mlp_sync_transport(
                disable_overlap_schedule=True,
                offload_tags=set(),
                force_cpu_mlp_sync=False,
            )
        )

    def test_overlap_defaults_to_cpu_transport(self):
        with patch(
            "sglang.srt.managers.scheduler_components.dp_attn.envs."
            "SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get",
            return_value=False,
        ):
            self.assertFalse(
                _use_device_mlp_sync_transport(
                    disable_overlap_schedule=False,
                    offload_tags=set(),
                    force_cpu_mlp_sync=False,
                )
            )


if __name__ == "__main__":
    unittest.main()
