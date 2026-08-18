import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.dp_attn import (  # noqa: E402
    _use_device_mlp_sync_transport,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDPAttnMLPSyncTransport(unittest.TestCase):
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
