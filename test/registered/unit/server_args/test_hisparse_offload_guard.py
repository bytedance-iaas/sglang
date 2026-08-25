"""Guard: --disaggregation-decode-enable-offload-kvcache is incompatible with
--enable-hisparse.

HiSparse committed KV is compressed and owned by the HiSparseCoordinator, whose
retract_req frees the device buffer / host slots *before* release_req runs. The
decode retract offload path (release_req -> allocator.get_cpu_copy) therefore has
no stable device tensor to copy for HiSparse pools -- the HiSparse device pools
(HiSparse[C4]DevicePool) raise NotImplementedError there on purpose. HiSparse
decode-side retraction recovers by recomputing the prefix on the prefill worker
(PD true-retraction rebootstrap), which needs no host offload.

This is the second half of the bs-cap fix: fix(a) gates the offload in
release_req on the flag and routes offload-disabled retracts to rebootstrap;
this guard makes the incompatible flag combination fail fast at startup instead
of crashing mid-serving inside get_cpu_copy.
"""

import unittest
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# Mock get_device() so the guard runs on CPU-only CI runners.
_mock_device = patch("sglang.srt.server_args.get_device", return_value="cuda")
_mock_device.start()


class TestHiSparseOffloadGuard(CustomTestCase):
    def _make_args(self, **overrides) -> ServerArgs:
        args = ServerArgs(model_path="dummy")
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_hisparse_plus_decode_offload_is_rejected(self):
        args = self._make_args(
            disaggregation_mode="decode",
            disaggregation_decode_enable_offload_kvcache=True,
            hicache_storage_backend="mooncake",
            enable_hisparse=True,
        )
        with self.assertRaisesRegex(
            ValueError,
            r"disaggregation-decode-enable-offload-kvcache is incompatible "
            r"with --enable-hisparse",
        ):
            args._handle_cache_compatibility()

    def test_decode_offload_without_hisparse_is_allowed(self):
        # Same offload config but no HiSparse: the HiSparse-specific guard must
        # not fire (the generic decode/backend checks already passed).
        args = self._make_args(
            disaggregation_mode="decode",
            disaggregation_decode_enable_offload_kvcache=True,
            hicache_storage_backend="mooncake",
            enable_hisparse=False,
        )
        # Should not raise for the HiSparse reason. It may proceed to the later
        # swa_full_tokens_ratio check, which is satisfied by the default 1.0.
        args._handle_cache_compatibility()

    def test_decode_offload_requires_storage_backend_before_hisparse_check(self):
        # Ordering guard: the storage-backend requirement is checked first, so a
        # missing backend surfaces its own clear error even with HiSparse on.
        args = self._make_args(
            disaggregation_mode="decode",
            disaggregation_decode_enable_offload_kvcache=True,
            hicache_storage_backend=None,
            enable_hisparse=True,
        )
        with self.assertRaisesRegex(
            ValueError,
            r"only supported when hicache-storage-backend is provided",
        ):
            args._handle_cache_compatibility()


if __name__ == "__main__":
    unittest.main()
