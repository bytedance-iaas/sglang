import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _RecordingPool:
    def __init__(self):
        self.calls = []

    def set_kv_buffer(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class TestTritonBackendDCPKVWrite(CustomTestCase):
    @staticmethod
    def _backend(*, use_mla: bool, dcp_size: int = 8, dcp_rank: int = 3):
        backend = TritonAttnBackend.__new__(TritonAttnBackend)
        backend.use_mla = use_mla
        backend.dcp_size = dcp_size
        backend.dcp_rank = dcp_rank
        backend.token_to_kv_pool = _RecordingPool()
        return backend

    @staticmethod
    def _forward_batch():
        return SimpleNamespace(
            out_cache_loc=torch.arange(16, dtype=torch.int64),
            positions=torch.arange(16, dtype=torch.int64),
            dcp_kv_mask=None,
        )

    def test_dcp_mla_passes_logical_locations_to_pool(self):
        backend = self._backend(use_mla=True)
        forward_batch = self._forward_batch()
        logical_loc = torch.arange(100, 116, dtype=torch.int64)
        layer = object()
        k = torch.randn(16, 1)
        v = torch.randn(16, 1)

        backend._set_kv_buffer(forward_batch, layer, logical_loc, k, v)

        args, kwargs = backend.token_to_kv_pool.calls[0]
        self.assertIs(args[1], logical_loc)
        self.assertEqual(kwargs, {})

    def test_dcp_mha_keeps_pretranslated_masked_write(self):
        backend = self._backend(use_mla=False)
        forward_batch = self._forward_batch()

        backend._set_kv_buffer(
            forward_batch,
            object(),
            torch.arange(100, 116, dtype=torch.int64),
            torch.randn(16, 1),
            torch.randn(16, 1),
        )

        args, kwargs = backend.token_to_kv_pool.calls[0]
        torch.testing.assert_close(args[1], forward_batch.out_cache_loc // 8)
        torch.testing.assert_close(
            kwargs["dcp_kv_mask"], forward_batch.positions % 8 == 3
        )


if __name__ == "__main__":
    unittest.main()
