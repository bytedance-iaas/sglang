import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch


def _make_batch(req, evictions):
    def evict_swa(_current_req, pre_len):
        evictions.append(pre_len)

    return SimpleNamespace(
        tree_cache=SimpleNamespace(
            sliding_window_size=128,
            page_size=256,
            supports_swa=lambda: True,
            is_chunk_cache=lambda: False,
        ),
        forward_mode=SimpleNamespace(
            is_decode=lambda: True,
            is_extend=lambda: False,
        ),
        reqs=[req],
        _evict_swa=evict_swa,
    )


def _make_req(seq_len, *, evicted=0, decode_batch_idx=1):
    return SimpleNamespace(
        seqlen=seq_len,
        swa_evicted_seqlen=evicted,
        decode_batch_idx=decode_batch_idx,
        swa_prefix_lock_released=True,
        swa_uuid_for_lock=None,
        last_node=None,
    )


class TestSWATokenEvictionGate(unittest.TestCase):
    def _run_gate(self, batch):
        with (
            patch.object(
                envs.SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER,
                "get",
                return_value=1.0,
            ),
            patch.object(
                envs.SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW,
                "get",
                return_value=False,
            ),
            patch(
                "sglang.srt.managers.schedule_batch.get_global_server_args",
                return_value=SimpleNamespace(chunked_prefill_size=0),
            ),
        ):
            ScheduleBatch.maybe_evict_swa(batch)

    def test_speculative_acceptance_crosses_token_boundary(self):
        for accepted_tokens in (1, 3, 6):
            evictions = []
            req = _make_req(385 - accepted_tokens)
            batch = _make_batch(req, evictions)

            self._run_gate(batch)
            self.assertEqual(evictions, [])

            req.seqlen += accepted_tokens
            self._run_gate(batch)
            self.assertEqual(evictions, [384])

    def test_gate_uses_last_evicted_frontier(self):
        evictions = []
        req = _make_req(640, evicted=256)
        batch = _make_batch(req, evictions)

        self._run_gate(batch)
        self.assertEqual(evictions, [])

        req.seqlen = 641
        self._run_gate(batch)
        self.assertEqual(evictions, [640])

    def test_first_decode_iteration_remains_protected(self):
        evictions = []
        req = _make_req(4096, decode_batch_idx=0)
        batch = _make_batch(req, evictions)

        self._run_gate(batch)
        self.assertEqual(evictions, [])


if __name__ == "__main__":
    unittest.main()
