import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest
from sglang.srt.disaggregation.fake.conn import FakeKVReceiver
from sglang.srt.disaggregation.base.conn import KVPoll


class TestDecodeBootstrapState(unittest.TestCase):
    def test_fake_receiver_exposes_decode_runtime_flags(self):
        receiver = FakeKVReceiver(mgr=None, bootstrap_addr="fake", bootstrap_room=0)

        self.assertIsNone(receiver.conclude_state)
        self.assertFalse(receiver.require_staging)

    def test_mark_receiver_initialized_accepts_receiver_without_conclude_state(self):
        receiver = FakeKVReceiver(mgr=None, bootstrap_addr="fake", bootstrap_room=0)
        decode_req = DecodeRequest(req=SimpleNamespace(), kv_receiver=receiver)
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)

        DecodePreallocQueue._mark_receiver_initialized(queue, decode_req, 0)

        self.assertTrue(decode_req.bootstrap_ready)

    def test_mark_receiver_initialized_skips_failed_receiver(self):
        class FailedReceiver:
            conclude_state = None

            def init(self, _prefill_dp_rank: int):
                self.conclude_state = KVPoll.Failed

        decode_req = DecodeRequest(
            req=SimpleNamespace(),
            kv_receiver=FailedReceiver(),
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)

        DecodePreallocQueue._mark_receiver_initialized(queue, decode_req, 0)

        self.assertFalse(decode_req.bootstrap_ready)
