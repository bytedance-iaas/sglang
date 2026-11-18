"""UCX peer publication, registration ordering, and failure contracts."""

import base64
import dataclasses
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import PrefillRankInfo
from sglang.srt.disaggregation.nixl.bootstrap import (
    MAX_ENCODED_METADATA,
    PrefillPeerRegistrar,
    publish_peer_metadata,
)
from sglang.srt.disaggregation.nixl.conn import NixlKVReceiver
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeAgent:
    def __init__(self, metadata=b"prefill", events=None):
        self.metadata = metadata
        self.name = metadata.decode()
        self.events = [] if events is None else events
        self.loads = []
        self.valid = set()
        self.fail = False

    def get_agent_metadata(self):
        self.events.append("export")
        return self.metadata

    def check_remote_metadata(self, peer):
        return peer in self.valid

    def add_remote_agent(self, raw):
        self.events.append("import")
        self.loads.append(raw)
        if self.fail:
            raise RuntimeError("remote registration failed")
        peer = raw.decode()
        self.valid.add(peer)
        return peer


class TestNixlBootstrapMetadata(CustomTestCase):
    def metadata(self, raw=b"prefill"):
        return {
            "version": "1",
            "backend": "UCX",
            "agent_metadata": base64.b64encode(raw).decode(),
        }

    def test_publish_does_not_load_another_backend(self):
        agent = FakeAgent()
        self.assertIsNone(publish_peer_metadata(agent, "LIBFABRIC"))
        self.assertEqual(agent.events, [])

    def test_rank_info_preserves_metadata_and_old_routes(self):
        old = dataclasses.asdict(PrefillRankInfo("192.168.0.6", "1234"))
        self.assertIsNone(old["transfer_metadata"])
        new = dataclasses.asdict(
            PrefillRankInfo("192.168.0.6", "1234", self.metadata())
        )
        self.assertEqual(new["transfer_metadata"], self.metadata())
        self.assertEqual(new["rank_port"], 1234)

    def test_publish_bound_prevents_oversized_bootstrap_request(self):
        agent = FakeAgent(b"x" * MAX_ENCODED_METADATA)
        with self.assertRaises(ValueError):
            publish_peer_metadata(agent, "UCX")

    def test_legacy_and_other_backend_do_not_register(self):
        agent = FakeAgent()
        self.assertFalse(PrefillPeerRegistrar(agent, "UCX").register(None))
        self.assertFalse(
            PrefillPeerRegistrar(agent, "LIBFABRIC").register(self.metadata())
        )
        self.assertEqual(agent.loads, [])

    def test_invalid_metadata_never_reaches_nixl(self):
        agent = FakeAgent()
        registrar = PrefillPeerRegistrar(agent, "UCX")
        invalid = [
            [],
            {},
            {**self.metadata(), "version": "2"},
            {**self.metadata(), "backend": "LIBFABRIC"},
            {**self.metadata(), "agent_metadata": "%%"},
            {**self.metadata(), "agent_metadata": ""},
            {**self.metadata(), "agent_metadata": "a" * (MAX_ENCODED_METADATA + 1)},
        ]
        for data in invalid:
            with self.subTest(data_type=type(data).__name__):
                with self.assertRaises(ValueError):
                    registrar.register(data)
        self.assertEqual(agent.loads, [])

    def test_repeated_registration_reuses_valid_peer(self):
        agent = FakeAgent()
        registrar = PrefillPeerRegistrar(agent, "UCX")
        self.assertTrue(registrar.register(self.metadata()))
        self.assertFalse(registrar.register(self.metadata()))
        self.assertEqual(agent.loads, [b"prefill"])

    def test_invalidated_peer_is_loaded_again(self):
        agent = FakeAgent()
        registrar = PrefillPeerRegistrar(agent, "UCX")
        registrar.register(self.metadata())
        agent.valid.clear()
        self.assertTrue(registrar.register(self.metadata()))
        self.assertEqual(agent.loads, [b"prefill", b"prefill"])

    def test_failed_load_is_not_cached(self):
        agent = FakeAgent()
        registrar = PrefillPeerRegistrar(agent, "UCX")
        agent.fail = True
        with self.assertRaises(RuntimeError):
            registrar.register(self.metadata())
        agent.fail = False
        self.assertTrue(registrar.register(self.metadata()))
        self.assertEqual(len(agent.loads), 2)

    def test_new_producer_generation_is_independent(self):
        agent = FakeAgent()
        registrar = PrefillPeerRegistrar(agent, "UCX")
        registrar.register(self.metadata(b"prefill-generation-1"))
        registrar.register(self.metadata(b"prefill-generation-2"))
        self.assertEqual(len(agent.loads), 2)

    def test_concurrent_same_peer_is_registered_once(self):
        agent = FakeAgent()
        registrar = PrefillPeerRegistrar(agent, "UCX")
        barrier = threading.Barrier(8)

        def register(_):
            barrier.wait(timeout=5)
            return registrar.register(self.metadata())

        with ThreadPoolExecutor(max_workers=8) as pool:
            outcomes = list(pool.map(register, range(8)))
        self.assertEqual(sum(outcomes), 1)
        self.assertEqual(agent.loads, [b"prefill"])

    def make_receiver(self, metadata):
        events = []
        agent = FakeAgent(b"decode", events)
        sock = MagicMock()
        sock.send_multipart.side_effect = lambda _: events.append("send")
        receiver = NixlKVReceiver.__new__(NixlKVReceiver)
        receiver.bootstrap_room = 42
        receiver.conclude_state = None
        receiver.bootstrap_infos = [{"transfer_metadata": metadata}]
        args = SimpleNamespace(
            kv_data_ptrs=[],
            kv_data_mem_kinds=[],
            kv_item_lens=[],
            kv_layer_ids=[],
            aux_data_ptrs=[],
            state_data_ptrs=[],
            state_item_lens=[],
            state_dim_per_tensor=[],
            state_layer_ids=[],
            gpu_id=6,
            engine_rank=6,
        )
        receiver.kv_mgr = SimpleNamespace(
            kv_args=args,
            agent=agent,
            _prefill_peer_registrar=PrefillPeerRegistrar(agent, "UCX"),
            enable_staging=False,
            local_ip="192.168.0.7",
            rank_port=1234,
            attn_tp_size=1,
            dcp_size=1,
            dcp_rank=0,
            record_failure=MagicMock(),
            update_status=MagicMock(),
        )
        receiver._connect_to_bootstrap_server = MagicMock(
            return_value=(sock, threading.Lock())
        )
        return receiver, agent, sock, events

    def test_import_finishes_before_decode_metadata_is_published(self):
        receiver, _, sock, events = self.make_receiver(self.metadata())
        self.assertTrue(receiver._register_kv_args())
        self.assertEqual(events, ["import", "export", "send"])
        self.assertEqual(sock.send_multipart.call_args.args[0][5], b"decode")

    def test_import_failure_fails_room_without_publishing_decode(self):
        receiver, agent, sock, events = self.make_receiver(self.metadata())
        agent.fail = True
        self.assertFalse(receiver._register_kv_args())
        self.assertEqual(events, ["import"])
        sock.send_multipart.assert_not_called()
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
        receiver.kv_mgr.update_status.assert_called_once_with(42, KVPoll.Failed)

    def test_legacy_producer_registration_is_unchanged(self):
        receiver, agent, _, events = self.make_receiver(None)
        self.assertTrue(receiver._register_kv_args())
        self.assertEqual(events, ["export", "send"])
        self.assertEqual(agent.loads, [])


if __name__ == "__main__":
    unittest.main()
