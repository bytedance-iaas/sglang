"""Reciprocal refresh fencing, identity bounds, replay and notification safety."""

import base64
import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.nixl.bootstrap import (
    PrefillPeerRegistrar,
    publish_peer_metadata,
)
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.disaggregation.nixl.recovery import ACK, REQUEST, UCXPeerRecovery
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
MODULE = "sglang.srt.disaggregation.nixl.recovery"


class Agent:
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.valid = set()
        self.fail_load = False

    def get_agent_metadata(self):
        return self.name.encode()

    def add_remote_agent(self, raw):
        peer = raw.decode()
        self.events.append("import-" + peer)
        if self.fail_load:
            raise RuntimeError("failed import")
        self.valid.add(peer)
        return peer.encode()

    def remove_remote_agent(self, peer):
        self.events.append("remove-" + peer)
        self.valid.discard(peer)

    def check_remote_metadata(self, peer):
        return peer in self.valid


def pair():
    events, requests, replies = [], [], []
    p, d = Agent("producer", events), Agent("consumer", events)
    pr, dr = PrefillPeerRegistrar(p, "UCX"), PrefillPeerRegistrar(d, "UCX")
    metadata = publish_peer_metadata(p, "UCX")
    dr.register(metadata)
    producer = consumer = None

    def to_consumer(endpoint, msg):
        assert endpoint == "tcp://consumer:1234"
        requests.append(list(msg))
        consumer.handle(msg)

    def to_producer(endpoint, msg):
        assert endpoint == "tcp://producer:1234"
        replies.append(list(msg))
        events.append("ack")
        producer.handle(msg)

    producer = UCXPeerRecovery(
        p, "tcp://producer:1234", to_consumer, pr, lambda *_: None
    )
    consumer = UCXPeerRecovery(
        d, "tcp://consumer:1234", to_producer, dr, dr.refresh_known
    )
    consumer.remember_producer(metadata, "tcp://producer:1234")
    events.clear()
    return producer, consumer, p, d, pr, dr, events, requests, replies


class TestReciprocalFence(CustomTestCase):
    def test_consumer_import_precedes_ack_and_sender_return(self):
        p, _, _, _, _, _, events, _, _ = pair()
        result = p.request("consumer", "tcp://consumer:1234")
        self.assertEqual(events, ["remove-producer", "import-producer", "ack"])
        self.assertEqual(base64.b64decode(result["agent_metadata"]), b"consumer")
        self.assertEqual(p._pending, {})

    def test_sender_waits_until_consumer_refresh_completes(self):
        p, d, _, _, _, dr, _, _, _ = pair()
        entered, resume = threading.Event(), threading.Event()

        def refresh(meta, peer):
            entered.set()
            self.assertTrue(resume.wait(2))
            dr.refresh_known(meta, peer)

        d.refresh = refresh
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(p.request, "consumer", "tcp://consumer:1234")
            self.assertTrue(entered.wait(2))
            self.assertFalse(future.done())
            resume.set()
            self.assertEqual(future.result()["backend"], "UCX")

    def test_timeout_clears_pending_without_a_success_result(self):
        p, _, _, _, _, _, _, _, _ = pair()
        p.send = lambda *_: None
        with patch(MODULE + ".REFRESH_TIMEOUT_S", 0.001):
            with self.assertRaises(TimeoutError):
                p.request("consumer", "tcp://consumer:1234")
        self.assertEqual(p._pending, {})

    def test_error_ack_never_releases_success_metadata(self):
        p, _, _, agent, _, _, _, _, _ = pair()
        agent.fail_load = True
        with self.assertRaisesRegex(RuntimeError, "Consumer UCX refresh failed"):
            p.request("consumer", "tcp://consumer:1234")
        self.assertEqual(p._pending, {})

    def test_failed_refresh_can_retry_using_previously_trusted_identity(self):
        p, _, _, agent, _, _, _, _, _ = pair()
        agent.fail_load = True
        with self.assertRaises(RuntimeError):
            p.request("consumer", "tcp://consumer:1234")
        agent.fail_load = False
        self.assertEqual(p.request("consumer", "tcp://consumer:1234")["backend"], "UCX")

    def test_send_failure_clears_pending(self):
        p, _, _, _, _, _, _, _, _ = pair()
        p.send = MagicMock(side_effect=RuntimeError("socket failed"))
        with self.assertRaisesRegex(RuntimeError, "socket failed"):
            p.request("consumer", "tcp://consumer:1234")
        self.assertEqual(p._pending, {})


class TestProtocolIdentity(CustomTestCase):
    def _pending_request(self):
        p, d, pa, da, pr, dr, events, requests, replies = pair()
        sent = threading.Event()

        def save(endpoint, msg):
            requests.append(list(msg))
            sent.set()

        p.send = save
        return p, d, sent, events, requests, replies

    def test_wrong_nonce_peer_sequence_or_metadata_does_not_unblock_sender(self):
        p, d, sent, _, requests, _ = self._pending_request()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(p.request, "consumer", "tcp://consumer:1234")
            self.assertTrue(sent.wait(2))
            req = requests[0]
            valid = [
                ACK,
                b"1",
                req[2],
                req[3],
                b"producer",
                b"consumer",
                b"OK",
                json.dumps(publish_peer_metadata(d.agent, "UCX")).encode(),
            ]
            for field, replacement in [
                (2, b"0" * 32),
                (3, b"999"),
                (4, b"other"),
                (5, b"other"),
                (6, b"UNKNOWN"),
                (7, b"{}"),
            ]:
                bad = list(valid)
                bad[field] = replacement
                self.assertTrue(p.handle(bad))
                self.assertFalse(future.done())
            d.handle(req)
            self.assertEqual(future.result()["version"], "1")

    def test_untrusted_route_or_metadata_never_refreshes_consumer(self):
        p, d, _, _, _, _, events, requests, _ = pair()
        p.request("consumer", "tcp://consumer:1234")
        req = requests[0]
        events.clear()
        for field, value in [
            (4, b"unknown"),
            (5, b"unknown"),
            (6, b"tcp://other:1234"),
            (
                7,
                json.dumps(
                    {
                        "version": "1",
                        "backend": "UCX",
                        "agent_metadata": base64.b64encode(b"other").decode(),
                    }
                ).encode(),
            ),
        ]:
            bad = list(req)
            bad[field] = value
            d.handle(bad)
        self.assertEqual(events, [])

    def test_duplicate_request_reuses_ack_without_resetting_connection(self):
        p, d, _, _, _, _, events, requests, replies = pair()
        p.request("consumer", "tcp://consumer:1234")
        d.handle(requests[0])
        self.assertEqual(events.count("remove-producer"), 1)
        self.assertEqual(len(replies), 2)

    def test_old_sequence_with_different_nonce_cannot_reset_new_connection(self):
        p, d, _, _, _, _, events, requests, _ = pair()
        p.request("consumer", "tcp://consumer:1234")
        p.request("consumer", "tcp://consumer:1234")
        old = list(requests[0])
        old[2] = b"a" * 32
        events.clear()
        d.handle(old)
        self.assertEqual(events, [])

    def test_reply_cache_is_bounded_and_evicted_replay_remains_stale(self):
        p, d, _, _, _, _, events, requests, _ = pair()
        for _ in range(130):
            p.request("consumer", "tcp://consumer:1234")
        self.assertEqual(len(d._replies), 128)
        events.clear()
        d.handle(requests[0])
        self.assertEqual(events, [])

    def test_malformed_known_messages_are_consumed_without_refresh(self):
        _, d, _, _, _, _, events, _, _ = pair()
        for msg in [
            [REQUEST],
            [ACK, b"2"],
            [REQUEST] + [b"x"] * 7,
            [REQUEST] + [b"x" * 100000] * 7,
        ]:
            self.assertTrue(d.handle(msg))
        self.assertFalse(d.handle([b"ABORT", b"1"]))
        self.assertEqual(events, [])


class TestNotificationPreservation(CustomTestCase):
    def test_queued_notifs_are_saved_before_connection_refresh(self):
        mgr = NixlKVManager.__new__(NixlKVManager)
        mgr._notification_lock = threading.Lock()
        mgr._preserved_notifications = {"producer": [b"earlier"]}
        mgr.agent = MagicMock()
        mgr.agent.get_new_notifs.return_value = {"producer": [b"1_aux"]}
        mgr._prefill_peer_registrar = MagicMock()

        def refresh(meta, peer):
            self.assertEqual(mgr._preserved_notifications[peer], [b"earlier", b"1_aux"])

        mgr._prefill_peer_registrar.refresh_known.side_effect = refresh
        mgr._refresh_producer_endpoint({}, "producer")
        mgr._prefill_peer_registrar.refresh_known.assert_called_once()


class TestManagerReciprocalIntegration(CustomTestCase):
    def manager_pair(self):
        producer, consumer, agent, _, registrar, _, events, _, _ = pair()
        mgr = NixlKVManager.__new__(NixlKVManager)
        mgr.agent = agent
        mgr._prefill_peer_registrar = registrar
        mgr._peer_recovery = producer
        mgr._peer_reload_lock = threading.Lock()
        mgr._peer_reload_times = {}
        mgr._unsettled_peers = set()
        mgr.decode_kv_args_table = {
            "consumer": SimpleNamespace(
                agent_metadata=b"consumer", kv_xfer_segments=["old"]
            )
        }
        mgr.prep_handles = {"consumer": "old"}
        mgr.prep_handles_slice_dst = {"consumer": "old"}
        mgr._prepare_payload_xfer = MagicMock(
            side_effect=lambda _: events.append("prepare")
        )
        info = {"consumer": SimpleNamespace(endpoint="consumer", dst_port=1234)}
        return mgr, producer, consumer, events, info

    def test_manager_rebuilds_only_after_consumer_ack_then_metadata_import(self):
        mgr, _, _, events, info = self.manager_pair()
        mgr._reload_invalidated_peers(info)
        self.assertEqual(
            events,
            ["remove-producer", "import-producer", "ack", "import-consumer", "prepare"],
        )
        self.assertEqual(mgr._unsettled_peers, set())
        mgr._prepare_payload_xfer.assert_called_once()

    def test_timeout_blocks_next_writes_even_when_native_invalidation_raises(self):
        mgr, p, _, _, info = self.manager_pair()
        p.send = lambda *_: None
        mgr.agent.remove_remote_agent = MagicMock(
            side_effect=RuntimeError("native invalidation failed")
        )
        with patch(MODULE + ".REFRESH_TIMEOUT_S", 0.001):
            mgr._reload_invalidated_peers(info)
        self.assertEqual(mgr._unsettled_peers, {"consumer"})
        self.assertNotIn("consumer", mgr.prep_handles)
        mgr._prepare_payload_xfer.assert_not_called()

    def test_consumer_ack_send_failure_does_not_kill_listener_or_allow_write(self):
        mgr, p, d, _, info = self.manager_pair()
        d.send = MagicMock(side_effect=RuntimeError("ack link failed"))
        with patch(MODULE + ".REFRESH_TIMEOUT_S", 0.001):
            mgr._reload_invalidated_peers(info)
        self.assertEqual(mgr._unsettled_peers, {"consumer"})
        self.assertEqual(p._pending, {})
        mgr._prepare_payload_xfer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
