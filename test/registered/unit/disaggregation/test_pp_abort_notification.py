"""PP aborts notify Decode only after the global GPU/source-read drain."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import requests
from test_pp_prefill_control import Pipeline

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, KVTransferError
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
from sglang.srt.disaggregation.pp_prefill_control import (
    PPAbortNotifications,
    apply_pp_boundary,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def make_sender(room, lease, receive, rank=0):
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager._pp_abort_notifications = PPAbortNotifications(300)
    manager._pp_abort_notifications.open(room)
    manager._pp_transfer_leases = lease
    manager.request_status = {room: KVPoll.Transferring}
    manager.failure_records = {}
    manager.failure_lock = threading.Lock()
    manager.deferred_bootstrap = None
    manager.transfer_infos = {}
    manager.req_to_decode_prefix_len = {}
    manager.enable_deferred_decode_kv_release = False
    manager._staging_outstanding = {}
    manager._deferred_ack_targets = {}
    manager._prefill_unique_rank = lambda: rank
    manager.resolve_kv_replica_factor = lambda infos: None
    manager._send_multipart_locked = lambda endpoint, parts, **kw: receive(parts)
    info = NS(
        room=room,
        endpoint="127.0.0.1",
        dst_port=10000,
        mooncake_session_id="decode",
        is_dummy=False,
        decode_prefix_len=0,
    )
    manager._register_decode_transfer_info(info, 1)
    sender = MooncakeKVSender.__new__(MooncakeKVSender)
    sender.kv_mgr = manager
    sender.bootstrap_room = room
    sender.conclude_state = None
    sender.trace_ctx = NS(abort=lambda **kw: None, trace_req_finish=lambda: None)
    return sender


class DrainFixture:
    def __init__(self, size=2, terminal=lambda: None):
        self.pipeline = Pipeline(size)
        self.wire = []
        self.retired = []
        self.decoder = CommonKVManager.__new__(CommonKVManager)
        self.decoder.request_status = {123: KVPoll.Transferring}
        self.decoder.failure_records = {}
        self.decoder.failure_lock = threading.Lock()
        self.schedulers = []
        for rank, (control, req, leases) in enumerate(
            zip(self.pipeline.controls, self.pipeline.reqs, self.pipeline.leases)
        ):

            def receive(parts):
                # A single Failed lets Decode release its destination. EVERY
                # source rank must already be safe before the first wire send.
                assert all(not c.gpu_refs for c in self.pipeline.controls)
                assert all(
                    l.drained(123) or rank in self.retired
                    for rank, l in enumerate(self.pipeline.leases)
                )
                self.wire.append(parts)
                room, status, source, reason = self.decoder.parse_kv_status_message(
                    parts
                )
                self.decoder.apply_prefill_status(
                    bootstrap_room=room,
                    status=status,
                    prefill_rank=source,
                    failure_reason=reason,
                )
                terminal()

            req.disagg_kv_sender = make_sender(123, leases, receive, rank)

            def retire(req, pp_retire_commit, rank=rank):
                assert pp_retire_commit and self.wire
                assert req.bootstrap_room in req.disagg_kv_sender.kv_mgr.transfer_infos
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception:
                    pass
                assert (
                    req.bootstrap_room not in req.disagg_kv_sender.kv_mgr.transfer_infos
                )
                self.retired.append(rank)

            self.schedulers.append(
                NS(
                    pp_prefill_control=control,
                    waiting_queue=[req],
                    disagg_prefill_bootstrap_queue=NS(queue=[]),
                    disagg_prefill_inflight_queue=[],
                    chunked_req=None,
                    attn_tp_cpu_group=None,
                    attn_cp_cpu_group=None,
                    handle_bootstrap_failure=retire,
                )
            )

    def boundary(self, deliver=True):
        frame = None
        with patch(
            "sglang.srt.disaggregation.utils._all_reduce_polls",
            side_effect=lambda polls, group: polls,
        ):
            for control, scheduler in zip(self.pipeline.controls, self.schedulers):
                control.log.begin(frame)
                apply_pp_boundary(scheduler)
                frame = control.log.finish()
        if deliver:
            self.pipeline.deliver()


class TestPPAbortNotification(CustomTestCase):
    def test_first_wire_failure_waits_for_every_gpu_and_queued_read(self):
        for size in (2, 4, 8):
            with self.subTest(size=size):
                f = DrainFixture(size)
                p = f.pipeline
                batch = NS(reqs=[p.reqs[-1]])
                p.controls[-1].bind_batch(batch)
                p.leases[-1].acquire(123)
                p.leases[-1].acquire(123)
                sender = p.reqs[-1].disagg_kv_sender
                # Worker errors must not bypass the global retirement barrier.
                sender.kv_mgr.conclude_failure(
                    bootstrap_room=123, failure_reason="worker error"
                )
                p.controls[-1].propose(p.reqs[-1], "abort")
                for _ in range(3):
                    f.boundary(deliver=False)  # PP0 has not received proposal.
                self.assertFalse(f.wire)
                p.deliver()
                f.boundary()
                p.controls[-1].complete_batch(batch)
                p.leases[-1].release(123)
                for _ in range(3):
                    f.boundary()
                self.assertFalse(f.wire)
                self.assertEqual(f.decoder.request_status[123], KVPoll.Transferring)
                p.leases[-1].release(123)
                for _ in range(4):
                    f.boundary()
                self.assertEqual(len(f.wire), size)
                self.assertEqual(sorted(f.retired), list(range(size)))
                self.assertEqual(f.decoder.request_status[123], KVPoll.Failed)

    def test_late_bootstrap_gets_failure_without_recreating_cleared_state(self):
        f = DrainFixture()
        f.pipeline.controls[0].propose(f.pipeline.reqs[0], "abort")
        for _ in range(4):
            f.boundary()
        sender = f.pipeline.reqs[0].disagg_kv_sender
        manager = sender.kv_mgr
        before = len(f.wire)
        info = NS(
            room=123,
            endpoint="127.0.0.2",
            dst_port=10001,
            mooncake_session_id="late-decode",
            is_dummy=False,
        )
        manager._register_decode_transfer_info(info, 1)
        manager._register_decode_transfer_info(info, 1)
        self.assertEqual(len(f.wire), before + 2)
        self.assertNotIn(123, manager.transfer_infos)
        self.assertNotIn(123, manager.request_status)

    def test_compaction_waits_for_fenced_gpu_and_source_read_owners(self):
        from sglang.srt.disaggregation.utils import (
            DisaggregationMode,
            unified_memory_disagg_move_gate,
        )

        for size in (2, 4, 8):
            with self.subTest(size=size):
                f = DrainFixture(size)
                p = f.pipeline
                batch = NS(reqs=[p.reqs[-1]])
                p.controls[-1].bind_batch(batch)
                p.leases[-1].acquire(123)
                gates = []
                for scheduler, req in zip(f.schedulers, p.reqs):
                    scheduler.disaggregation_mode = DisaggregationMode.PREFILL
                    scheduler.disagg_prefill_pending_chunk_rids = set()
                    scheduler.waiting_queue = []
                    scheduler.disagg_prefill_inflight_queue = [req]
                    gates.append(unified_memory_disagg_move_gate(scheduler))
                self.assertTrue(all(not gate() for gate in gates))
                p.controls[0].propose(p.reqs[0], "abort")
                f.boundary()
                self.assertTrue(
                    all(not s.disagg_prefill_inflight_queue for s in f.schedulers)
                )
                self.assertTrue(all(not gate() for gate in gates))
                p.controls[-1].complete_batch(batch)
                for _ in range(3):
                    f.boundary()
                self.assertTrue(all(not gate() for gate in gates))
                self.assertFalse(f.wire)
                p.leases[-1].release(123)
                for _ in range(4):
                    f.boundary()
                self.assertEqual(sorted(f.retired), list(range(size)))
                self.assertTrue(all(gate() for gate in gates))

    def test_failed_socket_send_is_retried_by_same_bootstrap_endpoint(self):
        from sglang.srt.disaggregation.pp_prefill_control import TransferLeases

        received = []
        attempts = []
        sender = make_sender(123, TransferLeases(), received.append)
        manager = sender.kv_mgr
        send = manager._send_multipart_locked

        def fail_once(*args, **kwargs):
            attempts.append(args[0])
            if len(attempts) == 1:
                raise OSError("injected transient send failure")
            send(*args, **kwargs)

        manager._send_multipart_locked = fail_once
        sender.pp_abort_fence()
        sender.pp_abort_notify()
        self.assertFalse(received)
        with self.assertRaises(KVTransferError):
            sender.failure_exception()
        info = NS(
            room=123,
            endpoint="127.0.0.1",
            dst_port=10000,
            mooncake_session_id="decode",
            is_dummy=False,
        )
        # Each same-endpoint retry must remain eligible after local cleanup.
        for _ in range(2):
            manager._register_decode_transfer_info(info, 1)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(len(received), 2)
        self.assertTrue(
            all(
                manager.parse_kv_status_message(msg)[1] == KVPoll.Failed
                for msg in received
            )
        )
        self.assertNotIn(123, manager.transfer_infos)
        self.assertNotIn(123, manager.request_status)

    def test_failed_room_reuse_rejects_intake_and_preserves_late_peer_tombstone(self):
        from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
        from sglang.srt.disaggregation.utils import TransferBackend
        from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.sampling.sampling_params import SamplingParams

        f = DrainFixture()
        f.pipeline.controls[0].propose(f.pipeline.reqs[0], "abort")
        for _ in range(4):
            f.boundary()
        manager = f.pipeline.reqs[0].disagg_kv_sender.kv_mgr
        gate = manager._pp_abort_notifications
        tombstone = gate.rooms[123]
        with self.assertRaisesRegex(ValueError, "new bootstrap_room"):
            gate.open(123)
        outputs, cache_finishes = [], []
        scheduler = NS(
            pp_prefill_control=f.pipeline.controls[0],
            beam_coordinator=NS(retire_group=lambda req: None),
            tree_cache=NS(finish=lambda *args: cache_finishes.append(args)),
            output_streamer=NS(stream_output=lambda reqs, *args: outputs.extend(reqs)),
        )
        scheduler._release_aborted_request = Scheduler._release_aborted_request.__get__(
            scheduler
        )
        scheduler.retire_unadmitted_request = (
            Scheduler.retire_unadmitted_request.__get__(scheduler)
        )
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.scheduler, queue.kv_manager = scheduler, manager
        queue.max_total_num_tokens, queue.bootstrap_port = 1000, 8998
        queue.transfer_backend, queue.queue = TransferBackend.MOONCAKE, []
        req = Req("retry", "", [1, 2], SamplingParams())
        req.bootstrap_room, req.bootstrap_host = 123, "127.0.0.1"
        req.disagg_prefill_dp_rank = None
        queue.add(req, 1)
        self.assertEqual(outputs, [req])
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.status_code, 400)
        self.assertIn("new bootstrap_room", req.finished_reason.message)
        self.assertEqual(len(cache_finishes), 1)
        self.assertFalse(queue.queue)
        self.assertFalse(scheduler.pp_prefill_control.requests)
        self.assertIs(gate.rooms[123], tombstone)
        info = NS(
            room=123,
            endpoint="127.0.0.2",
            dst_port=10001,
            mooncake_session_id="old-decode",
            is_dummy=False,
        )
        before = len(f.wire)
        manager._register_decode_transfer_info(info, 1)
        self.assertEqual(len(f.wire), before + 1)
        self.assertNotIn(123, manager.transfer_infos)
        self.assertNotIn(123, manager.request_status)
        # Success still permits explicit room reuse after ownership is released.
        gate.open(456)
        with self.assertRaisesRegex(ValueError, "new bootstrap_room"):
            gate.open(456)
        gate.forget_success(456)
        gate.open(456)
        self.assertTrue(gate.rooms[456]["owned"])

    def test_notification_rejects_undrained_local_source(self):
        f = DrainFixture()
        sender = f.pipeline.reqs[0].disagg_kv_sender
        sender.pp_abort_fence()
        # A closed room with an outstanding queued read is not drained.
        f.pipeline.leases[1].acquire(123)
        other = f.pipeline.reqs[1].disagg_kv_sender
        other.pp_abort_fence()
        with self.assertRaisesRegex(RuntimeError, "source drain"):
            other.pp_abort_notify()
        self.assertFalse(f.wire)

    def test_pending_owner_never_expires_and_success_drops_saved_targets(self):
        now = [0]
        gate = PPAbortNotifications(5, lambda: now[0])
        gate.open(123)
        self.assertEqual(gate.failure(123, [("decode", 1)], "error")[0], [])
        now[0] = 100
        gate.open(456)
        self.assertIn(123, gate.rooms)
        self.assertEqual(gate.commit(123), [("decode", 1)])
        self.assertEqual(
            gate.failure(123, [("decode", 1)], None), ([("decode", 1)], "error")
        )
        self.assertEqual(gate.failure(123, [], None)[0], [])
        now[0] = 106
        gate.open(789)
        self.assertNotIn(123, gate.rooms)
        gate.forget_success(456)
        self.assertNotIn(456, gate.rooms)

    def test_frozen_success_and_late_abort_do_not_send_failure(self):
        f = DrainFixture()
        for control, req in zip(f.pipeline.controls, f.pipeline.reqs):
            sender = req.disagg_kv_sender
            sender.conclude_state = (
                KVPoll.Failed
            )  # Late local abort after frozen Success.
            control.forget_success(req)
            self.assertNotIn(123, sender.kv_mgr._pp_abort_notifications.rooms)
            sender.clear()
            self.assertNotIn(123, sender.kv_mgr._pp_abort_notifications.rooms)
            from sglang.srt.managers.scheduler import Scheduler

            Scheduler.abort_request(
                NS(pp_prefill_control=control),
                NS(abort_all=True, rid="", abort_message="late abort"),
            )
        for _ in range(3):
            f.boundary()
        self.assertFalse(f.wire)
        self.assertFalse(f.retired)


@unittest.skipUnless(
    importlib.util.find_spec("sglang_router"), "sglang-router is required"
)
class TestPPAbortRouterTermination(CustomTestCase):
    def test_sse_200_abort_returns_only_after_global_drain(self):
        """Real Router + HTTP SSE + real PP decisions and KV status codec.

        Model/RDMA work is represented by explicitly held GPU/read leases.
        Full model coverage is also exercised by the original-attachment replay.
        """
        terminal = threading.Event()
        p_received, d_received = threading.Event(), threading.Event()
        f = DrainFixture(terminal=terminal.set)
        p = f.pipeline
        batch = NS(reqs=[p.reqs[-1]])
        p.controls[-1].bind_batch(batch)
        p.leases[-1].acquire(123)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if "health" in self.path:
                    body = b"OK"
                elif "models" in self.path:
                    body = json.dumps(
                        {"object": "list", "data": [{"id": "mock", "object": "model"}]}
                    ).encode()
                else:
                    body = json.dumps(
                        dict(
                            model_path="mock",
                            served_model_name="mock",
                            is_generation=True,
                            tp_size=1,
                            dp_size=1,
                            attn_dp_size=1,
                            disaggregation_mode=self.server.role,
                        )
                    ).encode()
                self.respond(body, "application/json")

            def respond(self, body, content_type):
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if self.server.role == "prefill":
                    p_received.set()
                else:
                    d_received.set()
                    if not terminal.wait(15):
                        return
                self.respond(
                    b'data: {"error":{"message":"KV transfer aborted","code":500}}\n\ndata: [DONE]\n\n',
                    "text/event-stream",
                )

        servers = []
        router = None
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(
            max_workers=1
        ) as executor:
            try:
                for role in ("prefill", "decode"):
                    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                    server.daemon_threads = True
                    server.role = role
                    threading.Thread(target=server.serve_forever, daemon=True).start()
                    servers.append(server)
                # The OS chooses an available port for the probe listener.
                import socket

                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                with (Path(directory) / "router.log").open("w+") as log:
                    router = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "sglang_router.launch_router",
                            "--pd-disaggregation",
                            "--prefill",
                            f"http://127.0.0.1:{servers[0].server_port}",
                            "--decode",
                            f"http://127.0.0.1:{servers[1].server_port}",
                            "--host",
                            "127.0.0.1",
                            "--port",
                            str(port),
                            "--disable-circuit-breaker",
                            "--health-check-interval-secs",
                            "999999",
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        self.assertIsNone(router.poll(), "Router exited during startup")
                        try:
                            response = requests.get(
                                f"http://127.0.0.1:{port}/workers", timeout=0.2
                            )
                            if (
                                response.status_code == 200
                                and response.text.count("healthy") >= 2
                            ):
                                break
                        except requests.RequestException:
                            pass
                        time.sleep(0.1)
                    else:
                        log.seek(0)
                        self.fail(
                            "Router did not register workers: " + log.read()[-4000:]
                        )
                    job = executor.submit(
                        requests.post,
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        json={
                            "model": "mock",
                            "messages": [{"role": "user", "content": "test"}],
                            "stream": True,
                            "max_tokens": 16,
                        },
                        timeout=(2, 10),
                    )
                    self.assertTrue(p_received.wait(5))
                    self.assertTrue(d_received.wait(5))
                    p.controls[1].propose(p.reqs[1], "client abort")
                    for _ in range(3):
                        f.boundary(deliver=False)
                    self.assertFalse(terminal.is_set())
                    self.assertFalse(job.done())
                    p.deliver()
                    for _ in range(3):
                        f.boundary()
                    self.assertFalse(terminal.is_set())
                    p.controls[-1].complete_batch(batch)
                    for _ in range(3):
                        f.boundary()
                    self.assertFalse(terminal.is_set())
                    p.leases[-1].release(123)
                    start = time.monotonic()
                    for _ in range(4):
                        f.boundary()
                    response = job.result(timeout=3)
                    self.assertLess(time.monotonic() - start, 3)
                    self.assertEqual(response.status_code, 200)
                    self.assertIn('"error"', response.text)
                    self.assertIn("[DONE]", response.text)
                    self.assertEqual(f.decoder.request_status[123], KVPoll.Failed)
                    self.assertEqual(sorted(f.retired), [0, 1])
            finally:
                terminal.set()
                if router is not None:
                    router.terminate()
                    try:
                        router.wait(5)
                    except subprocess.TimeoutExpired:
                        router.kill()
                        router.wait()
                for server in servers:
                    server.shutdown()
                    server.server_close()


if __name__ == "__main__":
    unittest.main()
