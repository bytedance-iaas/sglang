#!/usr/bin/env python3
"""Standalone SiDP Direction A admission proxy (single client entry).

Why a standalone proxy (not the Rust sgl-router): the router
(`python -m sglang_router.launch_router`) is a compiled Rust binary with no
Python routing hook, so custom cohort-grouping cannot be added without a Rust
rebuild. The cohort logic must live in Python regardless, and -- crucially --
dynamic nptr (Phase 2b) already makes the GPU side correct for ANY arrival
pattern, so this proxy is a pure *performance* layer: at worst a bug here slows
cohort formation, it cannot corrupt results.

Flow:
    client --/v1/chat/completions--> proxy :8080
        proxy computes shape_signature (no tokenize; image H*W from headers)
        proxy buckets same-signature requests into cohorts (n_min per worker)
        on cohort-full or timeout, proxy POSTs each request to worker 8100+k
        proxy relays each worker response back to its caller

Passthrough mode (``--passthrough``): skip signature bucketing and cohort
assembly entirely. Each request is dispatched the instant it arrives to the
next worker via a global round-robin counter (worker = rolling % world_size).
This is the right front-end when the device barrier is OFF (G0): cohort
assembly only exists to keep worker forward shapes aligned for the barrier, so
without coordination it is pure latency overhead that flattens the arrival
burst. Passthrough matches sglang-router's round_robin policy but keeps this
proxy's deep accept backlog and dispatch pool.

Run as a package module (so it ships under the force-synced sglang/ tree):
    python -m sglang.srt.layers.sidp.admission.proxy --world-size 8 ...
Stdlib + PIL only; no torch/CUDA dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Import sibling modules WITHOUT triggering the heavy ``sglang`` package
# __init__ (which imports orjson/torch). We add this package dir to sys.path and
# import the flat module names, so the proxy starts with stdlib + PIL only.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from cohort import CohortAssembler, Ticket  # noqa: E402
from signature import SignatureConfig  # noqa: E402

HOST = "127.0.0.1"
WORKER_BASE_PORT = 8100


class AdmissionHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a deep accept backlog and daemon threads.

    The default listen backlog is 5, so a burst of concurrent client connects
    (the benchmark opens ~1024 at once) overflows the accept queue and the OS
    resets the excess -- the ConnectionReset failures we saw at start. Raise the
    backlog so all clients can connect; allow_reuse_address avoids TIME_WAIT
    bind failures on repeated runs.
    """

    request_queue_size = 4096
    daemon_threads = True
    allow_reuse_address = True


class ProxyState:
    """Shared proxy state: worker URLs, assembler, dispatch pool."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.worker_urls = [
            f"http://{HOST}:{WORKER_BASE_PORT + k}" for k in range(args.world_size)
        ]
        self.world_size = args.world_size
        self.request_timeout = args.request_timeout_s
        self.passthrough = args.passthrough
        # Dispatch pool sizing is critical: _send_one blocks (synchronous urlopen)
        # for the WHOLE lifetime of a worker request -- prefill + all decode steps
        # -- because the proxy relays non-streaming responses. So the number of
        # pool threads == the max number of requests that can be in flight on the
        # workers at once. Too small (e.g. world_size) throttles the whole system
        # to that many concurrent requests regardless of how much the workers can
        # actually run, starving them and timing out the rest in their buckets.
        # Size it to the client concurrency the workers are expected to sustain
        # (their combined max_running_requests), configurable via --dispatch-workers.
        self._pool = ThreadPoolExecutor(
            max_workers=max(args.dispatch_workers, args.world_size),
            thread_name_prefix="dispatch",
        )
        # Passthrough: no signature/cohort layer. A global rolling counter assigns
        # each arriving request to the next worker; dispatch is immediate.
        self._rr_lock = threading.Lock()
        self._rr = 0
        self._passthrough_stats = {"submitted": 0, "dispatched": 0}
        if self.passthrough:
            self.assembler = None
        else:
            sig_cfg = SignatureConfig(
                img_rel_tol=args.img_rel_tol,
                text_rel_tol=args.text_rel_tol,
                text_chars_per_token=args.text_chars_per_token,
                fetch_remote_images=args.fetch_remote_images,
            )
            self.assembler = CohortAssembler(
                service_num=args.world_size,
                n_min=args.cohort_min_per_service,
                timeout_ms=args.cohort_wait_timeout_ms,
                dispatch_fn=self._dispatch,
                sig_cfg=sig_cfg,
                duplicate_fill=args.duplicate_fill,
            )

    @property
    def stats(self) -> dict:
        return self._passthrough_stats if self.passthrough else self.assembler.stats

    def submit_passthrough(self, body: dict) -> Ticket:
        """Immediate round-robin dispatch: one request -> next worker, no cohort."""
        with self._rr_lock:
            worker = self._rr
            self._rr = (self._rr + 1) % self.world_size
            self._passthrough_stats["submitted"] += 1
            self._passthrough_stats["dispatched"] += 1
        ticket = Ticket(body=body, signature=(), submit_ts=time.monotonic())
        ticket.worker_index = worker
        self._pool.submit(self._send_one, ticket, worker)
        return ticket

    def _dispatch(self, ticket: Ticket, worker: int) -> None:
        """Consumer callback: send one ticket to its assigned worker.

        The assembler's send loop pulls sequentially (deterministic rolling) and
        calls this per ticket; we hand the actual HTTP POST to the thread pool so
        a batch's requests still start together instead of serializing.
        """
        self._pool.submit(self._send_one, ticket, worker)

    def _send_one(self, ticket: Ticket, worker: int) -> None:
        url = f"{self.worker_urls[worker]}/v1/chat/completions"
        data = json.dumps(ticket.body).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as r:
                body = r.read()
                ticket.set_result(r.status, body)
        except urllib.error.HTTPError as e:
            ticket.set_result(e.code, e.read())
        except Exception as e:  # noqa: BLE001
            ticket.set_error(repr(e))

    def close(self) -> None:
        if self.assembler is not None:
            self.assembler.close()
        self._pool.shutdown(wait=False)


def make_handler(state: ProxyState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence per-request stderr spam
            pass

        def _send(self, status: int, body: bytes, ctype="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/health", "/v1/health"):
                self._send(200, b'{"status":"ok"}')
            elif self.path == "/stats":
                self._send(200, json.dumps(state.stats).encode())
            else:
                self._send(404, b'{"error":"not found"}')

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._send(404, b'{"error":"unsupported path"}')
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                body = json.loads(raw)
            except Exception as e:  # noqa: BLE001
                self._send(400, json.dumps({"error": f"bad body: {e!r}"}).encode())
                return

            # Passthrough dispatches immediately (round-robin); cohort mode buckets
            # by signature and releases on cohort-full/timeout. Both return a Ticket
            # the handler thread blocks on (ThreadingHTTPServer = one thread/req).
            if state.passthrough:
                ticket = state.submit_passthrough(body)
            else:
                ticket = state.assembler.submit(body)
            if not ticket.done.wait(timeout=state.request_timeout + 30):
                self._send(504, b'{"error":"cohort/admission timeout"}')
                return
            if ticket.error is not None:
                self._send(502, json.dumps({"error": ticket.error}).encode())
                return
            resp = ticket.response
            if isinstance(resp, str):
                resp = resp.encode()
            self._send(ticket.status or 200, resp or b"")

    return Handler


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--cohort-min-per-service", type=int, default=1,
                   help="N_min: min requests per worker per cohort")
    p.add_argument("--cohort-wait-timeout-ms", type=int, default=2000,
                   help="Bounded wait before releasing a partial cohort")
    p.add_argument("--dispatch-workers", type=int, default=1024,
                   help="Dispatch thread pool size = max concurrent requests the "
                        "proxy keeps in flight on workers. Set >= client "
                        "concurrency (workers' combined max_running_requests) so "
                        "the proxy relays, not throttles. Default 1024.")
    p.add_argument("--img-rel-tol", type=float, default=0.05,
                   help="Relative tolerance for sum(H*W) image bucketing (0.05 "
                        "=> within ~5% share a bucket)")
    p.add_argument("--text-rel-tol", type=float, default=0.15,
                   help="Relative tolerance for estimated-text-token bucketing")
    p.add_argument("--text-chars-per-token", type=float, default=4.0)
    p.add_argument("--fetch-remote-images", action="store_true",
                   help="Fetch http(s) images to measure H*W (default off)")
    p.add_argument("--duplicate-fill", action="store_true",
                   help="On timeout, duplicate bucket requests to keep all "
                        "workers active (pure-bandwidth runs). Default: rely on "
                        "dynamic nptr and dispatch a partial cohort.")
    p.add_argument("--request-timeout-s", type=float, default=1800.0,
                   help="Per-request upper bound for BOTH the proxy->worker "
                        "urlopen and the client-facing handler wait (the latter is "
                        "this + 30s). Default 1800s to match the client's own "
                        "timeout so the proxy is never the tighter gate -- a lower "
                        "value here aborts requests the workers would still finish "
                        "(seen as 'cohort/admission timeout' at ~timeout+30s).")
    p.add_argument("--passthrough", action="store_true",
                   help="Immediate round-robin dispatch: skip signature bucketing "
                        "and cohort assembly, send each request to the next worker "
                        "the moment it arrives. Use when the device barrier is off "
                        "(no coordination) so cohort assembly's latency is not paid "
                        "for nothing.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    state = ProxyState(args)
    server = AdmissionHTTPServer((HOST, args.port), make_handler(state))
    print(f"[admission] proxy listening on http://{HOST}:{args.port}")
    print(f"[admission] workers: {state.worker_urls}")
    if args.passthrough:
        print("[admission] mode=passthrough (immediate round-robin, no cohort)")
    else:
        print(f"[admission] cohort={args.cohort_min_per_service}/worker "
              f"(size={args.world_size * args.cohort_min_per_service}), "
              f"timeout={args.cohort_wait_timeout_ms}ms, "
              f"duplicate_fill={args.duplicate_fill}")
    print("启动ok", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.close()
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
