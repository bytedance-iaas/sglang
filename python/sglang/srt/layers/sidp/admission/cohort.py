"""Assemble shape-similar requests into cohorts for balanced worker dispatch.

Goal (design §5.4): keep the SiDP workers' per-step compute shapes similar so the
coordinated device barrier does not idle waiting on a slow member. We bucket
incoming requests by shape_signature and feed them to the workers in balanced
batches.

Correctness note: with dynamic nptr (Phase 2b) the GPU side is correct for ANY
arrival pattern -- if only M workers have work, the barrier uses nptr=M. So this
assembler is purely a *performance* layer. That lets us default to "dispatch what
we have" on timeout (rely on dynamic nptr) instead of fabricating padding work;
duplicate-fill remains available (opt-in) for pure-bandwidth measurement runs
where keeping all workers active matters more than avoiding wasted compute.

Producer / consumer design:
  * submit()  -- producer front door: compute signature, append to that
    signature's bucket. Does NOT dispatch. Returns a Ticket the caller waits on.
  * scan loop -- producer: periodically drains each bucket into a single shared
    send queue, one *batch* at a time, whenever the bucket has >= cohort_size
    requests, or its oldest request has waited past the timeout (partial batch,
    optionally duplicate-filled). After draining, the bucket count drops and the
    next threshold crossing enqueues the next batch.
  * send loop -- consumer: pops tickets from the head of the send queue one by
    one and assigns each to a worker via a GLOBAL rolling counter
    (worker = rolling; rolling = (rolling+1) % service_num). The counter
    continues across batches, so even a stream of small/partial batches spreads
    evenly over all workers instead of always starting at worker 0. Actual I/O
    is handed to dispatch_fn, which sends concurrently (see proxy) so a batch's
    requests still start together.

This module is HTTP-free and deterministic so it can be unit-tested offline; the
proxy injects a ``dispatch_fn`` that performs the actual worker POSTs.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

try:  # works both as a package module and as a flat module (see proxy.py)
    from .signature import SignatureConfig, compute_signature
except ImportError:  # noqa: BLE001
    from signature import SignatureConfig, compute_signature


@dataclass
class Ticket:
    """One in-flight request awaiting cohort dispatch and a worker response."""

    body: dict
    signature: tuple
    submit_ts: float
    done: threading.Event = field(default_factory=threading.Event)
    response: Any = None
    status: int = 0
    error: Optional[str] = None
    worker_index: int = -1
    is_padding: bool = False  # duplicate fill; response is discarded

    def set_result(self, status: int, response: Any) -> None:
        self.status = status
        self.response = response
        self.done.set()

    def set_error(self, error: str) -> None:
        self.error = error
        self.status = 502
        self.done.set()


# dispatch_fn(ticket, worker_index): perform the actual send for one ticket.
DispatchFn = Callable[[Ticket, int], None]


class CohortAssembler:
    """Producer/consumer shape-bucketing cohort former with rolling dispatch."""

    def __init__(
        self,
        *,
        service_num: int,
        n_min: int,
        timeout_ms: int,
        dispatch_fn: DispatchFn,
        sig_cfg: Optional[SignatureConfig] = None,
        duplicate_fill: bool = False,
        scan_interval_ms: int = 20,
    ) -> None:
        if service_num < 1:
            raise ValueError("service_num must be >= 1")
        if n_min < 1:
            raise ValueError("n_min must be >= 1")
        self.service_num = service_num
        self.n_min = n_min
        self.cohort_size = n_min * service_num
        self.timeout_s = timeout_ms / 1000.0
        self.dispatch_fn = dispatch_fn
        self.sig_cfg = sig_cfg or SignatureConfig()
        self.duplicate_fill = duplicate_fill
        self.scan_interval_s = scan_interval_ms / 1000.0

        # Buckets (producer side) and the shared send queue (consumer side) have
        # separate locks so scanning and sending never block each other.
        self._bucket_lock = threading.Lock()
        self._buckets: Dict[tuple, Deque[Ticket]] = defaultdict(deque)

        self._queue_lock = threading.Lock()
        self._send_queue: Deque[Ticket] = deque()
        self._queue_ready = threading.Event()  # signaled when queue non-empty
        self._rolling = 0  # GLOBAL worker rolling counter (consumer only)

        # Stats for observability (design §10 requires reporting cohort behavior).
        self.stats = {
            "submitted": 0,
            "batches_full": 0,
            "batches_timeout": 0,
            "padding_created": 0,
            "dispatched": 0,
        }

        self._stop = threading.Event()
        self._scanner = threading.Thread(
            target=self._scan_loop, name="cohort-scanner", daemon=True
        )
        self._sender = threading.Thread(
            target=self._send_loop, name="cohort-sender", daemon=True
        )
        self._scanner.start()
        self._sender.start()

    # -- producer front door ----------------------------------------------
    def submit(self, body: dict) -> Ticket:
        """Bucket a request by signature. Dispatch happens in the scan/send loops."""
        sig = compute_signature(body, self.sig_cfg)
        ticket = Ticket(body=body, signature=sig, submit_ts=time.monotonic())
        with self._bucket_lock:
            self.stats["submitted"] += 1
            self._buckets[sig].append(ticket)
        return ticket

    # -- producer: scan buckets -> send queue -----------------------------
    def _enqueue_batch(self, batch: List[Ticket]) -> None:
        with self._queue_lock:
            self._send_queue.extend(batch)
            self._queue_ready.set()

    def _scan_once(self) -> None:
        now = time.monotonic()
        batches: List[List[Ticket]] = []
        with self._bucket_lock:
            for bucket in self._buckets.values():
                # Full batches: while enough for a cohort, cut one cohort off.
                while len(bucket) >= self.cohort_size:
                    batch = [bucket.popleft() for _ in range(self.cohort_size)]
                    self.stats["batches_full"] += 1
                    batches.append(batch)
                # Timeout: oldest remaining request waited too long -> release a
                # partial batch (optionally duplicate-filled to a full cohort).
                if bucket and (now - bucket[0].submit_ts) >= self.timeout_s:
                    take = len(bucket)
                    batch = [bucket.popleft() for _ in range(take)]
                    self.stats["batches_timeout"] += 1
                    if self.duplicate_fill:
                        self._duplicate_fill(batch)
                    batches.append(batch)
        for batch in batches:
            self._enqueue_batch(batch)

    def _duplicate_fill(self, batch: List[Ticket]) -> None:
        """Pad a partial batch up to a full cohort with copies (same signature =>
        same shape); padding responses are discarded by the proxy."""
        n_real = len(batch)
        if n_real == 0:
            return
        i = 0
        while len(batch) < self.cohort_size:
            src = batch[i % n_real]
            batch.append(
                Ticket(
                    body=dict(src.body),
                    signature=src.signature,
                    submit_ts=time.monotonic(),
                    is_padding=True,
                )
            )
            self.stats["padding_created"] += 1
            i += 1

    def _scan_loop(self) -> None:
        while not self._stop.wait(self.scan_interval_s):
            self._scan_once()

    # -- consumer: send queue -> workers (global rolling) -----------------
    def _send_loop(self) -> None:
        while not self._stop.is_set():
            if not self._queue_ready.wait(timeout=self.scan_interval_s):
                continue
            while True:
                with self._queue_lock:
                    if not self._send_queue:
                        self._queue_ready.clear()
                        break
                    ticket = self._send_queue.popleft()
                    worker = self._rolling
                    self._rolling = (self._rolling + 1) % self.service_num
                ticket.worker_index = worker
                self.stats["dispatched"] += 1
                # dispatch_fn sends concurrently, so pulling stays sequential
                # (deterministic rolling) while a batch still starts together.
                self.dispatch_fn(ticket, worker)

    def close(self) -> None:
        self._stop.set()
        self._queue_ready.set()
