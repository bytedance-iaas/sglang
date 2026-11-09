"""Ordered PP-prefill decisions, carried by the existing request message.

Transport observations are proposals. Only a recorded PP0 decision may change
scheduler ownership. No PP collective is added to the forward critical path.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def request_key(req):
    # A new transport room is a new request incarnation, even if RID is reused.
    return req.rid, req.bootstrap_room


@dataclass(frozen=True)
class PPControlFrame:
    epoch: str
    seq: int
    events: tuple


class PPDecisionLog:
    """One immutable decision tape per logical pipeline scheduling slot."""

    def __init__(self, leader, epoch, broadcast=lambda value: value):
        self.leader = leader
        self.epoch = epoch
        self.seq = -1
        self.broadcast = broadcast
        self.events = []
        self.cursor = 0

    def begin(self, frame=None):
        if self.leader:
            if frame is not None:
                raise RuntimeError("PP0 received an upstream control frame")
        elif frame is None or frame.epoch != self.epoch or frame.seq != self.seq + 1:
            raise RuntimeError("Missing, stale, or out-of-order PP control frame")
        self.seq += 1
        self.events = [] if self.leader else frame.events
        self.cursor = 0

    def decide(self, phase, keys, produce):
        keys = tuple(keys)
        if self.leader:
            value = self.broadcast(produce())
            self.events.append((phase, keys, value))
            return value
        if self.cursor >= len(self.events):
            raise RuntimeError(f"PP control tape exhausted at {phase}")
        event = self.events[self.cursor]
        self.cursor += 1
        if event[:2] != (phase, keys):
            raise RuntimeError(f"PP control phase/input mismatch at {phase}")
        return event[2]

    def check(self, phase, value):
        expected = self.decide(phase, (), lambda: value)
        if expected != value:
            raise RuntimeError(f"PP logical batch mismatch at {phase}")

    def finish(self):
        if not self.leader and self.cursor != len(self.events):
            raise RuntimeError("Unconsumed PP control decisions")
        return PPControlFrame(self.epoch, self.seq, tuple(self.events))


class TransferLeases:
    """Count queued AND executing reads, independently of transport status.

    Closing a room and acquiring a lease use the same lock. A worker keeps its
    lease across requeue/defer; Failed alone never constitutes a drain proof.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.refs = Counter()
        self.closed = set()

    def acquire(self, room):
        with self.lock:
            if room in self.closed:
                return False
            self.refs[room] += 1
            return True

    def release(self, room):
        with self.lock:
            if self.refs[room] <= 0:
                raise RuntimeError("Unbalanced PP transfer lease")
            self.refs[room] -= 1
            if not self.refs[room]:
                del self.refs[room]

    def close(self, room):
        with self.lock:
            self.closed.add(room)

    def drained(self, room):
        with self.lock:
            return room in self.closed and self.refs.get(room, 0) == 0

    def idle(self, room):
        with self.lock:
            return self.refs.get(room, 0) == 0

    def forget(self, room):
        with self.lock:
            if self.refs.get(room, 0):
                raise RuntimeError("Retiring a live PP transfer")
            self.closed.discard(room)


class PPBootstrapRoomReuseError(ValueError):
    """A live/retired room cannot distinguish late metadata from a new peer."""


class PPAbortNotifications:
    """Hold decode failures until the immutable, globally drained PP retire.

    Targets survive worker/sender cleanup. Retired tombstones answer late
    bootstrap peers for the transport timeout; pending owners never expire.
    The same lock serializes target registration with the retire commit.
    """

    def __init__(self, timeout_s, clock=time.monotonic):
        self.timeout_s = timeout_s
        self.clock = clock
        self.lock = threading.RLock()
        self.rooms = {}
        self._next_cleanup = 0.0

    def _room(self, room):
        now = self.clock()
        if now >= self._next_cleanup:
            self.rooms = {
                key: state
                for key, state in self.rooms.items()
                if state["owned"] or state["expires"] > now
            }
            self._next_cleanup = now + min(1.0, self.timeout_s)
        return self.rooms.setdefault(
            room,
            dict(
                owned=False,
                committed=False,
                targets=set(),
                reason=None,
                expires=now + self.timeout_s,
            ),
        )

    def open(self, room):
        with self.lock:
            state = self._room(room)
            if state["owned"] or state["committed"]:
                # The wire carries only room, not an incarnation. Preserve the
                # old owner/tombstone so its late metadata cannot admit a peer
                # into a new transfer. Successful rooms are forgotten normally.
                raise PPBootstrapRoomReuseError(
                    f"Bootstrap room {room} is active or recently aborted; "
                    "retry with a new bootstrap_room"
                )
            state["owned"] = True

    def register(self, room, target):
        # Caller holds lock through insertion into transfer_infos. A late
        # registration can never recreate transport state after retire/clear.
        with self.lock:
            state = self._room(room)
            if target is not None:
                state["targets"].add(target)
            return state["committed"]

    def failure(self, room, targets, reason):
        with self.lock:
            state = self._room(room)
            state["targets"].update(targets)
            state["reason"] = state["reason"] or reason or "PP prefill aborted"
            if not state["committed"]:
                return [], state["reason"]
            # Failed is idempotent at Decode. Do not acknowledge delivery here:
            # socket I/O may still fail, and a late bootstrap must be retryable.
            return sorted(set(targets)), state["reason"]

    def commit(self, room):
        with self.lock:
            state = self._room(room)
            state["committed"] = True
            state["owned"] = False
            state["expires"] = self.clock() + self.timeout_s
            return sorted(state["targets"])

    def forget_success(self, room):
        with self.lock:
            self.rooms.pop(room, None)


class PPPrefillControl:
    """Abort fence / drain / retire over an asynchronous proposal store.

    Delayed proposals and delayed decisions retain all ownership. Readiness is
    monotone: a fenced request can never acquire a new GPU or transport lease.
    The caller applies retire commits at the same pre-admission boundary.
    """

    def __init__(self, log, store, timeout_s, clock=time.monotonic, max_pending=1024):
        self.log = log
        self.store = store
        self.clock = clock
        self.timeout_s = timeout_s
        self.max_pending = max_pending
        self.requests = {}
        self.proposals = {}
        self.fenced = {}
        self.gpu_refs = Counter()
        self.batches = {}
        self._deadline_lock = threading.Lock()
        self._deadlines = {}

    def _key(self, kind, *args):
        return ("pp-prefill-control", self.log.epoch, kind, *args)

    def register(self, req):
        key = request_key(req)
        previous = self.requests.setdefault(key, req)
        if previous is not req:
            raise RuntimeError("Duplicate live PP request incarnation")
        return key

    def propose(self, req, reason):
        key = self.register(req)
        if key not in self.proposals and key not in self.fenced:
            if len(self.proposals) >= self.max_pending:
                raise RuntimeError(
                    "PP abort proposal capacity exceeded; ownership retained"
                )
            self.proposals[key] = reason
            with self._deadline_lock:
                self._deadlines[key] = self.clock() + self.timeout_s
            self.store[self._key("proposals")] = tuple(self.proposals.items())

    def is_fenced(self, req):
        return request_key(req) in self.fenced

    def bind_batch(self, batch):
        if batch is None:
            return
        keys = tuple(self.register(req) for req in batch.reqs)
        if any(key in self.fenced for key in keys):
            raise RuntimeError("New forward references a fenced PP request")
        if id(batch) in self.batches:
            raise RuntimeError("PP batch already bound")
        self.batches[id(batch)] = keys
        self.gpu_refs.update(keys)

    def complete_batch(self, batch):
        # Called AFTER result copy/synchronization and result processing.
        for key in self.batches.pop(id(batch)):
            self.gpu_refs[key] -= 1
            if self.gpu_refs[key] == 0:
                del self.gpu_refs[key]

    def boundary(self, fence, retire, locally_ready, peer_failures=lambda: ()):
        # locally_ready includes TP/CP agreement and the transport drain proof.
        for key, seq in self.fenced.items():
            if not self.gpu_refs.get(key) and locally_ready(self.requests[key]):
                ready_key = self._key("ready", key, seq)
                if not self.store.get(ready_key, False):
                    self.store[ready_key] = True

        def produce():
            proposals = {}
            for key in peer_failures():
                if key in self.requests and key not in self.fenced:
                    proposals[key] = "PP transport failed"
            for snapshot in self.store.collect(self._key("proposals")):
                for key, reason in snapshot or ():
                    if key in self.requests and key not in self.fenced:
                        proposals.setdefault(key, reason)
            retiring = tuple(
                key
                for key, seq in self.fenced.items()
                if all(self.store.collect(self._key("ready", key, seq)))
            )
            return tuple(sorted(proposals.items())), retiring

        aborts, retiring = self.log.decide("boundary", (), produce)
        if len(self.fenced) + len(aborts) > self.max_pending:
            raise RuntimeError("PP abort fence capacity exceeded; ownership retained")
        for key, reason in aborts:
            if key not in self.requests:
                raise RuntimeError("PP abort references an unknown incarnation")
            if key in self.fenced:
                raise RuntimeError("Duplicate PP abort fence")
            self.fenced[key] = self.log.seq
            logger.info("PP abort fence seq=%s key=%s", self.log.seq, key)
            with self._deadline_lock:
                self._deadlines.setdefault(key, self.clock() + self.timeout_s)
            fence(self.requests[key], reason)
        for key in retiring:
            if key not in self.fenced or self.gpu_refs.get(key):
                raise RuntimeError("PP retire before local GPU drain")
            req = self.requests[key]
            # Ready was monotone; recheck to catch an adapter violating its lease.
            if not locally_ready(req):
                raise RuntimeError("PP retire before local transport drain")
            retire(req)
            logger.info("PP abort retire seq=%s key=%s", self.log.seq, key)
            seq = self.fenced.pop(key)
            self.store.pop(self._key("ready", key, seq), None)
            self.proposals.pop(key, None)
            self.requests.pop(key)
            with self._deadline_lock:
                self._deadlines.pop(key, None)
        if retiring:
            self.store[self._key("proposals")] = tuple(self.proposals.items())

    def forget_success(self, req):
        key = request_key(req)
        # An immutable successful release may race an as-yet uncommitted abort.
        # Drop saved notification targets even if a late local abort latched
        # sender.conclude_state=Failed after the immutable Success decision.
        manager = getattr(getattr(req, "disagg_kv_sender", None), "kv_mgr", None)
        gate = getattr(manager, "_pp_abort_notifications", None)
        if gate is not None:
            gate.forget_success(req.bootstrap_room)
        self.requests.pop(key, None)
        self.proposals.pop(key, None)
        with self._deadline_lock:
            self._deadlines.pop(key, None)
        self.store[self._key("proposals")] = tuple(self.proposals.items())
        self.store.pop(self._key("metadata", key), None)

    def check_deadlines(self):
        with self._deadline_lock:
            expired = len(self._deadlines) > self.max_pending or any(
                self.clock() >= end for end in self._deadlines.values()
            )
        if expired:
            raise TimeoutError(
                "PP abort decision/drain deadline exceeded; ownership retained"
            )

    def start_watchdog(self, failure_probe=lambda req: False):
        def watch():
            while True:
                time.sleep(min(1.0, self.timeout_s / 4))
                # Also arm a deadline when this scheduler is blocked awaiting
                # PP0's next request frame. This thread only retains ownership;
                # it never changes scheduling state or submits a collective.
                failed = [
                    key
                    for key, req in tuple(self.requests.items())
                    if failure_probe(req)
                ]
                with self._deadline_lock:
                    for key in failed:
                        if key in self.requests:
                            self._deadlines.setdefault(
                                key, self.clock() + self.timeout_s
                            )
                try:
                    self.check_deadlines()
                except TimeoutError:
                    # Match scheduler fatal handling. Never synchronize a possibly
                    # wedged GPU, or depend on PP0 to authorize a local free.
                    logger.critical(
                        "PP control deadline expired; terminating the scheduler group without freeing KV"
                    )
                    os.kill(os.getppid(), signal.SIGQUIT)
                    os._exit(1)

        threading.Thread(target=watch, daemon=True, name="pp-control-watchdog").start()


def init_pp_prefill_control(scheduler):
    from sglang.srt.disaggregation.base import KVPoll
    from sglang.srt.disaggregation.pp_consensus_store import PPConsensusStore
    from sglang.srt.distributed.communication_op import attn_cp_tp_broadcast_pyobj
    from sglang.srt.runtime_context import get_parallel

    manager = getattr(scheduler.disagg_prefill_bootstrap_queue, "kv_manager", None)
    if not hasattr(manager, "_pp_transfer_leases"):
        # V1 is scoped to Mooncake, whose source reads are instrumented below.
        return None
    parallel = get_parallel()
    store = manager.request_status
    if not isinstance(store, PPConsensusStore):
        raise RuntimeError("PP-prefill control requires the replicated status store")
    epoch = scheduler.pp_group.broadcast_object(uuid.uuid4().hex, src=0)

    # The existing helper accepts lists (including its empty-message fast path).
    # Keep a one-element envelope even for None/False/empty-tuple decisions.
    def broadcast(value):
        return attn_cp_tp_broadcast_pyobj([value])[0]

    epoch = broadcast(epoch)
    control = PPPrefillControl(
        PPDecisionLog(parallel.pp_rank == 0, epoch, broadcast),
        store,
        min(scheduler.server_args.watchdog_timeout, 300),
    )
    scheduler.pp_prefill_control = control
    scheduler.policy.pp_decision_log = control.log
    control.start_watchdog(
        lambda req: getattr(req.disagg_kv_sender, "kv_mgr", None) is not None
        and req.disagg_kv_sender.kv_mgr.request_status.get(req.bootstrap_room)
        == KVPoll.Failed
    )
    return control


def batch_signature(batch):
    if batch is None:
        return None
    return (
        int(batch.forward_mode),
        tuple((request_key(req), req.prefix_len, req.extend_end) for req in batch.reqs),
    )


def poll_prefill_controlled(scheduler, reqs, phase):
    """Observe locally, decide once on PP0, replay without a local veto."""
    from sglang.srt.disaggregation.base import KVPoll
    from sglang.srt.disaggregation.utils import (
        _all_reduce_polls,
        poll_and_all_reduce_attn_cp_tp_group,
    )

    control = scheduler.pp_prefill_control
    local = poll_and_all_reduce_attn_cp_tp_group(
        [req.disagg_kv_sender for req in reqs],
        scheduler.attn_cp_cpu_group,
        scheduler.attn_tp_cpu_group,
    )
    for req, poll in zip(reqs, local):
        control.register(req)
        if poll == KVPoll.Failed:
            control.propose(req, f"Local transport failure at {phase}")

    def produce():
        polls = [req.disagg_kv_sender.poll_pp_consensus() for req in reqs]
        polls = _all_reduce_polls(polls, scheduler.attn_tp_cpu_group)
        polls = _all_reduce_polls(polls, scheduler.attn_cp_cpu_group)
        for i, (req, poll) in enumerate(zip(reqs, polls)):
            if poll == KVPoll.Failed:
                control.propose(req, f"PP transport failure at {phase}")
                # No scheduler-side failure before the next abort fence.
                polls[i] = KVPoll.Bootstrapping
            elif phase == "bootstrap" and poll == KVPoll.WaitingForInput:
                if not all(
                    control.store.collect(control._key("metadata", request_key(req)))
                ):
                    polls[i] = KVPoll.Bootstrapping
        return tuple(polls)

    return control.log.decide(phase, map(request_key, reqs), produce)


def reserve_bootstrap_metadata(queue):
    from sglang.srt.disaggregation.utils import _all_reduce_polls

    scheduler = queue.scheduler
    control = scheduler.pp_prefill_control
    credits = [int(queue.ensure_metadata_buffer(req)) for req in queue.queue]
    credits = _all_reduce_polls(credits, scheduler.attn_tp_cpu_group)
    credits = _all_reduce_polls(credits, scheduler.attn_cp_cpu_group)
    for req, ready in zip(queue.queue, credits):
        key = control._key("metadata", control.register(req))
        if control.store.get(key) != bool(ready):
            control.store[key] = bool(ready)


def apply_pp_boundary(scheduler):
    from sglang.srt.disaggregation.base import KVPoll
    from sglang.srt.disaggregation.fake.conn import FakeKVSender
    from sglang.srt.disaggregation.utils import _all_reduce_polls

    control = scheduler.pp_prefill_control

    def fence(req, reason):
        req._pp_abort_reason = reason
        # Close the source-read lease BEFORE marking the transport Failed.
        sender = req.disagg_kv_sender
        getattr(sender, "pp_abort_fence", sender.abort)()
        scheduler.waiting_queue = [r for r in scheduler.waiting_queue if r is not req]
        queue = scheduler.disagg_prefill_bootstrap_queue
        queue.queue = [r for r in queue.queue if r is not req]
        scheduler.disagg_prefill_inflight_queue = [
            r for r in scheduler.disagg_prefill_inflight_queue if r is not req
        ]
        if scheduler.chunked_req is req:
            scheduler.chunked_req = None

    def locally_ready(req):
        # An uninstrumented backend cannot supply a drain proof. Retain its
        # ownership and let the deadline fail the group rather than guess.
        sender = req.disagg_kv_sender
        ready = int(
            isinstance(sender, FakeKVSender)
            or getattr(sender, "pp_abort_drained", lambda: False)()
        )
        ready = _all_reduce_polls([ready], scheduler.attn_tp_cpu_group)
        ready = _all_reduce_polls(ready, scheduler.attn_cp_cpu_group)
        return bool(ready[0])

    def retire(req):
        # All PP GPU/transport reads are drained in the frozen retire decision.
        # Notify before failure_exception/clear discards the decode endpoints.
        getattr(req.disagg_kv_sender, "pp_abort_notify", lambda: None)()
        # This is the only bypass of proposal-only bootstrap failure handling.
        scheduler.handle_bootstrap_failure(req, pp_retire_commit=True)
        getattr(req.disagg_kv_sender, "pp_abort_forget", lambda: None)()
        control.store.pop(control._key("metadata", request_key(req)), None)

    def peer_failures():
        # Normal status replication already reaches PP0 in the background.
        # Include failures of active chunks whose bootstrap is already done.
        return (
            key
            for key, req in control.requests.items()
            if KVPoll.Failed in control.store.collect(req.bootstrap_room)
        )

    control.boundary(fence, retire, locally_ready, peer_failures)
