"""Exhaustive interleaving check of the EIC PP load-back verdict protocol.

Runs the real EICPagedHiRadixCache protocol code (check_load_back_progress,
loading_check, release_load_admit) on PP stages whose only fakes are the
transport (a shared store, a per-round verdict buffer) and the load itself. A
depth-first search enumerates every interleaving the protocol must tolerate:
stage skew in the pipeline, when each EIC ack lands and how much it loaded, and
per scenario the per-stage host hit, which stages fail to kick their load, when
the request is released, whether its rid re-gates, and whether a second request
competes for a verdict stream capped at one row per round. Each explored run ends
with every ack delivered and extra lockstep rounds, then must satisfy:

- every stage admits each incarnation at the same round with the same length,
  and holds the same number of prefix slots (PP-uniform allocators);
- the code's own asserts hold;
- nothing is left behind: no per-req state, locks, in-flight loads, reports,
  verdicts or store keys; every non-released incarnation got admitted.
"""

import itertools
import os
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.eic_hiradix_cache import EICPagedHiRadixCache as C
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-test-cpu")

THRESHOLD = 10  # load_back_threshold
D = 4  # device match; device trees are PP-uniform, so every stage has the same
HOST_HITS = (0, 12, 16)  # per-stage host hit beyond the device match
COMPLETES = (0, 8, None)  # ack sizes; None = the whole allocation


class Violation(AssertionError):
    pass


class Store(dict):
    def set(self, k, v):
        self[k] = v

    def get(self, k):
        return self[k]

    def check(self, keys):
        return all(k in self for k in keys)

    def delete_key(self, k):
        return self.pop(k, None) is not None


class Stage:
    def __init__(self, world, pp_rank, pp_size):
        self.world, self.rank_pp = world, pp_rank
        c = self.c = object.__new__(C)
        c.pp_size, c.pp_rank, c.pp_group = pp_size, pp_rank, object()
        c.tp_size, c.rank, c.tp_group = 1, 0, None
        c.load_back_threshold, c.load_back_reserve = THRESHOLD, 0
        c.evictable_size_ = 0
        c.root_node = SimpleNamespace(evicted=False)
        for name in ("ongoing_load_admit", "ongoing_load_back", "_admit_verdict"):
            setattr(c, name, {})
        for name in ("_h_rid", "_rid_epoch", "_loadback_rid", "_report_outbox"):
            setattr(c, name, {})
        for name in ("_span_reports", "_load_reports", "_await_load", "_tombstone"):
            setattr(c, name, {})
        c._next_seq, c._verdict_outbox = {}, []
        c._pub_seq = c._round = 0
        c._VERDICT_CAP = 1  # one verdict per round, so overflow waits in the outbox
        c._store_handle = world.store
        self.locks = []
        c.inc_lock_ref = lambda node: self.locks.append(node)
        c.dec_lock_ref = lambda node: self.locks.remove(node)
        c._free_failed_loadback = self._free
        c._clip_host_chain = lambda node, excess, quota: node
        c.load_back = self._load_back
        c._pp_bcast_from_first = self._bcast
        c.cache_controller = SimpleNamespace(
            ack_load_queue=Queue(),
            mem_pool_device_allocator=SimpleNamespace(available_size=lambda: 10**9),
        )
        self.iteration = 0
        self.waiting = []  # (incarnation, req)
        self.admits = {}  # incarnation -> (iteration, prefix length)
        self.held = {}  # incarnation -> device slots its load still holds
        self.inflight = {}  # node_id -> (incarnation, alloc)

    def _bcast(self, buf, tag=None):
        # PP0 sends round k's verdicts; stage s receives exactly round k's.
        rounds = self.world.verdict_rounds
        if self.rank_pp == 0:
            rounds[self.c._round] = buf.clone()
        else:
            buf.copy_(rounds[self.c._round])

    def _load_back(self, node, allow_evict=None):
        inc = node.incarnation
        if not self.world.sc["load_ok"][self.rank_pp]:
            return None
        quota = self.world.span - self.world.sc["d"][self.rank_pp]
        self.c.ongoing_load_back[node.id] = (node, node, quota)
        self.inflight[node.id] = (inc, quota)
        self.held[inc] = quota
        return torch.arange(quota, dtype=torch.int64)

    def _free(self, node_id, complete):
        self.c.ongoing_load_back.pop(node_id)
        self.inflight.pop(node_id, None)
        self.held[node_id // 10] = complete  # frees [complete, alloc)

    def footprint(self):
        # Device slots each incarnation ends up with: an admitted one owns its
        # device match plus what its load kept; a released one must keep none.
        d = self.world.sc["d"][self.rank_pp]
        incs = set(self.admits) | set(self.held)
        out = {i: (d if i in self.admits else 0) + self.held.get(i, 0) for i in incs}
        return {i: n for i, n in out.items() if n}

    def make_req(self, inc):
        d, hh = self.world.sc["d"][self.rank_pp], self.world.sc["hh"][self.rank_pp]
        node = mock.Mock(evicted=False, value=[1], key=list(range(max(hh, 1))))
        node.parent, node.id, node.incarnation = (
            self.c.root_node,
            inc * 10 + self.rank_pp,
            inc,
        )
        rid = self.world.arrivals[inc][0]
        req = mock.Mock(rid=rid, host_hit_length=hh, best_match_node=node)
        req.prefix_indices = torch.arange(d, dtype=torch.int64)
        req.needs_host_load_back = lambda: hh > 0
        req.last_node = node
        return req

    def run_iteration(self):
        # One get_new_batch_prefill: the heartbeat, then the gate per waiting req.
        self.iteration += 1
        k, sc = self.iteration, self.world.sc
        for inc, (_, at) in enumerate(self.world.arrivals):
            if at == k:
                self.waiting.append((inc, self.make_req(inc)))
        C.loading_check(self.c)
        for inc, req in list(self.waiting):
            if sc["release"] == k and inc == 0:
                C.release_load_admit(self.c, "r")
                self.waiting.remove((inc, req))
            elif C.check_load_back_progress(self.c, req):
                self.admits[inc] = (k, len(req.prefix_indices))
                self.waiting.remove((inc, req))


class World:
    def __init__(self, sc, pp_size, max_iter, skew):
        self.sc, self.max_iter, self.skew = sc, max_iter, skew
        self.store, self.verdict_rounds = Store(), {}
        # (rid, arrival round) per incarnation: "r" may be released and re-gate
        # the round after; "q" competes with it for the verdict stream.
        self.arrivals = [("r", 1)]
        if sc["retry"]:
            self.arrivals.append(("r", sc["release"] + 1))
        if sc["other"]:
            self.arrivals.append(("q", 1))
        # the load quota follows SPAN = min over stages of d + hh
        self.span = min(d + hh for d, hh in zip(sc["d"], sc["hh"]))
        self.stages = [Stage(self, r, pp_size) for r in range(pp_size)]
        self.acked = set()

    def actions(self):
        out = []
        it = [s.iteration for s in self.stages]
        for r, s in enumerate(self.stages):
            if it[r] == self.max_iter:
                continue
            if r == 0 and it[0] - it[-1] < self.skew or r > 0 and it[r] < it[r - 1]:
                out.append(("step", r))
        for r, s in enumerate(self.stages):
            for nid, (inc, alloc) in s.inflight.items():
                if (r, nid) not in self.acked:
                    out += [("ack", r, nid, c) for c in COMPLETES]
        return out

    def do(self, a):
        if a[0] == "step":
            self.stages[a[1]].run_iteration()
        else:
            _, r, nid, c = a
            s = self.stages[r]
            s.c.cache_controller.ack_load_queue.put(
                (nid, s.inflight[nid][1] if c is None else c)
            )
            self.acked.add((r, nid))

    def fingerprint(self):
        parts = [tuple(sorted(self.store)), tuple(sorted(self.acked))]
        for s in self.stages:
            c = s.c
            parts.append(
                (
                    s.iteration,
                    tuple(i for i, _ in s.waiting),
                    tuple(sorted(s.admits.items())),
                    tuple(sorted(s.held.items())),
                    tuple(sorted(s.inflight)),
                    c.cache_controller.ack_load_queue.qsize(),
                    repr(
                        sorted(
                            (k, v["epoch"], v["alloc"], v["complete"])
                            for k, v in c.ongoing_load_admit.items()
                        )
                    ),
                    repr(sorted(c._admit_verdict.items())),
                    repr(sorted(c._report_outbox.items())),
                    repr(sorted((k, v[0]) for k, v in c._span_reports.items())),
                    repr(sorted((k, v[0]) for k, v in c._load_reports.items())),
                    repr(sorted((k, v[:3]) for k, v in c._await_load.items())),
                    repr(c._verdict_outbox),
                    repr(sorted((h, t[0]) for h, t in c._tombstone.items())),
                )
            )
        return hash(tuple(parts))

    def drain_and_check(self):
        # Fair ending: every load, including ones kicked while draining, acks.
        for _ in range(8):
            for r, s in enumerate(self.stages):
                for nid in list(s.inflight):
                    if (r, nid) not in self.acked:
                        self.do(("ack", r, nid, None))
            for s in self.stages:
                s.run_iteration()
        s0 = self.stages[0]
        for s in self.stages[1:]:
            if s.admits != s0.admits:
                raise Violation(
                    f"admits differ: stage0 {s0.admits}, stage{s.rank_pp} {s.admits}"
                )
            if s.footprint() != s0.footprint():
                raise Violation(
                    f"prefix slots differ: stage0 {s0.footprint()}, "
                    f"stage{s.rank_pp} {s.footprint()}"
                )
        released = {0} if self.sc["release"] else set()
        for s in self.stages:
            c = s.c
            left = {
                "waiting": [i for i, _ in s.waiting],
                "ongoing_load_admit": list(c.ongoing_load_admit),
                "ongoing_load_back": list(c.ongoing_load_back),
                "locks": len(s.locks),
                "admit_verdict": c._admit_verdict,
                "h_rid": c._h_rid,
                "loadback_rid": c._loadback_rid,
                "report_outbox": c._report_outbox,
                "span/load/await": (c._span_reports, c._load_reports, c._await_load),
                "verdict_outbox": c._verdict_outbox,
            }
            left = {k: v for k, v in left.items() if v and v != ({}, {}, {})}
            if left:
                raise Violation(f"stage{s.rank_pp} leftovers {left}")
            missing = set(range(len(self.arrivals))) - released - set(s.admits)
            if missing:
                raise Violation(f"stage{s.rank_pp} never admitted {missing}")
        if self.store:
            raise Violation(f"store keys left {sorted(self.store)}")


def scenarios(pp_size, max_iter, others=(False, True)):
    d = (D,) * pp_size
    for hh, load_ok in itertools.product(
        itertools.product(HOST_HITS, repeat=pp_size),
        itertools.product((True, False), repeat=pp_size),
    ):
        for release in (None, *range(1, max_iter)):
            for retry in (False, True) if release else (False,):
                for other in others:
                    yield dict(
                        d=d,
                        hh=hh,
                        load_ok=load_ok,
                        release=release,
                        retry=retry,
                        other=other,
                    )


def explore(sc, pp_size=2, max_iter=7, skew=2):
    """DFS over interleavings with state-hash pruning; returns (states, None) or
    (states, (schedule, error))."""
    seen, stack, states = set(), [[]], 0
    while stack:
        sched = stack.pop()
        w = World(sc, pp_size, max_iter, skew)
        try:
            for a in sched:
                w.do(a)
            fp = w.fingerprint()
            if fp in seen:
                continue
            seen.add(fp)
            states += 1
            acts = w.actions()
            if not acts:
                w.drain_and_check()
            stack += [sched + [a] for a in acts]
        except Exception as e:  # the code's asserts, a Violation, or a crash
            return states, (sched, e)
    return states, None


class TestEICPPProtocol(unittest.TestCase):
    def test_all_interleavings_two_stages(self):
        total = 0
        # ponytail: the competing-request scenarios take ~3 min (114k states);
        # CI runs the single-request space (~15s), EIC_PP_MODEL_FULL=1 runs all.
        full = os.environ.get("EIC_PP_MODEL_FULL") == "1"
        for sc in scenarios(2, 7, others=(False, True) if full else (False,)):
            states, bad = explore(sc)
            total += states
            if bad:
                sched, err = bad
                self.fail(f"{sc}\nschedule {sched}\n{type(err).__name__}: {err}")
        print(f"explored {total} states")


if __name__ == "__main__":
    unittest.main()
