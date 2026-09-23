"""Slot-ownership property test for EICPagedHiRadixCache.

Random admit / finish / backup / evict / load / settle sequences drive the real
cache over a mocked device allocator. Every device slot is free ("f"), loading
("l", load-back DMA not yet acked) or cached with a lock count (int). Each step
declares the per-slot transitions it performs; the check applies them to the
slots before the step and requires the cache to land on exactly the result, with
its avail / evictable / protected counters equal to the slot counts.
"""

import random
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.eic_cache_controller import get_content_hash
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.eic_hiradix_cache import EICPagedHiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-test-cpu")

TOTAL, PAGE = 64, 4
# Two families sharing an 8-token head, so requests hit, split and diverge.
BASES = [list(range(32)), list(range(8)) + list(range(100, 124))]

# Legal per-slot moves; None is a move the cache must never make. The comments
# name the PR whose bug the refusal rules out.
TRANSITIONS = {
    # adopt a slot: never one still loading (#768)
    "lock": lambda s: s + 1 if type(s) is int else None,
    # release a hold: never on a slot nobody holds (#748)
    "unlock": lambda s: s - 1 if type(s) is int and s > 0 else None,
    "startLoad": lambda s: "l" if s == "f" else None,
    "ackOk": lambda s: 0 if s == "l" else None,
    # a failed load frees only its own loading slots (#768)
    "ackFail": lambda s: "f" if s == "l" else None,
    "insert": lambda s: 0 if s == "f" else None,
    # eviction never takes a held or loading slot
    "evict": lambda s: "f" if s == 0 else None,
}


class Violation(AssertionError):
    pass


def check_step(label, prims, pre, post, counters):
    if "x" in pre or "x" in post:
        side = "before" if "x" in pre else "after"
        slots = [i for i, s in enumerate(pre if side == "before" else post) if s == "x"]
        raise Violation(f"{label}: slots {slots} lost or double-owned {side} the step")
    cur = list(pre)
    for name, slots in prims:
        for i in slots:
            nxt = TRANSITIONS[name](cur[i])
            if nxt is None:
                raise Violation(
                    f"{label}: illegal {name} on slot {i} in state {cur[i]}"
                )
            cur[i] = nxt
    if cur != post:
        diff = {i: (a, b) for i, (a, b) in enumerate(zip(cur, post)) if a != b}
        raise Violation(f"{label}: slot (expected, actual) {diff}")
    expect = (
        sum(s == "f" for s in post),
        sum(s == 0 for s in post),
        sum(s == "l" or (type(s) is int and s > 0) for s in post),
    )
    if counters != expect:
        raise Violation(
            f"{label}: avail/evictable/protected {counters}, slots {expect}"
        )


class Driver:
    def __init__(self, seed, adopt_inflight=False):
        self.rng = random.Random(seed)
        self.adopt_inflight = adopt_inflight  # re-opens the #768 window
        self.free_ids = set(range(TOTAL))
        self.reqs = []
        self.steps = 0
        self.c = self._make_cache()

    def _alloc(self, n):
        if len(self.free_ids) < n:
            return None
        ids = sorted(self.free_ids)[:n]
        self.free_ids.difference_update(ids)
        return torch.tensor(ids, dtype=torch.int64)

    def _free(self, t):
        self.free_ids.update(t.tolist())

    def _make_cache(self):
        c = object.__new__(EICPagedHiRadixCache)
        c.disable, c.is_eagle, c.page_size, c.device = False, False, PAGE, "cpu"
        c.sliding_window_size, c.pp_size = None, 1
        c.evictable_size_ = c.protected_size_ = 0
        c.evictable_leaves = set()
        c.write_through_threshold = 10**9
        c.load_back_threshold, c.load_back_check = 0, False
        c.ongoing_load_back = {}
        c.disable_finished_insert = False
        c.calculate_hash_fn = get_content_hash
        c._backup_unbacked_path = lambda node: None  # backups are their own step
        c.cache_controller = SimpleNamespace(
            write_policy="write_through",
            mem_pool_device_allocator=SimpleNamespace(free=self._free),
            evict_device=lambda dev, host: (self._free(dev), len(dev))[1],
            load_page=lambda host_indices, node_id, content_hash: self._alloc(
                len(host_indices)
            ),
        )
        c.token_to_kv_pool_allocator = SimpleNamespace(free=self._free)
        self.r2t = torch.zeros((1, 64), dtype=torch.int64)
        c.req_to_token_pool = SimpleNamespace(req_to_token=self.r2t)
        root = TreeNode()
        root.key, root.value, root.lock_ref = RadixKey([], None), [], 1
        c.root_node = root
        return c

    # ---- projection -------------------------------------------------------

    def _nodes(self):
        stack = list(self.c.root_node.children.values())
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children.values())

    def _loading(self):
        ids = set()
        for start, end, _ in self.c.ongoing_load_back.values():
            while end is not start:
                ids.add(end.id)
                end = end.parent
        return ids

    def snapshot(self):
        owner = [[] for _ in range(TOTAL)]
        for i in self.free_ids:
            owner[i].append("f")
        loading = self._loading()
        for n in self._nodes():
            if n.value is not None:
                s = "l" if n.id in loading else n.lock_ref
                for i in n.value.tolist():
                    owner[i].append(s)
        return [o[0] if len(o) == 1 else "x" for o in owner]

    def counters(self):
        c = self.c
        return (len(self.free_ids), c.evictable_size_, c.protected_size_)

    def _path_slots(self, node):
        out = []
        while node is not self.c.root_node:
            if node.value is not None:
                out += node.value.tolist()
            node = node.parent
        return out

    def _chain_slots(self, start, end):
        # Device order of a load: top-down along the chain.
        nodes = []
        while end is not start:
            nodes.append(end)
            end = end.parent
        return [i for n in reversed(nodes) for i in n.value.tolist()]

    def record(self, label, prims, pre):
        self.steps += 1
        check_step(
            f"step {self.steps} [{label}]", prims, pre, self.snapshot(), self.counters()
        )

    # ---- operations -------------------------------------------------------

    def _key(self):
        base = self.rng.choice(BASES)
        return RadixKey(base[: self.rng.choice(range(8, 33, PAGE))], None)

    def admit(self):
        key = self._key()
        m = self.c.match_prefix(MatchPrefixParams(key=key))
        node = m.last_device_node
        if not self.adopt_inflight and self.c.prefix_loading(node):
            return  # scheduler defers until the load settles
        pre, slots = self.snapshot(), self._path_slots(node)
        self.c.inc_lock_ref(node)
        self.reqs.append((key, m.device_indices, node))
        self.record("admit", [("lock", slots)], pre)

    def finish(self):
        if not self.reqs:
            return
        key, prefix, node = self.reqs.pop(self.rng.randrange(len(self.reqs)))
        pre, locked = self.snapshot(), self._path_slots(node)
        new = self._alloc(len(key) - len(prefix))
        if new is None:
            self.reqs.append((key, prefix, node))
            return
        kv = torch.cat([prefix, new])
        self.r2t[0, : len(kv)] = kv
        req = SimpleNamespace(
            origin_input_ids=list(key.token_ids),
            output_ids=[],
            extra_key=None,
            priority=0,
            prefix_indices=prefix,
            last_node=node,
            kv=SimpleNamespace(req_pool_idx=0, cache_protected_len=len(prefix)),
        )
        self.c.cache_finished_req(req, kv_len_to_handle=len(kv))
        cached = {
            i for n in self._nodes() if n.value is not None for i in n.value.tolist()
        }
        kept = [i for i in new.tolist() if i in cached]
        self.record("finish", [("unlock", locked), ("insert", kept)], pre)

    def backup(self):
        cands = [
            n
            for n in self._nodes()
            if n.value is not None
            and n.host_value is None
            and (n.parent is self.c.root_node or n.parent.backuped)
        ]
        if not cands:
            return
        n = self.rng.choice(cands)
        pre = self.snapshot()
        self.c._calculate_content_hash(n)
        n.host_value = torch.arange(len(n.key))
        self.record("backup", [], pre)

    def evict(self):
        cands = [n for n in self.c.evictable_leaves if n.lock_ref == 0]
        if not cands:
            return
        n = self.rng.choice(sorted(cands, key=lambda n: n.id))
        pre, slots = self.snapshot(), n.value.tolist()
        if n.backuped:
            self.c._evict_backuped(n)
        else:
            self.c._evict_regular(n)
        self.record("evict", [("evict", slots)], pre)

    def load(self):
        m = self.c.match_prefix(MatchPrefixParams(key=self._key()))
        if m.host_hit_length == 0:
            return
        if not self.adopt_inflight and self.c.prefix_loading(m.last_device_node):
            return  # same scheduler defer as admit
        pre = self.snapshot()
        if self.c.load_back(m.best_match_node, allow_evict=False) is None:
            self.record("load-refused", [], pre)
            return
        start, end, _ = self.c.ongoing_load_back[m.best_match_node.id]
        self.record(
            "load",
            [
                ("startLoad", self._chain_slots(start, end)),
                ("lock", self._path_slots(start)),
            ],
            pre,
        )

    def settle(self):
        if not self.c.ongoing_load_back:
            return
        node_id = self.rng.choice(sorted(self.c.ongoing_load_back))
        start, end, total = self.c.ongoing_load_back[node_id]
        complete = self.rng.choice([total, 0, self.rng.randrange(0, total + 1, PAGE)])
        pre, chain = self.snapshot(), self._chain_slots(start, end)
        held = self._path_slots(start)
        self.c._free_failed_loadback(node_id, complete)
        self.record(
            "settle",
            [
                ("ackOk", chain[:complete]),
                ("ackFail", chain[complete:]),
                ("unlock", held),
            ],
            pre,
        )

    def run(self, steps):
        ops = [self.admit, self.finish, self.backup, self.evict, self.load, self.settle]
        for _ in range(steps):
            self.rng.choice(ops)()


class TestEICSlotOwnership(unittest.TestCase):
    def test_random_sequences_keep_slot_ownership(self):
        for seed in range(200):
            with self.subTest(seed=seed):
                Driver(seed).run(300)

    def test_adopting_inflight_load_is_caught(self):
        # Without the prefix_loading defer (#768) some seed adopts a loading slot.
        caught = 0
        for seed in range(200):
            try:
                Driver(seed, adopt_inflight=True).run(300)
            except Violation as e:
                self.assertIn("illegal lock", str(e))
                caught += 1
        self.assertTrue(caught)

    def test_unlocking_unheld_node_is_caught(self):
        # #748: dec_lock_ref on a node nobody holds.
        d = Driver(0)
        d.c.insert(InsertParams(key=RadixKey(list(range(8)), None), value=d._alloc(8)))
        node = next(iter(d.c.root_node.children.values()))
        pre = d.snapshot()
        d.c.dec_lock_ref(node)
        with self.assertRaises(Violation):
            d.record("bad-unlock", [("unlock", node.value.tolist())], pre)


if __name__ == "__main__":
    unittest.main()
