"""Differential test: EICPagedHiRadixCache against the Lean spec in lean/eic.

Random admit / finish / backup / evict / load / settle sequences drive the real
cache over a mocked device allocator. Each step records the per-slot ownership
before and after, the spec transitions the step claims to perform, and the
cache's avail/evictable/protected counters; `eic_replay` (lean/eic/Main.lean)
checks every step against EicSpec.lean. Needs `lake` on PATH or EIC_REPLAY
pointing at a built `eic_replay`; skipped otherwise.
"""

import os
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.managers.eic_cache_controller import get_content_hash
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.eic_hiradix_cache import EICPagedHiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-test-cpu")

LEAN_DIR = Path(__file__).resolve().parents[4] / "lean" / "eic"
TOTAL, PAGE = 64, 4
# Two families sharing an 8-token head, so requests hit, split and diverge.
BASES = [list(range(32)), list(range(8)) + list(range(100, 124))]


def replay_binary():
    if os.environ.get("EIC_REPLAY"):
        return os.environ["EIC_REPLAY"]
    if shutil.which("lake") is None:
        return None
    subprocess.run(["lake", "build"], cwd=LEAN_DIR, check=True, capture_output=True)
    return str(LEAN_DIR / ".lake" / "build" / "bin" / "eic_replay")


class Driver:
    def __init__(self, seed, adopt_inflight=False):
        self.rng = random.Random(seed)
        self.adopt_inflight = adopt_inflight  # re-opens the #768 window
        self.free_ids = set(range(TOTAL))
        self.reqs = []
        self.lines = []
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
                s = "l" if n.id in loading else f"c{n.lock_ref}"
                for i in n.value.tolist():
                    owner[i].append(s)
        # A lost or double-owned slot has no spec state: "x" fails the replay.
        return " ".join(o[0] if len(o) == 1 else "x" for o in owner)

    def counters(self):
        c = self.c
        return f"{len(self.free_ids)} {c.evictable_size_} {c.protected_size_}"

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
        p = ";".join(f"{k}:{','.join(map(str, v))}" for k, v in prims if v) or "-"
        self.lines.append(f"{label}\t{p}\t{pre}\t{self.snapshot()}\t{self.counters()}")

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
        return self.lines


class TestEICLeanSpec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = replay_binary()
        if cls.replay is None:
            raise unittest.SkipTest("lake not on PATH and EIC_REPLAY unset")

    def _replay(self, lines):
        with tempfile.NamedTemporaryFile("w", suffix=".trace", delete=False) as f:
            f.write("\n".join(lines) + "\n")
        try:
            return subprocess.run([self.replay, f.name], capture_output=True, text=True)
        finally:
            os.unlink(f.name)

    def test_random_traces_match_spec(self):
        lines = []
        for seed in range(200):
            lines += Driver(seed).run(300)
        r = self._replay(lines)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_adopting_inflight_load_is_caught(self):
        # Without the prefix_loading defer (#768) some seed adopts a loading slot.
        caught = [
            s
            for s in range(200)
            if self._replay(Driver(s, adopt_inflight=True).run(300)).returncode
        ]
        self.assertTrue(caught)

    def test_unlocking_unheld_node_is_caught(self):
        # #748: dec_lock_ref on a node nobody holds.
        d = Driver(0)
        d.c.insert(InsertParams(key=RadixKey(list(range(8)), None), value=d._alloc(8)))
        node = next(iter(d.c.root_node.children.values()))
        pre = d.snapshot()
        d.c.dec_lock_ref(node)
        d.record("bad-unlock", [("unlock", node.value.tolist())], pre)
        r = self._replay(d.lines)
        self.assertEqual(r.returncode, 1)


if __name__ == "__main__":
    unittest.main()
