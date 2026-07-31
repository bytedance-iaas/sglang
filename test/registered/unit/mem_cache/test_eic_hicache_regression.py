import sys
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.eic_cache_controller import (
    EICCacheController,
    EICCacheOperation,
)
from sglang.srt.mem_cache.eic_chunk_cache import EICChunkCache
from sglang.srt.mem_cache.eic_pp_reconcile import EICPPReconciler
from sglang.srt.mem_cache.eic_hiradix_cache import EICPagedHiRadixCache
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class FakeTreeNode:
    # Minimal radix node for the finalize last_node walk: len(key), value (None ==
    # evicted), parent. Depth bookkeeping uses the sum of key lengths on the path.
    def __init__(self, key_len: int, resident: bool, parent):
        self.key = list(range(key_len))
        self.value = [0] if resident else None
        self.parent = parent


class FakeHostPool:
    def alloc(self, size: int):
        return torch.arange(size, dtype=torch.int32)


class FakeDeviceAllocator:
    def alloc(self, size: int):
        return torch.arange(100, 100 + size, dtype=torch.int32)


class TestEICHiCacheRegression(unittest.TestCase):
    def test_match_from_remote_skips_zero_committed_tokens(self):
        cache = object.__new__(EICPagedHiRadixCache)
        cache.match_req_set = {}
        cache.tp_size = 1
        cache.pp_size = 1
        cache.pp_group = None
        cache.cache_controller = mock.Mock()
        cache._match_for_remote_fetch = mock.Mock(
            side_effect=AssertionError("empty remote key should be skipped")
        )

        req = mock.Mock(
            rid="health-rid",
            origin_input_ids=[0],
            output_ids=[],
            extra_key=None,
        )

        cache.match_from_remote([req])

        cache._match_for_remote_fetch.assert_not_called()
        cache.cache_controller.batch_find_longest_prefix_in_eic.assert_not_called()
        self.assertEqual(cache.match_req_set, {})

    def test_match_from_remote_gates_and_indexes_by_queue(self):
        # match_from_remote first MIN-reduces the queue length (num_ready gate),
        # then reduces a vector indexed by queue position -- both PP-invariant
        # lengths. Here we capture the CP/TP reduce (PP uses p2p, tested
        # separately); the shapes are what keep the reduces in lockstep.
        cache = object.__new__(EICPagedHiRadixCache)
        cache.match_req_set = {}
        cache.tp_size = 2
        cache.tp_group = object()
        cache.pp_size = 1
        cache.pp_group = None
        cache.page_size = 1
        cache.load_remote_threshold = 1
        cache.eic_check_max_num = 0
        cache.root_node = object()
        cache._insert_remote_node = mock.Mock()
        node = SimpleNamespace(id=1, content_hash=None)

        # Fetch only the middle req; the others' prefix already covers the key.
        def match(root, key):
            match.i += 1
            return (0, 0, node) if match.i == 2 else (len(key), 0, node)

        match.i = 0
        cache._match_for_remote_fetch = match
        cache.cache_controller = mock.Mock()
        cache.cache_controller.batch_find_longest_prefix_in_eic.return_value = [8]
        reqs = [
            mock.Mock(rid=r, origin_input_ids=list(range(8)), output_ids=[9], extra_key=None)
            for r in ("a", "b", "c")
        ]

        captured = []

        def fake_all_reduce(t, op=None, group=None):
            if group is cache.tp_group:
                captured.append(t.tolist())

        with mock.patch(
            "sglang.srt.mem_cache.eic_hiradix_cache._need_calculate_hash",
            return_value=False,
        ), mock.patch.object(
            torch.distributed, "all_reduce", side_effect=fake_all_reduce
        ):
            EICPagedHiRadixCache.match_from_remote(cache, reqs)

        # First the num_ready gate (scalar == queue length), then the per-slot
        # vector (length == queue, only the fetched middle slot carries the hit).
        self.assertEqual(captured, [3, [0, 8, 0]])

    def test_eic_controller_write_uses_queue_api(self):
        controller = object.__new__(EICCacheController)
        controller.mem_pool_host = FakeHostPool()
        controller.write_queue = Queue()

        device_indices = torch.tensor([7, 8, 9], dtype=torch.int32)
        host_indices = controller.write(device_indices, priority=-2, node_id=11)

        self.assertEqual(host_indices.tolist(), [0, 1, 2])
        operation = controller.write_queue.get_nowait()
        self.assertIsInstance(operation, EICCacheOperation)
        self.assertEqual(operation.host_indices.tolist(), [0, 1, 2])
        self.assertIs(operation.device_indices, device_indices)
        self.assertEqual(operation.node_id, 11)
        self.assertEqual(operation.node_ids, [11])
        self.assertIsNone(operation.content_hash)
        self.assertEqual(operation.priority, -2)

    def test_eic_controller_load_uses_queue_api(self):
        controller = object.__new__(EICCacheController)
        controller.mem_pool_device_allocator = FakeDeviceAllocator()
        controller.load_queue = Queue()

        host_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
        with mock.patch(
            "sglang.srt.managers.eic_cache_controller.torch.cuda.current_stream"
        ) as current_stream:
            stream = mock.Mock()
            current_stream.return_value = stream
            device_indices = controller.load(host_indices, priority=-3, node_id=12)

        stream.synchronize.assert_called_once()
        self.assertEqual(device_indices.tolist(), [100, 101, 102])
        operation = controller.load_queue.get_nowait()
        self.assertIsInstance(operation, EICCacheOperation)
        self.assertIs(operation.host_indices, host_indices)
        self.assertEqual(operation.device_indices.tolist(), [100, 101, 102])
        self.assertEqual(operation.node_id, 12)
        self.assertEqual(operation.node_ids, [12])
        self.assertIsNone(operation.content_hash)
        self.assertEqual(operation.priority, -3)

    def test_unified_tree_node_exposes_storage_hash_helpers(self):
        root = UnifiedTreeNode((ComponentType.FULL, ComponentType.SWA))
        child = UnifiedTreeNode((ComponentType.FULL, ComponentType.SWA))
        child.parent = root
        root.hash_value = ["root-hash"]
        child.hash_value = ["child-hash-0", "child-hash-1"]

        self.assertEqual(root.get_last_hash_value(), "root-hash")
        self.assertEqual(child.get_last_hash_value(), "child-hash-1")
        self.assertEqual(child.get_prefix_hash_values(child.parent), ["root-hash"])
        self.assertFalse(
            hasattr(root, "hicache_storage_pass_prefix_keys"),
            "storage pass-prefix config belongs to the cache, not tree nodes",
        )

    def test_batch_exists_impl_failed_batch_keeps_cardinality(self):
        # A failed top-level mexist must yield exactly one False per input key.
        # The EIC outcome object may still carry per-object status codes; if the
        # loop below falls through to them we would return 2*N results, which
        # later overflows component_keys indexing in _batch_io_v2. See the
        # `continue` guard in EICStorage._batch_exists_impl.
        fake_eic = SimpleNamespace(
            StatusCode=SimpleNamespace(SUCCESS=0, FAILED=1),
            StringVector=list,
            ExistOption=SimpleNamespace,
        )
        with mock.patch.dict(sys.modules, {"eic": fake_eic}):
            from sglang.srt.mem_cache.storage.eic.eic_storage import EICStorage

            storage = object.__new__(EICStorage)
            storage.eic_namespace = "poc"
            storage._get_eic_key = lambda keys: list(keys)

            outcome = SimpleNamespace(status_codes=[fake_eic.StatusCode.SUCCESS] * 3)
            storage.connection = mock.Mock()
            storage.connection.mexist.return_value = (
                fake_eic.StatusCode.FAILED,
                outcome,
            )

            keys = [f"k{i}" for i in range(3)]
            result = storage._batch_exists_impl(keys)

        self.assertEqual(result, [False, False, False])
        self.assertEqual(len(result), len(keys))

    def test_eic_chunk_cache_passes_tp_group_to_controller(self):
        cache = object.__new__(EICChunkCache)
        cache.page_size = 16
        cache.load_cache_event = object()
        cache.token_to_kv_pool_host = object()
        params = SimpleNamespace(
            token_to_kv_pool_allocator=object(), tp_cache_group=object()
        )
        server_args = SimpleNamespace()

        with mock.patch(
            "sglang.srt.mem_cache.eic_chunk_cache.EICCacheController"
        ) as controller_cls:
            EICChunkCache._init_cache_controller(cache, params, server_args)

        controller_cls.assert_called_once_with(
            params.token_to_kv_pool_allocator,
            cache.token_to_kv_pool_host,
            cache.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=cache.load_cache_event,
            write_policy="write_through",
            server_args=server_args,
        )

    def test_free_swa_skips_reserved_page_and_dedups_aliases(self):
        # free_swa must drop reserved-page sentinels + dedup aliases under EIC.
        from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator

        alloc = object.__new__(SWATokenToKVPoolAllocator)
        alloc.page_size = 256
        alloc.swa_attn_allocator = mock.Mock()
        alloc.full_to_swa_index_mapping = torch.tensor(
            [6, 8, 10, 21, 512, 512, 512, 768], dtype=torch.int64
        )
        alloc._expand_to_full_pages = lambda idx: idx

        with mock.patch(
            "sglang.srt.mem_cache.swa_memory_pool.get_global_server_args",
            return_value=SimpleNamespace(enable_eic_cache=True),
        ):
            SWATokenToKVPoolAllocator.free_swa(alloc, torch.arange(8))

        freed = alloc.swa_attn_allocator.free.call_args[0][0]
        self.assertEqual(sorted(freed.tolist()), [512, 768])


    def test_pp_bcast_from_first_is_nonblocking_isend(self):
        # PP consensus must be a non-blocking isend down the pipeline: the stages
        # run out of phase, so a collective or a blocking send/recv (which makes
        # one stage wait for another) deadlocks. First stage isends its value
        # downstream (queued in work_list) and keeps it; a later stage overwrites
        # with the upstream (PP0's) value via recv. No all_reduce anywhere.
        from sglang.srt.mem_cache.eic_hiradix_cache import EICPagedHiRadixCache

        # first stage (rank 0 of 2): isend downstream, never recv
        c0 = object.__new__(EICPagedHiRadixCache)
        c0.pp_size, c0.pp_rank, c0.pp_group, c0.work_list = 2, 0, object(), []
        sent = []

        def fake_isend(t, group_dst=None, group=None, tag=None):
            sent.append((group_dst, t.item()))
            return "work"

        with mock.patch.object(
            torch.distributed, "isend", side_effect=fake_isend
        ), mock.patch.object(
            torch.distributed, "recv", side_effect=AssertionError("source never recvs")
        ), mock.patch.object(
            torch.distributed, "all_reduce", side_effect=AssertionError("no collective")
        ):
            t = torch.tensor(5, dtype=torch.int64)
            EICPagedHiRadixCache._pp_bcast_from_first(c0, t)
        self.assertEqual(sent, [(1, 5)])
        self.assertEqual(c0.work_list, ["work"])  # drained next round
        self.assertEqual(t.item(), 5)  # authoritative source keeps its value

        # last stage (rank 1 of 2): recv overwrites with PP0's value, no isend
        c1 = object.__new__(EICPagedHiRadixCache)
        c1.pp_size, c1.pp_rank, c1.pp_group, c1.work_list = 2, 1, object(), []

        def fake_recv(t, group_src=None, group=None, tag=None):
            t.fill_(3)  # PP0's authoritative value

        with mock.patch.object(
            torch.distributed, "recv", side_effect=fake_recv
        ), mock.patch.object(
            torch.distributed, "isend", side_effect=AssertionError("last rank sends nothing")
        ), mock.patch.object(
            torch.distributed, "all_reduce", side_effect=AssertionError("no collective")
        ):
            t = torch.tensor(8, dtype=torch.int64)
            EICPagedHiRadixCache._pp_bcast_from_first(c1, t)
        self.assertEqual(t.item(), 3)


    def test_drain_acks_single_stage_resolves_locally(self):
        # With one PP stage the local ack IS the verdict: d + complete.
        cache = object.__new__(EICPagedHiRadixCache)
        cache.pp_size, cache.pp_group, cache.tp_size = 1, None, 1
        cache._admit_verdict = {}
        cache._loadback_rid = {55: "r0"}
        cache.ongoing_load_admit = {
            "r0": {"h": 10, "d": 4, "load_node": SimpleNamespace(id=55), "complete": None}
        }
        cache.ongoing_load_back = {}
        q = Queue()
        q.put((55, 8))
        cache.cache_controller = SimpleNamespace(ack_load_queue=q)
        EICPagedHiRadixCache._drain_local_acks(cache)
        self.assertEqual(cache._admit_verdict, {10: 12})
        self.assertEqual(cache.ongoing_load_admit["r0"]["complete"], 8)

    def test_finalize_cross_pp_clamp_keeps_spanning_resident_last_node(self):
        # Device-only warm req, cross-PP MIN 10 lands MID-node (below this stage's
        # device match 12). last_node must stay the resident node SPANNING depth 10
        # (bottom 12 >= 10) so inc_lock_ref protects every slot the req reads.
        cache = object.__new__(EICPagedHiRadixCache)
        root = FakeTreeNode(0, True, None)
        cache.root_node = root
        cache.dec_lock_ref = lambda node: None
        cache._h_rid = {7: "r0"}
        node_a = FakeTreeNode(8, True, root)  # covers (0, 8]
        node_b = FakeTreeNode(4, True, node_a)  # covers (8, 12], spans clamp 10
        req = mock.Mock()
        req.rid = "r0"
        req.prefix_indices = torch.arange(12, dtype=torch.int64)
        req.fill_ids = list(range(20))
        req.last_node = node_b
        cache.ongoing_load_admit = {
            "r0": {
                "h": 7,
                "d": 12,
                "alloc": 0,
                "load_node": None,
                "new_indices": None,
                "deferred_lock": node_b,
                "complete": None,
            }
        }
        cache._admit_verdict = {7: 10}
        ok = EICPagedHiRadixCache._finalize_load_admit(cache, req)
        self.assertTrue(ok)
        self.assertEqual(len(req.prefix_indices), 10)
        self.assertIs(req.last_node, node_b)  # spanning resident node kept
        req.set_extend_input_len.assert_called_once_with(20 - 10)
        self.assertEqual(req.eic_loaded_len, 0)
        self.assertEqual(cache.ongoing_load_admit, {})
        self.assertEqual(cache._admit_verdict, {})
        self.assertEqual(cache._h_rid, {})

    def test_finalize_partial_load_lands_on_resident_not_evicted_tail(self):
        # Partial load-back (loaded 10 < span alloc 16): the failed tail node is
        # evicted. last_node must land on the deepest RESIDENT node at depth 14
        # (device 4 + loaded 10), NOT the evicted tail the walk skips over; the
        # whole [14, 20) tail is freed at the verdict boundary.
        cache = object.__new__(EICPagedHiRadixCache)
        root = FakeTreeNode(0, True, None)
        cache.root_node = root
        cache.dec_lock_ref = lambda node: None
        cache._h_rid = {9: "r1"}
        node_a = FakeTreeNode(4, True, root)  # device match, (0, 4]
        node_c = FakeTreeNode(10, True, node_a)  # loaded host, (4, 14]
        node_d = FakeTreeNode(6, False, node_c)  # failed tail, evicted, (14, 20]
        node_d.id = 55
        req = mock.Mock()
        req.rid = "r1"
        req.prefix_indices = torch.arange(4, dtype=torch.int64)
        req.fill_ids = list(range(20))
        req.last_node = node_d
        cache.ongoing_load_admit = {
            "r1": {
                "h": 9,
                "d": 4,
                "alloc": 16,
                "load_node": node_d,
                "new_indices": torch.arange(100, 116, dtype=torch.int64),
                "deferred_lock": node_a,
                "complete": 10,
            }
        }
        cache._admit_verdict = {9: 14}  # FINAL == this stage's loaded length
        freed = []
        cache._free_failed_loadback = lambda nid, c: freed.append((nid, c))
        ok = EICPagedHiRadixCache._finalize_load_admit(cache, req)
        self.assertTrue(ok)
        self.assertEqual(freed, [(55, 10)])  # free [14, 20) at the verdict boundary
        self.assertEqual(len(req.prefix_indices), 14)  # 4 device + 10 loaded
        self.assertEqual(req.prefix_indices[4:].tolist(), list(range(100, 110)))
        self.assertIs(req.last_node, node_c)  # resident node at depth 14, not node_d
        req.set_extend_input_len.assert_called_once_with(20 - 14)
        self.assertEqual(req.eic_loaded_len, 10)

    # ---- two-stage lockstep protocol tests --------------------------------

    class FakeStore:
        def __init__(self):
            self.kv = {}

        def set(self, key, value):
            self.kv[key] = value

        def get(self, key):
            return self.kv[key]

        def check(self, keys):
            return all(k in self.kv for k in keys)

        def delete_key(self, key):
            return self.kv.pop(key, None) is not None

    def _make_stage(self, pp_rank, store, dag_box, pp_size=2):
        cache = object.__new__(EICPagedHiRadixCache)
        cache.pp_size, cache.pp_rank = pp_size, pp_rank
        cache.pp_group = object() if pp_size > 1 else None
        cache.tp_size, cache.rank, cache.tp_group = 1, 0, None
        cache.load_back_threshold = 10
        cache.root_node = FakeTreeNode(0, True, None)
        cache.ongoing_load_admit = {}
        cache.ongoing_load_back = {}
        cache._admit_verdict = {}
        cache._h_rid = {}
        cache._rid_epoch = {}
        cache._loadback_rid = {}
        cache._report_outbox = {}
        cache._span_reports = {}
        cache._load_reports = {}
        cache._await_load = {}
        cache._verdict_outbox = []
        cache._tombstone = {}
        cache._pub_seq = 0
        cache._next_seq = {}
        cache._round = 0
        cache._store_handle = store
        cache.locks = []
        cache.inc_lock_ref = lambda node: cache.locks.append(node)
        cache.dec_lock_ref = lambda node: cache.locks.remove(node)
        cache.freed = []
        cache._free_failed_loadback = lambda nid, c: cache.freed.append((nid, c))
        cache._clip_host_chain = lambda node, excess, quota: node
        if pp_rank == 0:
            # rank0 isends: record the packed verdicts for this round.
            cache._pp_bcast_from_first = lambda buf, tag=None: dag_box.__setitem__(
                "buf", buf.clone()
            )
        else:
            # rank>0 blocking-recv: the k-th recv pairs with the k-th send.
            cache._pp_bcast_from_first = lambda buf, tag=None: buf.copy_(
                dag_box["buf"]
            )
        q = Queue()
        cache.cache_controller = SimpleNamespace(
            ack_load_queue=q, mem_pool_device_allocator=SimpleNamespace()
        )
        return cache

    def _make_req(self, rid, d, hh, load_node_id):
        node = mock.Mock()
        node.evicted = False
        node.value = [1]
        node.key = list(range(max(hh, 1)))
        node.parent = None
        node.id = load_node_id
        req = mock.Mock()
        req.rid = rid
        req.prefix_indices = torch.arange(d, dtype=torch.int64)
        req.fill_ids = list(range(64))
        req.host_hit_length = hh
        req.needs_host_load_back = lambda: hh > 0
        req.best_match_node = node
        req.last_node = node
        return req

    def test_two_stage_load_back_reconciles_span_and_final(self):
        # Host metadata diverges (hh 16 vs 12): SPAN = min(4+16, 4+12) = 16, both
        # stages allocate quota 12 in the SAME round; loads land 12 vs 8: FINAL =
        # min(16, 4+12, 4+8) = 12; both admit 12 in the SAME round and free the
        # [12, 16) tail with identical arguments.
        store, box = self.FakeStore(), {}
        s0 = self._make_stage(0, store, box)
        s1 = self._make_stage(1, store, box)
        for s in (s0, s1):
            s.load_back = lambda node, allow_evict=None: torch.arange(
                12, dtype=torch.int64
            )
        r0 = self._make_req("req-a", 4, 16, load_node_id=100)
        r1 = self._make_req("req-a", 4, 12, load_node_id=200)
        self.assertFalse(EICPagedHiRadixCache.check_load_back_progress(s0, r0))
        self.assertFalse(EICPagedHiRadixCache.check_load_back_progress(s1, r1))
        # round 1: s1 publishes its span report; s0 sees only its own.
        EICPagedHiRadixCache.loading_check(s0)
        EICPagedHiRadixCache.loading_check(s1)
        self.assertEqual(s0.ongoing_load_admit["req-a"]["load_node"], None)
        # round 2: s0 drains the store, forms SPAN=16, both stages kick alloc 12.
        EICPagedHiRadixCache.loading_check(s0)
        EICPagedHiRadixCache.loading_check(s1)
        self.assertEqual(s0.ongoing_load_admit["req-a"]["alloc"], 12)
        self.assertEqual(s1.ongoing_load_admit["req-a"]["alloc"], 12)
        # acks land: stage0 loads 12/12, stage1 only 8/12.
        s0.cache_controller.ack_load_queue.put((100, 12))
        s1.cache_controller.ack_load_queue.put((200, 8))
        # round 3: acks drain; s1 publishes its LOADED report.
        EICPagedHiRadixCache.loading_check(s0)
        EICPagedHiRadixCache.loading_check(s1)
        self.assertFalse(EICPagedHiRadixCache.check_load_back_progress(s0, r0))
        # round 4: s0 drains it, forms FINAL=12, both apply in the same round.
        EICPagedHiRadixCache.loading_check(s0)
        EICPagedHiRadixCache.loading_check(s1)
        self.assertTrue(EICPagedHiRadixCache.check_load_back_progress(s0, r0))
        self.assertTrue(EICPagedHiRadixCache.check_load_back_progress(s1, r1))
        self.assertEqual(len(r0.prefix_indices), 12)
        self.assertEqual(len(r1.prefix_indices), 12)
        r0.set_extend_input_len.assert_called_once_with(64 - 12)
        r1.set_extend_input_len.assert_called_once_with(64 - 12)
        # identical uniform-boundary frees: [FINAL, SPAN) == complete arg 8.
        self.assertEqual(s0.freed, [(100, 8)])
        self.assertEqual(s1.freed, [(200, 8)])
        # no leaked state, no leaked locks.
        for s in (s0, s1):
            self.assertEqual(s.ongoing_load_admit, {})
            self.assertEqual(s._admit_verdict, {})
            self.assertEqual(s._h_rid, {})
            self.assertEqual(s.locks, [])
        self.assertEqual(store.kv, {})  # every store batch consumed and deleted

    def test_two_stage_cold_req_single_round_trip(self):
        # No stage has host data: SPAN decision short-circuits to FINAL=min(d)
        # (one round trip), both admit device-only in the same round.
        store, box = self.FakeStore(), {}
        s0 = self._make_stage(0, store, box)
        s1 = self._make_stage(1, store, box)
        r0 = self._make_req("req-b", 8, 0, load_node_id=0)
        r1 = self._make_req("req-b", 8, 0, load_node_id=0)
        self.assertFalse(EICPagedHiRadixCache.check_load_back_progress(s0, r0))
        self.assertFalse(EICPagedHiRadixCache.check_load_back_progress(s1, r1))
        EICPagedHiRadixCache.loading_check(s0)
        EICPagedHiRadixCache.loading_check(s1)
        EICPagedHiRadixCache.loading_check(s0)
        EICPagedHiRadixCache.loading_check(s1)
        self.assertTrue(EICPagedHiRadixCache.check_load_back_progress(s0, r0))
        self.assertTrue(EICPagedHiRadixCache.check_load_back_progress(s1, r1))
        self.assertEqual(len(r0.prefix_indices), 8)
        self.assertEqual(len(r1.prefix_indices), 8)
        for s in (s0, s1):
            self.assertEqual(s.locks, [])

    def test_release_tombstones_and_drops_straggler_verdicts(self):
        # Release after the span reports are in flight: the late verdict and any
        # straggler reports must die on arrival (epoch/tombstone), leaving no
        # state and no locks behind.
        store, box = self.FakeStore(), {}
        s0 = self._make_stage(0, store, box)
        s1 = self._make_stage(1, store, box)
        r0 = self._make_req("req-c", 4, 16, load_node_id=100)
        r1 = self._make_req("req-c", 4, 16, load_node_id=200)
        EICPagedHiRadixCache.check_load_back_progress(s0, r0)
        EICPagedHiRadixCache.check_load_back_progress(s1, r1)
        EICPagedHiRadixCache.loading_check(s0)
        EICPagedHiRadixCache.loading_check(s1)  # s1's span report is now in the store
        EICPagedHiRadixCache.release_load_admit(s0, "req-c")
        EICPagedHiRadixCache.release_load_admit(s1, "req-c")
        h = s0._rid_hash("req-c")
        self.assertIn(h, s0._tombstone)
        # rank0 drains the straggler AFTER the release: tombstone drops it.
        EICPagedHiRadixCache.loading_check(s0)
        EICPagedHiRadixCache.loading_check(s1)
        self.assertEqual(s0._span_reports, {})
        self.assertEqual(s0._verdict_outbox, [])
        for s in (s0, s1):
            self.assertEqual(s.ongoing_load_admit, {})
            self.assertEqual(s.locks, [])

    def test_rid_reuse_after_release_still_reconciles(self):
        # A client retry reuses the rid within the tombstone TTL. The tombstone
        # must kill only the RELEASED incarnation's stragglers; the retry's
        # higher-epoch reports must pass or the retry wedges forever.
        store, box = self.FakeStore(), {}
        s0 = self._make_stage(0, store, box)
        s1 = self._make_stage(1, store, box)
        EICPagedHiRadixCache.check_load_back_progress(
            s0, self._make_req("req-d", 8, 0, load_node_id=0)
        )
        EICPagedHiRadixCache.check_load_back_progress(
            s1, self._make_req("req-d", 8, 0, load_node_id=0)
        )
        EICPagedHiRadixCache.loading_check(s0)
        EICPagedHiRadixCache.loading_check(s1)  # epoch-1 span report published
        EICPagedHiRadixCache.release_load_admit(s0, "req-d")
        EICPagedHiRadixCache.release_load_admit(s1, "req-d")
        # retry: same rid re-enters the gate (epoch 2) while tombstoned.
        r0 = self._make_req("req-d", 8, 0, load_node_id=0)
        r1 = self._make_req("req-d", 8, 0, load_node_id=0)
        self.assertFalse(EICPagedHiRadixCache.check_load_back_progress(s0, r0))
        self.assertFalse(EICPagedHiRadixCache.check_load_back_progress(s1, r1))
        for _ in range(3):
            EICPagedHiRadixCache.loading_check(s0)
            EICPagedHiRadixCache.loading_check(s1)
        self.assertTrue(EICPagedHiRadixCache.check_load_back_progress(s0, r0))
        self.assertTrue(EICPagedHiRadixCache.check_load_back_progress(s1, r1))
        self.assertEqual(len(r0.prefix_indices), 8)

    def test_stale_ack_after_release_is_orphaned_not_attributed(self):
        # Release with the load still in flight, then the same rid re-gates.
        # The OLD incarnation's ack must be orphan-freed (whole span), never
        # attributed to the new incarnation (whose node_id differs).
        store, box = self.FakeStore(), {}
        s0 = self._make_stage(0, store, box)
        s1 = self._make_stage(1, store, box)
        for s in (s0, s1):
            s.load_back = lambda node, allow_evict=None, _s=s: (
                _s.ongoing_load_back.__setitem__(node.id, (node, node, 12)),
                torch.arange(12, dtype=torch.int64),
            )[1]
        r0 = self._make_req("req-e", 4, 16, load_node_id=100)
        r1 = self._make_req("req-e", 4, 16, load_node_id=200)
        EICPagedHiRadixCache.check_load_back_progress(s0, r0)
        EICPagedHiRadixCache.check_load_back_progress(s1, r1)
        for _ in range(2):  # span verdict forms and applies -> loads kicked
            EICPagedHiRadixCache.loading_check(s0)
            EICPagedHiRadixCache.loading_check(s1)
        self.assertEqual(s0.ongoing_load_admit["req-e"]["load_node"].id, 100)
        EICPagedHiRadixCache.release_load_admit(s0, "req-e")
        EICPagedHiRadixCache.release_load_admit(s1, "req-e")
        # same rid re-gates (epoch 2, no load kicked yet: node_id None)
        EICPagedHiRadixCache.check_load_back_progress(
            s0, self._make_req("req-e", 4, 16, load_node_id=101)
        )
        # the old incarnation's ack lands now: must orphan-free the WHOLE span.
        s0.cache_controller.ack_load_queue.put((100, 12))
        EICPagedHiRadixCache.loading_check(s0)
        self.assertEqual(s0.freed, [(100, 0)])
        self.assertIsNone(s0.ongoing_load_admit["req-e"]["complete"])

    def test_pp1_cold_req_admits_with_zero_deferral(self):
        # pp<=1 keeps the old behavior: a cold/device-only req admits on its
        # FIRST gate pass, no verdict round trip.
        s = self._make_stage(0, self.FakeStore(), {}, pp_size=1)
        req = self._make_req("req-f", 8, 0, load_node_id=0)
        self.assertTrue(EICPagedHiRadixCache.check_load_back_progress(s, req))
        self.assertEqual(len(req.prefix_indices), 8)
        self.assertEqual(s.locks, [])
        self.assertEqual(s.ongoing_load_admit, {})

    def test_pp1_warm_req_kicks_at_gate_and_admits_on_ack(self):
        # pp<=1 warm path: load kicks at gate entry, req defers, the local ack
        # IS the verdict (d + complete), failed tail freed at that boundary.
        s = self._make_stage(0, self.FakeStore(), {}, pp_size=1)
        s.load_back = lambda node, allow_evict=None: torch.arange(
            16, dtype=torch.int64
        )
        req = self._make_req("req-g", 4, 16, load_node_id=100)
        self.assertFalse(EICPagedHiRadixCache.check_load_back_progress(s, req))
        self.assertEqual(s.ongoing_load_admit["req-g"]["load_node"].id, 100)
        s.cache_controller.ack_load_queue.put((100, 10))
        EICPagedHiRadixCache.loading_check(s)
        self.assertTrue(EICPagedHiRadixCache.check_load_back_progress(s, req))
        self.assertEqual(len(req.prefix_indices), 14)  # 4 device + 10 loaded
        self.assertEqual(s.freed, [(100, 10)])
        self.assertEqual(s.locks, [])

    def test_finalize_asserts_on_verdict_before_ack(self):
        # I5 enforcement: a FINAL verdict may never land before the local ack
        # when a load was kicked -- silent free of in-flight DMA otherwise.
        s = self._make_stage(0, self.FakeStore(), {}, pp_size=1)
        s.dec_lock_ref = lambda n: None
        node = self._make_req("x", 4, 16, load_node_id=100).best_match_node
        req = self._make_req("req-h", 4, 16, load_node_id=100)
        s.ongoing_load_admit = {
            "req-h": {
                "h": 5,
                "d": 4,
                "alloc": 12,
                "load_node": node,
                "new_indices": torch.arange(12, dtype=torch.int64),
                "deferred_lock": node,
                "complete": None,
            }
        }
        s._h_rid = {5: "req-h"}
        s._admit_verdict = {5: 10}
        with self.assertRaises(AssertionError):
            EICPagedHiRadixCache._finalize_load_admit(s, req)

    def _make_load_controller(self, page_data):
        cc = object.__new__(EICCacheController)
        cc.tp_world_size = 1
        cc.page_size = 4
        cc.ack_load_queue = Queue()
        cc.mem_pool_host = SimpleNamespace(
            get_page_data=mock.Mock(side_effect=page_data)
        )
        return cc

    def _load_op(self):
        op = object.__new__(EICCacheOperation)
        op.node_id = 7
        op.content_hash = ["h0", "h1", "h2"]
        op.host_indices = torch.arange(12, dtype=torch.int64)
        op.device_indices = torch.arange(12, dtype=torch.int64)
        return op

    def test_load_acks_zero_when_backend_raises(self):
        # A missing ack strands the request holding its KV until abort, and
        # under PP it also blocks the verdict for every other stage.
        cc = self._make_load_controller(RuntimeError("eic client blew up"))
        EICCacheController.load_operation_shared(cc, self._load_op())
        self.assertEqual(cc.ack_load_queue.get_nowait(), (7, 0))

    def test_short_all_true_mask_is_not_full_success(self):
        # The backend bails out early on failure, returning fewer mask entries
        # than pages; all(mask) would read that as a complete hit.
        cc = self._make_load_controller([[True]])
        EICCacheController.load_operation_shared(cc, self._load_op())
        self.assertEqual(cc.ack_load_queue.get_nowait(), (7, 4))

        cc = self._make_load_controller([[]])
        EICCacheController.load_operation_shared(cc, self._load_op())
        self.assertEqual(cc.ack_load_queue.get_nowait(), (7, 0))

    def test_full_mask_acks_every_token(self):
        cc = self._make_load_controller([[True, True, True]])
        EICCacheController.load_operation_shared(cc, self._load_op())
        self.assertEqual(cc.ack_load_queue.get_nowait(), (7, 12))
    def _make_reconciler(self, pp_rank, store, pp_size=2):
        r = EICPPReconciler(
            prefix="hipf", pp_rank=pp_rank, pp_size=pp_size, pp_group=object(), rank=0
        )
        r.eic = True
        r._store_handle = store
        return r

    def test_reconciler_min_across_stages(self):
        store = self.FakeStore()
        s0, s1 = self._make_reconciler(0, store), self._make_reconciler(1, store)
        h = EICPPReconciler.rid_hash("r")
        s0.report(h, 1, 512)
        s1.report(h, 1, 256)
        self.assertEqual(s1.collect(), [])
        self.assertEqual(s0.collect(), [(h, 1, 256)])

    def test_reconciler_waits_for_every_stage(self):
        s0 = self._make_reconciler(0, self.FakeStore())
        s0.report(EICPPReconciler.rid_hash("r"), 1, 512)
        self.assertEqual(s0.collect(), [])

    def test_reconciler_tombstone_is_epoch_aware(self):
        store = self.FakeStore()
        s0, s1 = self._make_reconciler(0, store), self._make_reconciler(1, store)
        h = EICPPReconciler.rid_hash("r")
        s0.release(h, 1)
        s1.report(h, 1, 256)
        s1.report(h, 2, 128)
        s1.collect()
        s0._drain_peers()
        self.assertNotIn((h, 1), s0._reports)
        self.assertIn((h, 2), s0._reports)

    def test_reconciler_epoch_never_reused(self):
        r = self._make_reconciler(0, self.FakeStore())
        self.assertEqual([r.bump_epoch("r"), r.bump_epoch("r")], [1, 2])


if __name__ == "__main__":
    unittest.main()
