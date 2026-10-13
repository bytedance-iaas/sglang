"""DPBudget + DataParallelController dispatch tests.

The e2e counterpart, over the real scheduler load-report path, is
test/registered/disaggregation/test_disaggregation_dp_attention.py.

Fragility: scheduler tests bypass `DataParallelController.__init__` via
`__new__` and inject only the attrs the schedulers read (`workers`, `status`,
`_active_workers`, `round_robin_counter`, `dp_budget`). Update `_make_controller`
if a scheduler starts reading another attr. `maybe_external_dp_rank_routing`
is exercised as the real method, no mock. The refresh-throttle tests inject
`load_snapshot_reader` and `_last_refresh_time` on top of those.
"""

import threading
import time
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec.structs

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.data_parallel_controller import (
    DataParallelController,
    DPBudget,
    LoadBalanceMethod,
)
from sglang.srt.managers.load_snapshot import LoadSnapshot
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


_BASE_LOAD = msgspec.structs.replace(
    LoadSnapshot(dp_rank=0),
    max_total_num_tokens=4096,
    max_running_requests=128,
)


def _load(**overrides) -> LoadSnapshot:
    return msgspec.structs.replace(_BASE_LOAD, **overrides)


def _make_controller(num_dp_ranks: int) -> DataParallelController:
    """Bypass __init__; inject only the attrs dispatch methods read."""
    ctl = DataParallelController.__new__(DataParallelController)
    ctl.workers = [MagicMock(name=f"worker_{i}") for i in range(num_dp_ranks)]
    ctl.status = [True] * num_dp_ranks
    ctl._active_workers = list(range(num_dp_ranks))
    ctl.round_robin_counter = 0
    ctl.dp_budget = DPBudget(num_dp_ranks=num_dp_ranks)
    return ctl


def _req(
    routed_dp_rank=None,
    bootstrap_room=None,
    input_ids=None,
    *,
    routing_key=None,
    session_id=None,
    session_params=None,
):
    """Req stand-in; SimpleNamespace avoids pinning to the Req dataclass schema."""
    return SimpleNamespace(
        routed_dp_rank=routed_dp_rank,
        bootstrap_room=bootstrap_room,
        input_ids=input_ids or [],
        routing_key=routing_key,
        session_id=session_id,
        session_params=session_params,
    )


def _make_affinity_controller(
    num_dp_ranks: int,
    *,
    fallback: str = "total_tokens",
    max_load_skew: float = 1.5,
    hash_tokens: int = 4096,
    disable_token_fallback: bool = False,
) -> DataParallelController:
    ctl = _make_controller(num_dp_ranks)
    ctl.load_balance_method = LoadBalanceMethod.PREFIX_AFFINITY
    ctl._affinity_fallback = LoadBalanceMethod.from_str(fallback)
    ctl._affinity_max_load_skew = max_load_skew
    ctl._affinity_hash_tokens = hash_tokens
    ctl._affinity_disable_token_fallback = disable_token_fallback
    ctl._incremental_session_owners = {}
    ctl._open_incremental_sessions = set()
    return ctl


def _dispatched_rank(ctl: DataParallelController) -> int:
    called = [i for i, worker in enumerate(ctl.workers) if worker.send_pyobj.called]
    assert len(called) == 1, called
    return called[0]


class TestLocalKvEventSources(CustomTestCase):
    @staticmethod
    def _source(rank, block_size=64):
        return {
            "dp_rank": rank,
            "endpoint": f"tcp://127.0.0.1:{5557 + rank}",
            "topic": "kv",
            "block_size": block_size,
        }

    def test_scheduler_reports_logical_block_size_or_no_source(self):
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.managers.scheduler_components.kv_events_publisher import (
            SchedulerKvEventsPublisher,
        )

        component = SchedulerKvEventsPublisher.__new__(SchedulerKvEventsPublisher)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_total_num_tokens = 64
        scheduler.max_req_input_len = 32
        scheduler.startup_time = {}
        scheduler.page_size = 64
        scheduler.kv_events_publisher = component
        publisher = MagicMock()
        publisher.describe_local_source.return_value = self._source(4, 256)
        for owned_publisher, expected in (
            (publisher, [self._source(4, 256)]),
            (None, []),
        ):
            with (
                self.subTest(has_publisher=owned_publisher is not None),
                get_context().override_server_args(grpc_port=50051, dcp_size=4),
            ):
                component.kv_event_publisher = owned_publisher
                self.assertEqual(
                    scheduler.get_init_info()["kv_event_sources"], expected
                )
        publisher.describe_local_source.assert_called_once_with(256)

    def test_controller_collects_all_scheduler_sources(self):
        from sglang.srt.managers import data_parallel_controller as module

        controller = DataParallelController.__new__(DataParallelController)
        controller.env_lock = threading.Lock()
        controller.scheduler_procs = []
        controller.local_kv_event_sources = []
        controller.run_scheduler_process_func = MagicMock()
        # TP8/DP4 over two nodes: this follower's four schedulers own ranks 2,3.
        readers = [
            MagicMock(
                recv=MagicMock(
                    return_value={
                        "max_total_num_tokens": 64,
                        "max_req_input_len": 32,
                        "kv_event_sources": sources,
                    }
                )
            )
            for sources in ([self._source(2)], [], [self._source(3)], [])
        ]
        with (
            get_context().override_server_args(
                tp_size=8,
                dp_size=4,
                enable_dp_attention=True,
                nnodes=2,
                node_rank=1,
                pp_size=1,
                attn_cp_size=1,
            ),
            patch.object(module.mp, "Process"),
            patch.object(
                module.mp,
                "Pipe",
                side_effect=[(reader, MagicMock()) for reader in readers],
            ),
            patch.object(module.PortArgs, "init_new", return_value=MagicMock()),
            patch.object(module.TorchMemorySaverAdapter, "create"),
            patch.object(module.numa_utils, "configure_subprocess"),
            patch.object(
                module, "maybe_reindex_device_id", return_value=nullcontext(0)
            ),
        ):
            controller.launch_tensor_parallel_group(
                MagicMock(), MagicMock(), 0, None, [7000, 7001, 7002, 7003]
            )
        self.assertEqual(
            controller.local_kv_event_sources, [self._source(2), self._source(3)]
        )


class TestDPBudgetUpdateBudget(CustomTestCase):
    def test_maps_running_plus_waiting_to_total_requests(self):
        budget = DPBudget(num_dp_ranks=2)
        budget.update_budget(
            [
                _load(dp_rank=0, timestamp=1.0, num_running_reqs=3, num_waiting_reqs=2),
                _load(dp_rank=1, timestamp=1.0, num_running_reqs=5, num_waiting_reqs=1),
            ]
        )
        self.assertEqual(budget.total_requests, [5, 6])

    def test_maps_num_total_tokens_not_num_used_tokens(self):
        budget = DPBudget(num_dp_ranks=2)
        budget.update_budget(
            [
                _load(
                    dp_rank=0, timestamp=1.0, num_used_tokens=100, num_total_tokens=150
                ),
                _load(
                    dp_rank=1, timestamp=1.0, num_used_tokens=80, num_total_tokens=80
                ),
            ]
        )
        self.assertEqual(budget.total_tokens, [150, 80])

    def test_partial_update_only_affects_reported_rank(self):
        budget = DPBudget(num_dp_ranks=3)
        budget.update_budget(
            [
                _load(
                    dp_rank=0, timestamp=1.0, num_running_reqs=10, num_total_tokens=100
                ),
                _load(
                    dp_rank=1, timestamp=1.0, num_running_reqs=20, num_total_tokens=200
                ),
                _load(
                    dp_rank=2, timestamp=1.0, num_running_reqs=30, num_total_tokens=300
                ),
            ]
        )
        budget.update_budget(
            [
                _load(
                    dp_rank=1,
                    timestamp=2.0,
                    num_running_reqs=1,
                    num_waiting_reqs=1,
                    num_total_tokens=50,
                )
            ]
        )
        self.assertEqual(budget.total_requests, [10, 2, 30])
        self.assertEqual(budget.total_tokens, [100, 50, 300])


class TestDPBudgetDispatch(CustomTestCase):
    """DPBudget.dispatch picks a rank from current state and updates counters."""

    def test_total_requests_dispatch_picks_min_and_increments(self):
        budget = DPBudget(num_dp_ranks=3)
        budget.total_requests = [4, 2, 7]
        rank = budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        self.assertEqual(rank, 1)
        self.assertEqual(
            budget.total_requests[1],
            3,
            "dispatch should increment chosen worker's request count",
        )

    def test_total_tokens_dispatch_applies_estimated_tokens(self):
        budget = DPBudget(num_dp_ranks=3)
        budget.total_tokens = [100, 50, 200]
        budget.total_requests = [0, 0, 0]
        rank = budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS, estimated_tokens=30)
        self.assertEqual(rank, 1, "should pick worker with min total_tokens")
        self.assertEqual(
            budget.total_tokens[1],
            80,
            "dispatch should add estimated_tokens to chosen worker",
        )
        self.assertEqual(
            budget.total_requests[1],
            1,
            "dispatch should also increment request count",
        )

    def test_total_tokens_tie_breaks_on_total_requests(self):
        budget = DPBudget(num_dp_ranks=3)
        budget.total_tokens = [50, 50, 50]
        budget.total_requests = [4, 2, 7]
        rank = budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS, estimated_tokens=10)
        self.assertEqual(
            rank, 1, "tie on total_tokens should fall back to min total_requests"
        )

    def test_dispatch_returns_none_for_methods_not_handled(self):
        """Round-robin and follow_bootstrap_room dispatch elsewhere; DPBudget
        only handles the load-aware variants."""
        budget = DPBudget(num_dp_ranks=3)
        self.assertIsNone(budget.dispatch(LoadBalanceMethod.ROUND_ROBIN))
        self.assertIsNone(budget.dispatch(LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM))


class TestRoundRobinScheduler(CustomTestCase):
    def test_cycles_through_active_workers_in_order(self):
        ctl = _make_controller(num_dp_ranks=4)
        for _ in range(8):
            ctl.round_robin_scheduler(_req())
        # 8 reqs across 4 active workers — 2 each, in round-robin order
        for i, worker in enumerate(ctl.workers):
            self.assertEqual(worker.send_pyobj.call_count, 2, f"worker {i} call count")

    def test_first_dispatch_picks_worker_zero(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.round_robin_scheduler(_req())
        ctl.workers[0].send_pyobj.assert_called_once()
        for i in (1, 2, 3):
            ctl.workers[i].send_pyobj.assert_not_called()

    def test_skips_inactive_workers(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.status[1] = False
        ctl.status[3] = False
        for _ in range(6):
            ctl.round_robin_scheduler(_req())
        # Only workers 0 and 2 are active — should split 6 reqs evenly
        self.assertEqual(ctl.workers[0].send_pyobj.call_count, 3)
        ctl.workers[1].send_pyobj.assert_not_called()
        self.assertEqual(ctl.workers[2].send_pyobj.call_count, 3)
        ctl.workers[3].send_pyobj.assert_not_called()

    def test_routed_dp_rank_bypasses_counter(self):
        """External dp-rank routing must not advance the counter."""
        ctl = _make_controller(num_dp_ranks=4)
        ctl.round_robin_scheduler(_req(routed_dp_rank=2))
        ctl.workers[2].send_pyobj.assert_called_once()
        self.assertEqual(
            ctl.round_robin_counter,
            0,
            "external routing must not advance the round-robin counter",
        )
        # Subsequent round-robin req still lands on worker 0
        ctl.round_robin_scheduler(_req())
        ctl.workers[0].send_pyobj.assert_called_once()


class TestFollowBootstrapRoomScheduler(CustomTestCase):
    def test_dispatches_by_bootstrap_room_modulo(self):
        ctl = _make_controller(num_dp_ranks=4)
        for room, expected_rank in [
            (0, 0),
            (1, 1),
            (4, 0),
            (5, 1),
            (100, 0),
            (101, 1),
        ]:
            ctl.follow_bootstrap_room_scheduler(_req(bootstrap_room=room))
            ctl.workers[expected_rank].send_pyobj.assert_called()

    def test_requires_bootstrap_room(self):
        ctl = _make_controller(num_dp_ranks=4)
        with self.assertRaises(AssertionError):
            ctl.follow_bootstrap_room_scheduler(_req(bootstrap_room=None))

    def test_routed_dp_rank_bypasses_bootstrap_room(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.follow_bootstrap_room_scheduler(_req(routed_dp_rank=3, bootstrap_room=1))
        ctl.workers[3].send_pyobj.assert_called_once()
        ctl.workers[1].send_pyobj.assert_not_called()


class TestTotalRequestsScheduler(CustomTestCase):
    def test_dispatches_to_min_request_worker(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.dp_budget.total_requests = [5, 3, 1, 4]
        ctl.total_requests_scheduler(_req())
        ctl.workers[2].send_pyobj.assert_called_once()
        for i in (0, 1, 3):
            ctl.workers[i].send_pyobj.assert_not_called()
        self.assertEqual(
            ctl.dp_budget.total_requests[2],
            2,
            "DPBudget must record the dispatch by incrementing the counter",
        )

    def test_routed_dp_rank_bypasses_budget(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.dp_budget.total_requests = [5, 3, 1, 4]
        ctl.total_requests_scheduler(_req(routed_dp_rank=0))
        ctl.workers[0].send_pyobj.assert_called_once()
        # DPBudget must not be touched when bypassed
        self.assertEqual(
            ctl.dp_budget.total_requests,
            [5, 3, 1, 4],
            "external routing must not mutate DPBudget state",
        )


class TestStatusAwareLoadBalancing(CustomTestCase):
    def test_total_requests_skips_inactive_rank(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.dp_budget.total_requests = [5, 3, 1, 4]
        ctl.status[2] = False
        ctl._active_workers = [0, 1, 3]
        ctl.total_requests_scheduler(_req())
        ctl.workers[1].send_pyobj.assert_called_once()
        ctl.workers[2].send_pyobj.assert_not_called()

    def test_total_tokens_skips_inactive_rank(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.dp_budget.total_tokens = [50, 30, 10, 40]
        ctl.status[2] = False
        ctl._active_workers = [0, 1, 3]
        ctl.total_tokens_scheduler(_req(input_ids=[1, 2]))
        ctl.workers[1].send_pyobj.assert_called_once()
        ctl.workers[2].send_pyobj.assert_not_called()


class TestPrefixAffinity(CustomTestCase):
    @staticmethod
    def _mark_open(ctl: DataParallelController, session_id: str) -> None:
        ctl._open_incremental_sessions.add(session_id)

    def test_routed_dp_rank_remains_authoritative_for_full_prompt(self):
        ctl = _make_affinity_controller(4)
        ctl.prefix_affinity_scheduler(
            _req(routed_dp_rank=2, routing_key="session-a", input_ids=[1, 2])
        )
        ctl.workers[2].send_pyobj.assert_called_once()
        self.assertEqual(ctl.dp_budget.total_requests, [0, 0, 0, 0])

    def test_top_level_session_id_is_deterministic(self):
        ranks = set()
        for _ in range(8):
            ctl = _make_affinity_controller(8)
            ctl.prefix_affinity_scheduler(
                _req(session_id="session-a", input_ids=[1, 2, 3])
            )
            ranks.add(_dispatched_rank(ctl))
        self.assertEqual(len(ranks), 1)

    def test_routing_key_has_priority_over_top_level_session_id(self):
        ctl = _make_affinity_controller(8)
        ctl.prefix_affinity_scheduler(
            _req(routing_key="project-a", session_id="session-a", input_ids=[1])
        )
        expected = ctl._rendezvous_ranked("project-a", list(range(8)))[0]
        self.assertEqual(_dispatched_rank(ctl), expected)

    def test_full_prompt_spills_from_overloaded_preferred_rank(self):
        ctl = _make_affinity_controller(4, max_load_skew=1.5)
        preferred = ctl._rendezvous_ranked("session:session-a", list(range(4)))[0]
        ctl.dp_budget.total_requests[preferred] = 100

        ctl.prefix_affinity_scheduler(_req(session_id="session-a", input_ids=[1, 2, 3]))

        self.assertNotEqual(_dispatched_rank(ctl), preferred)

    def test_huge_resident_prefix_is_not_an_overload_signal(self):
        ctl = _make_affinity_controller(4)
        preferred = ctl._rendezvous_ranked("session:session-a", list(range(4)))[0]
        ctl.dp_budget.total_tokens[preferred] = 800_000

        ctl.prefix_affinity_scheduler(_req(session_id="session-a", input_ids=[1, 2, 3]))

        self.assertEqual(_dispatched_rank(ctl), preferred)

    def test_incremental_session_stays_on_owner_despite_overload(self):
        ctl = _make_affinity_controller(4)
        params = SimpleNamespace(id="incremental-a")
        self._mark_open(ctl, params.id)
        ctl.prefix_affinity_scheduler(_req(session_params=params, input_ids=[1]))
        owner = ctl._incremental_session_owners["incremental-a"]
        for worker in ctl.workers:
            worker.reset_mock()
        ctl.dp_budget.total_requests[owner] = 100

        ctl.prefix_affinity_scheduler(_req(session_params=params, input_ids=[2]))

        self.assertEqual(_dispatched_rank(ctl), owner)

    def test_incremental_session_explicit_rank_sets_owner(self):
        ctl = _make_affinity_controller(4)
        params = SimpleNamespace(id="incremental-a")
        self._mark_open(ctl, params.id)
        ctl.prefix_affinity_scheduler(
            _req(routed_dp_rank=3, session_params=params, input_ids=[1])
        )
        self.assertEqual(ctl._incremental_session_owners["incremental-a"], 3)
        ctl.workers[3].send_pyobj.assert_called_once()

    def test_incremental_session_rejects_explicit_rank_change(self):
        ctl = _make_affinity_controller(4)
        params = SimpleNamespace(id="incremental-a")
        self._mark_open(ctl, params.id)
        ctl.prefix_affinity_scheduler(
            _req(routed_dp_rank=1, session_params=params, input_ids=[1])
        )
        for worker in ctl.workers:
            worker.reset_mock()
        req = _req(routed_dp_rank=2, session_params=params, input_ids=[2])

        ctl.prefix_affinity_scheduler(req)

        ctl.workers[0].send_pyobj.assert_called_once()
        self.assertIs(ctl.workers[0].send_pyobj.call_args.args[0], req)
        self.assertIn(
            "Scheduler-local history migration is unsupported",
            req.dp_routing_error,
        )

    def test_incremental_session_owner_inactive_fails_closed(self):
        ctl = _make_affinity_controller(4)
        params = SimpleNamespace(id="incremental-a")
        self._mark_open(ctl, params.id)
        ctl.prefix_affinity_scheduler(_req(session_params=params, input_ids=[1]))
        owner = ctl._incremental_session_owners["incremental-a"]
        ctl._active_workers.remove(owner)
        for worker in ctl.workers:
            worker.reset_mock()
        req = _req(session_params=params, input_ids=[2])

        ctl.prefix_affinity_scheduler(req)

        fallback_rank = ctl._active_workers[0]
        ctl.workers[fallback_rank].send_pyobj.assert_called_once()
        self.assertIs(ctl.workers[fallback_rank].send_pyobj.call_args.args[0], req)
        self.assertIn("owner DP rank", req.dp_routing_error)

    def test_close_invalidates_incremental_owner(self):
        ctl = _make_affinity_controller(4)
        ctl.send_control_message = MagicMock()
        params = SimpleNamespace(id="incremental-a")
        self._mark_open(ctl, params.id)
        ctl.prefix_affinity_scheduler(_req(session_params=params, input_ids=[1]))

        ctl.close_session(SimpleNamespace(session_id="incremental-a"))

        self.assertNotIn("incremental-a", ctl._incremental_session_owners)
        self.assertNotIn("incremental-a", ctl._open_incremental_sessions)
        ctl.send_control_message.assert_called_once()

    def test_duplicate_open_preserves_existing_incremental_owner(self):
        ctl = _make_affinity_controller(4)
        ctl.send_control_message = MagicMock()
        ctl._incremental_session_owners["incremental-a"] = 2

        ctl.open_session(SimpleNamespace(session_id="incremental-a"))

        self.assertEqual(ctl._incremental_session_owners["incremental-a"], 2)
        self.assertIn("incremental-a", ctl._open_incremental_sessions)
        ctl.send_control_message.assert_called_once()

    def test_first_incremental_request_only_dispatches_to_owner(self):
        ctl = _make_affinity_controller(4)
        params = SimpleNamespace(id="incremental-a")
        self._mark_open(ctl, params.id)

        ctl.prefix_affinity_scheduler(_req(session_params=params, input_ids=[1]))

        owner = ctl._incremental_session_owners[params.id]
        for rank, worker in enumerate(ctl.workers):
            if rank == owner:
                self.assertEqual(worker.send_pyobj.call_count, 1)
            else:
                worker.send_pyobj.assert_not_called()

    def test_unopened_incremental_session_is_rejected_without_owner_guess(self):
        ctl = _make_affinity_controller(4)
        req = _req(session_params=SimpleNamespace(id="incremental-a"), input_ids=[1])

        ctl.prefix_affinity_scheduler(req)

        self.assertNotIn("incremental-a", ctl._incremental_session_owners)
        self.assertIn("call /open_session first", req.dp_routing_error)

    def test_inactive_rank_only_remaps_keys_owned_by_that_rank(self):
        ctl = _make_affinity_controller(4)
        keys = [f"session-{i}" for i in range(256)]
        before = {key: ctl._rendezvous_ranked(key, [0, 1, 2, 3])[0] for key in keys}
        after = {key: ctl._rendezvous_ranked(key, [0, 1, 3])[0] for key in keys}
        for key in keys:
            if before[key] == 2:
                self.assertNotEqual(after[key], 2)
            else:
                self.assertEqual(after[key], before[key])


class TestRefreshLoadBudgetThrottle(CustomTestCase):
    @staticmethod
    def _controller_with_reader(num_dp_ranks, snapshots):
        ctl = _make_controller(num_dp_ranks)
        ctl.load_snapshot_reader = MagicMock()
        ctl.load_snapshot_reader.read_all.return_value = snapshots
        return ctl

    def test_throttled_refresh_spreads_a_burst_across_ranks(self):
        idle = [_load(dp_rank=i, timestamp=1.0, num_total_tokens=0) for i in range(4)]
        ctl = self._controller_with_reader(num_dp_ranks=4, snapshots=idle)
        # A refresh stamp in the future keeps every call inside the window, so
        # the burst runs entirely on speculative counters.
        ctl._last_refresh_time = time.perf_counter() + 3600.0

        for _ in range(8):
            ctl.refresh_load_budget()
            ctl.total_tokens_scheduler(_req(input_ids=[0] * 100))

        ctl.load_snapshot_reader.read_all.assert_not_called()
        self.assertEqual(
            ctl.dp_budget.total_tokens,
            [200, 200, 200, 200],
            "speculative increments should spread the burst evenly",
        )
        for i, worker in enumerate(ctl.workers):
            self.assertEqual(
                worker.send_pyobj.call_count, 2, f"worker {i} should get 2 of 8 reqs"
            )

    def test_refresh_outside_window_overwrites_speculative_increments(self):
        reported = [
            _load(dp_rank=0, timestamp=2.0, num_total_tokens=10),
            _load(dp_rank=1, timestamp=2.0, num_total_tokens=20),
        ]
        ctl = self._controller_with_reader(num_dp_ranks=2, snapshots=reported)
        ctl._last_refresh_time = 0.0  # window has long passed
        ctl.dp_budget.total_tokens = [999, 999]

        ctl.refresh_load_budget()

        ctl.load_snapshot_reader.read_all.assert_called_once()
        self.assertEqual(
            ctl.dp_budget.total_tokens,
            [10, 20],
            "a fresh snapshot must replace the speculative state",
        )
        self.assertGreater(ctl._last_refresh_time, 0.0)

    def test_unchanged_snapshot_does_not_reset_the_burst(self):
        frozen = [_load(dp_rank=i, timestamp=1.0, num_total_tokens=0) for i in range(2)]
        ctl = self._controller_with_reader(num_dp_ranks=2, snapshots=frozen)
        ctl._last_refresh_time = 0.0
        ctl.refresh_load_budget()  # adopts timestamp 1.0

        for _ in range(4):
            ctl.total_tokens_scheduler(_req(input_ids=[0] * 50))
        after_burst = list(ctl.dp_budget.total_tokens)

        ctl._last_refresh_time = 0.0  # let the next refresh through the throttle
        ctl.refresh_load_budget()  # same timestamp -> update_budget skips it

        self.assertEqual(
            after_burst, [100, 100], "burst should have spread over both ranks"
        )
        self.assertEqual(
            ctl.dp_budget.total_tokens,
            after_burst,
            "a stale-timestamp snapshot must not wipe the speculative state",
        )


if __name__ == "__main__":
    unittest.main()
