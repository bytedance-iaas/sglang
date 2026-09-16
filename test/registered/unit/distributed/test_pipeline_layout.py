import unittest

from sglang.srt.distributed.pipeline_layout import (
    PipelineControlEnvelope,
    PipelineControlKind,
    PipelineLayout,
    PipelinePrefixRegistry,
    PipelineRankSchedule,
    PipelineReadyQueueSchedule,
    PipelineReplicaIdentity,
    PipelineReplicaRegistry,
    PipelineResourceGate,
    PipelineResourceSnapshot,
    PipelineWavefrontSchedule,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPipelineLayout(unittest.TestCase):
    def test_vpp2_maps_two_non_contiguous_chunks_per_rank(self):
        layout = PipelineLayout.build(
            num_hidden_layers=40,
            physical_size=4,
            virtual_stages=2,
        )

        self.assertEqual(
            layout.layer_ids_for_rank(0),
            tuple(range(0, 5)) + tuple(range(20, 25)),
        )
        self.assertEqual(
            layout.layer_ids_for_rank(3),
            tuple(range(15, 20)) + tuple(range(35, 40)),
        )
        self.assertEqual(
            [stage.physical_rank for stage in layout.stages],
            [0, 1, 2, 3, 0, 1, 2, 3],
        )

    def test_digest_is_stable_and_changes_with_partition(self):
        default = PipelineLayout.build(40, 4, 2)
        same = PipelineLayout.build(40, 4, 2)
        uneven = PipelineLayout.build(
            40,
            4,
            2,
            partition=(4, 5, 5, 6, 5, 5, 5, 5),
        )

        self.assertEqual(default.digest, same.digest)
        self.assertNotEqual(default.digest, uneven.digest)

    def test_rejects_incomplete_partition_manifest(self):
        with self.assertRaisesRegex(ValueError, "does not match logical pipeline size"):
            PipelineLayout.build(40, 4, 2, partition=(10, 10, 10, 10))

        with self.assertRaisesRegex(ValueError, "does not match num_hidden_layers"):
            PipelineLayout.build(40, 4, 2, partition=(5,) * 7 + (4,))

    def test_vpp2_wavefront_schedule(self):
        schedule = PipelineWavefrontSchedule.build(4, 2)
        expected = (
            ((0, 0), None, None, None),
            ((1, 0), (0, 1), None, None),
            ((2, 0), (1, 1), (0, 2), None),
            ((3, 0), (2, 1), (1, 2), (0, 3)),
            ((0, 4), (3, 1), (2, 2), (1, 3)),
            ((1, 4), (0, 5), (3, 2), (2, 3)),
            ((2, 4), (1, 5), (0, 6), (3, 3)),
            ((3, 4), (2, 5), (1, 6), (0, 7)),
            (None, (3, 5), (2, 6), (1, 7)),
            (None, None, (3, 6), (2, 7)),
            (None, None, None, (3, 7)),
        )

        for tick, expected_actions in enumerate(expected):
            actual = []
            for rank in range(4):
                action = schedule.action(tick, rank)
                actual.append(
                    None if action is None else (action.batch_seq, action.stage_id)
                )
            self.assertEqual(tuple(actual), expected_actions)

    def test_vpp2_wavefront_preserves_dependencies_and_reuses_slots(self):
        schedule = PipelineWavefrontSchedule.build(4, 2)
        stage_ticks = {}
        first_batch_seq = 8
        for tick in range(schedule.num_ticks):
            for rank in range(4):
                action = schedule.action(tick, rank, first_batch_seq)
                if action is None:
                    continue
                self.assertEqual(action.physical_rank, action.stage_id % 4)
                self.assertEqual(action.slot_id, action.batch_seq % 4)
                stage_ticks[(action.batch_seq, action.stage_id)] = tick

        self.assertEqual(list(schedule.batch_seqs(first_batch_seq)), list(range(8, 12)))
        for batch_seq in schedule.batch_seqs(first_batch_seq):
            ticks = [stage_ticks[(batch_seq, stage_id)] for stage_id in range(8)]
            batch_offset = batch_seq - first_batch_seq
            self.assertEqual(ticks, list(range(batch_offset, batch_offset + 8)))
            self.assertEqual(
                schedule.completion_batch_seq(ticks[-1], first_batch_seq),
                batch_seq,
            )

    def test_vpp2_wavefront_rejects_unsupported_shapes(self):
        with self.assertRaisesRegex(ValueError, "VPP2 only"):
            PipelineWavefrontSchedule.build(4, 3)
        with self.assertRaisesRegex(ValueError, "one slot per PP rank"):
            PipelineWavefrontSchedule.build(4, 2, wave_size=8)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            PipelineWavefrontSchedule.build(4, 2).batch_seqs(-1)
        with self.assertRaisesRegex(ValueError, "must be in"):
            PipelineWavefrontSchedule.build(4, 2).action(-1, 0)

    def test_ready_queue_continuously_reuses_completed_slots(self):
        schedule = PipelineReadyQueueSchedule(4, 2, max_inflight=4)
        next_batch_seq = 0
        for _ in range(4):
            schedule.admit(next_batch_seq)
            next_batch_seq += 1

        actual = []
        for tick in range(12):
            actions = schedule.actions(tick)
            actual.append(
                tuple(
                    None if action is None else (action.batch_seq, action.stage_id)
                    for action in actions
                )
            )
            completed = schedule.complete(actions)
            for _ in completed:
                self.assertTrue(schedule.can_admit(next_batch_seq))
                schedule.admit(next_batch_seq)
                next_batch_seq += 1

        self.assertEqual(
            actual,
            [
                ((0, 0), None, None, None),
                ((1, 0), (0, 1), None, None),
                ((2, 0), (1, 1), (0, 2), None),
                ((3, 0), (2, 1), (1, 2), (0, 3)),
                ((0, 4), (3, 1), (2, 2), (1, 3)),
                ((1, 4), (0, 5), (3, 2), (2, 3)),
                ((2, 4), (1, 5), (0, 6), (3, 3)),
                ((3, 4), (2, 5), (1, 6), (0, 7)),
                ((4, 0), (3, 5), (2, 6), (1, 7)),
                ((5, 0), (4, 1), (3, 6), (2, 7)),
                ((6, 0), (5, 1), (4, 2), (3, 7)),
                ((7, 0), (6, 1), (5, 2), (4, 3)),
            ],
        )
        self.assertEqual(schedule.inflight_count, 4)

    def test_ready_queue_applies_backpressure_and_decode_priority(self):
        schedule = PipelineReadyQueueSchedule(4, 2, max_inflight=5)
        for batch_seq in range(5):
            schedule.admit(batch_seq)
        self.assertFalse(schedule.can_admit(5))

        for tick in range(4):
            schedule.complete(schedule.actions(tick))

        actions = schedule.actions(4)
        self.assertEqual(
            (actions[0].batch_seq, actions[0].stage_id),
            (0, 4),
        )

    def test_ready_queue_keeps_busy_rank_reserved_during_delayed_completion(self):
        schedule = PipelineReadyQueueSchedule(4, 2, max_inflight=4)
        for batch_seq in range(3):
            schedule.admit(batch_seq)

        first = schedule.actions(0)
        schedule.complete(first)

        second = schedule.actions(1)
        self.assertEqual(
            [
                (action.batch_seq, action.stage_id) if action else None
                for action in second
            ],
            [(1, 0), (0, 1), None, None],
        )

        # Rank 0 is still computing B1/S0 while rank 1 completes B0/S1.
        schedule.complete((None, second[1], None, None))
        third = schedule.actions(2)

        self.assertIsNone(third[0])
        self.assertEqual((third[2].batch_seq, third[2].stage_id), (0, 2))

    def test_ready_queue_skips_transport_blocked_decode_for_ready_encode(self):
        schedule = PipelineReadyQueueSchedule(4, 2, max_inflight=5)
        for batch_seq in range(5):
            schedule.admit(batch_seq)

        for tick in range(4):
            schedule.complete(schedule.actions(tick))

        blocked = {(0, 4)}
        actions = schedule.actions(
            4,
            is_ready=lambda _rank, batch_seq, stage_id: (
                (
                    batch_seq,
                    stage_id,
                )
                not in blocked
            ),
        )

        self.assertEqual((actions[0].batch_seq, actions[0].stage_id), (4, 0))
        schedule.complete(actions)

        actions = schedule.actions(5)
        self.assertEqual((actions[0].batch_seq, actions[0].stage_id), (0, 4))

    def test_control_envelope_round_trip_and_forward(self):
        envelope = PipelineControlEnvelope(
            protocol_version=1,
            runtime_epoch=11,
            layout_digest="layout",
            kind=PipelineControlKind.ADMIT,
            source_rank=0,
            batch_seq=8,
            generation=2,
            slot_id=0,
            payload={"requests": ("r0",)},
        )

        decoded = PipelineControlEnvelope.from_dict(envelope.to_dict())

        self.assertEqual(decoded, envelope)
        self.assertEqual(decoded.forwarded().hops, 1)
        self.assertEqual(decoded.forwarded().payload, envelope.payload)

    def test_rank_local_scheduler_drains_cancelled_batch_before_retire(self):
        schedule = PipelineRankSchedule(
            physical_rank=3,
            physical_size=4,
            virtual_stages=2,
            max_inflight=4,
        )
        schedule.admit(0)
        self.assertIsNone(schedule.next_action(0))

        schedule.mark_ready(0, 3)
        first = schedule.next_action(1)
        first_transition = schedule.complete(first)
        self.assertTrue(first_transition.first_pass_done)
        self.assertEqual(first_transition.successor_stage_id, 4)

        schedule.cancel(0)
        schedule.mark_ready(0, 7)
        final = schedule.next_action(2)
        final_transition = schedule.complete(final)
        self.assertTrue(final_transition.batch_complete)
        self.assertTrue(final_transition.cancelled)
        self.assertFalse(schedule.can_admit(4))

        schedule.retire(0)
        self.assertTrue(schedule.can_admit(4))

    def test_rank_local_schedulers_follow_transport_events_through_all_stages(self):
        schedules = [
            PipelineRankSchedule(
                physical_rank=rank,
                physical_size=4,
                virtual_stages=2,
                max_inflight=4,
            )
            for rank in range(4)
        ]
        for schedule in schedules:
            schedule.admit(0)

        visited = []
        for stage_id in range(8):
            owner = stage_id % 4
            action = schedules[owner].next_action(stage_id)
            self.assertIsNotNone(action)
            self.assertEqual(action.stage_id, stage_id)
            visited.append((owner, stage_id))
            transition = schedules[owner].complete(action)
            if transition.successor_stage_id is not None:
                next_owner = transition.successor_stage_id % 4
                schedules[next_owner].mark_ready(
                    transition.batch_seq,
                    transition.successor_stage_id,
                )

        self.assertEqual(
            visited,
            [(0, 0), (1, 1), (2, 2), (3, 3), (0, 4), (1, 5), (2, 6), (3, 7)],
        )
        for rank, schedule in enumerate(schedules):
            if rank != 3:
                schedule.mark_completed(0)
            schedule.retire(0)
            self.assertEqual(schedule.inflight_count, 0)

    def test_resource_gate_uses_hysteresis_and_all_rank_watermarks(self):
        gate = PipelineResourceGate(
            ranks=range(4),
            activation_low_watermark=100,
            activation_high_watermark=200,
            max_pending_sends=4,
        )
        for rank in range(4):
            gate.update(
                rank,
                PipelineResourceSnapshot(
                    request_slots=2,
                    kv_tokens=8192,
                    activation_bytes=50,
                    pending_sends=1,
                ),
            )
        self.assertTrue(gate.can_admit(1, 4096, 100))

        gate.update(
            2,
            PipelineResourceSnapshot(
                request_slots=2,
                kv_tokens=8192,
                activation_bytes=200,
                pending_sends=1,
            ),
        )
        self.assertFalse(gate.can_admit(1, 4096, 0))

        gate.update(
            2,
            PipelineResourceSnapshot(
                request_slots=2,
                kv_tokens=8192,
                activation_bytes=150,
                pending_sends=1,
            ),
        )
        self.assertFalse(gate.can_admit(1, 4096, 0))

        gate.update(
            2,
            PipelineResourceSnapshot(
                request_slots=2,
                kv_tokens=8192,
                activation_bytes=100,
                pending_sends=1,
            ),
        )
        self.assertTrue(gate.can_admit(1, 4096, 0))

    def test_prefix_registry_commits_only_common_materialized_frontier(self):
        registry = PipelinePrefixRegistry()
        entry = registry.plan(
            rid="r0",
            request_generation=3,
            residency_generation=7,
            start=0,
            end=4096,
            required_stages=range(8),
        )
        self.assertTrue(entry.locked)

        for stage_id in range(7):
            registry.mark_materialized("r0", 3, stage_id, 4096)
        self.assertEqual(entry.materialized_end, 0)
        with self.assertRaisesRegex(RuntimeError, "materialized"):
            registry.commit("r0", 3, 4096)

        registry.mark_materialized("r0", 3, 7, 4096)
        registry.commit("r0", 3, 4096)
        self.assertEqual(entry.committed_end, 4096)
        self.assertFalse(entry.locked)

        registry.invalidate_residency("r0", 3, residency_generation=8)
        self.assertEqual(entry.planned_end, 4096)
        self.assertEqual(entry.materialized_end, 4096)

    def test_replica_registry_requires_contiguous_generation_matched_ranges(self):
        registry = PipelineReplicaRegistry()
        replicas = [
            PipelineReplicaIdentity("prefix", 20, 1, rank) for rank in (1, 2, 3)
        ]
        for identity in replicas:
            registry.install(identity, 4, 0, 2048)
        registry.install(replicas[0], 4, 2048, 4096)

        self.assertEqual(registry.common_boundary(replicas, 4), 2048)
        self.assertEqual(
            registry.missing_range(replicas[1], 4, 4096),
            (2048, 4096),
        )
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            registry.install(replicas[1], 4, 1024, 4096)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            registry.install(replicas[1], 3, 2048, 4096)

        registry.lock(replicas[0], 4096)
        with self.assertRaisesRegex(RuntimeError, "locked"):
            registry.evict(replicas[0], 4)
        registry.unlock(replicas[0], 4096)
        self.assertEqual(registry.evict(replicas[0], 4), 5)
        self.assertEqual(registry.common_boundary(replicas, 4), 0)

    def test_replica_registry_keeps_replica_locked_for_other_batch_owner(self):
        registry = PipelineReplicaRegistry()
        identity = PipelineReplicaIdentity("prefix", 20, 1, 1)
        registry.install(identity, 4, 0, 4096)
        registry.lock(identity, 4096, owner_id=7)
        registry.lock(identity, 2048, owner_id=8)

        registry.unlock(identity, 4096, owner_id=7)

        with self.assertRaisesRegex(RuntimeError, "locked"):
            registry.evict(identity, 4)
        registry.unlock(identity, 2048, owner_id=8)
        self.assertEqual(registry.evict(identity, 4), 5)


if __name__ == "__main__":
    unittest.main()
