import dataclasses
import unittest
from array import array

from sglang.srt.speculative.paired_prefix_contract import (
    PairedPrefixContext,
    PairedPrefixFallback,
    PairedPrefixIdentity,
    PairedPrefixRegistry,
    PairedPrefixState,
    guarded_prompt_radix_key,
    plan_paired_prompt_prefix,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Event:
    def __init__(self, ready=False):
        self.ready = ready

    def query(self):
        return self.ready


class TestPairedPrefixPlanner(unittest.TestCase):
    def setUp(self):
        self.identity = PairedPrefixIdentity(
            rid="req-a",
            request_generation=3,
            owner_epoch=7,
            room_id="room-2",
            endpoint_id=("decode", 0),
            token_digest="sha256:prompt-a",
            target_cache_version=11,
            draft_cache_version=13,
            target_layout_version="target-dsa-v1",
            draft_layout_version="draft-mtp-v1",
        )
        self.context = PairedPrefixContext(True, True, True, True)

    def state(self, **changes):
        state = PairedPrefixState(
            identity=self.identity,
            prompt_len=100,
            target_committed_len=120,
            draft_committed_len=90,
            target_ack_generation=5,
            draft_ack_generation=7,
            target_ack_complete=True,
            draft_ack_complete=True,
            target_page_generation=9,
            draft_page_generation=12,
        )
        return dataclasses.replace(state, **changes)

    def test_guarded_key_partitions_prompt_tail_namespace_and_producer(self):
        prompt = array("q", [1, 2, 3, 4])
        agents = ("source-0", "source-1")
        key = guarded_prompt_radix_key("adapter", prompt, agents)
        self.assertEqual(key, guarded_prompt_radix_key("adapter", prompt[:], agents))
        self.assertNotEqual(
            key, guarded_prompt_radix_key("adapter", array("q", [1, 2, 3, 9]), agents)
        )
        self.assertNotEqual(key, guarded_prompt_radix_key("other", prompt, agents))
        self.assertNotEqual(
            key, guarded_prompt_radix_key("adapter", prompt, ("source-0", "restarted"))
        )

    def test_frontier_is_common_committed_prompt_only(self):
        plan = plan_paired_prompt_prefix(
            source=self.state(),
            destination=self.state(draft_committed_len=95),
            context=self.context,
        )
        self.assertTrue(plan.accepted)
        self.assertEqual(plan.reusable_tokens, 90)

        plan = plan_paired_prompt_prefix(
            source=self.state(draft_committed_len=140),
            destination=self.state(target_committed_len=130, draft_committed_len=150),
            context=self.context,
        )
        self.assertEqual(plan.reusable_tokens, 100)

    def test_rid_reuse_owner_room_endpoint_and_layout_fail_closed(self):
        fields = {
            "request_generation": 4,
            "owner_epoch": 8,
            "room_id": "room-3",
            "endpoint_id": ("decode", 1),
            "token_digest": "sha256:other",
            "target_cache_version": 12,
            "draft_cache_version": 14,
            "target_layout_version": "target-dsa-v2",
            "draft_layout_version": "draft-mtp-v2",
        }
        for field, value in fields.items():
            with self.subTest(field=field):
                other = dataclasses.replace(
                    self.identity,
                    **{field: value},
                )
                plan = plan_paired_prompt_prefix(
                    source=self.state(),
                    destination=self.state(identity=other),
                    context=self.context,
                )
                self.assertEqual(
                    plan.fallback_reason, PairedPrefixFallback.IDENTITY_MISMATCH
                )

    def test_ack_and_page_generations_are_atomic_pair_gates(self):
        plan = plan_paired_prompt_prefix(
            source=self.state(),
            destination=self.state(draft_ack_complete=False),
            context=self.context,
        )
        self.assertEqual(plan.fallback_reason, PairedPrefixFallback.ACK_INCOMPLETE)

        plan = plan_paired_prompt_prefix(
            source=self.state(),
            destination=self.state(draft_ack_generation=8),
            context=self.context,
        )
        self.assertEqual(
            plan.fallback_reason, PairedPrefixFallback.ACK_GENERATION_MISMATCH
        )

        plan = plan_paired_prompt_prefix(
            source=self.state(),
            destination=self.state(draft_page_generation=13),
            context=self.context,
        )
        self.assertEqual(
            plan.fallback_reason, PairedPrefixFallback.PAGE_GENERATION_MISMATCH
        )

    def test_only_decode_paired_prompt_path_is_supported(self):
        contexts = (
            PairedPrefixContext(False, True, True, True),
            PairedPrefixContext(True, False, True, True),
            PairedPrefixContext(True, True, False, True),
            PairedPrefixContext(True, True, True, False),
        )
        for context in contexts:
            with self.subTest(context=context):
                plan = plan_paired_prompt_prefix(
                    source=self.state(), destination=self.state(), context=context
                )
                expected = (
                    PairedPrefixFallback.FEATURE_DISABLED
                    if not context.enabled
                    else PairedPrefixFallback.UNSUPPORTED_PATH
                )
                self.assertEqual(plan.fallback_reason, expected)


class TestPairedPrefixRegistry(unittest.TestCase):
    def identity(self, rid="req-a", generation=1):
        return PairedPrefixIdentity(
            rid=rid,
            request_generation=generation,
            owner_epoch=2,
            room_id="room",
            endpoint_id="decode-0",
            token_digest="sha256:prompt",
            target_cache_version=3,
            draft_cache_version=4,
            target_layout_version="target-v1",
            draft_layout_version="draft-v1",
        )

    def test_cancel_waits_for_target_and_draft_users(self):
        registry = PairedPrefixRegistry[str]()
        identity = self.identity()
        target_event = _Event(ready=True)
        draft_event = _Event(ready=False)
        registry.put(
            identity,
            "frontier",
            target_page_generation=6,
            draft_page_generation=8,
        )
        registry.record_use(identity, target_event)
        registry.record_use(identity, draft_event)

        registry.release_request("req-a", 1)
        self.assertIsNone(
            registry.lookup(
                identity,
                target_page_generation=6,
                draft_page_generation=8,
            )
        )
        self.assertEqual(registry.collect_retired(), ())
        self.assertEqual(registry.retired_count, 1)

        draft_event.ready = True
        self.assertEqual(registry.collect_retired(), (identity,))
        self.assertEqual(registry.retired_count, 0)

    def test_rid_reuse_and_other_request_do_not_alias(self):
        registry = PairedPrefixRegistry[str]()
        old = self.identity(generation=1)
        new = self.identity(generation=2)
        other = self.identity(rid="req-b", generation=1)
        registry.put(old, "old", target_page_generation=6, draft_page_generation=8)
        registry.put(other, "other", target_page_generation=6, draft_page_generation=8)

        registry.release_request("req-a", 1)
        self.assertIsNone(
            registry.lookup(old, target_page_generation=6, draft_page_generation=8)
        )
        self.assertEqual(
            registry.lookup(other, target_page_generation=6, draft_page_generation=8),
            "other",
        )
        self.assertEqual(registry.collect_retired(), (old,))
        registry.put(new, "new", target_page_generation=6, draft_page_generation=8)
        self.assertEqual(
            registry.lookup(new, target_page_generation=6, draft_page_generation=8),
            "new",
        )

    def test_page_reuse_generation_invalidates_old_frontier(self):
        registry = PairedPrefixRegistry[str]()
        stale = self.identity("stale")
        current = self.identity("current")
        registry.put(stale, "stale", target_page_generation=5, draft_page_generation=7)
        registry.put(
            current, "current", target_page_generation=6, draft_page_generation=8
        )

        registry.invalidate_page_generations(6, 8)
        self.assertIsNone(
            registry.lookup(stale, target_page_generation=5, draft_page_generation=7)
        )
        self.assertEqual(
            registry.lookup(current, target_page_generation=6, draft_page_generation=8),
            "current",
        )
        self.assertEqual(registry.collect_retired(), (stale,))


if __name__ == "__main__":
    unittest.main()
