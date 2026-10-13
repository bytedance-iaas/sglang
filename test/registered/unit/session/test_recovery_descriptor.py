import unittest

from sglang.srt.session.recovery_descriptor import (
    RecoveryDescriptorError,
    RecoveryLifecycle,
    SessionRecoveryDescriptor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _descriptor(**overrides) -> SessionRecoveryDescriptor:
    values = dict(
        tenant_id="tenant-a",
        session_id="session-a",
        session_generation=3,
        slot_id=7,
        slot_generation=11,
        owner_dp_rank=2,
        cut=65_280,
        dsa_kv_valid_cut=65_280,
        dsa_index_valid_cut=65_280,
        dsa_kv_allocated_len=65_280,
        dsa_index_allocated_rows=16_320,
        kda_state_position=65_280,
        scheduler_credited_len=65_280,
        recompute_start=65_280,
        committed_draft_tokens=4,
        committed_bonus_tokens=1,
        uncommitted_tail_start=65_280,
        uncommitted_tail_end=65_284,
        page_size=256,
        kpool_size=4,
        store_complete=True,
        lifecycle=RecoveryLifecycle.LIVE,
    )
    values.update(overrides)
    return SessionRecoveryDescriptor(**values)


def _restore(descriptor: SessionRecoveryDescriptor, **overrides):
    values = dict(
        tenant_id="tenant-a",
        session_id="session-a",
        session_generation=3,
        slot_id=7,
        slot_generation=11,
        prompt_len=65_536,
    )
    values.update(overrides)
    return descriptor.validate_for_restore(**values)


class TestSessionRecoveryDescriptor(CustomTestCase):
    def test_exact_cut_returns_suffix_recompute_interval(self):
        self.assertEqual(_restore(_descriptor()), range(65_280, 65_536))

    def test_page256_and_kpool4_boundary(self):
        descriptor = _descriptor(
            cut=65_279,
            dsa_kv_valid_cut=65_279,
            dsa_index_valid_cut=65_279,
            dsa_kv_allocated_len=65_280,
            dsa_index_allocated_rows=16_320,
            kda_state_position=65_279,
            scheduler_credited_len=65_279,
            recompute_start=65_279,
            uncommitted_tail_start=65_279,
            uncommitted_tail_end=65_283,
        )
        self.assertEqual(descriptor.minimum_index_rows, 16_320)
        self.assertEqual(_restore(descriptor), range(65_279, 65_536))

    def test_last_token_recompute_requires_matching_state_cut(self):
        descriptor = _descriptor(
            scheduler_credited_len=65_279,
            recompute_start=65_279,
            uncommitted_tail_start=65_279,
        )
        with self.assertRaisesRegex(
            RecoveryDescriptorError, "scheduler_credited_len=65279"
        ):
            _restore(descriptor)

    def test_misaligned_kda_state_fails_closed(self):
        with self.assertRaisesRegex(
            RecoveryDescriptorError, "kda_state_position=65279"
        ):
            _restore(_descriptor(kda_state_position=65_279))

    def test_short_dsa_index_fails_closed(self):
        with self.assertRaisesRegex(RecoveryDescriptorError, "needs at least 16320"):
            _restore(_descriptor(dsa_index_allocated_rows=16_319))

    def test_unaligned_dsa_allocation_fails_closed(self):
        with self.assertRaisesRegex(
            RecoveryDescriptorError, "not page_size=256 aligned"
        ):
            _restore(_descriptor(dsa_kv_allocated_len=65_281))

    def test_partial_accept_and_bonus_fit_within_cut(self):
        descriptor = _descriptor(
            committed_draft_tokens=2,
            committed_bonus_tokens=1,
            uncommitted_tail_end=65_285,
        )
        self.assertEqual(_restore(descriptor), range(65_280, 65_536))

    def test_async_store_submission_is_not_completion(self):
        with self.assertRaisesRegex(RecoveryDescriptorError, "has not completed"):
            _restore(
                _descriptor(
                    store_complete=False,
                    lifecycle=RecoveryLifecycle.STORE_PENDING,
                )
            )

    def test_evict_close_and_release_are_not_restorable(self):
        for lifecycle in (
            RecoveryLifecycle.EVICTED,
            RecoveryLifecycle.CLOSED,
            RecoveryLifecycle.RELEASED,
        ):
            with self.subTest(lifecycle=lifecycle):
                with self.assertRaisesRegex(RecoveryDescriptorError, "not live"):
                    _restore(_descriptor(lifecycle=lifecycle))

    def test_tenant_session_generation_and_slot_generation_prevent_leak(self):
        for key, value in (
            ("tenant_id", "tenant-b"),
            ("session_id", "session-b"),
            ("session_generation", 4),
            ("slot_id", 8),
            ("slot_generation", 12),
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                    RecoveryDescriptorError, "identity mismatch"
                ):
                    _restore(_descriptor(), **{key: value})

    def test_prompt_shorter_than_cut_fails_closed(self):
        with self.assertRaisesRegex(RecoveryDescriptorError, "invalid recovery cut"):
            _restore(_descriptor(), prompt_len=65_000)


if __name__ == "__main__":
    unittest.main()
