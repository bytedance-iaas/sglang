import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.mooncake.registration import normalize_buffer_regions
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestMooncakeBufferRegistration(unittest.TestCase):
    def manager(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[1000],
            kv_data_lens=[64],
            aux_data_ptrs=[2000],
            aux_data_lens=[64],
            state_data_ptrs=[[300, 300, 350]],
            state_data_lens=[[100, 100, 100]],
        )
        manager.engine = Mock()
        manager.engine.batch_register.return_value = 0
        manager.engine.batch_deregister.return_value = 0
        return manager

    def test_compact_alias_and_partial_overlap_preserve_logical_layout(self):
        manager = self.manager()
        before = copy.deepcopy(vars(manager.kv_args))
        manager.register_buffer_to_engine()
        self.assertEqual(
            manager.engine.batch_register.call_args_list,
            [call([1000], [64]), call([2000], [64]), call([300], [150])],
        )
        self.assertEqual(vars(manager.kv_args), before)
        manager.deregister_buffer_to_engine()
        self.assertEqual(
            manager.engine.batch_deregister.call_args_list,
            [call([300]), call([2000]), call([1000])],
        )
        manager.deregister_buffer_to_engine()
        self.assertEqual(manager.engine.batch_deregister.call_count, 3)
        manager.register_buffer_to_engine()
        self.assertEqual(manager.engine.batch_register.call_count, 6)

    def test_overlap_across_categories_has_exact_union(self):
        batches = [
            ("kv", [100], [100]),
            ("aux", [300], [100]),
            ("state[0]", [150, 300, 200], [200, 50, 30]),
        ]
        self.assertEqual(
            normalize_buffer_regions(batches),
            [("kv+aux+state[0]", [100], [300])],
        )

    def test_adjacent_allocations_remain_separate(self):
        self.assertEqual(
            normalize_buffer_regions([("kv", [100, 200], [100, 100])]),
            [("kv", [100, 200], [100, 100])],
        )

    def test_same_base_with_different_lengths_covers_largest_view(self):
        self.assertEqual(
            normalize_buffer_regions([("state[0]", [100, 100, 100], [20, 80, 40])]),
            [("state[0]", [100], [80])],
        )

    def test_invalid_ranges_are_rejected_before_registration(self):
        for ptrs, lengths in (([100], []), ([0], [100]), ([100], [-1]), ([-1], [1])):
            with self.subTest(ptrs=ptrs, lengths=lengths):
                manager = self.manager()
                manager.kv_args.state_data_ptrs = [ptrs]
                manager.kv_args.state_data_lens = [lengths]
                with self.assertRaises(ValueError):
                    manager.register_buffer_to_engine()
                manager.engine.batch_register.assert_not_called()

    def test_zero_row_placeholders_preserve_layer_positions(self):
        manager = self.manager()
        manager.kv_args.kv_data_ptrs = [0]
        manager.kv_args.kv_data_lens = [0]
        manager.kv_args.state_data_ptrs = [[0, 300, 300]]
        manager.kv_args.state_data_lens = [[0, 100, 100]]
        before = copy.deepcopy(vars(manager.kv_args))
        manager.register_buffer_to_engine()
        self.assertEqual(
            manager.engine.batch_register.call_args_list,
            [call([2000], [64]), call([300], [100])],
        )
        self.assertEqual(vars(manager.kv_args), before)
        manager.deregister_buffer_to_engine()
        self.assertEqual(
            manager.engine.batch_deregister.call_args_list,
            [call([300]), call([2000])],
        )

    def test_state_component_count_mismatch_is_rejected(self):
        manager = self.manager()
        manager.kv_args.state_data_lens = []
        with self.assertRaisesRegex(ValueError, "state component counts"):
            manager.register_buffer_to_engine()
        manager.engine.batch_register.assert_not_called()

    def test_nonzero_status_stops_initialization_and_rolls_back_owned_batches(self):
        manager = self.manager()
        manager.engine.batch_register.side_effect = [0, -7]
        with self.assertRaisesRegex(RuntimeError, "category=aux batch=2/3: ret=-7"):
            manager.register_buffer_to_engine()
        self.assertEqual(manager.engine.batch_register.call_count, 2)
        manager.engine.batch_deregister.assert_called_once_with([1000])
        with self.assertRaisesRegex(
            RuntimeError, "previously failed; restart required"
        ):
            manager.register_buffer_to_engine()
        self.assertEqual(manager.engine.batch_register.call_count, 2)

    def test_backend_exception_keeps_cause_and_buffer_context(self):
        manager = self.manager()
        failure = OSError("injected RDMA registration failure")
        manager.engine.batch_register.side_effect = failure
        with self.assertRaisesRegex(RuntimeError, "category=kv batch=1/3") as ctx:
            manager.register_buffer_to_engine()
        self.assertIs(ctx.exception.__cause__, failure)

    def test_registration_cannot_be_repeated_before_deregistration(self):
        manager = self.manager()
        manager.register_buffer_to_engine()
        with self.assertRaisesRegex(RuntimeError, "already registered"):
            manager.register_buffer_to_engine()
        self.assertEqual(manager.engine.batch_register.call_count, 3)

    def test_deregistration_failure_requires_restart_without_repeating_batches(self):
        manager = self.manager()
        manager.register_buffer_to_engine()
        manager.engine.batch_deregister.side_effect = [-7, 0, 0]
        with self.assertRaisesRegex(
            RuntimeError, r"category=state\[0\] batch=3/3: ret=-7"
        ):
            manager.deregister_buffer_to_engine()
        manager.engine.batch_deregister.side_effect = None
        with self.assertRaisesRegex(RuntimeError, "deregistration previously failed"):
            manager.deregister_buffer_to_engine()
        with self.assertRaisesRegex(RuntimeError, "registration previously failed"):
            manager.register_buffer_to_engine()
        self.assertEqual(
            manager.engine.batch_deregister.call_args_list,
            [call([300]), call([2000]), call([1000])],
        )

    def test_staging_error_has_registration_context(self):
        manager = self.manager()
        manager.engine.batch_register.return_value = -7
        with self.assertRaisesRegex(RuntimeError, "category=staging batch=1/1: ret=-7"):
            manager._register_buffer_batch([100], [20], "staging", 1, 1)


class TestMooncakeBatchRegisterFailure(unittest.TestCase):
    def test_sdk_status_and_exception_propagate(self):
        engine = MooncakeTransferEngine.__new__(MooncakeTransferEngine)
        engine.engine = Mock()
        engine.engine.batch_register_memory.return_value = 0
        self.assertEqual(engine.batch_register([100], [20]), 0)
        engine.engine.batch_register_memory.return_value = -7
        with self.assertRaisesRegex(RuntimeError, "registration failed.*ret=-7"):
            engine.batch_register([100], [20])
        failure = OSError("injected SDK error")
        engine.engine.batch_register_memory.side_effect = failure
        with self.assertRaisesRegex(RuntimeError, "registration failed") as ctx:
            engine.batch_register([100], [20])
        self.assertIs(ctx.exception.__cause__, failure)

    def test_missing_sdk_api_retains_upgrade_message(self):
        engine = MooncakeTransferEngine.__new__(MooncakeTransferEngine)
        engine.engine = SimpleNamespace()
        with self.assertRaisesRegex(RuntimeError, "requires a newer version") as ctx:
            engine.batch_register([100], [20])
        self.assertIsInstance(ctx.exception.__cause__, AttributeError)


if __name__ == "__main__":
    unittest.main()
