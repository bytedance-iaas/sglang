"""Isolate registration policy from RDMA engine initialization for CPU tests."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from sglang.srt.environ import envs

source = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/disaggregation/mooncake/conn.py"
)
klass = next(
    node
    for node in ast.parse(source.read_text()).body
    if isinstance(node, ast.ClassDef) and node.name == "MooncakeKVManager"
)
method = next(
    node
    for node in klass.body
    if isinstance(node, ast.FunctionDef) and node.name == "register_buffer_to_engine"
)
namespace = {"envs": envs}
exec(
    compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace
)
register = namespace["register_buffer_to_engine"]


class RegistrationTests(unittest.TestCase):
    def test_batch_failure_is_reported(self):
        manager = SimpleNamespace(
            _registerable_regions=lambda: [(1, 64), (2, 128)], engine=Mock()
        )
        manager.engine.batch_register.return_value = -1
        with envs.SGLANG_MOONCAKE_REGISTER_SERIAL.override(False):
            with self.assertRaisesRegex(RuntimeError, "ret=-1"):
                register(manager)
        manager.engine.register.assert_not_called()

    def test_serial_registration_uses_checked_single_api(self):
        manager = SimpleNamespace(
            _registerable_regions=lambda: [(1, 64), (2, 128)], engine=Mock()
        )
        manager.engine.register.side_effect = [None, RuntimeError("register failed")]
        with envs.SGLANG_MOONCAKE_REGISTER_SERIAL.override(True):
            with self.assertRaisesRegex(RuntimeError, "register failed"):
                register(manager)
        self.assertEqual(manager.engine.register.call_count, 2)
        manager.engine.batch_register.assert_not_called()

    def test_empty_and_success(self):
        manager = SimpleNamespace(_registerable_regions=lambda: [], engine=Mock())
        with envs.SGLANG_MOONCAKE_REGISTER_SERIAL.override(False):
            register(manager)
            manager.engine.batch_register.assert_not_called()
            manager._registerable_regions = lambda: [(1, 64)]
            manager.engine.batch_register.return_value = 0
            register(manager)
            manager.engine.batch_register.assert_called_once_with([1], [64])


if __name__ == "__main__":
    unittest.main(verbosity=2)
