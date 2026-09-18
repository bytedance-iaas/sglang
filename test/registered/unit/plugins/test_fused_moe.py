import unittest
from unittest.mock import patch

from sglang.srt.plugins import fused_moe
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFusedMoERegistration(unittest.TestCase):
    def setUp(self):
        self.registry = patch.dict(fused_moe._factories, {}, clear=True)
        self.registry.start()
        self.loader = patch("sglang.srt.plugins.load_plugins")
        self.loader.start()

    def tearDown(self):
        self.loader.stop()
        self.registry.stop()

    def test_scoped_registration_and_duplicate(self):
        factory = lambda layer: layer
        fused_moe.register_fused_moe_backend("megamoe", (9, 0), factory)
        fused_moe.register_fused_moe_backend("megamoe", (9, 0), factory)
        layer = object()
        self.assertIs(
            fused_moe.create_fused_moe_backend("megamoe", (9, 0), layer), layer
        )
        self.assertIsNone(fused_moe.create_fused_moe_backend("megamoe", (10, 0), layer))
        self.assertIsNone(fused_moe.create_fused_moe_backend("none", (9, 0), layer))
        with self.assertRaisesRegex(ValueError, "already registered"):
            fused_moe.register_fused_moe_backend("megamoe", (9, 0), lambda _: None)

    def test_missing_sm90_plugin_fails_before_weight_repacking(self):
        with self.assertRaisesRegex(RuntimeError, "iaas-kernels"):
            fused_moe.create_fused_moe_backend("megamoe", (9, 0), object())


if __name__ == "__main__":
    unittest.main()
