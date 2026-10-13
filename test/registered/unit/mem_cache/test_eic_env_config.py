"""Unit tests for srt/mem_cache/storage/eic/eic_env_config.py (U1-U15)."""

import os
import tempfile
import unittest

import yaml

from sglang.srt.mem_cache.storage.eic.eic_env_config import (
    load_eic_config,
    parse_eic_remote_url,
    parse_eic_switch,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VALID_URL = "eic://http://1.2.3.4:12500"
VALID_ID = "eicyec229dyxxxxxxxzuu0"


def _write_yaml(directory, data):
    path = os.path.join(directory, "remote-eic.yaml")
    with open(path, "w") as fout:
        yaml.safe_dump(data, fout)
    return path


class TestEicEnvConfig(CustomTestCase):
    def test_u1_switch_true(self):
        self.assertTrue(parse_eic_switch("True"))

    def test_u2_switch_lowercase_is_false(self):
        self.assertFalse(parse_eic_switch("true"))
        self.assertFalse(parse_eic_switch("TRUE"))

    def test_u3_switch_strips_whitespace(self):
        self.assertTrue(parse_eic_switch("  True "))

    def test_u4_switch_other_values_are_false(self):
        for raw in ("1", "", "False", None):
            self.assertFalse(parse_eic_switch(raw))

    def test_u5_single_endpoint(self):
        endpoints, endpoint = parse_eic_remote_url(VALID_URL)
        self.assertEqual(endpoints, ["http://1.2.3.4:12500"])
        self.assertEqual(endpoint, "http://1.2.3.4:12500")

    def test_u6_multi_endpoint_with_spaces(self):
        endpoints, endpoint = parse_eic_remote_url(
            "eic:// http://1.2.3.4:12500 ; https://5.6.7.8:12501 ; http://9.10.11.12:12502 "
        )
        self.assertEqual(
            endpoints,
            [
                "http://1.2.3.4:12500",
                "https://5.6.7.8:12501",
                "http://9.10.11.12:12502",
            ],
        )
        self.assertEqual(endpoint, ";".join(endpoints))

    def test_u7_missing_prefix_raises(self):
        with self.assertRaisesRegex(ValueError, "must start with eic://"):
            parse_eic_remote_url("http://1.2.3.4:12500")

    def test_u8_invalid_endpoint_segment_raises(self):
        for raw in ("eic://1.2.3.4", "eic://http://no-port", "eic://ftp://1.2.3.4:1"):
            with self.assertRaises(ValueError):
                parse_eic_remote_url(raw)

    def test_u9_empty_endpoint_raises(self):
        with self.assertRaisesRegex(ValueError, "no valid endpoint"):
            parse_eic_remote_url("eic:// ; ")

    def test_u10_env_config_complete(self):
        env = {
            "EnableEicCache": "True",
            "EicRemoteUrl": VALID_URL,
            "EicInstanceId": VALID_ID,
        }
        cfg = load_eic_config(environ=env, config_file="/nonexistent/eic.yaml")
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.source, "env")
        self.assertEqual(cfg.instance_id, VALID_ID)
        self.assertEqual(cfg.endpoints, ["http://1.2.3.4:12500"])
        self.assertEqual(cfg.fixed["eic_log_dir"], "/tmp/eic-client-log")

    def test_u11_enabled_without_url_fails(self):
        env = {"EnableEicCache": "True", "EicInstanceId": VALID_ID}
        with self.assertRaisesRegex(ValueError, "EicRemoteUrl is required"):
            load_eic_config(environ=env, config_file="/nonexistent/eic.yaml")

    def test_u12_enabled_without_instance_id_fails(self):
        env = {"EnableEicCache": "True", "EicRemoteUrl": VALID_URL}
        with self.assertRaisesRegex(ValueError, "EicInstanceId is required"):
            load_eic_config(environ=env, config_file="/nonexistent/eic.yaml")

    def test_u13_no_env_falls_back_to_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(
                tmp, {"remote_url": VALID_URL, "eic_instance_id": VALID_ID}
            )
            cfg = load_eic_config(cli_enabled=True, environ={}, config_file=path)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.source, "yaml")
        self.assertEqual(cfg.instance_id, VALID_ID)

    def test_u14_no_env_no_cli_disabled(self):
        cfg = load_eic_config(environ={}, config_file="/nonexistent/eic.yaml")
        self.assertFalse(cfg.enabled)

    def test_u16_previous_bare_host_port_format(self):
        # The servingkit ConfigMap in production writes bare host:port
        # endpoints without an http scheme; that form must keep working.
        endpoints, endpoint = parse_eic_remote_url(
            "eic://192.168.30.90:12500;192.168.30.91:12500;192.168.30.92:12500"
        )
        self.assertEqual(
            endpoints,
            [
                "192.168.30.90:12500",
                "192.168.30.91:12500",
                "192.168.30.92:12500",
            ],
        )
        self.assertEqual(
            endpoint,
            "192.168.30.90:12500;192.168.30.91:12500;192.168.30.92:12500",
        )

    def test_u17_yaml_endpoint_verbatim_and_values_win(self):
        # Previous scheme first: yaml values are handed to the SDK
        # byte-for-byte and override the built-in defaults.
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(
                tmp,
                {
                    "remote_url": "eic://192.168.30.90:12500;192.168.30.91:12500",
                    "eic_instance_id": VALID_ID,
                    "eic_thread_num": 2,
                    "eic_trans_type": 2,
                },
            )
            cfg = load_eic_config(cli_enabled=True, environ={}, config_file=path)
        self.assertEqual(cfg.endpoint, "192.168.30.90:12500;192.168.30.91:12500")
        self.assertEqual(cfg.fixed["eic_thread_num"], 2)
        self.assertEqual(cfg.fixed["eic_trans_type"], 2)

    def test_u15_env_explicit_off_overrides_cli(self):
        env = {"EnableEicCache": "False"}
        cfg = load_eic_config(
            cli_enabled=True, environ=env, config_file="/nonexistent/eic.yaml"
        )
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.source, "env")


if __name__ == "__main__":
    unittest.main()
