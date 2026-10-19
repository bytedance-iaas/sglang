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
        # The SDK endpoint string is the bare host:port form proven by
        # the previous yaml path; an http scheme is input syntax only.
        self.assertEqual(endpoints, ["1.2.3.4:12500"])
        self.assertEqual(endpoint, "1.2.3.4:12500")

    def test_u6_multi_endpoint_with_spaces(self):
        endpoints, endpoint = parse_eic_remote_url(
            "eic:// http://1.2.3.4:12500 ; https://5.6.7.8:12501 ; http://9.10.11.12:12502 "
        )
        self.assertEqual(
            endpoints,
            [
                "1.2.3.4:12500",
                "5.6.7.8:12501",
                "9.10.11.12:12502",
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
        self.assertEqual(cfg.endpoints, ["1.2.3.4:12500"])
        self.assertEqual(cfg.fixed["eic_log_dir"], "/sgl-workspace/log")

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

    def test_u18_port_out_of_range_raises(self):
        for raw in ("eic://1.2.3.4:0", "eic://1.2.3.4:99999"):
            with self.assertRaisesRegex(ValueError, "1..65535"):
                parse_eic_remote_url(raw)

    def test_u19_whitespace_or_userinfo_endpoint_raises(self):
        for raw in ("eic://1.2.3.4 :12500", "eic://user@1.2.3.4:12500"):
            with self.assertRaises(ValueError):
                parse_eic_remote_url(raw)

    def test_u20_non_mapping_yaml_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.yaml")
            with open(path, "w") as fout:
                fout.write("- a\n- b\n")
            with self.assertRaisesRegex(ValueError, "must contain a mapping"):
                load_eic_config(cli_enabled=True, environ={}, config_file=path)

    def test_u21_instance_id_with_whitespace_raises(self):
        env = {
            "EnableEicCache": "True",
            "EicRemoteUrl": VALID_URL,
            "EicInstanceId": "eic x",
        }
        with self.assertRaisesRegex(ValueError, "must not contain whitespace"):
            load_eic_config(environ=env, config_file="/nonexistent/eic.yaml")

    def test_u22_yaml_null_counts_as_missing_for_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(
                tmp,
                {
                    "remote_url": VALID_URL,
                    "eic_instance_id": VALID_ID,
                    "eic_trans_type": None,
                },
            )
            cfg = load_eic_config(cli_enabled=True, environ={}, config_file=path)
        self.assertEqual(cfg.fixed["eic_trans_type"], 3)

    def test_u23_near_true_switch_warns_but_stays_disabled(self):
        env = {"EnableEicCache": "true"}
        with self.assertLogs(
            "sglang.srt.mem_cache.storage.eic.eic_env_config", level="WARNING"
        ):
            cfg = load_eic_config(
                cli_enabled=True, environ=env, config_file="/nonexistent/eic.yaml"
            )
        self.assertFalse(cfg.enabled)

    def test_u24_empty_env_url_does_not_fall_back_to_yaml(self):
        # A present-but-empty env value is an explicit injection error:
        # falling back would silently connect to the yaml cluster.
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(
                tmp, {"remote_url": VALID_URL, "eic_instance_id": VALID_ID}
            )
            env = {
                "EnableEicCache": "True",
                "EicRemoteUrl": "",
                "EicInstanceId": VALID_ID,
            }
            with self.assertRaisesRegex(ValueError, "EicRemoteUrl is required"):
                load_eic_config(environ=env, config_file=path)

    def test_u25_empty_env_id_does_not_fall_back_to_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(
                tmp, {"remote_url": VALID_URL, "eic_instance_id": VALID_ID}
            )
            env = {
                "EnableEicCache": "True",
                "EicRemoteUrl": VALID_URL,
                "EicInstanceId": "  ",
            }
            with self.assertRaisesRegex(ValueError, "EicInstanceId is required"):
                load_eic_config(environ=env, config_file=path)

    def test_u15_env_explicit_off_overrides_cli(self):
        env = {"EnableEicCache": "False"}
        cfg = load_eic_config(
            cli_enabled=True, environ=env, config_file="/nonexistent/eic.yaml"
        )
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.source, "env")

    def test_u26_env_only_defaults_match_rdma_template(self):
        # No yaml mounted: built-in defaults must equal the servingkit
        # SGLang + RDMA ConfigMap template, except NIC affinity.
        env = {
            "EnableEicCache": "True",
            "EicRemoteUrl": VALID_URL,
            "EicInstanceId": VALID_ID,
        }
        cfg = load_eic_config(environ=env, config_file="/nonexistent/eic.yaml")
        expected = {
            "eic_namespace": "",
            "eic_thread_num": 2,
            "eic_log_dir": "/sgl-workspace/log",
            "eic_log_level": 2,
            "save_decode_cache": False,
            "enable_kvset_gpu_direct": False,
            "eic_direct_backup": False,
            "eic_direct_writeback": False,
            "gdr_bounce_buffer_size": 2147483648,
            "load_back_check": True,
            "load_back_threshold": 100,
            "enable_gpu_nic_affinity": False,
        }
        for key, value in expected.items():
            self.assertEqual(cfg.fixed[key], value, key)
        # No flag file mounted: RDMA would run GDR reads with CRC on and
        # crash, so the default falls back to GDR transport (3).
        self.assertIsNone(cfg.fixed["eic_flag_file"])
        self.assertEqual(cfg.fixed["eic_trans_type"], 3)

    def test_u30_flag_file_keeps_rdma_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            flag = os.path.join(tmp, "eic_flag_file")
            with open(flag, "w") as fout:
                fout.write("--eic_client_kv_get_check_crc_type=0\n")
            path = _write_yaml(
                tmp,
                {
                    "remote_url": VALID_URL,
                    "eic_instance_id": VALID_ID,
                    "eic_flag_file": flag,
                },
            )
            cfg = load_eic_config(cli_enabled=True, environ={}, config_file=path)
        self.assertEqual(cfg.fixed["eic_trans_type"], 2)

    def test_u31_explicit_yaml_trans_type_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(
                tmp,
                {
                    "remote_url": VALID_URL,
                    "eic_instance_id": VALID_ID,
                    "eic_trans_type": 2,
                },
            )
            cfg = load_eic_config(cli_enabled=True, environ={}, config_file=path)
        self.assertEqual(cfg.fixed["eic_trans_type"], 2)

    def test_u27_legacy_yaml_full_template_still_loads(self):
        # Old ConfigMap (TCP template) keeps working; its values win.
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(
                tmp,
                {
                    "remote_url": "eic://192.168.30.90:12500;192.168.30.91:12500",
                    "eic_instance_id": VALID_ID,
                    "eic_namespace": "",
                    "eic_thread_num": 2,
                    "eic_trans_type": 0,
                    "eic_flag_file": "/sgl-workspace/config/eic_flag_file",
                    "save_decode_cache": False,
                    "enable_gpu_nic_affinity": False,
                    "load_back_check": True,
                    "load_remote_threshold": 400,
                },
            )
            cfg = load_eic_config(cli_enabled=True, environ={}, config_file=path)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.source, "yaml")
        self.assertEqual(cfg.endpoint, "192.168.30.90:12500;192.168.30.91:12500")
        self.assertEqual(cfg.fixed["eic_trans_type"], 0)
        self.assertEqual(
            cfg.fixed["eic_flag_file"], "/sgl-workspace/config/eic_flag_file"
        )
        self.assertEqual(cfg.fixed["load_back_threshold"], 400)

    def test_u28_new_threshold_key_wins_over_legacy_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(
                tmp,
                {
                    "remote_url": VALID_URL,
                    "eic_instance_id": VALID_ID,
                    "load_remote_threshold": 400,
                    "load_back_threshold": 64,
                },
            )
            cfg = load_eic_config(cli_enabled=True, environ={}, config_file=path)
        self.assertEqual(cfg.fixed["load_back_threshold"], 64)

    def test_u29_env_overrides_legacy_yaml_endpoint_and_id(self):
        # Mixed deployment: old ConfigMap mounted, new env injected.
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(
                tmp,
                {
                    "remote_url": "eic://10.0.0.1:12500",
                    "eic_instance_id": "old-instance",
                    "eic_trans_type": 0,
                },
            )
            env = {
                "EnableEicCache": "True",
                "EicRemoteUrl": "eic://192.168.0.53:10000;192.168.0.54:10000",
                "EicInstanceId": VALID_ID,
            }
            cfg = load_eic_config(environ=env, config_file=path)
        self.assertEqual(cfg.source, "env")
        self.assertEqual(cfg.endpoints, ["192.168.0.53:10000", "192.168.0.54:10000"])
        self.assertEqual(cfg.instance_id, VALID_ID)
        # Fixed parameters still come from the mounted yaml.
        self.assertEqual(cfg.fixed["eic_trans_type"], 0)


if __name__ == "__main__":
    unittest.main()
