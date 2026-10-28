"""Unit tests for srt/mem_cache/storage/eic/eic_env_config.py (U1-U15)."""

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from sglang.srt.mem_cache.storage.eic.eic_env_config import (
    DEFAULT_FIXED_CONFIG,
    FIXED_CONFIG_SPEC,
    SDK_DEFAULT_FLAGS,
    SDK_DEFAULTS_VERSION,
    SDK_FLAG_AUDIT,
    EicDefaultFlagFile,
    EicResolvedConfig,
    apply_eic_network_interfaces,
    load_eic_config,
    load_fixed_config,
    parse_eic_remote_url,
    parse_eic_switch,
    parse_network_interface_names,
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
            flag = os.path.join(tmp, "eic_flag_file")
            with open(flag, "w") as fout:
                fout.write("--eic_client_kv_get_check_crc_type=0\n")
            path = _write_yaml(
                tmp,
                {
                    "remote_url": "eic://192.168.30.90:12500;192.168.30.91:12500",
                    "eic_instance_id": VALID_ID,
                    "eic_namespace": "",
                    "eic_thread_num": 2,
                    "eic_trans_type": 0,
                    "eic_flag_file": flag,
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
        self.assertEqual(cfg.fixed["eic_flag_file"], flag)
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

    # --- File-existence semantics: missing falls back, present-but-broken fails ---

    def test_u32_corrupt_yaml_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.yaml")
            with open(path, "w") as fout:
                fout.write("remote_url: [unclosed\n")
            with self.assertRaisesRegex(ValueError, "corrupt YAML"):
                load_eic_config(cli_enabled=True, environ={}, config_file=path)

    def test_u33_unreadable_yaml_raises(self):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root bypasses permission bits")
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(
                tmp, {"remote_url": VALID_URL, "eic_instance_id": VALID_ID}
            )
            os.chmod(path, 0o000)
            try:
                with self.assertRaisesRegex(ValueError, "not readable"):
                    load_eic_config(cli_enabled=True, environ={}, config_file=path)
            finally:
                os.chmod(path, 0o644)

    def test_u34_pyyaml_missing_with_present_file_fails(self):
        # A mounted explicit config must not be silently dropped when
        # pyyaml is unavailable; only a missing file falls back to built-ins.
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(
                tmp, {"remote_url": VALID_URL, "eic_instance_id": VALID_ID}
            )
            with mock.patch.dict(sys.modules, {"yaml": None}):
                with self.assertRaisesRegex(ValueError, "pyyaml is unavailable"):
                    load_eic_config(cli_enabled=True, environ={}, config_file=path)

    # --- Explicit eic_flag_file validation (independent of NetworkInterfaceNames) ---

    def test_u35_explicit_missing_flag_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(
                tmp,
                {
                    "remote_url": VALID_URL,
                    "eic_instance_id": VALID_ID,
                    "eic_flag_file": "/nonexistent/eic.flags",
                },
            )
            with self.assertRaises(FileNotFoundError):
                load_fixed_config(environ={}, config_file=path)

    def test_u36_explicit_flag_file_not_a_regular_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "adir"))
            path = _write_yaml(
                tmp,
                {
                    "remote_url": VALID_URL,
                    "eic_instance_id": VALID_ID,
                    "eic_flag_file": os.path.join(tmp, "adir"),
                },
            )
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                load_fixed_config(environ={}, config_file=path)

    def test_u37_explicit_unreadable_flag_file_raises(self):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root bypasses permission bits")
        with tempfile.TemporaryDirectory() as tmp:
            flag = os.path.join(tmp, "f.flags")
            with open(flag, "w") as fout:
                fout.write("--x=1\n")
            os.chmod(flag, 0o000)
            path = _write_yaml(
                tmp,
                {
                    "remote_url": VALID_URL,
                    "eic_instance_id": VALID_ID,
                    "eic_flag_file": flag,
                },
            )
            try:
                with self.assertRaisesRegex(ValueError, "not readable"):
                    load_fixed_config(environ={}, config_file=path)
            finally:
                os.chmod(flag, 0o644)

    # --- Centralized registry single-source; SDK gflags emitted nowhere ---

    def test_u38_spec_is_single_source_for_defaults(self):
        for key, entry in FIXED_CONFIG_SPEC.items():
            self.assertEqual(DEFAULT_FIXED_CONFIG[key], entry[0])
            self.assertIn(entry[1], ("both", "pool", "hiradix"))
            self.assertIn(
                entry[3],
                (
                    "live-init",
                    "live-local",
                    "cache-tier",
                    "log-only",
                    "forced-off",
                    "unconsumed",
                ),
            )

    def test_u39_audit_carries_no_emission_marker(self):
        # Audit view is specification reference only; it must not carry an
        # emitted column, so it cannot disagree with SDK_DEFAULT_FLAGS.
        for _flag, row in SDK_FLAG_AUDIT.items():
            self.assertEqual(len(row), 3)  # (value, static status, note)

    # --- Versioned SDK defaults: conservative emitted subset, deferred audit-only ---

    def test_u40_sdk_default_flags_are_versioned_and_safe(self):
        self.assertTrue(SDK_DEFAULTS_VERSION)
        emitted = {k: v for k, v in SDK_DEFAULT_FLAGS.items() if v[2]}
        # Explicit conservative Set-GDR off, distinct from Python set-direct.
        self.assertEqual(emitted["--eic_client_kv_set_enable_gdr"][0], "false")
        # Unknown multi_nic is never emitted; the thread knob is never mapped.
        for flag, (_v, _s, emitted_flag, _n) in SDK_DEFAULT_FLAGS.items():
            self.assertNotIn("multi_nic", flag)
            self.assertNotIn("default_io_thread_num", flag)
        # Performance / memory tunables stay deferred: absent from emitted view.
        deferred = {
            "--eic_client_kv_get_enable_gdr",
            "--eic_use_polling_mode",
            "--eic_use_byte_express",
            "--eic_client_dram_mempool_limit_bytes",
            "--eic_client_split_kv_slice_size_byte",
        }
        for flag in deferred:
            self.assertNotIn(flag, SDK_DEFAULT_FLAGS)

    def test_u41_default_flag_file_carries_emitted_defaults_only(self):
        owner = EicDefaultFlagFile()
        try:
            with open(owner.path, encoding="utf-8") as fin:
                content = fin.read()
            for flag, (_v, _s, emitted, _n) in SDK_DEFAULT_FLAGS.items():
                if emitted:
                    self.assertIn(f"{flag}=", content)
                else:
                    self.assertNotIn(flag, content)
            # gflags may trim trailing whitespace; never emit trailing spaces,
            # and never emit leading whitespace on a flag line.
            for line in content.splitlines():
                self.assertEqual(line, line.strip())
                self.assertFalse(line.startswith((" ", "\t")))
            # No hardcoded NIC list, no transport, no unknown multi_nic.
            self.assertNotIn("eth1;eth2;eth3;eth4", content)
            self.assertNotIn("multi_nic", content)
            self.assertNotIn("transport", content)
        finally:
            owner.close()

    def test_u42_flag_file_path_space_hash_accepted_at_load(self):
        # Space and '#' are legal inside a flagfile path; gflags takes the
        # whole rest of line, no truncation, no blind quoting.
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "my dir#x.flags")
            with open(good, "w"):
                pass
            path = _write_yaml(
                tmp,
                {
                    "remote_url": VALID_URL,
                    "eic_instance_id": VALID_ID,
                    "eic_flag_file": good,
                },
            )
            fixed = load_fixed_config(environ={}, config_file=path)
            self.assertEqual(fixed["eic_flag_file"], os.path.abspath(good))

    def test_u43_yaml_directory_raises_not_silent_fallback(self):
        # A directory presented as the config file must raise, never fall back
        # to built-ins as if the file were absent.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                ValueError, "directory|not a regular|not accessible"
            ):
                load_eic_config(cli_enabled=True, environ={}, config_file=tmp)

    def test_u44_explicit_flag_file_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(
                tmp,
                {
                    "remote_url": VALID_URL,
                    "eic_instance_id": VALID_ID,
                    "eic_flag_file": tmp,
                },
            )
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                load_fixed_config(environ={}, config_file=path)

    def test_u45_falsy_explicit_flag_file_rejected_not_default(self):
        # 0 / False / "" / whitespace are invalid explicit values, not a
        # missing default; they must raise instead of hitting the default path.
        with tempfile.TemporaryDirectory() as tmp:
            for bad in (False, 0, "", "   "):
                with self.subTest(bad=bad):
                    path = _write_yaml(
                        tmp,
                        {
                            "remote_url": VALID_URL,
                            "eic_instance_id": VALID_ID,
                            "eic_flag_file": bad,
                        },
                    )
                    with self.assertRaisesRegex(ValueError, "non-empty path"):
                        load_fixed_config(environ={}, config_file=path)

    def test_u46_dangling_symlink_explicit_flag_file_raises_not_fallback(self):
        # A dangling symlink on an explicit path is present-but-broken: raise
        # ValueError; never silently fall back to the default path.
        with tempfile.TemporaryDirectory() as tmp:
            link = os.path.join(tmp, "broken.flags")
            os.symlink(os.path.join(tmp, "missing.target"), link)
            path = _write_yaml(
                tmp,
                {
                    "remote_url": VALID_URL,
                    "eic_instance_id": VALID_ID,
                    "eic_flag_file": link,
                },
            )
            with self.assertRaisesRegex(ValueError, "dangling symlink"):
                load_fixed_config(environ={}, config_file=path)

    def test_u47_dangling_symlink_default_auto_flag_raises_not_fallback(self):
        # A dangling symlink at the auto-discovered default path must NOT fall
        # back to "no flag file"; it raises as present-but-broken.
        import sglang.srt.mem_cache.storage.eic.eic_env_config as cfg_mod

        with tempfile.TemporaryDirectory() as tmp:
            link = os.path.join(tmp, "default.flags")
            os.symlink(os.path.join(tmp, "missing.target"), link)
            with mock.patch.object(cfg_mod, "DEFAULT_FLAG_FILE", link):
                with self.assertRaisesRegex(ValueError, "dangling symlink"):
                    load_fixed_config(environ={}, config_file="/nonexistent/eic.yaml")

    def test_u48_dangling_symlink_yaml_path_raises_not_empty(self):
        # A dangling symlink presented as the yaml config must raise, never
        # fall back to {} as if the file were absent.
        with tempfile.TemporaryDirectory() as tmp:
            link = os.path.join(tmp, "config.yaml")
            os.symlink(os.path.join(tmp, "missing.target"), link)
            with self.assertRaisesRegex(ValueError, "dangling symlink"):
                load_eic_config(cli_enabled=True, environ={}, config_file=link)


class TestEicNetworkInterfaces(unittest.TestCase):
    def setUp(self):
        self.env = {
            "EnableEicCache": "True",
            "EicRemoteUrl": VALID_URL,
            "EicInstanceId": VALID_ID,
        }
        self.interfaces = mock.patch(
            "sglang.srt.mem_cache.storage.eic.eic_env_config.socket.if_nameindex",
            return_value=[(1, "lo"), (2, "eth0"), (3, "eth1")],
        )
        self.interfaces.start()
        self.addCleanup(self.interfaces.stop)

    def test_comma_list_converts_to_sdk_semicolons(self):
        self.assertEqual(parse_network_interface_names("eth0,eth1"), "eth0;eth1")

    def test_single_name(self):
        self.assertEqual(parse_network_interface_names("eth0"), "eth0")

    def test_trim_and_deduplicate_preserves_order(self):
        self.assertEqual(
            parse_network_interface_names(" eth1, eth0 ,eth1 "), "eth1;eth0"
        )

    def test_common_device_names(self):
        self.assertEqual(
            parse_network_interface_names("enp65s0f0,eth0.100,bond-0"),
            "enp65s0f0;eth0.100;bond-0",
        )

    def test_reject_empty_items_and_sdk_delimiters(self):
        for raw in ("", " ", ",eth0", "eth0,", "eth0,,eth1", "eth0;eth1"):
            with self.subTest(raw=raw), self.assertRaisesRegex(
                ValueError, "NetworkInterfaceNames"
            ):
                parse_network_interface_names(raw)

    def test_reject_injection_and_invalid_device_names(self):
        for raw in (
            None,
            "eth 0",
            "eth0\n--flag=true",
            "../eth0",
            "eth0:1",
            "-bad",
            "eth0\x00",
            "x" * 16,
            "eth0@peer",
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_network_interface_names(raw)

    def test_trailing_control_chars_newline_tab_cr_rejected(self):
        # Trailing \n/\t/\r are control characters, not trim-able spaces:
        # reject instead of accepting or trimming them.
        for raw in ("eth0\n", "eth0\t", "eth0\r", "eth0,eth1\n\t"):
            with self.subTest(raw=raw), self.assertRaisesRegex(
                ValueError, "control character"
            ):
                parse_network_interface_names(raw)

    def test_comma_then_newline_rejected(self):
        with self.assertRaisesRegex(ValueError, "control character"):
            parse_network_interface_names("eth0,\neth1")

    def test_embedded_control_characters_rejected(self):
        for raw in ("eth0\x00", "eth0\x07", "eth0\x0e,eth1", "eth0\x7f"):
            with self.subTest(raw=raw), self.assertRaisesRegex(
                ValueError, "control character"
            ):
                parse_network_interface_names(raw)

    def test_load_from_env_without_yaml(self):
        cfg = load_eic_config(
            environ=dict(self.env, NetworkInterfaceNames="eth0,eth1"),
            config_file="/nonexistent/eic.yaml",
        )
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.network_interface_names, "eth0;eth1")
        self.assertFalse(cfg.fixed["enable_gpu_nic_affinity"])

    def test_unset_is_backward_compatible(self):
        cfg = load_eic_config(environ=self.env, config_file="/nonexistent/eic.yaml")
        self.assertIsNone(cfg.network_interface_names)
        option = SimpleNamespace(multi_net_local_interface_names="legacy")
        self.assertIsNone(apply_eic_network_interfaces(option, cfg))
        self.assertEqual(option.multi_net_local_interface_names, "legacy")

    def test_invalid_env_fails_instead_of_yaml_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(
                tmp, {"remote_url": VALID_URL, "eic_instance_id": VALID_ID}
            )
            with self.assertRaisesRegex(ValueError, "NetworkInterfaceNames"):
                load_eic_config(
                    cli_enabled=True,
                    environ={"NetworkInterfaceNames": ""},
                    config_file=path,
                )

    def test_disabled_eic_ignores_network_env(self):
        cfg = load_eic_config(
            environ={"EnableEicCache": "False", "NetworkInterfaceNames": ""},
            config_file="/nonexistent/eic.yaml",
        )
        self.assertFalse(cfg.enabled)
        self.assertIsNone(cfg.network_interface_names)

    def test_sets_init_option_without_flag_file_or_affinity(self):
        cfg = EicResolvedConfig(network_interface_names="eth0;eth1")
        option = SimpleNamespace()
        self.assertIsNone(apply_eic_network_interfaces(option, cfg))
        self.assertEqual(option.multi_net_local_interface_names, "eth0;eth1")
        self.assertFalse(hasattr(option, "flag_file"))

    def test_rejects_interface_missing_in_runtime_namespace(self):
        cfg = EicResolvedConfig(network_interface_names="eth0;eth9")
        option = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "missing interfaces: eth9"):
            apply_eic_network_interfaces(option, cfg)
        self.assertFalse(hasattr(option, "multi_net_local_interface_names"))

    def test_flag_wrapper_preserves_source_and_overrides_last(self):
        with tempfile.TemporaryDirectory(prefix="eic config ") as tmp:
            source = Path(tmp) / "source.flags"
            content = "--eic_client_rpc_timeout_in_ms=5000\n--eic_client_multi_net_local_interface_names=old\n"
            source.write_text(content, encoding="utf-8")
            cfg = EicResolvedConfig(
                fixed={"eic_flag_file": str(source)},
                network_interface_names="eth0;eth1",
            )
            option = SimpleNamespace(multi_net_local_interface_names="legacy")
            owner = apply_eic_network_interfaces(option, cfg)
            self.addCleanup(owner.close)
            self.assertEqual(option.multi_net_local_interface_names, "eth0;eth1")
            self.assertEqual(source.read_text(encoding="utf-8"), content)
            self.assertEqual(
                Path(option.flag_file).read_text(encoding="utf-8"),
                f"--flagfile={source}\n--eic_client_multi_net_local_interface_names=eth0;eth1\n",
            )
            self.assertEqual(stat.S_IMODE(os.stat(option.flag_file).st_mode), 0o600)
            self.assertEqual(
                stat.S_IMODE(os.stat(Path(option.flag_file).parent).st_mode), 0o700
            )
            owner.close()
            self.assertFalse(Path(option.flag_file).exists())
            self.assertTrue(source.exists())
            owner.close()

    def test_missing_external_flag_file_is_not_silently_dropped(self):
        cfg = EicResolvedConfig(
            fixed={"eic_flag_file": "/nonexistent/eic.flags"},
            network_interface_names="eth0",
        )
        with self.assertRaises(FileNotFoundError):
            apply_eic_network_interfaces(SimpleNamespace(), cfg)

    def test_nic_wrapper_refuses_comma_in_nested_flag_path(self):
        # gflags --flagfile=a,b splits ',' into a list; refuse it fail-closed
        # on the NIC wrapper only (the legacy direct SDK path is untouched).
        cfg = EicResolvedConfig(
            fixed={"eic_flag_file": "/tmp/a,b.flags"},
            network_interface_names="eth0",
        )
        with self.assertRaisesRegex(ValueError, "','"):
            apply_eic_network_interfaces(SimpleNamespace(), cfg)

    def test_fork_child_does_not_remove_parent_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.flags"
            source.write_text("", encoding="utf-8")
            owner = apply_eic_network_interfaces(
                SimpleNamespace(),
                EicResolvedConfig(
                    fixed={"eic_flag_file": str(source)}, network_interface_names="eth0"
                ),
            )
            self.addCleanup(owner.close)
            with mock.patch("os.getpid", return_value=owner._owner_pid + 1):
                owner.close()
            self.assertTrue(Path(owner.path).exists())


if __name__ == "__main__":
    unittest.main()
