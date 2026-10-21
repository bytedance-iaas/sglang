# SPDX-License-Identifier: Apache-2.0
"""Unified EIC configuration resolution.

Priority is fixed: environment variables > CLI arguments > yaml > defaults.
The switch, endpoint, instance id and network interface names may come from
the environment so servingkit does not need to pass EIC arguments. Fixed
parameters (thread count, log level, transport type, namespace, NIC affinity)
are read from yaml or the built-in defaults below.

This module depends only on the standard library (plus optional yaml) so
it can be unit-tested without the EIC SDK, torch or a GPU.
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import shutil
import socket
import tempfile
from dataclasses import dataclass, field
from typing import Mapping, Optional

logger = logging.getLogger(__name__)

ENV_ENABLE_EIC_CACHE = "EnableEicCache"
ENV_EIC_REMOTE_URL = "EicRemoteUrl"
ENV_EIC_INSTANCE_ID = "EicInstanceId"
ENV_NETWORK_INTERFACE_NAMES = "NetworkInterfaceNames"
ENV_REMOTE_EIC_YAML = "REMOTE_EIC_YAML"

EIC_URL_PREFIX = "eic://"
DEFAULT_CONFIG_FILE = "/sgl-workspace/config/remote-eic.yaml"
DEFAULT_FLAG_FILE = "/sgl-workspace/config/eic_flag_file"
RDMA_TRANS_TYPE = 2
NO_FLAG_FILE_TRANS_TYPE = 3


def _is_dangling_symlink(path: str) -> bool:
    """Return True when `path` itself exists as a symlink whose target is gone.

    os.stat(follow_symlinks=True) raises FileNotFoundError both for a truly
    absent path and for a dangling symlink; lstat distinguishes them. A
    dangling link is a present-but-broken path and must never be treated as
    "missing" for a silent fallback. A dangling component higher up in the
    parent chain cannot be classified here and is not specially reported.
    """
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        return False
    try:
        os.stat(path, follow_symlinks=True)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


# Centralized fixed-parameter registry: the single source for built-in
# defaults when no remote-eic.yaml is mounted (the env-only path). A mounted
# yaml overrides every key it sets; fixed parameters have no env channel.
#
# Tuple columns:
#   (built-in default, consumer, bind point, status)
# consumer: "both" = EICKVClient (eic_memory_pool.py) and EICStorage
#           (storage/eic/eic_storage.py); "pool" = EICKVClient only;
#           "hiradix" = EICHiRadixCache / EICPagedHiRadixCache (not a client).
# status:
#   live-init    -> reaches eic.InitOption in the named consumer(s)
#   live-local   -> consumed by local Python logic, never an InitOption field
#   cache-tier   -> consumed by the radix cache tier, not the clients
#   log-only     -> parsed and logged, no runtime effect in 1.5.2
#   forced-off   -> code forces the value regardless of yaml
#   unconsumed   -> read but never acted upon
#
# Values come from the current safe code, not a blind alignment with the
# servingkit ConfigMap template. Template-only SDK gflags (timeouts, CRC,
# GDR, polling, byte_express, multi_nic, mempool limits) have no Python
# injection API in 1.5.2 and are NOT emitted here; they only take effect
# from a mounted eic_flag_file parsed by the SDK itself.
FIXED_CONFIG_SPEC = {
    # --- InitOption fields consumed by BOTH clients ---
    "eic_log_dir": (
        "/sgl-workspace/log",
        "both",
        "InitOption.log_dir",
        "live-init",
    ),
    "eic_log_level": (2, "both", "InitOption.log_level", "live-init"),
    # Built-in 2 (RDMA) is rewritten to 3 (GDR) by load_fixed_config when no
    # flag file is mounted and yaml did not set eic_trans_type explicitly:
    # type 2 without the template flag file keeps KvGet CRC on over device
    # memory and segfaults in CalculateKvValueCRCsPerSlice / OnKvGetGDR.
    "eic_trans_type": (2, "both", "InitOption.transport_type", "live-init"),
    # Explicit yaml path wins; otherwise the standard default path is used
    # only when it exists (see load_fixed_config).
    "eic_flag_file": (None, "both", "InitOption.flag_file", "live-init"),
    "eic_namespace": ("", "both", "Set/Get/ExistOption.ns at IO time", "live-init"),
    # Template RDMA sets this True, but the affinity map depends on node
    # topology; only a node-specific ConfigMap may turn it on.
    "enable_gpu_nic_affinity": (
        False,
        "both",
        "G_EnableGPUNicAffinity NIC mapping",
        "live-local",
    ),
    # --- EICKVClient local behavior (never an InitOption field) ---
    "gdr_bounce_buffer_size": (
        2 * 1024 * 1024 * 1024,
        "pool",
        "local bounce tensor pool sizing",
        "live-local",
    ),
    "eic_direct_backup": (False, "pool", "local backup branch", "live-local"),
    "eic_direct_writeback": (
        False,
        "pool",
        "local writeback branch (ANDed with G_EnableKVGetGPUDirect)",
        "live-local",
    ),
    # Forced off in EICKVClient regardless of yaml (remote write path cannot
    # register CUDA memory; IO-thread segfault risk). EICStorage parses it
    # but does not consume it. This is the Python knob, NOT the SDK
    # kv_set_enable_gdr gflag; the two must not be equated.
    "enable_kvset_gpu_direct": (
        False,
        "both",
        "ignored/forced-off; not SDK kv_set_enable_gdr",
        "forced-off",
    ),
    # --- Radix cache tier (consumed by EICHiRadixCache, not the clients) ---
    # Read and logged in EICHiRadixCache but never used; EICChunkCache
    # hardcodes True and is not bound to this key.
    "save_decode_cache": (False, "hiradix", "read+log only", "unconsumed"),
    "load_back_check": (
        True,
        "hiradix",
        "EICPagedHiRadixCache pre-load existence gate",
        "cache-tier",
    ),
    # L2/PP load ADMISSION threshold; not the removed remote-probe gate.
    "load_back_threshold": (100, "hiradix", "load admission quota", "cache-tier"),
    "eic_probe_backoff_seconds": (30.0, "hiradix", "probe backoff base", "cache-tier"),
    "eic_probe_backoff_max_seconds": (
        300.0,
        "hiradix",
        "probe backoff cap",
        "cache-tier",
    ),
    "eic_probe_backoff_write_failures": (
        3,
        "hiradix",
        "write backoff streak",
        "cache-tier",
    ),
    # Parsed and logged only: the 1.5.2 client has no consumed io-thread
    # knob; 1.7.3's eic_client_default_io_thread_num has no evidence here,
    # so this key is never mapped to an SDK flag.
    "eic_thread_num": (2, "both", "log only", "log-only"),
}

DEFAULT_FIXED_CONFIG = {key: entry[0] for key, entry in FIXED_CONFIG_SPEC.items()}

# Template keys that the code reads under another name. The template key
# applies only when the yaml does not also set the new key.
#
# NOTE: load_remote_threshold (old remote-probe gate, removed in ad7b8d9) is
# NOT semantically equivalent to load_back_threshold (L2/PP admission).
# This alias is a best-effort redirect only; it does not restore the old
# 16384-token probe gate, and full legacy compatibility is not claimed.
LEGACY_KEY_ALIASES = {
    "load_remote_threshold": "load_back_threshold",
}

# Unified SDK gflag registry: the single source for both the emitted defaults
# (SDK_DEFAULT_FLAGS) and the specification reference view (SDK_FLAG_AUDIT).
# Emission state exists ONLY here (third tuple column) and in the derived
# SDK_DEFAULT_FLAGS; SDK_FLAG_AUDIT is reference-only and carries no emitted
# marker, so the two tables can never disagree on emission.
#
# Static status:
#   static-present  -> the 1.5.2 wheel registers/mangles the flag name; this
#                     does NOT imply Init has been verified against it live.
#   unverified      -> statically unconfirmed against 1.5.2; NOT a verdict that
#                      the flag is unsupported or inapplicable.
#   not-found-static-> current static evidence finds no target support; this is
#                     a per-profile choice, not a permanent SDK conclusion.
#
# Process-global caveat (verified against 1.7.3 source only, NOT 1.5.2):
# gflags are process-global and the EicClient singleton re-runs FLAGS on each
# Init. A private generated flag file does NOT give SDK state isolation: once
# the SDK parses a flag_file it writes into process-global FLAGS, whether or
# not Python calls a setter directly. A same-process second client / second
# Init is unverified; isolation across different NIC or default-flag configs is
# NOT guaranteed, and per-client flag isolation must not be assumed.
#
# tuple: (standard/default value, 1.5.2 static status, emitted this round, note)
_SDK_FLAG_REGISTRY = {
    "--eic_client_rpc_timeout_in_ms": (
        "5000",
        "static-present",
        True,
        "RPC timeout",
    ),
    "--eic_client_kv_req_timeout_in_ms": (
        "5000",
        "static-present",
        True,
        "KvReq timeout",
    ),
    "--eic_client_kv_req_rdma_timeout_in_ms": (
        "1000",
        "static-present",
        True,
        "KvReq RDMA timeout",
    ),
    "--eic_client_kv_rpc_rdma_timeout_in_ms": (
        "1000",
        "static-present",
        True,
        "KvRpc RDMA timeout",
    ),
    "--eic_client_enable_slice_task_copy_bypass": (
        "false",
        "static-present",
        True,
        "conservative: slice copy bypass off",
    ),
    "--eic_client_enable_slice_task_copy_thread": (
        "false",
        "static-present",
        True,
        "conservative: slice copy thread off",
    ),
    "--eic_client_enable_kv_set_crc": (
        "false",
        "static-present",
        True,
        "conservative: Set CRC off",
    ),
    "--eic_client_kv_get_check_crc_type": (
        "0",
        "static-present",
        True,
        "CRC check type 0",
    ),
    # Explicit conservative off; this SDK gflag is distinct from the Python
    # enable_kvset_gpu_direct forced-off knob and must not be equated.
    "--eic_client_kv_set_enable_gdr": (
        "false",
        "static-present",
        True,
        "conservative: Set GDR off; not Python set-direct",
    ),
    "--eic_client_kv_get_enable_gdr": (
        "true",
        "unverified",
        False,
        "deferred; no live evidence on 1.5.2",
    ),
    "--eic_use_polling_mode": (
        "true",
        "unverified",
        False,
        "deferred; no live evidence on 1.5.2",
    ),
    "--eic_use_byte_express": (
        "true",
        "unverified",
        False,
        "deferred; no live evidence on 1.5.2",
    ),
    "--eic_enable_multi_nic": (
        "true",
        "not-found-static",
        False,
        "not generated here; per-profile choice, no permanent SDK verdict",
    ),
    "--eic_client_multi_net_local_interface_names": (
        "eth1;eth2;eth3;eth4",
        "unverified",
        False,
        "NIC; set via InitOption.multi_net_local_interface_names when env present",
    ),
    "--eic_client_dram_mempool_limit_bytes": (
        "17179869184",
        "unverified",
        False,
        "deferred; no live evidence on 1.5.2",
    ),
    "--eic_client_split_kv_slice_size_byte": (
        "65536",
        "unverified",
        False,
        "deferred; no live evidence on 1.5.2",
    ),
}

# Emitted-default view: the conservative subset written to the private flag
# file on the missing-file path. External flags are passed through unchanged.
SDK_DEFAULT_FLAGS = {
    flag: (entry[0], entry[1], True, entry[3])
    for flag, entry in _SDK_FLAG_REGISTRY.items()
    if entry[2]
}

# Specification-reference view (standard value, static status, note) only.
# This is NOT an emission-status table; do not read an emitted flag from it.
SDK_FLAG_AUDIT = {
    flag: (entry[0], entry[1], entry[3]) for flag, entry in _SDK_FLAG_REGISTRY.items()
}

# Bump when SDK_DEFAULT_FLAGS changes; recorded for audit/rollback.
SDK_DEFAULTS_VERSION = "eic-eic-defaults/v1"

# Versioned, auditable SDK defaults written to a private flag file and passed
# to InitOption.flag_file ONLY on the missing-file path (no mounted external
# flag_file). They never modify, append to, or mix with an existing external
# file; on the external path the NIC env override remains the only change.
# No hardcoded NIC list is written: without NetworkInterfaceNames the existing
# NIC selection / GPU affinity is untouched. Only flags statically present in
# the eic 1.5.2 wheel AND assessed safe are emitted; performance and memory
# tunables without live evidence stay deferred in SDK_FLAG_AUDIT. "static-present"
# means the wheel registers/mangles the flag name; it does NOT imply Init has
# been verified against it on 1.5.2.
# tuple: (default, 1.5.2 static, emitted this round, note)
SDK_DEFAULT_FLAGS = {
    "--eic_client_rpc_timeout_in_ms": (
        "5000",
        "static-present",
        True,
        "RPC timeout",
    ),
    "--eic_client_kv_req_timeout_in_ms": (
        "5000",
        "static-present",
        True,
        "KvReq timeout",
    ),
    "--eic_client_kv_req_rdma_timeout_in_ms": (
        "1000",
        "static-present",
        True,
        "KvReq RDMA timeout",
    ),
    "--eic_client_kv_rpc_rdma_timeout_in_ms": (
        "1000",
        "static-present",
        True,
        "KvRpc RDMA timeout",
    ),
    "--eic_client_enable_slice_task_copy_bypass": (
        "false",
        "static-present",
        True,
        "conservative: slice copy bypass off",
    ),
    "--eic_client_enable_slice_task_copy_thread": (
        "false",
        "static-present",
        True,
        "conservative: slice copy thread off",
    ),
    "--eic_client_enable_kv_set_crc": (
        "false",
        "static-present",
        True,
        "conservative: Set CRC off",
    ),
    "--eic_client_kv_get_check_crc_type": (
        "0",
        "static-present",
        True,
        "CRC check type 0",
    ),
    # Explicit conservative off; this SDK gflag is distinct from the Python
    # enable_kvset_gpu_direct forced-off knob and must not be equated.
    "--eic_client_kv_set_enable_gdr": (
        "false",
        "static-present",
        True,
        "conservative: Set GDR off; not Python set-direct",
    ),
}

# Bump when SDK_DEFAULT_FLAGS changes; recorded for audit/rollback.
SDK_DEFAULTS_VERSION = "eic-eic-defaults/v1"


class EicDefaultFlagFile:
    """Own a private flag file carrying SDK_DEFAULT_FLAGS (missing-file path).

    The directory is removed at process exit; a forked child never deletes a
    parent's directory.
    """

    def __init__(self):
        self._owner_pid = os.getpid()
        self._directory = tempfile.mkdtemp(prefix="sglang-eic-defaults-")
        self.path = os.path.join(self._directory, "flags")
        try:
            fd, pending = tempfile.mkstemp(dir=self._directory)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                for flag, (
                    _value,
                    _static,
                    emitted,
                    _note,
                ) in SDK_DEFAULT_FLAGS.items():
                    if emitted:
                        stream.write(f"{flag}={_value}\n")
            os.replace(pending, self.path)
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def close(self):
        if os.getpid() == self._owner_pid:
            shutil.rmtree(self._directory, ignore_errors=True)


@dataclass
class EicResolvedConfig:
    enabled: bool = False
    instance_id: str = ""
    endpoint: str = ""
    endpoints: list = field(default_factory=list)
    source: str = "default"
    fixed: dict = field(default_factory=dict)
    network_interface_names: Optional[str] = None


def parse_network_interface_names(raw: str) -> str:
    """Convert a comma-separated interface list to the SDK's semicolon form."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            "NetworkInterfaceNames must be a non-empty comma-separated list"
        )
    # Reject every ASCII/Unicode control character before trimming: \x00-\x1f
    # and \x7f (including \t, \n, \r) are never legal in an interface name and
    # must not reach the name alphabet check. Normal spaces are still trimmed
    # by `.strip()` below.
    for ch in raw:
        code = ord(ch)
        if code <= 0x1F or code == 0x7F:
            raise ValueError("NetworkInterfaceNames contains a control character")
    names = []
    for value in raw.split(","):
        name = value.strip()
        # Linux interface names have at most 15 bytes. Keep the accepted
        # alphabet explicit so newlines and SDK delimiters cannot enter flags.
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,14}", name):
            raise ValueError(
                "NetworkInterfaceNames must contain interface names separated by "
                "commas, with no empty items or embedded whitespace"
            )
        if name not in names:
            names.append(name)
    return ";".join(names)


class EicNetworkFlagFile:
    """Own a private flag-file wrapper without changing the user's file."""

    def __init__(self, source: str, names: str):
        self._owner_pid = os.getpid()
        self._directory = tempfile.mkdtemp(prefix="sglang-eic-network-")
        self.path = os.path.join(self._directory, "flags")
        try:
            fd, pending = tempfile.mkstemp(dir=self._directory)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                # gflags processes the source first, then the explicit override.
                stream.write(f"--flagfile={source}\n")
                stream.write(f"--eic_client_multi_net_local_interface_names={names}\n")
            os.replace(pending, self.path)
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def close(self):
        # A forked child must not remove a parent's flag file.
        if os.getpid() == self._owner_pid:
            shutil.rmtree(self._directory, ignore_errors=True)


def apply_eic_network_interfaces(init_option, resolved: EicResolvedConfig):
    """Apply explicit env NICs before Client.init; return an optional file owner.

    No env override means no changes and no files. The caller retains the
    returned owner for the client's lifetime. A mounted source file is included
    before the NIC override because the SDK parses flags after InitOption.
    """
    names = resolved.network_interface_names
    if names is None:
        return None
    try:
        available = {name for _, name in socket.if_nameindex()}
    except OSError as exc:
        raise ValueError(
            "Cannot inspect NetworkInterfaceNames in this process"
        ) from exc
    missing = [name for name in names.split(";") if name not in available]
    if missing:
        raise ValueError(
            f"NetworkInterfaceNames contains missing interfaces: {', '.join(missing)}"
        )
    source = resolved.fixed.get("eic_flag_file")
    owner = None
    if source is not None:
        # gflags --flagfile=a,b splits ',' into a flag list; refuse it fail-closed
        # on the NIC wrapper only (the legacy direct SDK path is untouched).
        # Checked before existence so a comma path fails on the list rule.
        if "," in source:
            raise ValueError(
                "eic_flag_file path contains ','; gflags splits --flagfile on "
                "comma into a list, so the nested path is refused"
            )
        # load_fixed_config already validated; re-check defensively so a
        # missing/unreadable user config never drops its flags silently.
        source = _validate_flag_file_path(source, explicit=True)
        owner = EicNetworkFlagFile(source, names)
    try:
        init_option.multi_net_local_interface_names = names
        if owner is not None:
            init_option.flag_file = owner.path
    except BaseException:
        if owner is not None:
            owner.close()
        raise
    logger.info("EIC NetworkInterfaceNames applied: %s", names)
    return owner


def parse_eic_switch(raw) -> bool:
    """Strict switch rule: stripped value must equal 'True' exactly."""
    if raw is None:
        return False
    return str(raw).strip() == "True"


def parse_eic_remote_url(raw) -> tuple[list, str]:
    """Parse an EicRemoteUrl value into (endpoint list, SDK endpoint string).

    Both values are normalized to the bare host:port form that the
    previous yaml path handed to the SDK verbatim, joined with ';' for
    multiple endpoints. An http:// or https:// scheme on an endpoint is
    accepted as input syntax and stripped, so every input form reaches
    the SDK in the one shape already proven in production.
    """
    text = "" if raw is None else str(raw).strip()
    if not text:
        raise ValueError("EicRemoteUrl is required when EnableEicCache=True")
    if not text.startswith(EIC_URL_PREFIX):
        raise ValueError("EicRemoteUrl must start with eic://")
    body = text[len(EIC_URL_PREFIX) :]
    parts = [part.strip() for part in body.split(";")]
    parts = [part for part in parts if part]
    if not parts:
        raise ValueError("no valid endpoint in EicRemoteUrl")
    endpoints = [_validate_endpoint(part) for part in parts]
    return endpoints, ";".join(endpoints)


def _validate_endpoint(endpoint: str) -> str:
    # Previous servingkit yaml writes bare host:port endpoints, e.g.
    # eic://192.168.30.90:12500;192.168.30.91:12500 -- that form must keep
    # working. http:// and https:// forms are also accepted; any other
    # scheme is rejected.
    if "://" in endpoint:
        for scheme in ("http://", "https://"):
            if endpoint.startswith(scheme):
                host_port = endpoint[len(scheme) :]
                break
        else:
            raise ValueError(
                f"invalid endpoint in EicRemoteUrl: {endpoint!r} "
                "must start with http:// or https://"
            )
    else:
        host_port = endpoint
    if any(ch.isspace() for ch in endpoint) or "@" in host_port or "/" in host_port:
        raise ValueError(
            f"invalid endpoint in EicRemoteUrl: {endpoint!r} "
            "must be in host:port form without whitespace"
        )
    host, sep, port = host_port.rpartition(":")
    if not sep or not host or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError(
            f"invalid endpoint in EicRemoteUrl: {endpoint!r} "
            "must be in host:port form with a port in 1..65535"
        )
    return host_port


def _validate_flag_file_path(path, *, explicit: bool) -> str:
    """Validate an eic_flag_file path and return its absolute form.

    explicit=True: the path came from the operator's yaml, so a missing path
    is an explicit error -- it must NOT silently fall back to the standard
    auto-discovered path.
    explicit=False: the standard default path was auto-discovered; missing
    simply means "no flag file", but a present-but-broken path (directory,
    permission denied, or unreadable) still raises.

    Symlinks are resolved (follow_symlinks=True): a broken symlink on an
    explicit path raises FileNotFoundError; it never becomes a silent fallback.
    Only a true FileNotFoundError (genuinely absent) may skip validation;
    permission, directory, stat and IO errors always raise.
    """
    raw = os.fspath(path)
    if any(ch in raw for ch in "\r\n\x00"):
        raise ValueError("eic_flag_file contains an invalid path")
    abspath = os.path.abspath(raw)
    try:
        st = os.stat(abspath, follow_symlinks=True)
    except FileNotFoundError:
        if _is_dangling_symlink(abspath):
            raise ValueError(f"eic_flag_file is a dangling symlink: {abspath}")
        if explicit:
            raise FileNotFoundError(
                f"eic_flag_file does not exist: {abspath}"
            ) from None
        # Auto-discover path absent: there is simply no flag file.
        return None
    except PermissionError as exc:
        raise ValueError(f"eic_flag_file is not readable: {abspath}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"eic_flag_file cannot be accessed: {abspath}: {exc}") from exc
    import stat as _stat

    if not _stat.S_ISREG(st.st_mode):
        raise ValueError(f"eic_flag_file is not a regular file: {abspath}")
    try:
        with open(abspath, "rb"):
            pass
    except PermissionError as exc:
        raise ValueError(f"eic_flag_file is not readable: {abspath}: {exc}") from exc
    return abspath


def get_eic_config_file_path(environ: Optional[Mapping] = None) -> str:
    env = os.environ if environ is None else environ
    return env.get(ENV_REMOTE_EIC_YAML) or DEFAULT_CONFIG_FILE


def load_yaml_config(
    environ: Optional[Mapping] = None, config_file: Optional[str] = None
) -> dict:
    """Return the yaml config dict, or {} when the file does not exist."""
    path = config_file or get_eic_config_file_path(environ)
    # Only a genuinely absent file (FileNotFoundError) may fall back to {};
    # permission, directory, broken-symlink and stat errors must raise rather
    # than be swallowed into a silent built-in fallback.
    try:
        st = os.stat(path, follow_symlinks=True)
    except FileNotFoundError:
        if _is_dangling_symlink(path):
            raise ValueError(f"EIC config file {path} is a dangling symlink")
        return {}
    except PermissionError as exc:
        raise ValueError(f"EIC config file {path} is not accessible: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"failed to stat EIC config file {path}: {exc}") from exc
    import stat as _stat

    if _stat.S_ISDIR(st.st_mode):
        raise ValueError(f"EIC config file {path} is a directory, expected a file")
    try:
        import yaml
    except ImportError as exc:
        # A present explicit config must not be silently dropped: without
        # pyyaml it cannot be parsed, so fail instead of falling back to
        # built-ins (which would hide the mounted ConfigMap).
        raise ValueError(
            f"EIC config file {path} is present but pyyaml is unavailable; "
            "install pyyaml or remove the file"
        ) from exc
    try:
        with open(path, "r", encoding="utf-8") as fin:
            data = yaml.safe_load(fin)
    except PermissionError as exc:
        # Present-but-unreadable is an explicit error, not a fallback:
        # silently using built-ins would hide a mounted config that the
        # operator expected to apply.
        raise ValueError(f"EIC config file {path} is not readable: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"EIC config file {path} is corrupt YAML: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"failed to read EIC config file {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"EIC config file {path} must contain a mapping, "
            f"got {type(data).__name__}"
        )
    return data


def load_fixed_config(
    environ: Optional[Mapping] = None, config_file: Optional[str] = None
) -> dict:
    """Fixed parameters: yaml values over the built-in defaults.

    An explicit null in yaml counts as missing for a key that has a
    built-in default, so generated templates that write null
    placeholders do not turn into None values at client construction.
    Template keys listed in LEGACY_KEY_ALIASES map to the key the code
    reads. When no flag file is configured and the standard ConfigMap
    path exists, that flag file is used.
    """
    yaml_config = load_yaml_config(environ, config_file)
    fixed = dict(DEFAULT_FIXED_CONFIG)
    fixed.update(yaml_config)
    for old_key, new_key in LEGACY_KEY_ALIASES.items():
        if yaml_config.get(old_key) is not None and new_key not in yaml_config:
            fixed[new_key] = yaml_config[old_key]
    for key, default in DEFAULT_FIXED_CONFIG.items():
        if fixed.get(key) is None and default is not None:
            fixed[key] = default
    # Flag file resolution is unconditional (does not depend on
    # NetworkInterfaceNames): an explicit yaml path is validated and a
    # missing explicit path raises instead of silently falling back; the
    # standard auto-discovered path is used only when it exists, and a
    # present-but-broken standard path also raises.
    explicit_flag = yaml_config.get("eic_flag_file")
    if explicit_flag is None:
        # null explicitly means unset; keep the legacy compatible behavior.
        pass
    elif (
        isinstance(explicit_flag, bool)
        or not isinstance(explicit_flag, str)
        or explicit_flag.strip() == ""
    ):
        # 0 / False / empty string are invalid, not "missing": reject them
        # explicitly instead of treating them as the default path.
        raise ValueError(
            "eic_flag_file must be a non-empty path string; " f"got {explicit_flag!r}"
        )
    else:
        fixed["eic_flag_file"] = _validate_flag_file_path(explicit_flag, explicit=True)

    if not fixed.get("eic_flag_file"):
        try:
            fixed["eic_flag_file"] = _validate_flag_file_path(
                DEFAULT_FLAG_FILE, explicit=False
            )
        except FileNotFoundError:
            fixed["eic_flag_file"] = None
    # RDMA (2) forces the GDR read path on. Without a flag file the client
    # keeps KvGet CRC checks on and computes CRC over device memory, which
    # segfaults (CalculateKvValueCRCsPerSlice). The template always ships
    # trans_type 2 with a flag file that sets check_crc_type=0. The
    # centralized built-in value is 2; when no flag file is resolved and
    # yaml did not set eic_trans_type explicitly, this function rewrites it
    # to GDR (3) here. Both clients receive the final value through
    # resolved.fixed -- identical for EICKVClient and EICStorage -- and
    # neither recomputes or overrides it locally.
    if (
        not fixed.get("eic_flag_file")
        and yaml_config.get("eic_trans_type") is None
        and fixed.get("eic_trans_type") == RDMA_TRANS_TYPE
    ):
        fixed["eic_trans_type"] = NO_FLAG_FILE_TRANS_TYPE
        logger.info(
            "no eic_flag_file: eic_trans_type defaults to %d instead of %d",
            NO_FLAG_FILE_TRANS_TYPE,
            RDMA_TRANS_TYPE,
        )
    return fixed


def load_eic_config(
    cli_enabled: bool = False,
    environ: Optional[Mapping] = None,
    config_file: Optional[str] = None,
) -> EicResolvedConfig:
    """Resolve the effective EIC configuration.

    Environment variables win over CLI arguments, which win over yaml.
    When the switch is off, the endpoint and instance id are not read.
    When it is on, a missing or invalid endpoint or instance id raises
    ValueError instead of silently falling back to a disabled cache.
    """
    env = os.environ if environ is None else environ
    fixed = load_fixed_config(env, config_file)

    switch_from_env = ENV_ENABLE_EIC_CACHE in env
    if switch_from_env:
        enabled = parse_eic_switch(env.get(ENV_ENABLE_EIC_CACHE))
    else:
        enabled = bool(cli_enabled)

    if not enabled:
        source = "env" if switch_from_env else ("cli" if cli_enabled else "default")
        if switch_from_env:
            raw_switch = str(env.get(ENV_ENABLE_EIC_CACHE) or "").strip()
            if raw_switch.lower() in ("true", "1", "yes", "on"):
                logger.warning(
                    "EnableEicCache=%r is treated as disabled: only the "
                    "exact value 'True' enables EIC (case-sensitive)",
                    raw_switch,
                )
        resolved = EicResolvedConfig(enabled=False, source=source, fixed=fixed)
        logger.info("eic config resolved: enabled=False source=%s", source)
        return resolved

    network_interface_names = None
    if ENV_NETWORK_INTERFACE_NAMES in env:
        network_interface_names = parse_network_interface_names(
            env[ENV_NETWORK_INTERFACE_NAMES]
        )

    yaml_config = load_yaml_config(env, config_file)

    # A present-but-empty env value is an explicit injection error, not
    # an absence: parsing it raises instead of silently falling back to
    # a yaml endpoint that may name a different cluster.
    yaml_url = str(yaml_config.get("remote_url") or "").strip()
    if ENV_EIC_REMOTE_URL in env:
        endpoints, endpoint = parse_eic_remote_url(env.get(ENV_EIC_REMOTE_URL))
        url_source = "env"
    elif yaml_url:
        endpoints, endpoint = parse_eic_remote_url(yaml_url)
        url_source = "yaml"
    else:
        raise ValueError("EicRemoteUrl is required when EnableEicCache=True")

    yaml_id = str(yaml_config.get("eic_instance_id") or "").strip()
    if ENV_EIC_INSTANCE_ID in env:
        instance_id = (env.get(ENV_EIC_INSTANCE_ID) or "").strip()
        id_source = "env"
    else:
        instance_id = yaml_id
        id_source = "yaml"
    if not instance_id:
        raise ValueError("EicInstanceId is required when EnableEicCache=True")
    if any(ch.isspace() for ch in instance_id):
        raise ValueError("EicInstanceId must not contain whitespace")

    source = "env" if "env" in (url_source, id_source) else "yaml"
    resolved = EicResolvedConfig(
        enabled=True,
        instance_id=instance_id,
        endpoint=endpoint,
        endpoints=endpoints,
        source=source,
        fixed=fixed,
        network_interface_names=network_interface_names,
    )
    logger.info(
        "eic config resolved: enabled=True source=%s endpoints=%s instance_id=%s",
        source,
        endpoints,
        instance_id,
    )
    return resolved
