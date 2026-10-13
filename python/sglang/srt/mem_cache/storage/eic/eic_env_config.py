# SPDX-License-Identifier: Apache-2.0
"""Unified EIC configuration resolution.

Priority is fixed: environment variables > CLI arguments > yaml > defaults.
The switch, endpoint and instance id may come from the environment so that
servingkit does not need to pass EIC arguments. Fixed parameters (thread
count, log level, transport type, namespace, NIC affinity) stay bound to
the image version and are read from yaml or the built-in defaults below.

This module depends only on the standard library (plus optional yaml) so
it can be unit-tested without the EIC SDK, torch or a GPU.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Mapping, Optional

logger = logging.getLogger(__name__)

ENV_ENABLE_EIC_CACHE = "EnableEicCache"
ENV_EIC_REMOTE_URL = "EicRemoteUrl"
ENV_EIC_INSTANCE_ID = "EicInstanceId"
ENV_REMOTE_EIC_YAML = "REMOTE_EIC_YAML"

EIC_URL_PREFIX = "eic://"
DEFAULT_CONFIG_FILE = "/sgl-workspace/config/remote-eic.yaml"

# Built-in defaults, aligned with the current remote-eic.yaml values.
DEFAULT_FIXED_CONFIG = {
    "eic_log_dir": "/tmp/eic-client-log",
    "eic_log_level": 2,
    "eic_trans_type": 3,
    "eic_thread_num": 1,
    "eic_flag_file": None,
    "eic_namespace": "",
    "enable_gpu_nic_affinity": False,
    "save_decode_cache": True,
    "load_back_threshold": 10,
}


@dataclass
class EicResolvedConfig:
    enabled: bool = False
    instance_id: str = ""
    endpoint: str = ""
    endpoints: list = field(default_factory=list)
    source: str = "default"
    fixed: dict = field(default_factory=dict)


def parse_eic_switch(raw) -> bool:
    """Strict switch rule: stripped value must equal 'True' exactly."""
    if raw is None:
        return False
    return str(raw).strip() == "True"


def parse_eic_remote_url(raw) -> tuple[list, str]:
    """Parse an EicRemoteUrl value into (endpoint list, SDK endpoint string).

    The SDK endpoint string is the ';'-joined endpoint list. Whether the
    SDK accepts a multi-endpoint string is pending confirmation from the
    EIC SDK team; the full list is always kept on the resolved config.
    """
    text = "" if raw is None else str(raw).strip()
    if not text:
        raise ValueError("EicRemoteUrl is required when EnableEicCache=True")
    if not text.startswith(EIC_URL_PREFIX):
        raise ValueError("EicRemoteUrl must start with eic://")
    body = text[len(EIC_URL_PREFIX) :]
    endpoints = [part.strip() for part in body.split(";")]
    endpoints = [part for part in endpoints if part]
    if not endpoints:
        raise ValueError("no valid endpoint in EicRemoteUrl")
    for endpoint in endpoints:
        _validate_endpoint(endpoint)
    if len(endpoints) > 1:
        logger.warning(
            "EicRemoteUrl carries %d endpoints; passing the ';'-joined "
            "string to the EIC SDK, whose multi-endpoint support is not "
            "yet confirmed",
            len(endpoints),
        )
    return endpoints, ";".join(endpoints)


def _validate_endpoint(endpoint: str) -> None:
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
    host, sep, port = host_port.rpartition(":")
    if not sep or not host or not port.isdigit():
        raise ValueError(
            f"invalid endpoint in EicRemoteUrl: {endpoint!r} "
            "must be in host:port form"
        )


def get_eic_config_file_path(environ: Optional[Mapping] = None) -> str:
    env = os.environ if environ is None else environ
    return env.get(ENV_REMOTE_EIC_YAML) or DEFAULT_CONFIG_FILE


def load_yaml_config(
    environ: Optional[Mapping] = None, config_file: Optional[str] = None
) -> dict:
    """Return the yaml config dict, or {} when the file does not exist."""
    path = config_file or get_eic_config_file_path(environ)
    if not os.path.exists(path):
        return {}
    try:
        import yaml
    except ImportError:
        logger.warning("pyyaml is unavailable; ignoring EIC config %s", path)
        return {}
    with open(path, "r") as fin:
        return yaml.safe_load(fin) or {}


def load_fixed_config(
    environ: Optional[Mapping] = None, config_file: Optional[str] = None
) -> dict:
    """Fixed parameters: yaml values over the built-in defaults."""
    fixed = dict(DEFAULT_FIXED_CONFIG)
    fixed.update(load_yaml_config(environ, config_file))
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
        resolved = EicResolvedConfig(enabled=False, source=source, fixed=fixed)
        logger.info("eic config resolved: enabled=False source=%s", source)
        return resolved

    yaml_config = load_yaml_config(env, config_file)

    env_url = (env.get(ENV_EIC_REMOTE_URL) or "").strip()
    yaml_url = str(yaml_config.get("remote_url") or "").strip()
    if env_url:
        endpoints, endpoint = parse_eic_remote_url(env_url)
    elif yaml_url:
        endpoints, endpoint = parse_eic_remote_url(yaml_url)
        # Previous usage handed the yaml string to the SDK verbatim
        # (remote_url without the eic:// prefix, separators untouched).
        # Keep that byte-for-byte so existing ConfigMaps behave as before;
        # the normalized form is only used for the env-var scheme.
        endpoint = yaml_url[len(EIC_URL_PREFIX) :]
    else:
        raise ValueError("EicRemoteUrl is required when EnableEicCache=True")

    env_id = (env.get(ENV_EIC_INSTANCE_ID) or "").strip()
    yaml_id = str(yaml_config.get("eic_instance_id") or "").strip()
    instance_id = env_id or yaml_id
    if not instance_id:
        raise ValueError("EicInstanceId is required when EnableEicCache=True")

    source = "env" if (switch_from_env or env_url or env_id) else "yaml"
    resolved = EicResolvedConfig(
        enabled=True,
        instance_id=instance_id,
        endpoint=endpoint,
        endpoints=endpoints,
        source=source,
        fixed=fixed,
    )
    logger.info(
        "eic config resolved: enabled=True source=%s endpoints=%s instance_id=%s",
        source,
        endpoints,
        instance_id,
    )
    return resolved
