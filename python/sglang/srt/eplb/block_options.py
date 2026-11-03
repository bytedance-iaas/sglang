"""Read-only feature configuration; disabled by default."""

from sglang.srt.runtime_context import get_exec


def options():
    return get_exec().moe


def migration_enabled():
    return options().enable_expert_migration


def enabled():
    return options().ep_dispatch_algorithm == "block"
