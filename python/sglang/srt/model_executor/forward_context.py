"""Per-forward-call control context.

This is a lightweight compatibility layer for code backported from newer
SGLang branches. It is separate from the piecewise CUDA graph compilation
context in ``sglang.srt.compilation.piecewise_context_manager``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool


@dataclass(frozen=True, slots=True)
class ForwardContext:
    attn_backend: AttentionBackend


_current: Optional[ForwardContext] = None


def set_forward_context(ctx: Optional[ForwardContext]) -> Optional[ForwardContext]:
    global _current
    prev, _current = _current, ctx
    return prev


def has_forward_context() -> bool:
    return _current is not None


def get_forward_context() -> ForwardContext:
    assert _current is not None, (
        "no forward context active; call forward_context(...) or "
        "set_forward_context(...) before reading get_forward_context()."
    )
    return _current


def get_attn_backend() -> AttentionBackend:
    return get_forward_context().attn_backend


def get_token_to_kv_pool() -> KVCache:
    return get_attn_backend().token_to_kv_pool


def get_req_to_token_pool() -> ReqToTokenPool:
    return get_attn_backend().req_to_token_pool


@contextmanager
def forward_context(ctx: ForwardContext):
    prev = set_forward_context(ctx)
    try:
        yield
    finally:
        set_forward_context(prev)
