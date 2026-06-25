from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Optional

import torch

from sglang.srt.environ import envs

_traced_rids: set[str] = set()
_rejected_rids: set[str] = set()
_event_counts: dict[str, int] = {}
_ANSWER_PATTERN = re.compile(r"Answer:\s*([A-D])", re.IGNORECASE)
_TAIL_LETTER_PATTERN = re.compile(r"(?:^|</think>|\n)\s*([A-D])\s*$", re.IGNORECASE)


def trace_enabled() -> bool:
    return envs.SGLANG_DSV4_HISPARSE_ACCURACY_TRACE.get()


def _rid_of(req: Any) -> str:
    rid = getattr(req, "rid", None)
    return str(rid if rid is not None else id(req))


def _sampled_by_rate(rid: str) -> bool:
    rate = envs.SGLANG_DSV4_HISPARSE_ACCURACY_TRACE_SAMPLE_RATE.get()
    if rate <= 0:
        return True
    if rate >= 1:
        return True
    digest = hashlib.blake2b(rid.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "little") / float(1 << 64)
    return bucket < rate


def should_trace_req(req: Any) -> bool:
    if not trace_enabled() or req is None:
        return False

    rid = _rid_of(req)
    if rid in _traced_rids:
        return True
    if rid in _rejected_rids:
        return False

    max_reqs = envs.SGLANG_DSV4_HISPARSE_ACCURACY_TRACE_MAX_REQS.get()
    if max_reqs <= 0 or len(_traced_rids) >= max_reqs:
        _rejected_rids.add(rid)
        return False
    if not _sampled_by_rate(rid):
        _rejected_rids.add(rid)
        return False

    _traced_rids.add(rid)
    return True


def should_trace_event(event: str) -> bool:
    if not trace_enabled():
        return False
    max_events = envs.SGLANG_DSV4_HISPARSE_ACCURACY_TRACE_MAX_REQS.get()
    if max_events <= 0:
        return False
    count = _event_counts.get(event, 0)
    if count >= max_events:
        return False
    _event_counts[event] = count + 1
    return True


def emit_trace_event(
    logger: logging.Logger,
    event: str,
    **fields: Any,
) -> None:
    if not trace_enabled():
        return
    if not logger.isEnabledFor(logging.INFO):
        return
    field_msg = ", ".join(
        f"{field_key}={_format_value(value)}"
        for field_key, value in fields.items()
    )
    logger.info("DSV4 HiSparse accuracy trace: event=%s, %s", event, field_msg)


def tensor_digest(tensor: Any, *, max_items: int = 16) -> str:
    if not isinstance(tensor, torch.Tensor):
        return str(tensor)
    try:
        flat = tensor.detach().reshape(-1)
        sample = flat[: min(max_items, flat.numel())].to("cpu", non_blocking=False)
        digest = hashlib.blake2b(sample.numpy().tobytes(), digest_size=8).hexdigest()
        return (
            f"shape={tuple(tensor.shape)},dtype={tensor.dtype},device={tensor.device},"
            f"sample_items={sample.numel()},digest={digest}"
        )
    except Exception as exc:
        return (
            f"shape={tuple(tensor.shape)},dtype={tensor.dtype},device={tensor.device},"
            f"digest_error={type(exc).__name__}"
        )


def text_tail(text: Optional[str], *, max_chars: int = 512) -> Optional[str]:
    if text is None:
        return None
    tail = text[-max_chars:]
    return tail.replace("\n", "\\n").replace("\r", "\\r")


def answer_markers(text: Optional[str]) -> dict[str, Any]:
    if not text:
        return {
            "answer_pattern_count": 0,
            "last_answer_pattern": None,
            "tail_letter": None,
            "has_think_end": False,
        }
    matches = _ANSWER_PATTERN.findall(text)
    tail_match = _TAIL_LETTER_PATTERN.search(text)
    return {
        "answer_pattern_count": len(matches),
        "last_answer_pattern": matches[-1].upper() if matches else None,
        "tail_letter": tail_match.group(1).upper() if tail_match else None,
        "has_think_end": "</think>" in text,
    }


def _format_value(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)},dtype={value.dtype},"
            f"device={value.device})"
        )
    if isinstance(value, (list, tuple)):
        if len(value) > 8:
            return f"{type(value).__name__}(len={len(value)},head={list(value[:8])})"
        return repr(value)
    if isinstance(value, dict):
        if len(value) <= 8 and all(
            isinstance(k, (str, int, float, bool, type(None)))
            and isinstance(v, (str, int, float, bool, type(None)))
            for k, v in value.items()
        ):
            return repr(value)
        return f"dict(keys={list(value.keys())[:8]},len={len(value)})"
    return repr(value)


def trace_req(
    logger: logging.Logger,
    event: str,
    req: Any,
    **fields: Any,
) -> None:
    if not should_trace_req(req):
        return
    if not logger.isEnabledFor(logging.INFO):
        return
    base_fields = {
        "rid": getattr(req, "rid", None),
        "req_pool_idx": getattr(req, "req_pool_idx", None),
        "input_len": len(getattr(req, "origin_input_ids", []) or []),
        "output_len": len(getattr(req, "output_ids", []) or []),
        "cache_protected_len": getattr(req, "cache_protected_len", None),
        "kv_allocated_len": getattr(req, "kv_allocated_len", None),
        "kv_committed_len": getattr(req, "kv_committed_len", None),
    }
    base_fields.update(fields)
    field_msg = ", ".join(
        f"{key}={_format_value(value)}" for key, value in base_fields.items()
    )
    logger.info("DSV4 HiSparse accuracy trace: event=%s, %s", event, field_msg)


def trace_event(
    logger: logging.Logger,
    event: str,
    *,
    key: Optional[str] = None,
    **fields: Any,
) -> None:
    event_key = key or event
    if not should_trace_event(event_key):
        return
    emit_trace_event(logger, event, **fields)
