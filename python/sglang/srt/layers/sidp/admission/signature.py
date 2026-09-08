"""Compute a request's shape_signature without tokenizing or running vision.

Rationale (design §5): the device barrier aligns *time*, and each forward step's
compute cost depends only on shape (image token count, text token count), not on
content. So requests can share a cohort iff their shapes match. We approximate
shape cheaply on the host:

  * image tokens  ~ sum(H*W) over all images / patch^2  (patch is constant), so
    sum(H*W) is a proxy for image token count. H,W come from image headers only
    (PIL lazy open), so no base64 decode of pixels beyond the header, no vision
    run. Multiple images are summed; image *count* is not matched.
  * text tokens   ~ chars / chars_per_token  (rough estimate; kept in a coarser
    bucket than images to absorb the estimation noise).

signature = (has_image, forward_mode, img_bucket, text_bucket)

The signature is intentionally a plain tuple so it is hashable and cheap to use
as a dict key in the cohort assembler.
"""

from __future__ import annotations

import base64
import io
import math
import os
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlsplit

try:  # PIL is used for header-only H*W extraction; degrade gracefully if absent.
    from PIL import Image  # type: ignore

    _HAS_PIL = True
except Exception:  # noqa: BLE001
    _HAS_PIL = False


# Sentinel image area used when a request declares an image we cannot measure
# (missing PIL, unreadable file, remote URL we chose not to fetch). Requests that
# hit this path still bucket together deterministically instead of crashing.
UNKNOWN_IMAGE_AREA = -1


@dataclass(frozen=True)
class SignatureConfig:
    """Tunable bucketing tolerances (see design §10 for standardization TODO).

    Buckets use *relative* tolerance via log-space quantization: two requests
    share a bucket iff their values are within the tolerance ratio. This gives
    scale-adaptive grouping (small and large requests both bucket at "within
    x%") with zero state and O(1) cost, unlike a fixed absolute width that is
    too fine for small requests and too coarse for large ones.
    """

    # Image relative tolerance on sum(H*W). 0.05 => images whose total area is
    # within ~5% share a bucket. Tight, because H*W is measured exactly.
    img_rel_tol: float = 0.05
    # Text relative tolerance on estimated tokens. Coarser (0.15) because the
    # char->token estimate is noisy.
    text_rel_tol: float = 0.15
    # Rough chars-per-token for the text estimate. Overridable per tokenizer.
    text_chars_per_token: float = 4.0
    # Whether to fetch remote (http/https) images to measure them. Off by default
    # to keep admission cheap; remote images fall back to UNKNOWN_IMAGE_AREA.
    fetch_remote_images: bool = False


def _bucket(value: int, rel_tol: float) -> int:
    """Log-space relative bucketing: floor(log(v) / log(1+rel_tol)).

    Each bucket covers a fixed ratio (1+rel_tol), so any two values in the same
    bucket differ by at most ~rel_tol. Stateless and order-independent (bucket
    edges are fixed by the formula, so there is no representative to initialize
    or drift). Negatives (sentinels) and non-positive values pass through /
    map to 0.
    """
    if value < 0:
        return value
    if value == 0 or rel_tol <= 0:
        return 0
    return int(math.log(value) / math.log(1.0 + rel_tol))


def _image_area_from_bytes(data: bytes) -> Optional[int]:
    if not _HAS_PIL:
        return None
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size  # header read only; pixels not decoded
        return int(w) * int(h)
    except Exception:  # noqa: BLE001
        return None


def _image_area_from_path(path: str) -> Optional[int]:
    if not _HAS_PIL or not os.path.isfile(path):
        return None
    try:
        with Image.open(path) as im:
            w, h = im.size
        return int(w) * int(h)
    except Exception:  # noqa: BLE001
        return None


def _decode_data_uri(uri: str) -> Optional[bytes]:
    # data:image/png;base64,XXXX
    try:
        header, b64 = uri.split(",", 1)
    except ValueError:
        return None
    if "base64" not in header:
        return None
    try:
        return base64.b64decode(b64)
    except Exception:  # noqa: BLE001
        return None


def _image_area(url_or_path: str, cfg: SignatureConfig) -> int:
    """Best-effort H*W for one image reference; UNKNOWN_IMAGE_AREA on failure."""
    if url_or_path.startswith("data:"):
        data = _decode_data_uri(url_or_path)
        if data is not None:
            area = _image_area_from_bytes(data)
            if area is not None:
                return area
        return UNKNOWN_IMAGE_AREA

    scheme = urlsplit(url_or_path).scheme
    if scheme in ("http", "https"):
        if not cfg.fetch_remote_images:
            return UNKNOWN_IMAGE_AREA
        try:
            import urllib.request

            req = urllib.request.Request(
                url_or_path, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
                data = resp.read()
            area = _image_area_from_bytes(data)
            return area if area is not None else UNKNOWN_IMAGE_AREA
        except Exception:  # noqa: BLE001
            return UNKNOWN_IMAGE_AREA

    # Treat as a local filesystem path (run_jsonl_chat_batch rewrites URLs to
    # local cached paths before sending).
    area = _image_area_from_path(url_or_path)
    return area if area is not None else UNKNOWN_IMAGE_AREA


def _iter_image_refs(content: Any):
    """Yield image url/path strings from an OpenAI chat 'content' field.

    Supports the common shapes:
      content = "text"                                  (no images)
      content = [{"type":"text","text":...},
                 {"type":"image_url","image_url":{"url": ...}},
                 {"type":"image","image": ...}, ...]
    """
    if not isinstance(content, list):
        return
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "image_url":
            iu = part.get("image_url")
            if isinstance(iu, dict) and isinstance(iu.get("url"), str):
                yield iu["url"]
            elif isinstance(iu, str):
                yield iu
        elif ptype == "image":
            iv = part.get("image")
            if isinstance(iv, str):
                yield iv
            elif isinstance(iv, dict) and isinstance(iv.get("url"), str):
                yield iv["url"]


def _text_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                t = part.get("text")
                if isinstance(t, str):
                    total += len(t)
        return total
    return 0


@dataclass(frozen=True)
class ShapeStats:
    """Raw (pre-bucketing) measurements, useful for logging/standardization."""

    has_image: bool
    total_image_area: int  # sum(H*W); may be UNKNOWN_IMAGE_AREA if unmeasurable
    num_images: int
    text_chars: int
    est_text_tokens: int


def measure(body: dict, cfg: SignatureConfig) -> ShapeStats:
    """Measure raw shape stats from an OpenAI chat completion request body."""
    messages = body.get("messages") or []
    total_area = 0
    num_images = 0
    text_chars = 0
    any_unknown = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        text_chars += _text_chars(content)
        for ref in _iter_image_refs(content):
            num_images += 1
            area = _image_area(ref, cfg)
            if area == UNKNOWN_IMAGE_AREA:
                any_unknown = True
            else:
                total_area += area
    has_image = num_images > 0
    if has_image and any_unknown:
        total_area = UNKNOWN_IMAGE_AREA
    est_tokens = int(text_chars / max(cfg.text_chars_per_token, 1e-6))
    return ShapeStats(
        has_image=has_image,
        total_image_area=(total_area if has_image else 0),
        num_images=num_images,
        text_chars=text_chars,
        est_text_tokens=est_tokens,
    )


def compute_signature(body: dict, cfg: SignatureConfig) -> tuple:
    """Return the hashable shape_signature for an OpenAI chat request body.

    forward_mode is always 'prefill' at admission time (a fresh request begins
    with prefill); decode-time coordination is handled on the device side.
    """
    stats = measure(body, cfg)
    img_bucket = (
        _bucket(stats.total_image_area, cfg.img_rel_tol)
        if stats.has_image
        else 0
    )
    text_bucket = _bucket(stats.est_text_tokens, cfg.text_rel_tol)
    return (stats.has_image, "prefill", img_bucket, text_bucket)
