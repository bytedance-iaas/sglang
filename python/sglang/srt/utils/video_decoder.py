"""Unified video decoder: torchcodec preferred, decord as fallback."""

import logging
import os
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

try:
    from torchcodec.decoders import VideoDecoder

    _BACKEND = "torchcodec"
except (ImportError, RuntimeError):
    _BACKEND = "decord"


_cuda_backend_enabled: bool | None = None
_cuda_backend_lock = threading.Lock()


def _try_cuda_backend() -> bool:
    """Return whether this TorchCodec exposes the CUDA backend context manager."""
    global _cuda_backend_enabled
    if _cuda_backend_enabled is not None:
        return _cuda_backend_enabled
    try:
        from torchcodec.decoders import set_cuda_backend

        _cuda_backend_enabled = callable(set_cuda_backend)
    except Exception:
        _cuda_backend_enabled = False
    return _cuda_backend_enabled


def _create_torchcodec_decoder(source, kwargs):
    """Create a decoder, entering the beta CUDA context when requested.

    TorchCodec 0.11's ``set_cuda_backend`` is a context manager rather than a
    process-wide setter. The decoder must be constructed inside that context;
    decoding can happen after the context exits.
    """
    if kwargs.get("device") != "cuda":
        return VideoDecoder(source, **kwargs)

    if not _try_cuda_backend():
        raise RuntimeError("TorchCodec CUDA backend is unavailable")

    from torchcodec.decoders import set_cuda_backend

    # Decoder construction is short, and serializing the context transition
    # avoids races when several I/O workers initialize videos concurrently.
    with _cuda_backend_lock:
        with set_cuda_backend("beta"):
            return VideoDecoder(source, **kwargs)


class VideoDecoderWrapper:
    """Unified video decoder that uses torchcodec when available, decord as fallback.

    ``get_frames_at`` preserves the legacy NHWC uint8 NumPy contract.
    ``get_frames_as_tensor`` preserves the configured device and layout.
    """

    def __init__(
        self,
        source,
        device: str = "cpu",
        num_decode_threads: int = 0,
        dimension_order: str = "NHWC",
    ):
        """source: file path (str) or video bytes.
        device: "cpu" or "cuda". GPU decoding only supported with torchcodec.
        num_decode_threads: number of parallel decoder instances for frame
            extraction (torchcodec only). 0 = auto (capped at 16),
            1 = single decoder. Set > 1 to split frame indices across
            multiple decoders in parallel threads.
        dimension_order: ``NHWC`` for the legacy NumPy path or ``NCHW`` for
            device-preserving tensor consumers.
        """
        dimension_order = dimension_order.upper()
        if dimension_order not in {"NHWC", "NCHW"}:
            raise ValueError(
                f"dimension_order must be NHWC or NCHW, got {dimension_order!r}"
            )
        timing_enabled = os.getenv("SGLANG_GEMMA4_VIDEO_TIMING", "0") == "1"
        started = time.perf_counter()
        self._source = source
        self._dimension_order = dimension_order
        self._requested_device = device
        self._num_decode_threads = num_decode_threads
        self._source_bytes = source if isinstance(source, bytes) else None
        self._source_path = source if isinstance(source, str) else None
        self._tmp_path = None
        if _BACKEND == "torchcodec":
            kwargs = {"dimension_order": dimension_order}
            if device == "cuda":
                kwargs["device"] = "cuda"
            self._tc_kwargs = kwargs
            try:
                self._decoder = _create_torchcodec_decoder(source, kwargs)
            except Exception as exc:
                if "device" in kwargs:
                    logger.warning(
                        "CUDA video decoder initialization failed; falling back "
                        "to CPU: %s",
                        exc,
                    )
                    kwargs.pop("device")
                    self._tc_kwargs = kwargs
                    self._decoder = _create_torchcodec_decoder(source, kwargs)
                else:
                    raise

            fallback_status = str(getattr(self._decoder, "cpu_fallback", "unknown"))
            if self.is_cuda and "No fallback required" not in fallback_status:
                logger.warning(
                    "TorchCodec CUDA decoder reports CPU fallback: %s",
                    fallback_status,
                )
            elif self.is_cuda:
                logger.info("TorchCodec CUDA decoder status: %s", fallback_status)
        else:
            from decord import VideoReader, cpu

            if isinstance(source, bytes):
                import tempfile

                fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
                try:
                    os.write(fd, source)
                finally:
                    os.close(fd)
                self._tmp_path = tmp_path
                self._decoder = VideoReader(tmp_path, ctx=cpu(0))
            else:
                self._decoder = VideoReader(source, ctx=cpu(0))
        if timing_enabled:
            actual_device = (
                self._tc_kwargs.get("device", "cpu")
                if _BACKEND == "torchcodec"
                else "cpu"
            )
            logger.info(
                "[Gemma4VideoTiming] step=4_decoder_init elapsed_ms=%.3f "
                "backend=%s requested_device=%s actual_device=%s source_type=%s",
                (time.perf_counter() - started) * 1000,
                _BACKEND,
                device,
                actual_device,
                type(source).__name__,
            )

    def __len__(self):
        return len(self._decoder)

    def __getitem__(self, idx):
        """Return single frame as numpy NHWC uint8."""
        if _BACKEND == "torchcodec":
            frame = self._decoder[idx]
            frame = getattr(frame, "data", frame)
            if self._dimension_order == "NCHW":
                frame = frame.permute(1, 2, 0).contiguous()
            return frame.detach().cpu().numpy()
        else:
            frame = self._decoder[idx]
            return frame.asnumpy() if hasattr(frame, "asnumpy") else np.array(frame)

    @property
    def avg_fps(self) -> float:
        if _BACKEND == "torchcodec":
            return self._decoder.metadata.average_fps
        else:
            return self._decoder.get_avg_fps()

    def get_frames_at(self, indices: list) -> np.ndarray:
        """Return frames at given indices as numpy array with shape (N, H, W, C)."""
        if _BACKEND == "torchcodec":
            tensor = self._get_torchcodec_frames(indices)
            if self._dimension_order == "NCHW":
                tensor = tensor.permute(0, 2, 3, 1).contiguous()
            return tensor.detach().cpu().numpy()
        else:
            return self._decoder.get_batch(indices).asnumpy()

    def get_frames_as_tensor(self, indices: list):
        """Return frames as a uint8 tensor without leaving the decode device.

        The layout is ``self.dimension_order``. CPU outputs are pinned when CUDA
        is available; CUDA outputs are returned directly and never pinned.
        """
        import torch

        if (
            _BACKEND == "torchcodec"
            and not self.is_cuda
            and self._num_decode_threads != 1
            and len(indices) > 1
        ):
            num_threads = self._num_decode_threads
            if num_threads <= 0:
                num_threads = min(os.cpu_count() or 8, 16)
            num_threads = min(num_threads, len(indices))
            if num_threads > 1:
                return self._parallel_decode(indices, num_threads)

        if _BACKEND == "torchcodec":
            tensor = self._get_torchcodec_frames(indices)
            return self._pin_cpu_tensor(tensor)
        else:
            arr = self._decoder.get_batch(indices).asnumpy()
            tensor = torch.from_numpy(arr)
            if self._dimension_order == "NCHW":
                tensor = tensor.permute(0, 3, 1, 2).contiguous()
            return self._pin_cpu_tensor(tensor)

    def _get_torchcodec_frames(self, indices):
        try:
            return self._decoder.get_frames_at(indices).data
        except RuntimeError as exc:
            if not self.is_cuda:
                raise
            logger.warning(
                "CUDA video decoding failed during frame extraction; retrying "
                "on CPU: %s",
                exc,
            )
            self._tc_kwargs.pop("device", None)
            self._decoder = _create_torchcodec_decoder(
                self._source,
                self._tc_kwargs,
            )
            return self._decoder.get_frames_at(indices).data

    @staticmethod
    def _pin_cpu_tensor(tensor):
        import torch

        if tensor.is_cuda or not torch.cuda.is_available():
            return tensor
        return tensor.pin_memory()

    def _parallel_decode(self, indices, num_threads):
        """Decode frames using multiple VideoDecoder instances in parallel threads."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import torch

        chunks = [list(c) for c in np.array_split(indices, num_threads) if len(c) > 0]
        source = self._source
        kwargs = self._tc_kwargs

        def _decode_chunk(chunk):
            d = _create_torchcodec_decoder(source, kwargs)
            return d.get_frames_at(chunk).data

        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            future_to_idx = {
                executor.submit(_decode_chunk, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            results = [None] * len(chunks)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return self._pin_cpu_tensor(torch.cat(results, dim=0))

    @property
    def dimension_order(self) -> str:
        return self._dimension_order

    @property
    def is_cuda(self) -> bool:
        return _BACKEND == "torchcodec" and self._tc_kwargs.get("device") == "cuda"

    @property
    def decode_device(self) -> str:
        return "cuda" if self.is_cuda else "cpu"

    @property
    def cpu_fallback_status(self) -> str:
        return str(getattr(self._decoder, "cpu_fallback", "unknown"))

    @property
    def source_bytes(self) -> bytes | None:
        """Return raw video bytes if available (needed for audio extraction)."""
        if self._source_bytes is not None:
            return self._source_bytes
        path = self._tmp_path or self._source_path
        if path is not None:
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    return f.read()
        return None

    def close(self):
        """Explicitly clean up temporary files."""
        tmp_path = getattr(self, "_tmp_path", None)
        if tmp_path is not None:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            self._tmp_path = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
