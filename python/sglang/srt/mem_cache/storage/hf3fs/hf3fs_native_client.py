"""Tensor adapter for the optional 3FS native batch binding.

The binding owns registered staging memory; this path still copies CPU bytes.
It drains I/O before returning and deliberately has no early-return timeout.
"""

import torch

from sglang.srt.mem_cache.storage.hf3fs.hf3fs_client import Hf3fsClient


class Hf3fsNativeClient(Hf3fsClient):
    def __init__(self, path, size, bytes_per_page, read_entries, write_entries):
        try:
            from hf3fs_py_batch import Batch, Buffers
        except ImportError as exc:
            raise ImportError(
                "HF3FS io_backend=native requires the 3FS hf3fs_py_batch binding. "
                "Build src/lib/py/batch in 3FS and add its output to PYTHONPATH."
            ) from exc

        self._buffers = Buffers
        self.size = size
        self.bytes_per_page = bytes_per_page
        self.reader = Batch(path, size, bytes_per_page, read_entries, True)
        try:
            self.writer = Batch(path, size, bytes_per_page, write_entries, False)
        except BaseException:
            self.reader.close()
            raise

    def check(self, offsets, tensors):
        if len(offsets) != len(tensors):
            raise ValueError("HF3FS offsets and tensors must have the same length")
        for offset, tensor in zip(offsets, tensors):
            if tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise ValueError("HF3FS requires contiguous CPU tensors")
            size = tensor.numel() * tensor.element_size()
            if (
                size <= 0
                or size > self.bytes_per_page
                or not 0 <= offset <= self.size - size
            ):
                raise ValueError("HF3FS page is outside the file or staging buffer")

    def _io(self, ring, offsets, tensors):
        self.check(offsets, tensors)
        # numpy views export ownership without copying, including BF16 tensors.
        buffers = self._buffers(
            [tensor.detach().view(torch.uint8).numpy() for tensor in tensors]
        )
        return ring.io(offsets, buffers)

    def batch_read(self, offsets, tensors):
        return self._io(self.reader, offsets, tensors)

    def batch_write(self, offsets, tensors):
        return self._io(self.writer, offsets, tensors)

    def get_size(self):
        return self.size

    def flush(self):
        self.writer.flush()

    def close(self):
        try:
            self.reader.close()
        finally:
            self.writer.close()
