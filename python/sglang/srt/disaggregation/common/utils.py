import threading
from collections import deque
from typing import List, Tuple

import numpy as np
import numpy.typing as npt


class FastQueue:
    def __init__(self):
        self._buf = deque()
        self._cond = threading.Condition()

    def put(self, item):
        with self._cond:
            self._buf.append(item)
            # wake up a thread of wait()
            self._cond.notify()

    def get(self):
        with self._cond:
            # if queue is empty  ,block until is notified()
            while not self._buf:
                self._cond.wait()
            return self._buf.popleft()


def group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int32], dst_indices: npt.NDArray[np.int32]
) -> Tuple[List[npt.NDArray[np.int32]], List[npt.NDArray[np.int32]]]:
    """Vectorised NumPy implementation."""
    if src_indices.size == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


def get_dcp_compatible_transfer_page_size(
    page_size: int, prefill_dcp_size: int, decode_dcp_size: int
) -> int:
    if decode_dcp_size <= 1 or prefill_dcp_size == decode_dcp_size:
        return page_size
    if prefill_dcp_size == 1 and decode_dcp_size > 1:
        return page_size * decode_dcp_size
    raise RuntimeError(
        "PD disaggregation only supports mixed DCP transfer when prefill dcp_size=1 "
        f"and decode dcp_size>1. Got prefill dcp_size={prefill_dcp_size}, "
        f"decode dcp_size={decode_dcp_size}."
    )


def get_dcp_compatible_token_positions(
    page_size: int,
    prefill_dcp_size: int,
    decode_dcp_size: int,
    decode_dcp_rank: int,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], int]:
    token_positions = np.arange(page_size, dtype=np.int64)

    if decode_dcp_size <= 1 or prefill_dcp_size == decode_dcp_size:
        return token_positions, token_positions, 1

    if prefill_dcp_size == 1 and decode_dcp_size > 1:
        src_token_positions = token_positions * decode_dcp_size + decode_dcp_rank
        return src_token_positions, token_positions, decode_dcp_size

    raise RuntimeError(
        "PD disaggregation only supports mixed DCP transfer when prefill dcp_size=1 "
        f"and decode dcp_size>1. Got prefill dcp_size={prefill_dcp_size}, "
        f"decode dcp_size={decode_dcp_size}."
    )
