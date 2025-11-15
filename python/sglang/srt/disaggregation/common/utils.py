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

class StepCounter:
    """
    用于协调模型计算步骤与KV缓存传输步骤的计数器
    在异步传输场景中，确保只有当计算完成后才开始对应层的传输
    """
    def __init__(self):
        # 缓存传输相关的步骤计数器
        self._cache_step = 0
        # 辅助数据传输相关的步骤计数器
        self._aux_step = 0
        # 线程安全锁
        self._lock = threading.Lock()

    def current_step(self) -> Tuple[int, int]:
        """获取当前的缓存步骤和辅助数据步骤"""
        with self._lock:
            return self._cache_step, self._aux_step

    def advance_step(self, delta_cache_step: int = 0, delta_aux_step: int = 0) -> None:
        """
        推进步骤计数器
        
        参数:
            delta_cache_step: 缓存步骤的增量
            delta_aux_step: 辅助数据步骤的增量
        """
        with self._lock:
            if delta_cache_step > 0:
                self._cache_step += delta_cache_step
            if delta_aux_step > 0:
                self._aux_step += delta_aux_step

    def query_ready_cache_step(self) -> int:
        """查询当前已就绪的缓存步骤"""
        with self._lock:
            return self._cache_step

    def query_ready_aux_step(self) -> int:
        """查询当前已就绪的辅助数据步骤"""
        with self._lock:
            return self._aux_step

    @staticmethod
    def is_step_ready(current_ready_step: int, target_step: int) -> bool:
        """
        检查目标步骤是否已就绪
        
        参数:
            current_ready_step: 当前已就绪的步骤
            target_step: 需要检查的目标步骤
            
        返回:
            如果目标步骤已就绪则返回True，否则返回False
        """
        return current_ready_step >= target_step

    def reset(self) -> None:
        """重置所有步骤计数器"""
        with self._lock:
            self._cache_step = 0
            self._aux_step = 0

