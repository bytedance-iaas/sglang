import hashlib
import logging
import pickle
import threading
import time
from queue import Empty, PriorityQueue, Queue
from typing import Iterable, List, Optional

import torch

from sglang.srt.managers.cache_controller import (
    CacheOperation,
    HiCacheController,
    LayerDoneCounter,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost

logger = logging.getLogger(__name__)


def get_content_hash(
    content: Iterable, page_size: int, prev_hash: Optional[int] = None
) -> List[int]:
    """
    Get the hash of the content.
    """

    def hash_func(input):
        input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
        return int.from_bytes(hashlib.sha256(input_bytes).digest(), byteorder="big")

    if prev_hash is None:
        prev_hash = 0
    result = []
    for i in range(len(content) // page_size):
        page = content[i * page_size : (i + 1) * page_size]
        page_hash = hash_func((prev_hash, page))
        prev_hash = page_hash
        result.append(page_hash)
    return result


class OffloadCacheOperation(CacheOperation):
    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        content_hash: Optional[List[int]] = None,
        priority: Optional[int] = None,
    ):
        self.content_hash = content_hash
        super().__init__(
            host_indices=host_indices,
            device_indices=device_indices,
            node_id=node_id,
            priority=priority,
        )


class OffloadCacheController(HiCacheController):
    def __init__(
        self,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        mem_pool_host: MHATokenToKVPoolHost,
        page_size: int,
        load_cache_event: threading.Event = None,
        write_policy: str = "write_through",
    ):
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        self.mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy
        self.page_size = page_size
        self.load_cache_event = load_cache_event
        self.layer_done_counter = LayerDoneCounter(self.mem_pool_device.layer_num)
        # self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        self.write_queue = Queue()
        self.load_queue = Queue()

        self.ack_write_queue = Queue()
        self.ack_load_queue = Queue()

        self.stop_event = threading.Event()

        self.write_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

        self.write_parallel = 1
        self.load_parallel = 1

        self.write_thread_pool = [
            threading.Thread(target=self.write_thread_func_direct, daemon=True)
            for _ in range(self.write_parallel)
        ]
        self.load_thread_pool = [
            threading.Thread(target=self.load_thread_func_direct, daemon=True)
            for _ in range(self.load_parallel)
        ]

        for th in self.write_thread_pool:
            th.start()
        for th in self.load_thread_pool:
            th.start()

    def reset(self):
        self.stop_event.set()

        for th in self.write_thread_pool:
            th.join()
        for th in self.load_thread_pool:
            th.join()

        self.write_queue.queue.clear()
        self.load_queue.queue.clear()
        self.ack_write_queue.queue.clear()
        self.ack_load_queue.queue.clear()

        self.write_thread_pool = [
            threading.Thread(target=self.write_thread_func_direct, daemon=True)
            for _ in range(self.write_parallel)
        ]
        self.load_thread_pool = [
            threading.Thread(target=self.load_thread_func_direct, daemon=True)
            for _ in range(self.load_parallel)
        ]

        self.stop_event.clear()

        for th in self.write_thread_pool:
            th.start()
        for th in self.load_thread_pool:
            th.start()

    def write_token_wise(self, operation: OffloadCacheOperation):
        """
        Write the KV cache to host memory.
        """
        operation.data = self.mem_pool_device.get_flat_data(operation.device_indices)
        ret = self.mem_pool_host.transfer(operation.host_indices, operation.data)
        if not ret:
            logger.error(f"Failed to write to host memory {operation.node_ids}")
            self.mem_pool_host.free(operation.host_indices)
            for node_id in operation.node_ids:
                if node_id != 0:
                    self.ack_write_queue.put((node_id, False))
            return
        self.mem_pool_host.complete_io(operation.host_indices)
        for node_id in operation.node_ids:
            if node_id != 0:
                self.ack_write_queue.put((node_id, True))

    def load_token_wise(self, operation: OffloadCacheOperation):
        """
        Load the KV cache from host memory to device memory.
        """
        operation.data, mask = self.mem_pool_host.get_flat_data(operation.host_indices)
        if operation.data is None:
            logger.error(f"Failed to load from host memory {operation.node_ids}")
            for node_id in operation.node_ids:
                if node_id != 0:
                    self.ack_load_queue.put((node_id, False))
            return
        self.mem_pool_device.transfer(operation.device_indices, operation.data)
        self.mem_pool_host.complete_io(operation.host_indices)
        for node_id in operation.node_ids:
            if node_id != 0:
                self.ack_load_queue.put((node_id, True))

    def write_page_wise(self, operation: OffloadCacheOperation):
        """
        Write the KV cache to host memory.
        """
        assert len(operation.host_indices) == self.page_size * len(
            operation.content_hash
        )
        operation.data = self.mem_pool_device.get_flat_data(operation.device_indices)
        ret = self.mem_pool_host.assign_page_data(
            operation.content_hash, operation.data
        )
        if not ret:
            logger.error(f"Failed to write to host memory {operation.node_ids}")
            self.mem_pool_host.free(operation.host_indices, operation.content_hash)
            for node_id in operation.node_ids:
                if node_id != 0:
                    self.ack_write_queue.put((node_id, False))
            return
        self.mem_pool_host.complete_io(operation.host_indices)
        for node_id in operation.node_ids:
            if node_id != 0:
                self.ack_write_queue.put((node_id, True))

    def load_page_wise(self, operation: OffloadCacheOperation):
        """
        Load the KV cache from host memory to device memory.
        """
        assert len(operation.host_indices) == self.page_size * len(
            operation.content_hash
        )
        operation.data, mask = self.mem_pool_host.get_page_data(operation.content_hash)
        if operation.data is None:
            logger.error(f"Failed to load from host memory {operation.node_ids}")
            for node_id in operation.node_ids:
                if node_id != 0:
                    self.ack_load_queue.put((node_id, False))
            return
        self.mem_pool_device.transfer(operation.device_indices, operation.data)
        self.mem_pool_host.complete_io(operation.host_indices)
        for node_id in operation.node_ids:
            if node_id != 0:
                self.ack_load_queue.put((node_id, True))

    def write_to_offload(self, operation: OffloadCacheOperation):
        """
        Write the KV cache to host memory.
        """
        if self.page_size == 1:
            self.write_token_wise(operation)
        else:
            self.write_page_wise(operation)

    def load_from_offload(self, operation: OffloadCacheOperation):
        """
        Load the KV cache from host memory to device memory.
        """
        if self.page_size == 1:
            self.load_token_wise(operation)
        else:
            self.load_page_wise(operation)

    def write_thread_func_direct(self):
        """
        Directly write through KV caches to host memory without buffering.
        """
        torch.cuda.set_stream(self.write_stream)
        while not self.stop_event.is_set():
            # logger.debug("wirte thread eventloop running")
            try:
                operation = self.write_queue.get(block=True, timeout=1)
                if self.write_policy == "write_through":
                    torch.cuda.synchronize()
                self.write_to_offload(operation)
            except Empty:
                continue
            except Exception as e:
                logger.error(e, e.with_traceback(e.__traceback__))

    def load_thread_func_direct(self):
        """
        Directly load KV caches from host memory to device memory without buffering.
        """
        torch.cuda.set_stream(self.load_stream)
        while not self.stop_event.is_set():
            # logger.debug("load thread eventloop running")
            # self.load_cache_event.wait(timeout=1)
            # if not self.load_cache_event.is_set():
            #     continue
            # self.load_cache_event.clear()
            try:
                operation = self.load_queue.get(block=True, timeout=1)
                self.load_from_offload(operation)
            except Empty:
                continue
            except Exception as e:
                logger.error(e, e.with_traceback(e.__traceback__))

    def host_allocate(self, size):
        """
        Allocate memory on the host.
        """
        return self.mem_pool_host.alloc(size)

    def find_longest_prefix_in_offload(self, prompt):
        """
        Find the longest prefix in the Offload cache.
        """
        if len(prompt) == 0:
            return [], []
        content_hash = get_content_hash(prompt, self.page_size)
        exist_result = self.mem_pool_host.exist_page(content_hash)
        return exist_result, prompt[: len(exist_result) * self.page_size]

    def write_page(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = 0,
        content_hash: List[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Back up KV caches from device memory to host memory.
        """
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.mem_pool_host.protect_write(host_indices)
        self.write_queue.put(
            OffloadCacheOperation(
                host_indices, device_indices, node_id, content_hash, priority
            )
        )
        return host_indices

    def load_page(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = 0,
        content_hash: List[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from host memory to device memory.
        """
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        self.mem_pool_host.protect_load(host_indices)
        # to ensure the device indices are ready before accessed by another CUDA stream
        torch.cuda.current_stream().synchronize()
        self.load_queue.put(
            OffloadCacheOperation(
                host_indices, device_indices, node_id, content_hash, priority
            )
        )
        return device_indices

    def evict_host(
        self,
        host_indices: torch.Tensor,
        content_hash: Optional[List[int]],
        backup_only: bool = True,
    ) -> int:
        if not backup_only:
            raise ValueError("Other eviction policies are not supported yet.")

        if self.mem_pool_host.is_backup(host_indices):
            self.mem_pool_host.free(host_indices, content_hash)
            return len(host_indices)
        else:
            raise ValueError(
                f"Inconsistent states: {self.mem_pool_host.get_state(host_indices)}"
            )
