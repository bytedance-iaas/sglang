import logging
import threading
import time
from queue import Empty, PriorityQueue, Queue

import torch

from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPoolHost,
    TokenToKVPoolAllocator,
)

from .cache_controller import HiCacheController

logger = logging.getLogger(__name__)


class EICCacheController(HiCacheController):
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

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        self.write_queue = PriorityQueue()
        self.load_queue = PriorityQueue()

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

    def write_thread_func_direct(self):
        """
        Directly write through KV caches to host memory without buffering.
        """
        with torch.cuda.stream(self.write_stream):
            while not self.stop_event.is_set():
                logger.debug("wirte thread eventloop running")
                try:
                    operation = self.write_queue.get(block=True, timeout=1)
                    if self.write_policy == "write_through":
                        torch.cuda.synchronize()
                    operation.data = self.mem_pool_device.get_flat_data(
                        operation.device_indices
                    )
                    ret = self.mem_pool_host.transfer(
                        operation.host_indices, operation.data
                    )
                    if not ret:
                        logger.error(
                            f"Failed to write to host memory {operation.node_ids}"
                        )
                        self.mem_pool_host.free(operation.host_indices)
                        for node_id in operation.node_ids:
                            if node_id != 0:
                                self.ack_write_queue.put((node_id, False))
                        continue
                    self.mem_pool_host.complete_io(operation.host_indices)
                    for node_id in operation.node_ids:
                        if node_id != 0:
                            self.ack_write_queue.put((node_id, True))
                except Empty:
                    continue
                except Exception as e:
                    logger.error(e, e.with_traceback(e.__traceback__))

    def load_thread_func_direct(self):
        """
        Directly load KV caches from host memory to device memory without buffering.
        """
        torch.cuda.current_stream().synchronize()
        with torch.cuda.stream(self.load_stream):
            while not self.stop_event.is_set():
                logger.debug("load thread eventloop running")
                # self.load_cache_event.wait(timeout=1)
                # if not self.load_cache_event.is_set():
                #     continue
                # self.load_cache_event.clear()
                try:
                    operation = self.load_queue.get(block=True, timeout=1)
                    # time.sleep(18e-6 * len(operation.host_indices))
                    get_flat_data_start_time = time.perf_counter()
                    operation.data, mask = self.mem_pool_host.get_flat_data(
                        operation.host_indices
                    )
                    # [2, 80, 1993, 1, 128]
                    get_flat_data_end_time = time.perf_counter()
                    cost_second = get_flat_data_end_time - get_flat_data_start_time
                    logger.debug(
                        f"load thread func finish get_flat_data, indices {operation.host_indices.shape} cost {cost_second}"
                    )
                    if operation.data is None:
                        logger.error(
                            f"Failed to load from host memory {operation.node_ids}"
                        )
                        for node_id in operation.node_ids:
                            if node_id != 0:
                                self.ack_load_queue.put((node_id, False))
                        continue
                    self.mem_pool_device.transfer(
                        operation.device_indices, operation.data
                    )
                    load_transfer_end_time = time.perf_counter()
                    cost_second = load_transfer_end_time - get_flat_data_end_time
                    logger.debug(
                        f"load thread func finish transfer, indices {operation.device_indices.shape} cost {cost_second}"
                    )
                    self.mem_pool_host.complete_io(operation.host_indices)
                    for node_id in operation.node_ids:
                        if node_id != 0:
                            self.ack_load_queue.put((node_id, True))
                except Empty:
                    continue
                except Exception as e:
                    logger.error(e, e.with_traceback(e.__traceback__))
