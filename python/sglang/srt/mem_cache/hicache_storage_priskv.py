import hashlib
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pris
import torch
import yaml

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from python.sglang.srt.mem_cache.hicache_storage import HiCacheStorage
from python.sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)

logger = logging.getLogger(__name__)

G_TensorPoolSize = 2048

REMOTE_OFFLOAD_YAML_ENV_VAR = "REMOTE_OFFLOAD_YAML"

# GPU direct RDMA for KV set
G_EnableKVSetGPUDirect = False

# GPU direct RDMA for KV get
G_EnableKVGetGPUDirect = False


class FlexibleKVCacheMemoryPool:
    def __init__(
        self,
        client: pris.PrisClient,
        device: str,
        kv_cache_shape: tuple,
        kv_cache_dtype: torch.dtype,
    ):
        self._init = False
        self.client = client
        self.device = device
        """ (num_layer, 2, chunk_size, num_kv_head, head_size) """
        self.kv_cache_shape = kv_cache_shape
        self.kv_cache_dtype = kv_cache_dtype

        self.max_kv_cache_num = G_TensorPoolSize * 2

        self.mempool = torch.zeros(
            (self.max_kv_cache_num,) + kv_cache_shape,
            dtype=kv_cache_dtype,
            device=device,
        )
        self.kv_cache_idx = 0
        self.kv_cache_numel = 1
        for i in self.kv_cache_shape:
            self.kv_cache_numel *= i

        # Register memory with PrisClient
        self.mr_mem = self.client.reg_memory(
            self.mempool.data_ptr(), self.mempool.numel() * self.mempool.element_size()
        )
        if self.mr_mem == 0:
            logger.error("Failed to register memory with PrisClient")
            exit(1)

        logger.info(
            f"Registered memory pool shape {self.kv_cache_shape}, dtype {self.kv_cache_dtype}, "
            f"kv_cache_num {self.max_kv_cache_num}, device {device}, "
            f"single_kv_cache_size {np.prod(self.kv_cache_shape) * self.kv_cache_dtype.itemsize / 1024 / 1024:.2f}MB, "
            f"total_size {(self.max_kv_cache_num * (self.mempool[0].numel() * self.mempool[0].element_size())) / 1024 / 1024:.2f}MB"
        )

    def try_allocate_kv_cache(self, shape, dtype, count=1):
        if self.kv_cache_dtype != dtype or self.kv_cache_shape != shape:
            logger.error(
                f"Allocate from mempool failed, self.kv_cache_shape {self.kv_cache_shape}, "
                f"dtype {self.kv_cache_dtype}, require shape {shape}, dtype {dtype}"
            )
            return None

        if count > self.max_kv_cache_num:
            logger.error(
                f"Allocate from mempool failed, self.kv_cache_shape {self.kv_cache_shape}, "
                f"dtype {self.kv_cache_dtype}, require count {count}, max_kv_cache_num {self.max_kv_cache_num}"
            )
            return None

        if self.kv_cache_idx + count > self.max_kv_cache_num:
            self.kv_cache_idx = 0

        ret = self.mempool[self.kv_cache_idx : self.kv_cache_idx + count]
        self.kv_cache_idx = (self.kv_cache_idx + count) % self.max_kv_cache_num
        return ret


class PrisKVClient(HiCacheStorage):
    """
    The remote address and port are loaded from a YAML file, supporting environment variable override.
    """

    def __init__(
        self,
        endpoint: str,
        kv_cache_dtype: torch.dtype,
        kv_cache: KVCache,
        page_size: int,
        device="cpu",
    ):
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_suffix = f"_{tp_rank}_{tp_size}" if tp_size > 1 else ""
        self.kv_cache = kv_cache
        self.layer_num = self.kv_cache.layer_num
        self.page_size = page_size
        if isinstance(kv_cache, MLATokenToKVPool):
            self.kv_cache_shape = (
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache.kv_lora_rank + self.kv_cache.qk_rope_head_dim,
            )
        elif isinstance(kv_cache, MHATokenToKVPool):
            self.kv_cache_shape = (
                2,
                self.layer_num,
                self.page_size,
                self.kv_cache.head_num,
                self.kv_cache.head_dim,
            )
        else:
            logger.error(f"Not Support kvcache type {type(self.kv_cache)}")
            exit(1)

        global G_EnableKVSetGPUDirect, G_EnableKVGetGPUDirect, G_TensorPoolSize
        config_file = os.environ.get(
            REMOTE_OFFLOAD_YAML_ENV_VAR, "/sgl-workspace/config/remote-offload.yaml"
        )
        if not os.path.exists(config_file):
            logger.error(f"Config file {config_file} does not exist")
            exit(1)

        with open(config_file, "r") as fin:
            config = yaml.safe_load(fin)

        remote_url = config.get("remote_url", "hpkv://127.0.0.1-6379")
        address_part = remote_url.split("://")[1]
        raddr, rport = address_part.split("-")

        G_EnableKVSetGPUDirect = config.get("enable_kvset_gpu_direct", False)
        logger.info(f"offload enable_kvset_gpu_direct: {G_EnableKVSetGPUDirect}")

        G_EnableKVGetGPUDirect = config.get("enable_kvget_gpu_direct", False)
        logger.info(f"offload enable_kvget_gpu_direct: {G_EnableKVGetGPUDirect}")

        G_TensorPoolSize = config.get("tensor_pool_size", 2048)
        logger.info(f"offload tensor_pool_size: {G_TensorPoolSize}")

        password = config.get("pris_password", "")

        self.client = pris.PrisClient(raddr, int(rport), password)
        self.device = device
        self.kv_cache_dtype = kv_cache_dtype

        # Use GPU if G_EnableKVGetGPUDirect is True, else CPU
        kv_get_device = (
            "cuda" if G_EnableKVGetGPUDirect and torch.cuda.is_available() else "cpu"
        )

        self.kv_cache_read_mem_pool = FlexibleKVCacheMemoryPool(
            self.client,
            kv_get_device,
            self.kv_cache_shape,
            kv_cache_dtype,
        )

        # Use GPU if G_EnableKVSetGPUDirect is True, else CPU
        kv_set_device = (
            "cuda" if G_EnableKVSetGPUDirect and torch.cuda.is_available() else "cpu"
        )
        self.kv_cache_write_mem_pool = FlexibleKVCacheMemoryPool(
            self.client,
            kv_set_device,
            self.kv_cache_shape,
            kv_cache_dtype,
        )
        self.write_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()
    def _get_suffixed_key(self, key: str) -> str:
        return key + self.tp_suffix
    def get(
        self, key: str, target_location: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        key = self._get_suffixed_key(key)
        logger.debug(f"Pris get {key}")
        get_data_start_time = time.perf_counter()

        # Generate data keys and SGLs
        objs = []
        dtype = self.kv_cache_dtype
        shape = self.kv_cache_shape
        logger.debug(f"Get tensor shape {shape}, dtype {dtype}")
        item = self.kv_cache_read_mem_pool.try_allocate_kv_cache(shape, dtype)
        if item is None:
            logger.error("Cannot allocate tensor from FlexibleKVCacheMemoryPool")
            return None
        sgl = pris.SGL(
            item.data_ptr(),
            item.element_size() * item.numel(),
            self.kv_cache_read_mem_pool.mr_mem,
        )
        # Get data
        length = 0
        status = self.client.get(key, sgl, length)
        get_data_end_time = time.perf_counter()
        get_data_execution_time = (get_data_end_time - get_data_start_time) * 1e6
        logger.debug(
            "Pris mget操作详情:\n"
            f"- 总键数量: {len(key)}\n"
            f"- 状态码: {status}\n"
            f"- 完整键列表: {key}"
        )
        if status != 0:
            logger.error(f"Pris mget {key} failed, status {status}")
            return None
        logger.info(
            f"Pris mget | keys={len(key)} | shapes={self.kv_cache_shape} | status={status} | time={get_data_execution_time}µs"
        )

        # Ensure the output tensor is on the target device
        result = torch.stack([item])
        # if self.kv_cache_read_mem_pool.device != self.device:
        #     logger.debug(f"Moving get result from {self.kv_cache_read_mem_pool.device} to {self.device}")
        #     result = result.to(self.device)
        return result

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor | None]:
        keys = [self._get_suffixed_key(key) for key in keys]
        logger.debug(f"Pris get {len(keys)}")
        get_data_start_time = time.perf_counter()

        # Generate data keys and SGLs
        count = len(keys)
        items = self.kv_cache_read_mem_pool.try_allocate_kv_cache(
            self.kv_cache_shape, self.kv_cache_dtype, count
        )
        if items is None:
            logger.error("Cannot allocate tensor from FlexibleKVCacheMemoryPool")
            return None, []

        sgls = [
            pris.SGL(
                item.data_ptr(),
                item.element_size() * item.numel(),
                self.kv_cache_read_mem_pool.mr_mem,
            )
            for item in items
        ]

        status, _ = self.client.mget(keys, sgls, [0] * len(keys))
        if status == 0:
            success_mask = [True] * len(keys)
        else:
            success_mask = [False] * len(keys)

        get_data_execution_time = (time.perf_counter() - get_data_start_time) * 1e6
        logger.info(
            "Pris batch_get detail info:- total key nums: {len(keys)} status: {status}, mget cost: {get_data_execution_time:%.2f}us "
        )

        if target_locations != None:
            for i in range(len(keys)):
                if success_mask[i]:
                    target_locations[i].copy_(items[i], non_blocking=True)
        else:
            target_locations = troch.stack(items)
        return target_locations

    def set(self, key: str, value: torch.Tensor) -> bool:
        return self.batch_set([key], [value])

    def batch_set(self, keys: List[str], obj_inputs: List[torch.Tensor]) -> bool:
        start_time = time.perf_counter()
        keys = [self._get_suffixed_key(key) for key in keys]
        logger.debug(f"Pris set {len(keys)} keys")
        assert len(keys) == len(obj_inputs)
        count = len(keys)
        items = self.kv_cache_write_mem_pool.try_allocate_kv_cache(
            self.kv_cache_shape, self.kv_cache_dtype, count
        )
        if items is None:
            logger.error("Cannot allocate tensor from FlexibleKVCacheMemoryPool")
            return False

        torch.cuda.set_stream(self.write_stream)
        for i in range(len(keys)):
            temp = items[i].reshape(obj_inputs[i].shape).contiguous()
            temp.copy_(obj_inputs[i], non_blocking=True)
        self.write_stream.synchronize()
        sgls = [
            pris.SGL(
                item.data_ptr(),
                item.element_size() * item.numel(),
                self.kv_cache_write_mem_pool.mr_mem,
            )
            for item in items
        ]
        elapsed_copy_us = (time.perf_counter() - start_time) * 1e6
        status, _ = self.client.mset(keys, sgls)

        elapsed_us = (time.perf_counter() - start_time) * 1e6
        logger.info(
            f"Pris mset | keys={len(keys)} | shapes={self.kv_cache_shape} | status={status} | time={elapsed_us:.2f}µs, | copy_time={elapsed_copy_us:.2f}µs"
        )
        if (elapsed_us - elapsed_copy_us) / len(keys) > 1000:
            logger.warning(
                f"Pris slow mset | keys={len(keys)} | shapes={self.kv_cache_shape} | status={status} | time={elapsed_us:.2f}µs, | copy_time={elapsed_copy_us:.2f}µs"
            )
            for key in keys:
                logger.warning(f"Pris slow mset | key={key}")
        if status != 0:
            logger.error(f"Pris mset {len(keys)} failed, status {status}")
            return False
        return True

    def exists(self, key: str) -> bool:
        key = self._get_suffixed_key(key)
        status = self.client.exists(key)
        logger.info(f"Pris exists {key}, status {status}")
        return status == 0

    def delete(self, keys: List[str]) -> bool:
        keys = [self._get_suffixed_key(key) for key in keys]
        status, _ = self.client.mdel(keys)
        if status != 0:
            logger.error(
                f"Pris mdel {len(keys)} failed, status {status}\n"
                f"- status info: {status}"
            )
            return False
        logger.info(f"Pris mdel {len(keys)} keys successfully, status {status}")
        return True
