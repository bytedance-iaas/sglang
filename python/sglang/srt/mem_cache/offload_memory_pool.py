import logging
import os
import threading
import time
from typing import List, Optional, Tuple

import pris
import torch
import yaml
import numpy as np


from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    MemoryStateInt,
    MHATokenToKVPool,
    MLATokenToKVPool,
    debug_timing,
    synchronized,
)

logger = logging.getLogger(__name__)
G_TensorPoolSize = 2048

REMOTE_OFFLOAD_YAML_ENV_VAR = "REMOTE_OFFLOAD_YAML"

# GPU direct RDMA for KV set
G_EnableKVSetGPUDirect = False

# GPU direct RDMA for KV get
G_EnableKVGetGPUDirect = False


class FlexibleKVCacheMemoryPool:
    def __init__(self, client: pris.PrisClient, device: str, kv_cache_shape, kv_cache_dtype):
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
            self.mempool.data_ptr(),
            self.mempool.numel() * self.mempool.element_size()
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


class PrisKVClient:
    """
    The remote address and port are loaded from a YAML file, supporting environment variable override.
    """

    def __init__(self, endpoint: str, kv_cache_dtype, kv_cache_shape, device="cpu"):
        global G_EnableKVSetGPUDirect, G_EnableKVGetGPUDirect, G_TensorPoolSize
        config_file = os.environ.get(REMOTE_OFFLOAD_YAML_ENV_VAR, "/sgl-workspace/config/remote-offload.yaml")
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
        self.kv_cache_shape = kv_cache_shape
        self.kv_cache_dtype = kv_cache_dtype

        # Use GPU if G_EnableKVGetGPUDirect is True, else CPU
        kv_get_device = "cuda" if G_EnableKVGetGPUDirect and torch.cuda.is_available() else "cpu"

        self.kv_cache_mem_pool = FlexibleKVCacheMemoryPool(
            self.client,
            kv_get_device,
            kv_cache_shape,
            kv_cache_dtype,
        )

        # Use GPU if G_EnableKVSetGPUDirect is True, else CPU
        kv_set_device = "cuda" if G_EnableKVSetGPUDirect and torch.cuda.is_available() else "cpu"
        self.kv_cache_write_mem_pool = FlexibleKVCacheMemoryPool(
            self.client,
            kv_set_device,
            kv_cache_shape,
            kv_cache_dtype,
        )

    def exists(self, key: str) -> bool:
        logger.debug(f"Pris exists {key}")
        return self.client.exist(key)

    def exists_batch(self, keys: List[str]) -> List[bool]:
        logger.debug(f"Pris exists {len(keys)}")
        
        ret = []
        for key in keys:
            r = self.client.exist(key)
            if r:
                ret.append(True)
            else:
                break
        logger.debug(f"Pris exists ret {len(ret)}")
        return ret

    def get(self, keys: List[str]) -> Optional[torch.Tensor]:
        logger.debug(f"Pris get {keys}")
        get_data_start_time = time.perf_counter()

        # Generate data keys and SGLs
        objs = []
        sgls = []
        for i, key in enumerate(keys):
            dtype = self.kv_cache_dtype
            shape = self.kv_cache_shape
            logger.debug(f"Get tensor shape {shape}, dtype {dtype}")

            item = self.kv_cache_mem_pool.try_allocate_kv_cache(shape, dtype)
            if item is None:
                logger.error("Cannot allocate tensor from FlexibleKVCacheMemoryPool")
                return None
            else:
                obj = item
            objs.append(obj)
            sgls.append(pris.SGL(
                obj.data_ptr(),
                obj.element_size() * obj.numel(),
                self.kv_cache_mem_pool.mr_mem
            ))

        # Get data
        status, lengths = self.client.mget(keys, sgls)
        get_data_end_time = time.perf_counter()
        get_data_execution_time = (get_data_end_time - get_data_start_time) * 1e6
        logger.debug("Pris mget操作详情:\n"
             f"- 总键数量: {len(keys)}\n"
             f"- 状态码: {status}\n"
             f"- 完整键列表: {keys}")
        if status != 0:
            logger.error(f"Pris mget {keys} failed, status {status}")
            return None
        logger.info(f"Pris mget | keys={len(keys)} | shapes={self.kv_cache_shape} | status={status} | time={get_data_execution_time:.2f}µs")

        # Ensure the output tensor is on the target device
        result = torch.stack(objs)
        # if self.kv_cache_mem_pool.device != self.device:
        #     logger.debug(f"Moving get result from {self.kv_cache_mem_pool.device} to {self.device}")
        #     result = result.to(self.device)
        return result

    def batch_get(
        self, keys: List[str]
    ) -> Tuple[Optional[torch.Tensor], Optional[List[bool]]]:
        logger.debug(f"Pris get {len(keys)}")
        get_data_start_time = time.perf_counter()

        # Generate data keys and SGLs
        count = len(keys)
        success_mask = [True for _ in range(count)]
        items = self.kv_cache_mem_pool.try_allocate_kv_cache(
            self.kv_cache_shape, self.kv_cache_dtype, count
        )
        if items is None:
            logger.error("Cannot allocate tensor from FlexibleKVCacheMemoryPool")
            return None, []
        else:
            objs = items

        sgls = []
        for i in range(count):
            sgls.append(pris.SGL(
                objs[i].data_ptr(),
                objs[i].element_size() * objs[i].numel(),
                self.kv_cache_mem_pool.mr_mem
            ))

        ret = []
        for key,sgl in zip(keys, sgls):
            length= 0
            # 0 is ok, > 0  is not ok
            status = self.client.get(key, sgl, length)
            if status == 0:
                ret.append(True)
            else:
                break
        while len(ret) < len(keys):
            ret.append(False)
        success_mask = ret
        
        get_data_end_time = time.perf_counter()
        get_data_execution_time = (get_data_end_time - get_data_start_time) * 1e6
        logger.info("Pris batch_get detail info:\n"
             f"- total key nums: {len(keys)}\n"
             f"- success key nums: {len(ret)}")
        logger.debug(f"- success key list: {keys}")
        logger.debug(f"Pris get {count} keys data cost %.2f us", get_data_execution_time)

        # Ensure the output tensor is on the target device
        # if self.kv_cache_mem_pool.device != self.device:
        #     logger.debug(f"Moving batch_get result from {self.kv_cache_mem_pool.device} to {self.device}")
        #     objs = objs.to(self.device)
        return objs, success_mask

    def set(self, keys: List[str], obj_inputs: torch.Tensor) -> bool:
        start_time = time.perf_counter()
        logger.debug(f"Pris set {len(keys)} keys")
        count = len(keys)
        items = self.kv_cache_write_mem_pool.try_allocate_kv_cache(
            self.kv_cache_shape, self.kv_cache_dtype, count
        )
        if items is None:
            logger.error("Cannot allocate tensor from FlexibleKVCacheMemoryPool")
            return False
        else:
            objs = items

        sgls = []
        for i, _ in enumerate(keys):
            temp = objs[i].reshape(obj_inputs[i].shape).contiguous()
            temp.copy_(obj_inputs[i])
            sgls.append(pris.SGL(
                temp.data_ptr(),
                temp.element_size() * temp.numel(),
                self.kv_cache_write_mem_pool.mr_mem
            ))

        # 多线程set实现
        from concurrent.futures import ThreadPoolExecutor
        
        def _set_single(key, sgl):
            logger.debug(f"Pris set {key}")
            return self.client.set(key, sgl)
            
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_set_single, keys, sgls))

        success_count = sum(1 for result in results if result == 0)
        status = success_count == len(results)

        # # Set data
        # status = self.client.mset(keys, sgls)
        logger.debug("Pris mset info:\n"
             f"- total keys: {len(keys)}\n"
             f"- summary results: {status}\n"
             f"- keys info: {keys}\n"
             f"- status info: {results} \n"
             f"- success count: {success_count}")
        elapsed_us = (time.perf_counter() - start_time) * 1e6
        logger.info(f"Pris mset | keys={len(keys)} | shapes={self.kv_cache_shape} | status={status} | time={elapsed_us:.2f}µs")
        if not status:
            logger.error(f"Pris mset {len(keys)} failed, status {status}")
            return False
        return True
    
    def delete(self,  keys: List[str]) -> bool:
          # 多线程del实现
        from concurrent.futures import ThreadPoolExecutor
        
        def _del_single(key):
            logger.debug(f"Pris del single {key}")
            return self.client.delete(key)
            
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_del_single, keys))

        success_count = sum(1 for result in results if result == 0)
        status = success_count == len(results)
        logger.info(f"Pris mdel {len(keys)} keys,  successfully {success_count}")
        if not status:
            logger.error(f"Pris mdel {len(keys)} failed, status {status}\n"
            f"- status info: {results}")
            return False
        logger.info(f"Pris mdel {len(keys)} keys successfully, status {status}")
        return True



class OffloadBaseTokenToKVPoolHost:
    def __init__(
        self,
        device_pool: KVCache,
        host_to_device_ratio: float = 4.0,
        host_size: int = 10,
        device: str = "cpu",
        page_size: int = 1,
        rank: int = 0,
        extra_info: Optional[dict] = None,
    ):
        self.device_pool = device_pool
        self.host_to_device_ratio = host_to_device_ratio
        self.device = device
        self.dtype = device_pool.store_dtype
        self.page_size = page_size
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        self.size = self.size - (self.size % self.page_size)
        logger.info(f"OffloadBaseTokenToKVPoolHost init,{self.size=},{host_size=},{self.size_per_token=}")
        # Initialize memory states and tracking structures
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int32)
        self.can_use_mem_size = self.size

        # A lock for synchronized operations
        self.lock = threading.RLock()
        self.debug = logger.isEnabledFor(logging.DEBUG)

        self.rank = rank
        self.host_ip = self._get_host_ip()
        self.split_dim = 2
        self.extra_info = extra_info
        self.deploy_key = self._get_deploy_info()

    def _encode_key_exclusive(self, indices):
        return [
            f"{self.host_ip}_{self.rank}_{index}"
            for index in indices.to("cpu").tolist()
        ]

    def _get_host_ip(self):
        import socket
        return socket.gethostbyname(socket.gethostname())

    def _get_deploy_info(self):
        model_path = self.extra_info.get("model_path", "fake_model_path")
        world_size = self.extra_info.get("world_size", 1)
        rank = self.extra_info.get("tp_rank", 0)
        page_size = self.page_size
        framework = self.extra_info.get("framework", "sglang")
        deploy_key = f"{model_path}_{world_size}_{rank}_{page_size}@{framework}"
        return deploy_key

    def _encode_key_shared(self, content_hashs):
        return [f"{content_hash}@{self.deploy_key}" for content_hash in content_hashs]

    def get_keys(self, indices, content_hash):
        if self.page_size == 1:
            return self._encode_key_exclusive(indices)
        else:
            return self._encode_key_shared(content_hash)

    def get_flat_data(self, indices) -> Tuple[Optional[torch.Tensor], List[bool]]:
        logger.debug(f"Get flat data indices {indices}")
        keys = self._encode_key_exclusive(indices)
        bs = G_TensorPoolSize
        ret = []
        masks = []

        for i in range(0, len(keys), bs):
            key = keys[i : i + bs]
            objs, success_mask = self.pris_client.batch_get(key)
            if objs is None:
                logger.error(f"Get flat data keys {key} failed, pris_client returned None")
                return None, []
            copy_objs = objs.clone()
            ret.extend([copy_objs[i] for i in range(copy_objs.shape[0])])
            masks.extend(success_mask)

        if len(ret) == 0:
            logger.error(
                f"Get flat data keys size {len(keys)} failed, pris_client returned none, ret {ret}"
            )
            return None, []

        flat_data = torch.cat(ret, dim=self.split_dim)
        return flat_data, masks

    def assign_flat_data(self, indices, flat_data):
        logger.debug(f"Assign flat data indices {indices},shape={flat_data.shape},split_dim={self.split_dim}")
        start_time = time.perf_counter()

        keys = self._encode_key_exclusive(indices)
        flat_data = flat_data.contiguous()
        values = torch.split(flat_data, 1, dim=self.split_dim)

        bs = G_TensorPoolSize
        split_time = time.perf_counter()
        for i in range(0, len(keys), bs):
            key = keys[i : i + bs]
            value = values[i : i + bs]
            ret = self.pris_client.set(key, value)
            if not ret:
                logger.error(
                    f"Assign flat data keys {key} failed, pris_client returned None"
                )
                return False
        cost_time = time.perf_counter() - split_time
        if cost_time > 1:
            logger.warning(
                f"Finish assign flat data, total keys {len(keys)}, split time {split_time - start_time}, transfer time {cost_time}"
            )
        return True

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num
        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def exist_page(self, content_hashs):
        keys = self._encode_key_shared(content_hashs)
        ret = self.pris_client.exists_batch(keys)
        res = []
        for i, exist in enumerate(ret):
            if exist:
                res.append(content_hashs[i])
            else:
                break
        return res

    def get_page_data(self, content_hashs):
        logger.debug(f"Get flat data content_hashs {content_hashs}")
        keys = self._encode_key_shared(content_hashs)
        bs = G_TensorPoolSize
        ret = []
        masks = []

        for i in range(0, len(keys), bs):
            key = keys[i : i + bs]
            objs, success_mask = self.pris_client.batch_get(key)
            if objs is None:
                logger.error(f"Get flat data keys {key} failed, pris_client returned None")
                return None, []
            copy_objs = objs.clone()
            ret.extend([copy_objs[i] for i in range(copy_objs.shape[0])])
            masks.extend(success_mask)

        if len(ret) == 0:
            logger.error(
                f"Get flat data keys size {len(keys)} failed, pris_client returned none, ret {ret}"
            )
            return None, []

        flat_data = torch.cat(ret, dim=self.split_dim)
        return flat_data, masks

    def assign_page_data(self, content_hashs, flat_data):
        logger.debug(f"Assign flat data hashs {content_hashs}")
        keys = self._encode_key_shared(content_hashs)
        flat_data = flat_data.contiguous()
        values = torch.split(flat_data, self.page_size, dim=self.split_dim)
        bs = G_TensorPoolSize

        for i in range(0, len(keys), bs):
            key = keys[i : i + bs]
            value = values[i : i + bs]
            ret = self.pris_client.set(key, value)
            if not ret:
                logger.error(
                    f"Assign flat data keys {key} failed, pris_client returned None"
                )
                return False

        return True

    @debug_timing
    def transfer(self, indices, flat_data):
        return self.assign_flat_data(indices, flat_data)

    @synchronized()
    def clear(self):
        self.mem_state.fill_(0)
        self.can_use_mem_size = self.size
        self.free_slots = torch.arange(self.size, dtype=torch.int32)

    @synchronized()
    def get_state(self, indices: torch.Tensor) -> MemoryStateInt:
        assert len(indices) > 0, "The indices should not be empty"
        states = self.mem_state[indices]
        assert (
            states == states[0]
        ).all(), "The memory slots should have the same state {}".format(states)
        return MemoryStateInt(states[0].item())

    @synchronized()
    def alloc(self, need_size: int) -> torch.Tensor:
        if need_size > self.can_use_mem_size:
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        self.mem_state[select_index] = MemoryStateInt.RESERVED
        self.can_use_mem_size -= need_size
        return select_index

    @synchronized()
    def is_reserved(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.RESERVED

    @synchronized()
    def is_protected(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.PROTECTED

    @synchronized()
    def is_synced(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.SYNCED

    @synchronized()
    def is_backup(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.BACKUP

    @synchronized()
    def update_backup(self, indices: torch.Tensor):
        assert self.is_synced(indices) or (
            self.page_size > 1 and self.is_reserved(indices)
        ), (
            f"The host memory slots should be in SYNCED state before turning into BACKUP. "
            f"Current state: {self.get_state(indices)}"
        )
        self.mem_state[indices] = MemoryStateInt.BACKUP

    @synchronized()
    def update_synced(self, indices: torch.Tensor):
        self.mem_state[indices] = MemoryStateInt.SYNCED

    @synchronized()
    def protect_write(self, indices: torch.Tensor):
        assert self.is_reserved(indices), (
            f"The host memory slots should be RESERVED before write operations. "
            f"Current state: {self.get_state(indices)}"
        )
        self.mem_state[indices] = MemoryStateInt.PROTECTED

    @synchronized()
    def protect_load(self, indices: torch.Tensor):
        self.mem_state[indices] = MemoryStateInt.PROTECTED

    @synchronized()
    def complete_io(self, indices: torch.Tensor):
        assert self.is_protected(indices), (
            f"The host memory slots should be PROTECTED during I/O operations. "
            f"Current state: {self.get_state(indices)}"
        )
        self.mem_state[indices] = MemoryStateInt.SYNCED

    def available_size(self):
        return len(self.free_slots)

    @synchronized()
    def free(self, indices: torch.Tensor, content_hash) -> int:
        result = (self.mem_state[indices] == MemoryStateInt.BACKUP).all()
        logger.debug(f"check mem state: {result}, (indices: {indices})")
        keys = self.get_keys(indices, content_hash)
        logger.info(f"free keys[0]: {keys[0]}")
        self.pris_client.delete(keys)
        self.mem_state[indices] = MemoryStateInt.IDLE
        self.free_slots = torch.concat([self.free_slots, indices])
        self.can_use_mem_size += len(indices)
        return len(indices)


class OffloadMHATokenToKVPoolHost(OffloadBaseTokenToKVPoolHost):
    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        device: str = "cpu",
        page_size: int = 1,
        rank: int = 0,
        extra_info: Optional[dict] = None,
    ):
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            device,
            page_size,
            rank,
            extra_info,
        )
        self.head_num = device_pool.head_num
        self.head_dim = device_pool.head_dim
        self.layer_num = device_pool.layer_num
        self.size_per_token = (
            self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2
        )
        self.kvcache_shape = (
            2,
            self.layer_num,
            page_size,
            self.head_num,
            self.head_dim,
        )
        self.pris_client = PrisKVClient(
            None, self.dtype, self.kvcache_shape, device_pool.device
        )




class OffloadMLATokenToKVPoolHost(OffloadBaseTokenToKVPoolHost):
    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        device: str = "cpu",
        page_size: int = 1,
        rank: int = 0,
        extra_info: Optional[dict] = None,
    ):
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            device,
            page_size,
            rank,
            extra_info,
        )
        self.kv_lora_rank = self.device_pool.kv_lora_rank
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        self.layer_num = self.device_pool.layer_num
        self.size_per_token = (
            (self.kv_lora_rank + self.qk_rope_head_dim) * 1 * self.dtype.itemsize
        )
        self.kvcache_shape = (
            self.layer_num,
            page_size,
            1,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )
        self.pris_client = PrisKVClient(
            None, self.dtype, self.kvcache_shape, device_pool.device
        )
        self.split_dim = 1

    def get_size_per_token(self):
        self.kv_lora_rank = self.device_pool.kv_lora_rank
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        self.layer_num = self.device_pool.layer_num
        return (
            (self.kv_lora_rank + self.qk_rope_head_dim)
            * 1
            * self.dtype.itemsize
            * self.layer_num
        )