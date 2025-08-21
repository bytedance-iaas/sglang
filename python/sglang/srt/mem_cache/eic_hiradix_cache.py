import heapq
import logging
import os
import threading
import time
from functools import partial
from typing import List, Optional

import torch
import yaml

from sglang.srt.managers.eic_cache_controller import (
    EICCacheController,
    get_content_hash,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.eic_memory_pool import (
    EICMHATokenToKVPoolHost,
    EICMLATokenToKVPoolHost,
    MemoryStateInt,
    get_eic_config_file_path,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class EICHiRadixCacheBuilder:
    @staticmethod
    def build(
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_cache_group: torch.distributed.ProcessGroup,
        page_size: int,
        hicache_ratio: float,
        hicache_size: int,
        hicache_write_policy: str,
        server_args: ServerArgs,
    ):
        if server_args.disable_eic_shared:
            return EICHiRadixCache(
                req_to_token_pool,
                token_to_kv_pool_allocator,
                tp_cache_group,
                page_size,
                hicache_ratio,
                hicache_size,
                hicache_write_policy,
                server_args,
            )
        else:
            return EICPagedHiRadixCache(
                req_to_token_pool,
                token_to_kv_pool_allocator,
                tp_cache_group,
                page_size,
                hicache_ratio,
                hicache_size,
                hicache_write_policy,
                server_args,
            )


def mha_pool_get_flat_data(self: MHATokenToKVPool, indices: torch.Tensor):
    flatten = torch.stack(
        [
            torch.stack([self.k_buffer[i][indices] for i in range(self.layer_num)]),
            torch.stack([self.v_buffer[i][indices] for i in range(self.layer_num)]),
        ]
    )
    return flatten


def mha_pool_transfer(
    self: MHATokenToKVPool, indices: torch.Tensor, flat_data: torch.Tensor
):
    flat_data = flat_data.to(device=self.device, non_blocking=False)
    k_data, v_data = flat_data[0], flat_data[1]
    for i in range(self.layer_num):
        self.k_buffer[i][indices] = k_data[i]
        self.v_buffer[i][indices] = v_data[i]


def mla_pool_get_flat_data(self: MLATokenToKVPool, indices: torch.Tensor):
    return torch.stack([self.kv_buffer[i][indices] for i in range(self.layer_num)])


def mla_pool_transfer(
    self: MLATokenToKVPool, indices: torch.Tensor, flat_data: torch.Tensor
):
    flat_data = flat_data.to(device=self.device, non_blocking=False)
    for i in range(self.layer_num):
        self.kv_buffer[i][indices] = flat_data[i]


class EICHiRadixCache(RadixCache):

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_cache_group: torch.distributed.ProcessGroup,
        page_size: int,
        hicache_ratio: float,
        hicache_size: int,
        hicache_write_policy: str,
        server_args: ServerArgs,
    ):
        self.tp_group = tp_cache_group
        self.tp_size = self.tp_group.size()
        self.rank = self.tp_group.rank()
        self.kv_cache = token_to_kv_pool_allocator.get_kvcache()
        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = EICMHATokenToKVPoolHost(
                self.kv_cache,
                hicache_ratio,
                hicache_size,
                "cpu",
                page_size,
                self.rank,
                extra_info=self.get_extra_info(server_args),
            )
            self.kv_cache.get_flat_data = partial(mha_pool_get_flat_data, self.kv_cache)
            self.kv_cache.transfer = partial(mha_pool_transfer, self.kv_cache)
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = EICMLATokenToKVPoolHost(
                self.kv_cache,
                hicache_ratio,
                hicache_size,
                "cpu",
                page_size,
                self.rank,
                extra_info=self.get_extra_info(server_args),
            )
            self.kv_cache.get_flat_data = partial(mla_pool_get_flat_data, self.kv_cache)
            self.kv_cache.transfer = partial(mla_pool_transfer, self.kv_cache)
        else:
            raise ValueError(f"HiRadixCache only supports MHA and MLA yet")

        self.load_cache_event = threading.Event()
        self.cache_controller = EICCacheController(
            token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            page_size,
            load_cache_event=self.load_cache_event,
            write_policy=hicache_write_policy,
            server_args=server_args,
        )

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if hicache_write_policy == "write_through" else 3
        )
        self.load_back_threshold = 10
        super().__init__(
            req_to_token_pool, token_to_kv_pool_allocator, page_size, disable=False
        )

        self.save_decode_cache = True
        config_file = get_eic_config_file_path()
        if os.path.exists(config_file):
            with open(config_file, "r") as fin:
                config = yaml.safe_load(fin)
            self.init_hyper_params(config)

    def get_extra_info(self, server_args: ServerArgs):
        # TODO update when sglang support pp
        extra_info = {
            "model_path": server_args.model_path,
            "world_size": self.tp_size,
            "tp_rank": self.rank,
            "framework": "sglang",
        }
        return extra_info

    def init_hyper_params(self, config: dict):
        self.save_decode_cache = config.get("save_decode_cache", True)
        logger.info(
            f"EICHiRadixCache save_decode_cache set to {self.save_decode_cache}"
        )
        self.load_back_threshold = config.get("load_back_threshold", 10)
        logger.info(
            f"EICHiRadixCache load_back_threshold set to {self.load_back_threshold}"
        )

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        self.ongoing_load_back = {}
        self.ongoing_write_through = {}
        super().reset()

    def cache_finished_req(self, req: Req, is_decode: bool = False):
        """Cache request when it finishes."""
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.origin_input_ids) + len(req.output_ids) - 1
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].clone()
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.clone()

        # Radix Cache takes one ref in memory pool
        eic_backup = True
        if not self.save_decode_cache and is_decode:
            eic_backup = False

        new_prefix_len = self.insert(
            token_ids[:page_aligned_len], page_aligned_kv_indices, eic_backup=eic_backup
        )
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
        )

        # Remove req slot release the cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def write_backup(self, node: TreeNode, write_back=False):
        logger.debug(f"write backup for node {node.id}")
        if node.evicted:
            return 0
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            priority=-self.get_height(node),
            node_id=node.id,
        )
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                priority=-self.get_height(node),
                node_id=node.id,
            )
        if host_indices is not None:
            node.host_value = host_indices
            self.ongoing_write_through[node.id] = node
            if not write_back:
                self.inc_lock_ref(node)
        else:
            return 0

        return len(host_indices)

    def inc_hit_count(self, node: TreeNode):
        if node.backuped or self.cache_controller.write_policy == "write_back":
            return
        node.hit_count += 1
        if node.hit_count >= self.write_through_threshold:
            self.write_backup(node)
            node.hit_count = 0

    def get_tp_result(self, flag):
        if isinstance(flag, bool):
            flag = [flag]
        if self.tp_size <= 1:
            return flag
        # synchronize the result across TP workers
        temp = [0 if x else 1 for x in flag]
        temp_tensor = torch.tensor(temp, dtype=torch.int64, device="cpu")
        torch.distributed.all_reduce(
            temp_tensor, op=torch.distributed.ReduceOp.SUM, group=self.tp_group
        )
        result_list = temp_tensor.tolist()
        result = []
        for i in range(len(result_list)):
            result.append(result_list[i] == 0)
        return result

    def writing_check(self, write_back=False):
        if len(self.ongoing_write_through) == 0:
            return
        write_check_start_time = time.perf_counter()
        if write_back:
            while (
                len(self.ongoing_write_through)
                != self.cache_controller.ack_write_queue.qsize()
            ):
                time.sleep(0.01)
        queue_size = torch.tensor(
            self.cache_controller.ack_write_queue.qsize(), dtype=torch.int
        )
        if torch.distributed.get_world_size(group=self.tp_group) > 1:
            # synchrnoize TP workers to make the same update to radix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        ack_list = []
        flags = []
        for _ in range(queue_size.item()):
            ack_id, success = self.cache_controller.ack_write_queue.get_nowait()
            ack_list.append(ack_id)
            flags.append(success)
        flags = self.get_tp_result(flags)
        for ack_id, success in zip(ack_list, flags):
            if (
                not success
                and self.ongoing_write_through[ack_id].host_value is not None
            ):
                if (
                    self.cache_controller.mem_pool_host.get_state(
                        self.ongoing_write_through[ack_id].host_value
                    )
                    != MemoryStateInt.IDLE
                ):
                    self.cache_controller.mem_pool_host.free(
                        self.ongoing_write_through[ack_id].host_value
                    )
                self.ongoing_write_through[ack_id].host_value = None
            if not write_back:
                self.dec_lock_ref(self.ongoing_write_through[ack_id])
            # clear the reference
            del self.ongoing_write_through[ack_id]
        cost_time = time.perf_counter() - write_check_start_time
        if cost_time > 0.1:
            logger.warning(
                f"writing check cost {cost_time:.3f} seconds, "
                f"queue size {queue_size.item()}"
            )

    def loading_check(self):
        if len(self.ongoing_load_back) == 0:
            return
        loading_check_start_time = time.perf_counter()
        queue_size = torch.tensor(
            self.cache_controller.ack_load_queue.qsize(), dtype=torch.int
        )
        if torch.distributed.get_world_size(group=self.tp_group) > 1:
            # synchrnoize TP workers to make the same update to radix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        ack_list = []
        flags = []
        for _ in range(queue_size.item()):
            ack_id, success = self.cache_controller.ack_load_queue.get_nowait()
            ack_list.append(ack_id)
            flags.append(success)
        flags = self.get_tp_result(flags)
        for ack_id, success in zip(ack_list, flags):
            start_node, end_node = self.ongoing_load_back[ack_id]
            self.dec_lock_ref(end_node)
            while end_node != start_node:
                assert end_node.loading
                if not success:
                    self.cache_controller.mem_pool_device_allocator.free(end_node.value)
                    self.evictable_size_ -= len(end_node.value)
                    end_node.value = None
                end_node.loading = False
                end_node = end_node.parent
            # clear the reference
            del self.ongoing_load_back[ack_id]
        cost_time = time.perf_counter() - loading_check_start_time
        if cost_time > 0.1:
            logger.warning(
                f"loading check cost {cost_time:.3f} seconds, "
                f"queue size {queue_size.item()}"
            )

    # TODO: is not correct for eic, but neednt to be fixed rightnow
    def evictable_size(self):
        return self.evictable_size_

    def evict(self, num_tokens: int, evict_callback=None, retry_times: int = 3):
        while len(self.ongoing_write_through) > 50 or len(self.ongoing_load_back) > 50:
            self.writing_check()
            self.loading_check()
            time.sleep(0.1)

        num_evicted = 0
        while retry_times > 0:
            retry_times -= 1
            leaves = self._collect_leaves_device()
            heapq.heapify(leaves)

            write_back_nodes = []
            idx = 0

            logger.debug(
                f"evict {num_tokens} tokens, current evictable size {self.evictable_size_}, protect_size {self.protected_size_}, leaves {len(leaves)}"
            )
            while num_evicted < num_tokens and len(leaves):
                x = heapq.heappop(leaves)
                logger.debug(f"evicting {idx} node {x.id}, access {x.last_access_time}")
                idx += 1

                if x.lock_ref > 0:
                    logger.debug(f"node {x.id} is locked, skip eviction")
                    continue

                if not x.backuped:
                    if self.cache_controller.write_policy == "write_back":
                        # write to host if the node is not backuped
                        num_evicted += self.write_backup(x, write_back=True)
                        write_back_nodes.append(x)
                    else:
                        num_evicted += self._evict_regular(x)
                else:
                    num_evicted += self._evict_backuped(x)

                for child in x.parent.children.values():
                    if child in write_back_nodes:
                        continue
                    if not child.evicted:
                        break
                else:
                    # all children are evicted or no children
                    heapq.heappush(leaves, x.parent)

            if self.cache_controller.write_policy == "write_back":
                # blocking till all write back complete
                self.writing_check(write_back=True)
                for node in write_back_nodes:
                    if node.backuped:
                        self._evict_backuped(node)
                    else:
                        self._evict_regular(node)

            if num_evicted < num_tokens:
                logger.info(
                    f"only evicted {num_evicted} tokens, less than requested {num_tokens}"
                )
            else:
                return

    def _evict_backuped(self, node: TreeNode):
        if node.host_value is None:
            logger.error(f"host value is None for node {node.id}")
            return self._evict_regular(node)
        num_evicted = self.cache_controller.evict_device(node.value, node.host_value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None
        return num_evicted

    def _evict_regular(self, node: TreeNode):
        # evict a node not initiated write to host
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted

    def evict_host(self, num_tokens: int):
        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)
            if x == self.root_node:
                break
            # only evict the host value of evicted nodes
            if not x.evicted:
                continue
            assert x.lock_ref == 0 and x.host_value is not None

            assert self.cache_controller.evict_host(x.host_value) > 0
            for k, v in x.parent.children.items():
                if v == x:
                    break
            del x.parent.children[k]

            if len(x.parent.children) == 0 and x.parent.evicted:
                heapq.heappush(leaves, x.parent)

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        # todo: more loading policies

        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        delta = self.inc_lock_ref(ancester_node)

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None

        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=last_hit_node.id
        )
        if device_indices is None:
            self.evict(len(host_indices))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=last_hit_node.id
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            return None

        self.ongoing_load_back[last_hit_node.id] = (ancester_node, last_hit_node)
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)]
            offset += len(node.host_value)
            node.loading = True
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        return device_indices

    def loading_complete(self, node: TreeNode):
        self.loading_check()
        return node.loading == False

    def init_load_back(
        self,
        last_node: TreeNode,
        host_hit_length: int,
        mem_quota: Optional[int] = None,
    ):
        _ = host_hit_length  # unused, but kept for compatibility
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node

            while last_node.evicted:
                last_node = last_node.parent

        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )

    def ready_to_load_host_cache(self):
        producer_index = self.cache_controller.layer_done_counter.next_producer()
        self.load_cache_event.set()
        return producer_index

    def check_hicache_events(self):
        self.writing_check()
        self.loading_check()

    def match_prefix(self, key: List[int], **kwargs):
        empty_value = torch.empty((0,), dtype=torch.int64, device=self.device)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=empty_value,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = empty_value

        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
        )

    def _match_prefix_helper(self, node: TreeNode, key: List):
        node.last_access_time = time.monotonic()
        child_key = self.get_child_key_fn(key)
        value = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                self.inc_hit_count(new_node)
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                self.inc_hit_count(child)
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.loading = child.loading
        new_node.hit_count = child.hit_count

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len]
            child.value = child.value[split_len:]
        if child.backuped:
            new_node.host_value = child.host_value[:split_len]
            child.host_value = child.host_value[split_len:]
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        return new_node

    def insert(self, key: List, value=None, eic_backup=True):
        if self.disable:
            return 0

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value, eic_backup)

    def _insert_helper(self, node: TreeNode, key: List, value, eic_backup=True):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len == len(node.key):
                if node.evicted:
                    # change the reference if the node is evicted
                    # this often happens in the case of KV cache recomputation
                    node.value = value[:prefix_len]
                    if not isinstance(self, EICPagedHiRadixCache):
                        self.token_to_kv_pool_host.free(node.host_value)
                    self.evictable_size_ += len(node.value)
                    if eic_backup:
                        self.inc_hit_count(node)
                else:
                    if eic_backup:
                        self.inc_hit_count(node)
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                if new_node.evicted:
                    new_node.value = value[:prefix_len]
                    if not isinstance(self, EICPagedHiRadixCache):
                        self.token_to_kv_pool_host.free(new_node.host_value)
                    self.evictable_size_ += len(new_node.value)
                    if eic_backup:
                        self.inc_hit_count(new_node)
                else:
                    if eic_backup:
                        self.inc_hit_count(new_node)
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)

            if self.cache_controller.write_policy != "write_back" and eic_backup:
                self.inc_hit_count(new_node)
        return total_prefix_length

    def _collect_leaves_device(self):
        def is_leaf(node):
            if node.evicted:
                return False
            if node == self.root_node:
                return False
            if len(node.children) == 0:
                return True
            for child in node.children.values():
                if not child.evicted:
                    return False
            return True

        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            if is_leaf(cur_node):
                ret_list.append(cur_node)
            else:
                for cur_child in cur_node.children.values():
                    if not cur_child.evicted:
                        stack.append(cur_child)
        return ret_list


def _need_calculate_hash(node: TreeNode, page_size: int):
    if node is None or node.key is None or len(node.key) == 0:
        return False
    return node.content_hash is None or len(node.key) // page_size != len(
        node.content_hash
    )


class EICPagedHiRadixCache(EICHiRadixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_cache_group: torch.distributed.ProcessGroup,
        page_size: int,
        hicache_ratio: float,
        hicache_size: int,
        hicache_write_policy: str,
        server_args: ServerArgs,
    ):
        self.calculate_hash_fn = get_content_hash
        self.load_remote_threshold = 100
        self.match_req_set = []
        self.eic_check_max_num = -1
        super().__init__(
            req_to_token_pool,
            token_to_kv_pool_allocator,
            tp_cache_group,
            page_size,
            hicache_ratio,
            hicache_size,
            hicache_write_policy,
            server_args,
        )

    def init_hyper_params(self, config):
        super().init_hyper_params(config)
        self.load_remote_threshold = max(
            config.get("load_remote_threshold", 100), self.page_size
        )
        logger.info(
            f"EICPagedHiRadixCache load_remote_threshold set to {self.load_remote_threshold}"
        )
        self.eic_check_max_num = config.get("eic_check_max_num", -1)
        logger.info(
            f"EICPagedHiRadixCache eic_check_max_num set to {self.eic_check_max_num}"
        )

    def _calculate_content_hash(self, node: TreeNode):
        if _need_calculate_hash(node.parent, self.page_size):
            self._calculate_content_hash(node.parent)
        if node.parent is not None and node.parent.content_hash is not None:
            prev_node_hash = node.parent.content_hash[-1]
        else:
            prev_node_hash = None
        node.content_hash = self.calculate_hash_fn(
            node.key, self.page_size, prev_node_hash
        )

    def _split_node(self, key, child: TreeNode, split_len: int):
        assert (
            split_len % self.page_size == 0
        ), f"split_len {split_len} is not page aligned"
        # child node split into new_node -> child
        if _need_calculate_hash(child, self.page_size):
            self._calculate_content_hash(child)
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.loading = child.loading
        new_node.hit_count = child.hit_count
        split_hash_nums = split_len // self.page_size
        new_node.content_hash = child.content_hash[:split_hash_nums]
        child.content_hash = child.content_hash[split_hash_nums:]

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len]
            child.value = child.value[split_len:]
        if child.backuped:
            new_node.host_value = child.host_value[:split_len]
            child.host_value = child.host_value[split_len:]
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        return new_node

    def match_prefix_extend(self, key: List[int], last_node):
        cache_prefix_len = 0
        temp_node = last_node
        while temp_node:
            cache_prefix_len += len(temp_node.key)
            temp_node = temp_node.parent

        # if the cache prefix is too long, or the remaining key is too short, we can skip loading from eic
        if (len(key) - cache_prefix_len) < self.load_remote_threshold:
            return last_node

        logger.debug(
            f"few cache in radix, try load from eic, cache len {cache_prefix_len}, total len {len(key)}"
        )
        if _need_calculate_hash(last_node, self.page_size):
            self._calculate_content_hash(last_node)
        last_prev_hash = None
        if last_node.content_hash is not None and len(last_node.content_hash) > 0:
            last_prev_hash = last_node.content_hash[-1]
        need_compute_key = key[cache_prefix_len:]
        eic_hash, eic_key = self.cache_controller.find_longest_prefix_in_eic(
            need_compute_key, last_prev_hash
        )
        if self.tp_size > 1:
            eic_hash_len_tensor = torch.tensor(
                [len(eic_hash)], dtype=torch.int64, device="cpu"
            )
            torch.distributed.all_reduce(
                eic_hash_len_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            eic_hash_len = eic_hash_len_tensor.item()
            eic_hash = eic_hash[:eic_hash_len]
            eic_key = eic_key[: eic_hash_len * self.page_size]
        if len(eic_key) < self.load_remote_threshold:
            logger.debug(
                f"eic key is too short, skip loading from eic, eic cache len {len(eic_key)}, need compute key len {len(need_compute_key)}"
            )
            return last_node
        load_node = TreeNode()
        load_node.key = eic_key
        load_node.content_hash = eic_hash
        load_node.host_value = torch.arange(
            len(eic_key), dtype=torch.int32, device="cpu"
        )
        assert (
            last_node.children.get(self.get_child_key_fn(eic_key)) is None
        ), f"eic key {eic_key} already exists in radix cache"
        logger.debug(
            f"load token from eic: {len(eic_key)}, node {load_node.id}, parent {last_node.id}"
        )
        last_node.children[self.get_child_key_fn(eic_key)] = load_node
        load_node.parent = last_node
        return load_node

    def _match_for_remote_fetch(self, node: TreeNode, key: List):
        node.last_access_time = time.monotonic()
        child_key = self.get_child_key_fn(key)
        local_prefix_len = 0

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            local_prefix_len += prefix_len
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                node = new_node
                break
            else:
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)
        temp_node = node
        local_evict_len = 0
        while temp_node.evicted:
            local_evict_len += len(temp_node.host_value)
            temp_node = temp_node.parent
        return local_prefix_len, local_evict_len, node

    def _insert_remote_node(self, node: TreeNode, key: List):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len == len(node.key):
                if node.evicted and node.host_value is None:
                    node.host_value = torch.arange(
                        len(node.key), dtype=torch.int32, device="cpu"
                    )
                if not node.evicted:
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                if new_node.evicted and new_node.host_value is None:
                    new_node.host_value = torch.arange(
                        len(new_node.key), dtype=torch.int32, device="cpu"
                    )
                if not new_node.evicted:
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.host_value = torch.arange(
                len(key), dtype=torch.int32, device="cpu"
            )
            node.children[child_key] = new_node
            self._calculate_content_hash(new_node)
        return total_prefix_length

    def match_from_remote(self, waiting_queue: List[Req]):
        compute_keys = []
        prev_hashes = []
        fetch_list = []
        if len(self.match_req_set) > 1000:
            self.match_req_set = self.match_req_set[500:]
        eic_keys = 0
        for req in waiting_queue:
            logger.debug(f"req {req.rid} match from eic")
            if req.rid in self.match_req_set:
                continue
            key = req.adjust_max_prefix_ids()
            (local_prefix_len, local_evict_len, last_node) = (
                self._match_for_remote_fetch(self.root_node, key)
            )
            if (
                len(key) - local_prefix_len + local_evict_len
                < self.load_remote_threshold
            ):
                # skip loading from eic if the remaining key is too short
                logger.debug(f"req {req.rid} skip loading from eic")
                continue
            compute_keys.append(key[local_prefix_len:])
            if _need_calculate_hash(last_node, self.page_size):
                self._calculate_content_hash(last_node)
            prev_hashes.append(
                last_node.content_hash[-1] if last_node.content_hash else None
            )
            fetch_list.append((last_node, req, local_prefix_len, local_evict_len))
            self.match_req_set.append(req.rid)
            eic_keys += (len(key) - local_prefix_len) // self.page_size
            if self.eic_check_max_num > 0 and eic_keys >= self.eic_check_max_num:
                logger.info(
                    f"eic check max num {self.eic_check_max_num} reached, stop matching"
                )
                break
        if len(fetch_list) == 0:
            return
        # batch exist
        eic_prefix_lens = self.cache_controller.batch_find_longest_prefix_in_eic(
            compute_keys, prev_hashes
        )
        if len(eic_prefix_lens) == 0 or len(eic_prefix_lens) != len(fetch_list):
            return
        if self.tp_size > 1:
            eic_prefix_len_tensor = torch.tensor(
                eic_prefix_lens, dtype=torch.int64, device="cpu"
            )
            torch.distributed.all_reduce(
                eic_prefix_len_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            eic_prefix_lens = eic_prefix_len_tensor.tolist()
        for i, (last_node, req, local_prefix_len, local_evict_len) in enumerate(
            fetch_list
        ):
            eic_prefix_len = eic_prefix_lens[i]
            if eic_prefix_len + local_evict_len < self.load_remote_threshold:
                continue
            eic_key = compute_keys[i][:eic_prefix_len]
            logger.debug(
                f"req {req.rid} match from eic, "
                f"last node {last_node.id}, "
                f"local prefix len {local_prefix_len}, "
                f"eic prefix len {eic_prefix_len}"
            )
            self._insert_remote_node(last_node, eic_key)

    def match_prefix(self, key: List[int], **kwargs):
        empty_value = torch.empty((0,), dtype=torch.int64, device=self.device)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=empty_value,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = empty_value

        # last_node = self.match_prefix_extend(key, last_node)
        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
        )

    def write_backup(self, node: TreeNode, write_back=False):
        if node.evicted:
            return 0
        if _need_calculate_hash(node, self.page_size):
            self._calculate_content_hash(node)
        host_indices = self.cache_controller.write_page(
            device_indices=node.value,
            priority=-self.get_height(node),
            node_id=node.id,
            content_hash=node.content_hash,
        )
        if host_indices is not None:
            node.host_value = host_indices
            self.ongoing_write_through[node.id] = node
            if not write_back:
                self.inc_lock_ref(node)
        else:
            return 0

        return len(host_indices)

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        # todo: more loading policies

        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        delta = self.inc_lock_ref(ancester_node)

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None
        host_content_hash = []
        for n in nodes_to_load:
            host_content_hash.extend(n.content_hash)

        device_indices = self.cache_controller.load_page(
            host_indices=host_indices,
            node_id=last_hit_node.id,
            content_hash=host_content_hash,
        )
        if device_indices is None:
            self.evict(len(host_indices))
            device_indices = self.cache_controller.load_page(
                host_indices=host_indices,
                node_id=last_hit_node.id,
                content_hash=host_content_hash,
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            return None

        self.ongoing_load_back[last_hit_node.id] = (ancester_node, last_hit_node)
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)]
            offset += len(node.host_value)
            node.loading = True
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        return device_indices
