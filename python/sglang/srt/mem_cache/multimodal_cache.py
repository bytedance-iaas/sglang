import abc
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator


class MultimodalCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
    ):
        pass
        # self.hidden_size = hidden_size
        # self.max_size = max_size
        # self.page_size = page_size
        # self.dtype = dtype
        # self.device = device
        # if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        #     # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
        #     self.store_dtype = torch.uint8
        # else:
        #     self.store_dtype = dtype
        # # self.memory_saver_adapter = TorchMemorySaverAdapter.create(
        # #     enable=enable_memory_saver
        # # )
        # self.mem_usage = 0

        # # used for chunked cpu-offloading
        # self.cpu_offloading_chunk_size = 8192

    @abc.abstractmethod
    def get_mm_embedding(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """
        Extract the embedding with the hash-ids of the queried items. Try combined hash first, if missed, fallback to individual hashes
        The returned tensor may not be contiguous
        """
        raise NotImplementedError()

    @staticmethod
    def combine_hashes(mm_hashes: List[int]) -> Optional[int]:
        """
        Get a combined hash from individual mm item hashes
        """
        if not mm_hashes:
            return None
        return hash(tuple(mm_hashes))

    @abc.abstractmethod
    def set_mm_embedding(
        self,
        mm_hash: int,
        embedding: torch.Tensor,
        mm_embedding_allocator: BaseTokenToKVPoolAllocator,
    ) -> bool:
        """
        Set the embedding to the pre-allocated locations with a hash id
        """
        raise NotImplementedError()

    # @abc.abstractmethod
    # def get_pointers_from_locs(self, locs: torch.Tensor) -> torch.Tensor:
    #     """
    #     Given a tensor of locations (indices), returns a tensor of pointers
    #     to these locations in the multimodal buffer.
    #     """
    #     raise NotImplementedError()

    @abc.abstractmethod
    def has(self, mm_hash: int) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def free(
        self, mm_hash: int, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def available_size(self):
        raise NotImplementedError()


def _get_tensor_size(embedding: torch.Tensor):
    return embedding.element_size() * embedding.numel()


class MultiModalStaticCache(MultimodalCache):
    """MultiModalStaticCache is used to store precomputed multimodal embeddings.
    Embeddings will be computed prior, and this cache does not really pre-alloc
    """

    def __init__(
        self,
        max_size: int,
    ):
        super().__init__()
        self.max_size = max_size
        self.mm_cache: Dict[int, torch.Tensor] = {}
        self.current_size = 0

    def get_mm_embedding(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[torch.Tensor]:

        combined_hash = self.combine_hashes(mm_hashes)
        # MultiModalStaticCache does not fallback to individual item lookup
        return self.mm_cache.get(combined_hash)

    def set_mm_embedding(
        self, mm_hash: int, embedding: torch.Tensor, loc: Optional[torch.Tensor] = None
    ) -> bool:
        if mm_hash in self.mm_cache:
            return True
        data_size = _get_tensor_size(embedding)
        if self.current_size + data_size > self.max_size:
            return False
        self.mm_cache[mm_hash] = embedding
        self.current_size += data_size
        return True

    def has(self, mm_hash: int) -> bool:
        return mm_hash in self.mm_cache

    def free(
        self, mm_hash: int, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> bool:
        if mm_hash not in self.mm_cache:
            return False
        old_embedding = self.mm_cache.pop(mm_hash)
        self.current_size -= _get_tensor_size(old_embedding)
        return True

    def clear(self):
        self.mm_cache.clear()
        self.current_size = 0

    def __len__(self):
        return len(self.mm_cache)

    def available_size(self):
        return self.__len__()


class PagedMultiModalEmbeddingPool(MultimodalCache):
    """PagedMultiModalCache pre-allocates a buffer to store multimodal embeddings,
    and works with an external paged allocator. The allocator manages the allocation
    of token slots from this cache's buffer.
    """

    def __init__(
        self,
        size: int,
        hidden_size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.size = size  # Number of token slots
        self.hidden_size = hidden_size
        self.page_size = page_size

        self.mm_hash_to_indices: Dict[int, torch.Tensor] = {}
        self.mm_hash_count = Counter()

        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self.used_size = 0
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype

        self.capacity = self.size + self.page_size

        self.mm_buffer = torch.zeros(
            (self.capacity, self.hidden_size),
            dtype=self.store_dtype,
            device=self.device,
        )
        # self.memory_saver_adapter = TorchMemorySaverAdapter.create(
        #     enable=enable_memory_saver
        # )
        self.mem_usage = 0

        # used for chunked cpu-offloading
        # self.cpu_offloading_chunk_size = 8192

    # for disaggregation, aligned with get_contiguous_buf_infos
    def get_mm_buffer_info(self) -> Tuple[List[int], List[int], List[int]]:
        """Returns the pointer, size, and item length of the multimodal buffer."""
        return (
            [self.mm_buffer.data_ptr()],
            [self.mm_buffer.nbytes],
            [self.mm_buffer[0].nbytes * self.page_size],
        )

    def stored_mm_hashes(self) -> List[int]:
        return list(self.mm_hash_to_indices.keys())

    def get_pointers_from_locs(self, locs: torch.Tensor) -> torch.Tensor:
        """
        Given a tensor of locations (indices), returns a tensor of pointers
        to these locations in the multimodal buffer.
        """
        base_ptr = self.mm_buffer.data_ptr()
        item_size = self.hidden_size * self.mm_buffer.element_size()

        # Ensure locs is on the right device and is of type int64 for arithmetic
        locs_gpu = locs.to(device=self.device, dtype=torch.int64)

        # Calculate pointers
        pointers = base_ptr + locs_gpu * item_size
        return pointers

    def get_embedding_locs_from_hash(self, mm_hash: int) -> torch.Tensor:
        return self.mm_hash_to_indices[mm_hash]

    def get_embedding_locs_from_hashes(
        self, mm_hashes: List[int]
    ) -> List[torch.Tensor]:
        return [self.get_embedding_locs_from_hash(mm_hash) for mm_hash in mm_hashes]

    def try_get_mm_embedding(self, combined_hash: int) -> Optional[torch.Tensor]:
        indices = self.mm_hash_to_indices.get(combined_hash)
        if indices is None:
            return None

        embedding = self.mm_buffer.index_select(0, indices)

        if self.store_dtype != self.dtype:
            return embedding.view(self.dtype)
        return embedding

    def get_mm_embedding(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """
        Tries to get the multimodal embedding using a combined hash.
        If that fails, falls back to getting embeddings for each item individually
        and concatenating them.
        """
        # 1. Try with combined hash
        combined_hash = combined_hash or self.combine_hashes(mm_hashes)
        combined_embedding = self.try_get_mm_embedding(combined_hash)
        if combined_embedding is not None:
            return combined_embedding

        # 2. Fallback to individual item hashes
        individual_embeddings = []
        for h in mm_hashes:
            individual_embedding = self.try_get_mm_embedding(h)
            if individual_embedding is None:
                # If any part is missing, we can't reconstruct the full embedding.
                return None
            individual_embeddings.append(individual_embedding)

        # 3. Concatenate and return
        if not individual_embeddings:
            return None

        return torch.cat(individual_embeddings, dim=0)

    def set_mm_embedding(
        self,
        mm_hash: int,
        embedding: torch.Tensor,
        mm_embedding_allocator: BaseTokenToKVPoolAllocator,
    ) -> bool:
        if mm_hash in self.mm_hash_to_indices:
            self.mm_hash_count[mm_hash] += 1
            return True

        self.mm_hash_count[mm_hash] = 1
        loc = mm_embedding_allocator.alloc(embedding.shape[0])
        if loc is None:
            raise RuntimeError("Out of memory—needs to be handled.")

        if embedding.dtype != self.dtype:
            embedding = embedding.to(self.dtype)

        if self.store_dtype != self.dtype:
            embedding_to_store = embedding.view(self.store_dtype)
        else:
            embedding_to_store = embedding

        self.mm_buffer.index_put_((loc,), embedding_to_store)
        self.mm_hash_to_indices[mm_hash] = loc
        self.used_size += embedding.size(0)

        return True

    def reserve_mm_embedding(
        self,
        mm_hash: int,
        num_tokens: int,
        mm_embedding_allocator: BaseTokenToKVPoolAllocator,
    ) -> torch.Tensor:
        if mm_hash in self.mm_hash_to_indices:
            self.mm_hash_count[mm_hash] += 1
            return self.mm_hash_to_indices[mm_hash]
        self.mm_hash_count[mm_hash] = 1
        # Even if mm_hash exists in mm_hash_to_indices, it should still return loc
        # caching should be handled elsewhere
        loc = mm_embedding_allocator.alloc(num_tokens)
        if loc is None:
            raise RuntimeError("Out of memory—needs to be handled.")

        self.mm_hash_to_indices[mm_hash] = loc
        self.used_size += num_tokens
        return loc

    def has(self, mm_hash: int) -> bool:
        return mm_hash in self.mm_hash_to_indices

    def free(
        self, mm_hash: int, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> torch.Tensor:
        if mm_hash in self.mm_hash_to_indices:
            self.mm_hash_count[mm_hash] -= 1
            if self.mm_hash_count[mm_hash] <= 0:
                del self.mm_hash_count[mm_hash]
                indices = self.mm_hash_to_indices.pop(mm_hash)
                self.used_size -= indices.size(0)
                mm_embedding_allocator.free(indices)
                return indices
        return torch.tensor([])

    def clear(self):
        self.mm_hash_to_indices.clear()

    def __len__(self):
        return len(self.mm_hash_to_indices)

    def allocated(self) -> int:
        return self.used_size

    def available_size(self):
        return self.capacity - self.allocated()
