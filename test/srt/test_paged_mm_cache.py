import unittest

import torch

from sglang.srt.mem_cache.multimodal_cache import PagedMultiModalEmbeddingPool
from sglang.test.test_utils import CustomTestCase


class TestPagedMMCache(CustomTestCase):

    def test_paged_multimodal_cache(self):
        """
        A short code snippet to verify the functionality of PagedMultiModalCache.
        """
        # 1. Setup parameters
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_token_slots = 1024
        hidden_size = 128
        page_size = 16
        dtype = torch.float16

        print("--- 1. Initializing PagedMultiModalCache ---")
        cache = PagedMultiModalEmbeddingPool(
            size=num_token_slots,
            hidden_size=hidden_size,
            page_size=page_size,
            dtype=dtype,
            device=device,
        )
        print(f"Cache created on device: '{device}'")
        print(f"Buffer shape: {cache.mm_buffer.shape}\n")

        # 2. Simulate allocation for a new embedding
        print("--- 2. Simulating allocation from an allocator ---")
        embedding_len = 5
        # Let's assume the allocator gives us these non-contiguous token indices
        allocated_locs = torch.tensor([42, 15, 1, 888, 233], device=device)
        print(f"A new embedding (hash=12345) needs {embedding_len} tokens.")
        print(
            f"Allocator assigned non-contiguous locations: {allocated_locs.cpu().tolist()}\n"
        )

        # 3. Get pointers for the allocated locations
        print("--- 3. Verifying get_pointers_from_locs ---")
        pointers = cache.get_pointers_from_locs(allocated_locs)
        print(f"Retrieved {len(pointers)} pointers from the cache.")
        print(f"First pointer: {pointers[0]}")
        print(f"Pointers tensor (first 5): {pointers[:5].cpu().tolist()}\n")
        # A simple check to see if pointers are calculated correctly
        expected_first_ptr = (
            cache.mm_buffer.data_ptr()
            + allocated_locs[0] * hidden_size * cache.mm_buffer.element_size()
        )
        assert pointers[0] == expected_first_ptr, "Pointer calculation seems incorrect"
        print("Pointer calculation sanity check passed.\n")

        # 4. Create a dummy embedding and write to cache
        print("--- 4. Writing embedding to the cache ---")
        mm_hash = 12345
        dummy_embedding = torch.randn(
            embedding_len, hidden_size, dtype=dtype, device=device
        )
        is_success = cache.set_mm_embedding(
            mm_hash=mm_hash, embedding=dummy_embedding, loc=allocated_locs
        )
        assert is_success, "set_mm_embedding failed"
        print(f"Successfully called set_mm_embedding for hash {mm_hash}.\n")

        # 5. Read the embedding back
        print("--- 5. Reading embedding back from the cache ---")
        retrieved_embedding = cache.get_mm_embedding(mm_hash)
        print(f"Retrieved embedding of shape: {retrieved_embedding.shape}\n")

        # 6. Verification
        print("--- 6. Verifying correctness ---")
        are_equal = torch.allclose(dummy_embedding, retrieved_embedding)
        print(f"Is the retrieved embedding identical to the original? -> {are_equal}")
        assert (
            are_equal
        ), "Verification failed: retrieved embedding does not match the original."
        print("\nVerification successful!")

    def test_paged_multimodal_cache_direct_write(self):
        """
        A short code snippet to verify the functionality of PagedMultiModalCache,
        focusing on direct memory access simulation instead of using the setter API.
        """
        # 1. Setup parameters
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_token_slots = 1024
        hidden_size = 128
        page_size = 16
        dtype = torch.float16

        print("--- 1. Initializing PagedMultiModalCache ---")
        cache = PagedMultiModalEmbeddingPool(
            size=num_token_slots,
            hidden_size=hidden_size,
            page_size=page_size,
            dtype=dtype,
            device=device,
        )
        print(f"Cache created on device: '{device}'")
        print(f"Buffer shape: {cache.mm_buffer.shape}\n")

        # 2. Simulate allocation for a new embedding
        print("--- 2. Simulating allocation from an allocator ---")
        embedding_len = 5
        # Let's assume the allocator gives us these non-contiguous token indices
        allocated_locs = torch.tensor([42, 15, 1, 888, 233], device=device)
        mm_hash = 12345
        print(f"A new embedding (hash={mm_hash}) needs {embedding_len} tokens.")
        print(
            f"Allocator assigned non-contiguous locations: {allocated_locs.cpu().tolist()}\n"
        )

        # 3. Get pointers for the allocated locations to verify they can be retrieved
        print("--- 3. Verifying get_pointers_from_locs ---")
        pointers = cache.get_pointers_from_locs(allocated_locs)
        print(f"Successfully retrieved {len(pointers)} pointers from the cache.")
        print("This confirms that an external module could receive these pointers.\n")

        # 4. Simulate direct write to memory by bypassing the public API
        # In a real scenario, a C++/CUDA component would use the raw pointers.
        # Here, we simulate this by directly writing to the cache's internal buffer.
        print(
            "--- 4. Simulating direct write to cache buffer (bypassing setter API) ---"
        )
        dummy_embedding = torch.randn(
            embedding_len, hidden_size, dtype=dtype, device=device
        )

        # Prepare embedding for storage (e.g., handling float8)
        embedding_to_store = dummy_embedding
        if cache.store_dtype != cache.dtype:
            embedding_to_store = dummy_embedding.view(cache.store_dtype)

        # Directly write into the buffer at the allocated locations
        cache.mm_buffer.index_put_((allocated_locs,), embedding_to_store)
        print("Successfully wrote data directly into the internal buffer.")

        # After the external write, the manager needs to update the cache's metadata
        cache.mm_hash_to_indices[mm_hash] = allocated_locs
        print(f"Manually updated hash-to-indices mapping for hash {mm_hash}.\n")

        # 5. Read the embedding back
        print("--- 5. Reading embedding back from the cache ---")
        retrieved_embedding = cache.get_mm_embedding(mm_hash)
        print(f"Retrieved embedding of shape: {retrieved_embedding.shape}\n")

        # 6. Verification
        print("--- 6. Verifying correctness ---")
        are_equal = torch.allclose(dummy_embedding, retrieved_embedding)
        print(f"Is the retrieved embedding identical to the original? -> {are_equal}")
        assert (
            are_equal
        ), "Verification failed: retrieved embedding does not match the original."
        print("\nVerification successful!")


if __name__ == "__main__":
    unittest.main()
