import unittest

import torch

from sglang.kernels.ops.memory.allocator import (
    ALLOCATOR_BATCH_SIZE_STEP,
    alloc_decode_kernel,
    alloc_extend_kernel,
)
from sglang.srt.mem_cache.allocator.paged import alloc_extend_naive


class TestPagedAllocatorKernels(unittest.TestCase):
    def _inputs(self, batch_size: int, page_size: int):
        prefix_lens = torch.arange(batch_size, device="cuda", dtype=torch.int64)
        prefix_lens %= 2 * page_size + 1
        extend_lens = torch.arange(batch_size, device="cuda", dtype=torch.int64)
        extend_lens = extend_lens % (page_size + 3) + 1
        seq_lens = prefix_lens + extend_lens

        last_loc = torch.arange(batch_size, device="cuda", dtype=torch.int64)
        last_loc *= 4 * page_size
        last_loc += torch.clamp(prefix_lens - 1, min=0) % page_size
        last_loc = torch.where(prefix_lens == 0, -1, last_loc)

        pages_before = torch.div(
            prefix_lens + page_size - 1, page_size, rounding_mode="floor"
        )
        pages_after = torch.div(
            seq_lens + page_size - 1, page_size, rounding_mode="floor"
        )
        num_new_pages = int((pages_after - pages_before).sum().item())
        free_pages = torch.arange(
            10_000,
            10_000 + num_new_pages + batch_size + 1,
            device="cuda",
            dtype=torch.int64,
        )
        return prefix_lens, seq_lens, last_loc, free_pages

    def test_alloc_extend_reuses_module_across_batch_sizes(self):
        for page_size in (1, 32, 64):
            compiled_kernel = None
            for batch_size in (1, 7, 32, ALLOCATOR_BATCH_SIZE_STEP + 1):
                with self.subTest(page_size=page_size, batch_size=batch_size):
                    prefix_lens, seq_lens, last_loc, free_pages = self._inputs(
                        batch_size, page_size
                    )
                    extend_num_tokens = int((seq_lens - prefix_lens).sum().item())
                    actual = torch.empty(
                        extend_num_tokens, device="cuda", dtype=torch.int64
                    )
                    expected = torch.empty_like(actual)

                    current_kernel = alloc_extend_kernel[(batch_size,)](
                        prefix_lens,
                        seq_lens,
                        last_loc,
                        free_pages,
                        actual,
                        ALLOCATOR_BATCH_SIZE_STEP,
                        page_size,
                    )
                    alloc_extend_naive(
                        prefix_lens,
                        seq_lens,
                        last_loc,
                        free_pages,
                        expected,
                        page_size,
                        "cuda",
                    )
                    torch.cuda.synchronize()

                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if compiled_kernel is None:
                        compiled_kernel = current_kernel
                    else:
                        self.assertIs(current_kernel, compiled_kernel)

    def test_alloc_decode_reuses_module_across_batch_sizes(self):
        for page_size in (1, 32, 64):
            compiled_kernel = None
            for batch_size in (1, 7, 32, ALLOCATOR_BATCH_SIZE_STEP + 1):
                with self.subTest(page_size=page_size, batch_size=batch_size):
                    prefix_lens, _, last_loc, free_pages = self._inputs(
                        batch_size, page_size
                    )
                    seq_lens = prefix_lens + 1
                    actual = torch.empty(batch_size, device="cuda", dtype=torch.int64)
                    expected = torch.empty_like(actual)

                    current_kernel = alloc_decode_kernel[(batch_size,)](
                        seq_lens,
                        last_loc,
                        free_pages,
                        actual,
                        ALLOCATOR_BATCH_SIZE_STEP,
                        page_size,
                    )
                    alloc_extend_naive(
                        prefix_lens,
                        seq_lens,
                        last_loc,
                        free_pages,
                        expected,
                        page_size,
                        "cuda",
                    )
                    torch.cuda.synchronize()

                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if compiled_kernel is None:
                        compiled_kernel = current_kernel
                    else:
                        self.assertIs(current_kernel, compiled_kernel)


if __name__ == "__main__":
    unittest.main()
