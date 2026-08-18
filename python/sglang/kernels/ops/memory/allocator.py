import triton
import triton.language as tl

# Keep the batch reduction shape independent of the runtime batch size.  A
# constexpr next_power_of_2(batch_size) makes Triton load a new module for every
# batch bucket, which can stall distributed schedulers when the first real
# request reaches a previously unseen bucket.  A runtime loop over fixed-size
# blocks compiles once and still supports batches larger than one block.
ALLOCATOR_BATCH_SIZE_STEP = 512


# free_page_ptr aliases self.free_pages, which the paged allocator re-slices
# after every allocation (self.free_pages = self.free_pages[num_new_pages:]).
# Slicing only advances data_ptr() by num_new_pages * 8 bytes, so the pointer
# flips between 16-byte-aligned and unaligned across calls. Triton specializes
# on pointer alignment by default and bakes it into the cache key, compiling two
# kernel variants (one with tt.divisibility=16 on free_page_ptr, one without)
# so the second prefill on a fresh DCP server hits the alternate alignment and
# pays an extra ~100ms JIT for that kernel variant. do_not_specialize skips
# that specialization so only one kernel is ever compiled; the perf cost is
# negligible (this kernel runs in ~10us and only loads ~4KB through this ptr).
@triton.jit(do_not_specialize=["free_page_ptr"])
def alloc_extend_kernel(
    pre_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    batch_size_step: tl.constexpr,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    extend_len = seq_len - pre_len

    sum_extend_lens = 0
    sum_num_new_pages = 0
    sum_extend_lens = sum_extend_lens.to(tl.int64)
    sum_num_new_pages = sum_num_new_pages.to(tl.int64)
    num_batch_blocks = tl.cdiv(pid + 1, batch_size_step)
    for block_id in range(num_batch_blocks):
        offsets = block_id * batch_size_step + tl.arange(0, batch_size_step)
        mask = offsets <= pid
        block_seq_lens = tl.load(seq_lens_ptr + offsets, mask=mask, other=0)
        block_pre_lens = tl.load(pre_lens_ptr + offsets, mask=mask, other=0)
        sum_extend_lens += tl.sum(block_seq_lens - block_pre_lens)
        num_pages_after = (block_seq_lens + page_size - 1) // page_size
        num_pages_before = (block_pre_lens + page_size - 1) // page_size
        sum_num_new_pages += tl.sum(num_pages_after - num_pages_before)

    output_start_loc = sum_extend_lens - extend_len

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Part 1: fill the old partial page
    last_loc = tl.load(last_loc_ptr + pid)
    num_part1 = (
        min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
    )
    offset_one_page = tl.arange(0, page_size)
    tl.store(
        out_indices + output_start_loc + offset_one_page,
        last_loc + 1 + offset_one_page,
        mask=offset_one_page < num_part1,
    )
    if pre_len + num_part1 == seq_len:
        return

    # Part 2: fill the new full pages using a dynamic blocked loop.
    # The loop bound is derived from num_part2 (runtime value), so Triton
    # generates a real loop instead of unrolling -- no constexpr dependency
    # on extend size and only one kernel compilation.
    num_part2 = (
        seq_len // page_size * page_size
        - (pre_len + page_size - 1) // page_size * page_size
    )
    BLOCK_EXTEND: tl.constexpr = 4096
    num_blocks = (num_part2 + BLOCK_EXTEND - 1) // BLOCK_EXTEND
    for block_id in range(num_blocks):
        offset_in_block = tl.arange(0, BLOCK_EXTEND)
        offset = block_id * BLOCK_EXTEND + offset_in_block
        mask = offset < num_part2
        page_start = tl.load(
            free_page_ptr + new_page_start_loc + offset // page_size,
            mask=mask,
        )
        tl.store(
            out_indices + output_start_loc + num_part1 + offset,
            page_start * page_size + offset % page_size,
            mask=mask,
        )
    if pre_len + num_part1 + num_part2 == seq_len:
        return

    # Part 3: fill the new partial page
    num_part3 = seq_len - seq_len // page_size * page_size
    start_loc = tl.load(
        free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
        start_loc * page_size + offset_one_page,
        mask=offset_one_page < num_part3,
    )


# Same free_page_ptr alignment rationale as alloc_extend_kernel above.
@triton.jit(do_not_specialize=["free_page_ptr"])
def alloc_decode_kernel(
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    batch_size_step: tl.constexpr,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = seq_len - 1

    sum_num_new_pages = 0
    sum_num_new_pages = sum_num_new_pages.to(tl.int64)
    num_batch_blocks = tl.cdiv(pid + 1, batch_size_step)
    for block_id in range(num_batch_blocks):
        offsets = block_id * batch_size_step + tl.arange(0, batch_size_step)
        mask = offsets <= pid
        block_seq_lens = tl.load(seq_lens_ptr + offsets, mask=mask, other=0)
        block_pre_lens = tl.where(mask, block_seq_lens - 1, block_seq_lens)
        num_pages_after = (block_seq_lens + page_size - 1) // page_size
        num_pages_before = (block_pre_lens + page_size - 1) // page_size
        sum_num_new_pages += tl.sum(num_pages_after - num_pages_before)

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    if num_page_start_loc_self == 0:
        last_loc = tl.load(last_loc_ptr + pid)
        tl.store(out_indices + pid, last_loc + 1)
    else:
        page = tl.load(free_page_ptr + new_page_start_loc)
        tl.store(out_indices + pid, page * page_size)
