"""Registration ranges, independent of the logical per-layer transfer layout."""

import operator


def normalize_buffer_regions(batches):
    """Register overlapping views once, without spanning adjacent allocations.

    Each input batch is (category, pointers, lengths). Overlapping byte ranges
    are replaced by their exact union, owned by their earliest input batch.
    Logical KVArgs lists must remain untouched: their aliases are part of the
    transfer protocol, whereas registration describes physical address ranges.
    """
    regions = []
    for index, (category, ptrs, lengths) in enumerate(batches):
        ptrs = [] if ptrs is None else ptrs
        lengths = [] if lengths is None else lengths
        if len(ptrs) != len(lengths):
            raise ValueError(f"Mooncake {category}: pointer/length counts differ")
        for ptr, length in zip(ptrs, lengths):
            ptr, length = operator.index(ptr), operator.index(length)
            # Zero-row DSA placeholders preserve the logical layer positions
            # but have no physical memory region to register (upstream policy).
            if length == 0:
                continue
            if ptr <= 0 or length <= 0 or ptr + length > 1 << 64:
                raise ValueError(
                    f"Mooncake {category}: invalid range ptr={ptr} length={length}"
                )
            regions.append((ptr, ptr + length, index))

    merged = []
    for start, end, index in sorted(regions):
        if merged and start < merged[-1][1]:
            previous = merged[-1]
            previous[1] = max(previous[1], end)
            previous[2].add(index)
        else:
            merged.append([start, end, {index}])

    grouped = {}
    for start, end, indices in merged:
        owner = min(indices)
        ptrs, lengths, categories = grouped.setdefault(owner, ([], [], set()))
        ptrs.append(start)
        lengths.append(end - start)
        categories.update(indices)
    return [
        (
            "+".join(batches[index][0] for index in sorted(categories)),
            ptrs,
            lengths,
        )
        for _, (ptrs, lengths, categories) in sorted(grouped.items())
    ]
