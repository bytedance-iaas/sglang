import os
import weakref
from collections.abc import Callable

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    HiSparseC4DevicePool,
)
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.utils.common import get_num_new_pages


def _stable_unique_page_ids(page_ids: torch.Tensor) -> torch.Tensor:
    """Deduplicate page ids without changing their first-owner order."""
    if page_ids.numel() == 0:
        return page_ids.to(dtype=torch.int64)

    unique_page_ids, inverse = torch.unique(
        page_ids.to(dtype=torch.int64), sorted=False, return_inverse=True
    )
    positions = torch.arange(
        page_ids.numel(), dtype=torch.int64, device=page_ids.device
    )
    first_positions = torch.full_like(unique_page_ids, page_ids.numel())
    first_positions.scatter_reduce_(
        0, inverse, positions, reduce="amin", include_self=True
    )
    return unique_page_ids[torch.argsort(first_positions)]


def _released_page_ids(
    allocator: PagedTokenToKVPoolAllocator, *, device: torch.device
) -> torch.Tensor:
    """Return every page already reusable by a paged allocator.

    Disaggregated allocators stage newly returned pages in ``release_pages``
    until a later merge, so ``free_pages`` alone is not the free set.
    """
    released_page_sets = []
    for name in ("free_pages", "release_pages"):
        released = getattr(allocator, name, None)
        if isinstance(released, torch.Tensor):
            released_page_sets.append(released.to(device=device, dtype=torch.int64))
        elif isinstance(released, (list, tuple)) and released:
            released_page_sets.append(
                torch.as_tensor(released, device=device, dtype=torch.int64)
            )
    if not released_page_sets:
        return torch.empty(0, device=device, dtype=torch.int64)
    return torch.cat(released_page_sets)


class _HiSparsePageOwnership:
    """Release physical pages only after every logical/buffer owner is clear."""

    def __init__(
        self,
        *,
        mapping: torch.Tensor,
        child_allocator: PagedTokenToKVPoolAllocator,
        page_size: int,
    ) -> None:
        assert child_allocator.is_not_in_free_group
        assert page_size > 0
        self.mapping = mapping
        self.child_allocator = child_allocator
        self.page_size = page_size
        # Coordinator device buffers outlive individual logical mappings.  A
        # page can therefore have no mapping owner while it is still reachable
        # through req_to_device_buffer.  Keep those cross-transaction owners
        # here; inspecting only ``mapping`` during free creates a use-after-free
        # window in which the page can be reallocated and later returned twice.
        self._extra_owner_page_ids: set[int] = set()

    def clear(self) -> None:
        self._extra_owner_page_ids.clear()

    def debug_snapshot(
        self, extra_owned_coordinates: torch.Tensor | None = None
    ) -> dict[str, object]:
        """Return a synchronized ownership snapshot for lifecycle diagnosis.

        This helper is intentionally called only behind
        ``SGLANG_HISPARSE_DEBUG_LIFECYCLE``.  It performs device-to-host
        synchronization and a full mapping scan, so it must not be used by the
        normal allocation path.
        """
        allocator = self.child_allocator
        mapping_coordinates = self.mapping[self.mapping > 0]
        mapping_page_ids = self._page_ids(mapping_coordinates)
        request_page_ids = (
            self._page_ids(extra_owned_coordinates)
            if extra_owned_coordinates is not None
            else torch.empty(0, dtype=torch.int64, device=self.mapping.device)
        )
        request_page_id_set = set(request_page_ids.cpu().tolist())
        mapping_page_id_set = set(mapping_page_ids.cpu().tolist())

        def _page_count(name: str) -> int:
            pages = getattr(allocator, name, None)
            if isinstance(pages, torch.Tensor):
                return int(pages.numel())
            if isinstance(pages, (list, tuple)):
                return sum(
                    int(page.numel()) if isinstance(page, torch.Tensor) else 1
                    for page in pages
                )
            return 0

        return {
            "available": int(allocator.available_size()),
            "capacity": int(allocator.size),
            "free_pages": _page_count("free_pages"),
            "release_pages": _page_count("release_pages"),
            "mapping_slots": int(mapping_coordinates.numel()),
            "mapping_pages": len(mapping_page_id_set),
            "extra_owner_pages": len(self._extra_owner_page_ids),
            "request_pages": sorted(request_page_id_set),
            "request_claimed_pages": sorted(
                request_page_id_set & self._extra_owner_page_ids
            ),
            "request_mapping_pages": sorted(request_page_id_set & mapping_page_id_set),
        }

    def _page_ids(self, coordinates: torch.Tensor) -> torch.Tensor:
        positive_coordinates = coordinates[coordinates > 0].to(torch.int64)
        return _stable_unique_page_ids(positive_coordinates // self.page_size)

    def claim(self, coordinates: torch.Tensor) -> None:
        """Persist ownership held outside the logical mapping tensor."""
        page_ids = self._page_ids(coordinates)
        if page_ids.numel() == 0:
            return
        page_id_set = set(page_ids.cpu().tolist())
        duplicate = page_id_set & self._extra_owner_page_ids
        if duplicate:
            raise RuntimeError(
                "HiSparse physical pages acquired by multiple side-buffer owners: "
                f"pages={sorted(duplicate)}"
            )
        self._extra_owner_page_ids.update(page_id_set)

    def assert_unclaimed(self, coordinates: torch.Tensor) -> None:
        page_ids = set(self._page_ids(coordinates).cpu().tolist())
        claimed = page_ids & self._extra_owner_page_ids
        if claimed:
            raise RuntimeError(
                "HiSparse attempted direct free of side-buffer-owned pages: "
                f"pages={sorted(claimed)}"
            )

    def release_mapped_pages(self, page_ids: torch.Tensor) -> None:
        """Clear every logical alias before returning canonical pages.

        DSV4 speculative allocation reserves a full logical page and can leave
        aliases outside the currently visible verify window.  Once the whole
        physical page is host-backed, the physical owner -- rather than one
        logical slice -- must retire all of those aliases in one transaction.
        """
        assert self.child_allocator.is_not_in_free_group
        page_ids = _stable_unique_page_ids(page_ids.to(dtype=torch.int64))
        if page_ids.numel() == 0:
            return

        claimed = set(page_ids.cpu().tolist()) & self._extra_owner_page_ids
        if claimed:
            raise RuntimeError(
                "HiSparse attempted to retire coordinator-owned pages: "
                f"pages={sorted(claimed)}"
            )

        positive = self.mapping > 0
        aliases = positive & torch.isin(
            torch.div(self.mapping, self.page_size, rounding_mode="floor"),
            page_ids,
        )
        self.mapping[aliases] = 0

        # Physical page zero is the allocator sentinel. A non-page-aligned PD
        # prefix can make the first generated C4 tail use positive coordinates
        # 1..page_size-1 in that page; retire their mappings, but never return
        # the sentinel page to the allocator.
        page_ids = page_ids[page_ids > 0]
        already_released = _released_page_ids(
            self.child_allocator, device=page_ids.device
        )
        if already_released.numel() > 0:
            page_ids = page_ids[~torch.isin(page_ids, already_released)]
        if page_ids.numel() == 0:
            return

        offsets = torch.arange(
            self.page_size, dtype=torch.int64, device=page_ids.device
        )
        self.child_allocator.free(
            (page_ids[:, None] * self.page_size + offsets).reshape(-1)
        )

    def rehome_temporary_pages(
        self,
        *,
        mapping_indices: torch.Tensor,
        retained_page_ids: torch.Tensor,
        install_retained_owner: Callable[[], None],
    ) -> None:
        """Transfer or release complete temporary pages before remapping them."""
        assert self.child_allocator.is_not_in_free_group
        coordinates = self.mapping[mapping_indices]
        owned_page_ids = self._page_ids(coordinates)
        retained_page_ids = _stable_unique_page_ids(
            retained_page_ids.to(device=owned_page_ids.device, dtype=torch.int64)
        )
        if retained_page_ids.numel() > 0 and torch.any(
            ~torch.isin(retained_page_ids, owned_page_ids)
        ):
            raise RuntimeError(
                "Retained HiSparse pages must belong to the temporary mapping owner"
            )

        # Remove the temporary mapping owner first.  The callback then installs
        # the durable side-buffer owner before any released page can be reused.
        self.mapping[mapping_indices] = 0
        install_retained_owner()
        released_page_ids = owned_page_ids[
            ~torch.isin(owned_page_ids, retained_page_ids)
        ]
        self.release_mapped_pages(released_page_ids)

    def retire_replaced_mapping_pages(
        self,
        *,
        replaced_coordinates: torch.Tensor,
        completed_mapping_indices: torch.Tensor,
    ) -> None:
        """Return temporary pages after their complete logical page is rebound.

        Paged speculative decode reserves a whole physical page, while target
        verify replaces only the currently visible token window with stable
        coordinator slots.  The final replacement in a logical page removes
        the last mapping owner of that temporary physical page.  Retire that
        page here, at the ownership transition, rather than trying to discover
        orphaned pages when the request eventually finishes.

        ``completed_mapping_indices`` must contain complete logical pages.
        Coordinator-owned side pages and candidate pages that still have a
        mapping owner are explicitly excluded.  The latter is a valid state
        when a partial prompt tail and its generated continuation share one
        physical page.
        """
        candidate_page_ids = self._page_ids(replaced_coordinates)
        if candidate_page_ids.numel() == 0:
            return

        if self._extra_owner_page_ids:
            claimed_page_ids = torch.tensor(
                sorted(self._extra_owner_page_ids),
                dtype=torch.int64,
                device=candidate_page_ids.device,
            )
            candidate_page_ids = candidate_page_ids[
                ~torch.isin(candidate_page_ids, claimed_page_ids)
            ]
        candidate_page_ids = candidate_page_ids[candidate_page_ids > 0]
        if candidate_page_ids.numel() == 0:
            return

        completed_mapping_indices = completed_mapping_indices.to(torch.int64)
        if completed_mapping_indices.numel() % self.page_size != 0:
            raise RuntimeError(
                "HiSparse replaced-owner retirement requires complete logical pages"
            )
        completed_blocks = completed_mapping_indices.reshape(-1, self.page_size)
        expected_offsets = torch.arange(
            self.page_size,
            dtype=torch.int64,
            device=completed_mapping_indices.device,
        )
        if torch.any(
            completed_blocks
            != (completed_blocks[:, :1] // self.page_size) * self.page_size
            + expected_offsets
        ):
            raise RuntimeError(
                "HiSparse replaced-owner retirement received a partial logical page"
            )

        remaining_coordinates = self.mapping[completed_mapping_indices]
        remaining_page_ids = self._page_ids(remaining_coordinates)
        candidate_page_ids = candidate_page_ids[
            ~torch.isin(candidate_page_ids, remaining_page_ids)
        ]
        if candidate_page_ids.numel() == 0:
            return

        already_released = _released_page_ids(
            self.child_allocator, device=candidate_page_ids.device
        )
        if already_released.numel() > 0:
            candidate_page_ids = candidate_page_ids[
                ~torch.isin(candidate_page_ids, already_released)
            ]
        if candidate_page_ids.numel() == 0:
            return

        offsets = torch.arange(
            self.page_size, dtype=torch.int64, device=candidate_page_ids.device
        )
        self.child_allocator.free(
            (candidate_page_ids[:, None] * self.page_size + offsets).reshape(-1)
        )

    def release(
        self,
        *,
        mapping_indices: torch.Tensor,
        extra_owned_coordinates: torch.Tensor | None = None,
        clear_extra_owner: Callable[[], None] | None = None,
    ) -> None:
        # This physical allocator is not part of the logical allocator's free
        # transaction. Fail before mutating any owner if that invariant breaks.
        assert self.child_allocator.is_not_in_free_group
        coordinates = self.mapping[mapping_indices]
        extra_page_ids: set[int] = set()
        extra_page_ids_tensor = torch.empty(
            0, dtype=torch.int64, device=self.mapping.device
        )
        if extra_owned_coordinates is not None:
            coordinates = torch.cat([coordinates, extra_owned_coordinates])
            extra_page_ids_tensor = self._page_ids(extra_owned_coordinates)
            extra_page_ids = set(extra_page_ids_tensor.cpu().tolist())
            missing = extra_page_ids - self._extra_owner_page_ids
            if missing:
                raise RuntimeError(
                    "HiSparse released side-buffer pages without ownership: "
                    f"pages={sorted(missing)}"
                )
        page_ids = self._page_ids(coordinates)

        self.mapping[mapping_indices] = 0
        remaining_mask = self.mapping > 0
        remaining_indices = torch.nonzero(remaining_mask, as_tuple=False).flatten()
        remaining_coordinates = self.mapping[remaining_mask].to(torch.int64)
        if extra_page_ids_tensor.numel() > 0 and remaining_coordinates.numel() > 0:
            # A coordinator side buffer explicitly owns complete physical
            # pages. EAGLE verify can leave logical aliases outside the
            # request-visible kv_allocated_len, so request_finished cannot
            # enumerate them through mapping_indices. Once the canonical
            # side-buffer owner is released, every alias of those pages is
            # stale and must be retired in the same transaction; otherwise
            # the global remaining-owner scan below pins the pages forever.
            remaining_page_ids = torch.div(
                remaining_coordinates, self.page_size, rounding_mode="floor"
            )
            stale_extra_aliases = torch.isin(remaining_page_ids, extra_page_ids_tensor)
            self.mapping[remaining_indices[stale_extra_aliases]] = 0
            remaining_coordinates = remaining_coordinates[~stale_extra_aliases]
        self._extra_owner_page_ids.difference_update(extra_page_ids)
        if clear_extra_owner is not None:
            clear_extra_owner()

        # Physical page zero is the allocator sentinel.  A non-page-aligned
        # PD prefix can still expose positive coordinates 1..page_size-1 from
        # that page through the logical mapping.  Those aliases must be
        # cleared above, but page zero must never be returned to the child
        # allocator or its available size will exceed its physical capacity.
        page_ids = page_ids[page_ids > 0]
        if page_ids.numel() == 0:
            return

        # A page with a live coordinator-side owner must not become reusable,
        # even when all of its current logical aliases were cleared by an
        # earlier cache transaction.
        if self._extra_owner_page_ids:
            claimed_page_ids = torch.tensor(
                sorted(self._extra_owner_page_ids),
                dtype=torch.int64,
                device=page_ids.device,
            )
            page_ids = page_ids[~torch.isin(page_ids, claimed_page_ids)]
        if page_ids.numel() == 0:
            return

        # Paged allocations can expose only part of a physical page through one
        # logical range. Another range may therefore still own a different slot
        # in the same page and can be retired by a later release call. The page
        # allocator has no reference counts, so return only pages whose final
        # mapping owner was cleared by this transaction.
        if remaining_coordinates.numel() > 0:
            remaining_page_ids = torch.unique(remaining_coordinates // self.page_size)
            page_ids = page_ids[~torch.isin(page_ids, remaining_page_ids)]
        if page_ids.numel() == 0:
            return

        # In disaggregated mode PagedTokenToKVPoolAllocator stages returned
        # pages in release_pages until the next merge. A stale logical owner can
        # survive beyond the release which already returned its page; checking
        # both containers prevents that later owner from returning it twice.
        already_released = _released_page_ids(
            self.child_allocator, device=page_ids.device
        )
        if already_released.numel() > 0:
            page_ids = page_ids[~torch.isin(page_ids, already_released)]
        if page_ids.numel() == 0:
            return

        offsets = torch.arange(
            self.page_size, dtype=torch.int64, device=page_ids.device
        )
        full_page_blocks = (page_ids[:, None] * self.page_size + offsets).reshape(-1)
        self.child_allocator.free(full_page_blocks)


class HiSparseDemotionMixin:
    def set_demote_until_hisparse_available(self, callback):
        self._demote_until_hisparse_available = weakref.WeakMethod(callback)

    def set_schedulable_hisparse_available(self, callback):
        self._schedulable_hisparse_available = weakref.WeakMethod(callback)

    def _get_schedulable_hisparse_available(self) -> int:
        callback_ref = getattr(self, "_schedulable_hisparse_available", None)
        if callback_ref is None:
            return self.hisparse_attn_allocator.available_size()

        callback = callback_ref()
        if callback is None:
            return self.hisparse_attn_allocator.available_size()
        return callback()

    def _ensure_hisparse_available(self, need_tokens: int) -> bool:
        if self.hisparse_attn_allocator.available_size() >= need_tokens:
            return True

        callback_ref = getattr(self, "_demote_until_hisparse_available", None)
        if callback_ref is None:
            return False

        callback = callback_ref()
        return (
            callback is not None
            and callback(need_tokens)
            and self.hisparse_attn_allocator.available_size() >= need_tokens
        )


class HiSparseTokenToKVPoolAllocator(HiSparseDemotionMixin, BaseTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        kvcache: HiSparseDSATokenToKVPool,
        need_sort: bool,
        host_to_device_ratio: int = 2,
    ):
        self._kvcache = kvcache
        self._size_full = size * host_to_device_ratio
        self._size_hisparse = size
        self.compress_ratio = 1
        self.dtype = dtype
        self.device = device
        self.page_size = page_size
        self.need_sort = need_sort
        self.debug_validate_lifecycle = (
            os.environ.get("SGLANG_HISPARSE_DEBUG_LIFECYCLE", "0") == "1"
        )

        self.logical_attn_allocator = PagedTokenToKVPoolAllocator(
            self._size_full,
            self.page_size,
            self.dtype,
            self.device,
            kvcache,
            need_sort,
        )
        self.hisparse_attn_allocator = PagedTokenToKVPoolAllocator(
            self._size_hisparse,
            self.page_size,
            self.dtype,
            self.device,
            kvcache,
            need_sort,
        )
        self.full_to_hisparse_device_index_mapping = torch.cat(
            [
                torch.zeros(
                    self._size_full + self.page_size,
                    dtype=torch.int64,
                    device=self.device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=self.device),
            ]
        )
        self._page_ownership = _HiSparsePageOwnership(
            mapping=self.full_to_hisparse_device_index_mapping,
            child_allocator=self.hisparse_attn_allocator,
            page_size=self.page_size,
        )

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []
        self.clear()
        self._kvcache.register_mapping(
            weakref.proxy(self.full_to_hisparse_device_index_mapping)
        )

    @property
    def size_full(self) -> int:
        return self._size_full

    @property
    def size(self) -> int:
        return self._size_full

    @property
    def hisparse_device_page_size(self) -> int:
        return self.page_size

    def available_size(self) -> int:
        return min(
            self.logical_attn_allocator.available_size(),
            self._get_schedulable_hisparse_available(),
        )

    def get_kvcache(self):
        return self._kvcache

    def alloc(self, need_size: int):
        if self.page_size != 1:
            raise NotImplementedError(
                "HiSparse generic allocation is only supported for page_size=1. "
                "Use alloc_extend for paged allocation."
            )
        if not self._ensure_hisparse_available(need_size):
            return None

        logical_indices = self.logical_attn_allocator.alloc(need_size)
        if logical_indices is None:
            return None

        hisparse_indices = self.hisparse_attn_allocator.alloc(need_size)
        if hisparse_indices is None:
            self.logical_attn_allocator.free(logical_indices)
            return None

        self.full_to_hisparse_device_index_mapping[logical_indices] = hisparse_indices
        return logical_indices

    def alloc_logical_only(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        """Allocate only logical indices without hisparse device indices.

        Used in the direct-to-host transfer path where KV data is written
        directly to host memory by the prefill node, skipping GPU staging.
        """
        return self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )

    def alloc_device_buffer(self, allocated_indices, need_size: int):
        assert need_size % self.page_size == 0
        # clear original reference and isolate the buffer from outside addressing, allocate new buffer if needed
        hisparse_indices = self.full_to_hisparse_device_index_mapping[allocated_indices]
        self.full_to_hisparse_device_index_mapping[allocated_indices] = 0
        # Filter valid (non-zero) hisparse indices.
        # In the direct-to-host path, mapping is all zeros since no hisparse
        # device indices were pre-allocated.
        hisparse_indices = hisparse_indices[hisparse_indices > 0]
        if hisparse_indices.numel() > 0:
            # A speculative over-allocation can keep a logical mapping alive
            # beyond the request span handled by an earlier residency
            # transition. When that span becomes visible on a later demotion,
            # its physical page may already have been returned by the previous
            # transition. The mapping has just been detached above, so discard
            # these stale references before transferring ownership to the
            # fixed device buffer or releasing surplus pages. Otherwise the
            # surplus free below returns the same page twice; if the stale slot
            # falls in the retained prefix it becomes a use-after-free instead.
            mapped_pages = hisparse_indices // self.page_size
            stale = torch.isin(
                mapped_pages,
                _released_page_ids(
                    self.hisparse_attn_allocator, device=mapped_pages.device
                ),
            )
            if torch.any(stale):
                hisparse_indices = hisparse_indices[~stale]
        if len(hisparse_indices) >= need_size:
            buffer_indices = hisparse_indices[:need_size]
            surplus = hisparse_indices[need_size:]
            if surplus.numel() > 0:
                # ``PagedTokenToKVPoolAllocator.free`` releases whole pages.
                # A detached speculative mapping can leave holes in
                # ``hisparse_indices``, so a slot-count cut may split one
                # physical page between the retained buffer and the surplus.
                # Keep every page touched by the buffer alive and release only
                # pages owned exclusively by the surplus.
                buffer_pages = torch.unique(buffer_indices // self.page_size)
                surplus_pages = torch.unique(surplus // self.page_size)
                pure_surplus = surplus_pages[~torch.isin(surplus_pages, buffer_pages)]
                if pure_surplus.numel() > 0:
                    self.free_hisparse_indices(pure_surplus * self.page_size)
        else:
            # page alignment, claiming the residual space for an incomplete page
            page_residual_length = len(hisparse_indices) % self.page_size
            if page_residual_length != 0:
                hisparse_indices = torch.cat(
                    [
                        hisparse_indices,
                        torch.arange(
                            hisparse_indices[-1] + 1,
                            hisparse_indices[-1]
                            + self.page_size
                            - page_residual_length
                            + 1,
                            device=self.device,
                        ),
                    ]
                )
            extra_indices = self.hisparse_attn_allocator.alloc(
                need_size - len(hisparse_indices)
            )
            assert (
                extra_indices is not None
            ), "Hisparse allocation failed in alloc_device_buffer"
            buffer_indices = torch.cat([hisparse_indices, extra_indices])
        return buffer_indices

    def free_hisparse_indices(self, buffer_indices: torch.Tensor):
        # Device-page ownership is independent from the logical free group.
        # Never mutate the child allocator's transaction state implicitly.
        assert self.hisparse_attn_allocator.is_not_in_free_group
        buffer_indices = buffer_indices[buffer_indices > 0]
        if buffer_indices.numel() == 0:
            return
        self._page_ownership.assert_unclaimed(buffer_indices)
        if self.debug_validate_lifecycle:
            pages = torch.unique(buffer_indices // self.page_size)
            already_free = pages[
                torch.isin(
                    pages,
                    _released_page_ids(
                        self.hisparse_attn_allocator, device=pages.device
                    ),
                )
            ]
            if already_free.numel() > 0:
                raise RuntimeError(
                    "HiSparse physical page double-free detected: "
                    f"pages={already_free.tolist()} "
                    f"available={self.hisparse_attn_allocator.available_size()} "
                    f"capacity={self.hisparse_attn_allocator.size}"
                )
        self.hisparse_attn_allocator.free(buffer_indices)
        if self.debug_validate_lifecycle:
            free_pages = _released_page_ids(
                self.hisparse_attn_allocator, device=buffer_indices.device
            )
            if torch.unique(free_pages).numel() != free_pages.numel():
                raise RuntimeError(
                    "HiSparse physical free list contains duplicate pages after free"
                )

    def release_hisparse_ownership(
        self,
        *,
        mapping_indices: torch.Tensor,
        extra_owned_coordinates: torch.Tensor | None = None,
        clear_extra_owner: Callable[[], None] | None = None,
    ) -> None:
        self._page_ownership.release(
            mapping_indices=mapping_indices,
            extra_owned_coordinates=extra_owned_coordinates,
            clear_extra_owner=clear_extra_owner,
        )

    def debug_hisparse_ownership(
        self, extra_owned_coordinates: torch.Tensor | None = None
    ) -> dict[str, object]:
        return self._page_ownership.debug_snapshot(extra_owned_coordinates)

    def claim_hisparse_ownership(self, coordinates: torch.Tensor) -> None:
        self._page_ownership.claim(coordinates)

    def release_hisparse_mapped_pages(self, page_ids: torch.Tensor) -> None:
        self._page_ownership.release_mapped_pages(page_ids)

    def rehome_temporary_hisparse_pages(
        self,
        *,
        mapping_indices: torch.Tensor,
        retained_page_ids: torch.Tensor,
        install_retained_owner: Callable[[], None],
    ) -> None:
        self._page_ownership.rehome_temporary_pages(
            mapping_indices=mapping_indices,
            retained_page_ids=retained_page_ids,
            install_retained_owner=install_retained_owner,
        )

    def retire_replaced_hisparse_mapping_pages(
        self,
        *,
        replaced_coordinates: torch.Tensor,
        completed_mapping_indices: torch.Tensor,
    ) -> None:
        self._page_ownership.retire_replaced_mapping_pages(
            replaced_coordinates=replaced_coordinates,
            completed_mapping_indices=completed_mapping_indices,
        )

    def get_last_loc_compressed(self, last_locs: torch.Tensor):
        return last_locs

    def get_last_loc_hisparse_device(self, last_locs: torch.Tensor):
        return self._kvcache._translate_loc_to_hisparse_device(last_locs)

    def alloc_extend_with_device_mapping(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        device_slots: torch.Tensor,
        backup_state: bool = False,
    ):
        """Allocate logical tokens and bind them to coordinator-owned device slots.

        Speculative verification writes into the per-request HiSparse extra page.
        Those physical slots are owned by ``HiSparseCoordinator`` and must not be
        allocated or freed by the ordinary logical allocator lifecycle.
        """
        available = self.logical_attn_allocator.available_size()
        if available < extend_num_tokens:
            raise RuntimeError(
                "HiSparse logical allocation is exhausted: "
                f"need={extend_num_tokens}, available={available}"
            )

        logical_state = (
            self.logical_attn_allocator.backup_state() if backup_state else None
        )
        logical_indices = self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        if logical_indices is None:
            raise RuntimeError(
                "HiSparse logical alloc_extend failed for coordinator-owned "
                f"draft slots: need={extend_num_tokens}"
            )
        if logical_indices.numel() != device_slots.numel():
            if logical_state is not None:
                self.logical_attn_allocator.restore_state(logical_state)
            raise RuntimeError(
                "HiSparse draft-slot mapping size mismatch: "
                f"logical={logical_indices.numel()}, device={device_slots.numel()}"
            )

        self.full_to_hisparse_device_index_mapping[logical_indices] = device_slots
        if backup_state:
            return logical_indices, (logical_state, logical_indices.clone())
        return logical_indices

    def clear_device_mapping(self, logical_indices: torch.Tensor) -> None:
        """Detach coordinator-owned slots before logical token release."""
        if logical_indices.numel() > 0:
            self.full_to_hisparse_device_index_mapping[logical_indices] = 0

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
        extend_num_tokens: int,
    ):
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        if (
            num_new_pages
            > self.logical_attn_allocator.available_size() // self.page_size
        ):
            return None
        if (
            num_new_pages
            > self.hisparse_attn_allocator.available_size() // self.page_size
        ):
            if not self._ensure_hisparse_available(num_new_pages * self.page_size):
                return None

        logical_indices = self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        assert logical_indices is not None, "Logical allocation failed in alloc_extend"

        hisparse_last_loc = self.get_last_loc_hisparse_device(last_loc)
        hisparse_indices = self.hisparse_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            hisparse_last_loc,
            len(logical_indices),
            num_new_pages=num_new_pages,
        )
        assert (
            hisparse_indices is not None
        ), "Hisparse allocation failed in alloc_extend"
        self.full_to_hisparse_device_index_mapping[logical_indices] = hisparse_indices
        return logical_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        return self.logical_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )

    def free_hisparse(self, free_indices: torch.Tensor):
        self.release_hisparse_ownership(mapping_indices=free_indices)

    def clear(self):
        self.logical_attn_allocator.clear()
        self.hisparse_attn_allocator.clear()
        # Note: the last item is -1, we don't clear it, see the comment in __init__
        self.full_to_hisparse_device_index_mapping[:-1].fill_(0)
        self._page_ownership.clear()
        self.is_not_in_free_group = True
        self.free_group = []

    def free_group_begin(self):
        return

    def free_group_end(self):
        return

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if self.is_not_in_free_group:
            self.logical_attn_allocator.free(free_index)
            self.free_hisparse(free_index)
        else:
            self.free_group.append(self._copy_for_free_group(free_index))
        assert (
            self.logical_attn_allocator.available_size()
            <= self.logical_attn_allocator.size
        )
        assert (
            self.hisparse_attn_allocator.available_size()
            <= self.hisparse_attn_allocator.size
        )

    def backup_state(self):
        return (
            self.logical_attn_allocator.backup_state(),
            self.hisparse_attn_allocator.backup_state(),
            self.full_to_hisparse_device_index_mapping.clone(),
        )

    def restore_state(self, state):
        if len(state) == 2:
            # ``alloc_extend_with_device_mapping(..., backup_state=True)`` owns
            # the physical extra-page slots outside the allocator. Roll back
            # only the logical allocation and keep the mapping live until
            # accepted-token finalization clears it transactionally.
            self.logical_attn_allocator.restore_state(state[0])
            return

        logical_state, hisparse_state, mapping_snapshot = state
        self.logical_attn_allocator.restore_state(logical_state)
        self.hisparse_attn_allocator.restore_state(hisparse_state)
        self.full_to_hisparse_device_index_mapping[: mapping_snapshot.shape[0]].copy_(
            mapping_snapshot
        )
        if (
            mapping_snapshot.shape[0]
            < self.full_to_hisparse_device_index_mapping.shape[0]
        ):
            self.full_to_hisparse_device_index_mapping[mapping_snapshot.shape[0] :] = 0


class DeepSeekV4HiSparseTokenToKVPoolAllocator(
    HiSparseDemotionMixin, BaseTokenToKVPoolAllocator
):

    def __init__(
        self,
        logical_attn_allocator: BaseTokenToKVPoolAllocator,
    ):
        assert isinstance(logical_attn_allocator._kvcache, DeepSeekV4TokenToKVPool)
        assert isinstance(
            logical_attn_allocator._kvcache.c4_kv_pool, HiSparseC4DevicePool
        )
        self.compress_ratio = 4

        self.hisparse_kvcache = logical_attn_allocator._kvcache.c4_kv_pool
        self._size_full = logical_attn_allocator.size_full
        self._size_hisparse = self.hisparse_kvcache.size

        self.dtype = self.hisparse_kvcache.dtype
        self.device = self.hisparse_kvcache.device
        # Keep the public page_size as the logical DSV4 full/SWA page size.
        # C4 HiSparse allocation/device-buffer code must use the compressed page size.
        self.page_size = logical_attn_allocator.page_size
        self.hisparse_page_size = self.hisparse_kvcache.page_size

        self.logical_attn_allocator = logical_attn_allocator
        self._kvcache = logical_attn_allocator._kvcache
        self.hisparse_attn_allocator = PagedTokenToKVPoolAllocator(
            self._size_hisparse,
            self.hisparse_page_size,
            self.dtype,
            self.device,
            self.hisparse_kvcache,
            logical_attn_allocator.need_sort,
        )

        self.full_to_hisparse_device_index_mapping = torch.cat(
            [
                torch.zeros(
                    self._kvcache.c4_logical_size + self.hisparse_page_size,
                    dtype=torch.int64,
                    device=self.device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=self.device),
            ]
        )
        self._page_ownership = _HiSparsePageOwnership(
            mapping=self.full_to_hisparse_device_index_mapping,
            child_allocator=self.hisparse_attn_allocator,
            page_size=self.hisparse_page_size,
        )

        self.need_sort = logical_attn_allocator.need_sort
        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []
        self.clear()

        self.hisparse_kvcache.register_mapping(
            weakref.proxy(self.full_to_hisparse_device_index_mapping)
        )

    @property
    def size_full(self) -> int:
        return self._size_full

    @property
    def size(self) -> int:
        return self.logical_attn_allocator.size

    @property
    def size_swa(self) -> int:
        return self.logical_attn_allocator.size_swa

    @property
    def hisparse_device_page_size(self) -> int:
        return self.hisparse_page_size

    @property
    def full_to_swa_index_mapping(self):
        return self.logical_attn_allocator.full_to_swa_index_mapping

    def debug_print(self) -> str:
        msg = self.logical_attn_allocator.debug_print()
        msg += (
            f"#hisparse-available-size: "
            f"{self.hisparse_attn_allocator.available_size()}, "
        )
        return msg

    def get_kvcache(self):
        return self._kvcache

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        return self.logical_attn_allocator.translate_loc_from_full_to_swa(kv_indices)

    def full_available_size(self):
        return min(
            self.logical_attn_allocator.full_available_size(),
            self.hisparse_attn_allocator.available_size() * self.compress_ratio,
        )

    def schedulable_full_available_size(self):
        return min(
            self.logical_attn_allocator.full_available_size(),
            self._get_schedulable_hisparse_available() * self.compress_ratio,
        )

    def swa_available_size(self):
        return self.logical_attn_allocator.swa_available_size()

    def free_swa(self, free_indices: torch.Tensor):
        self.logical_attn_allocator.free_swa(free_indices)

    def available_size(self) -> int:
        return min(
            self.logical_attn_allocator.available_size(),
            self.hisparse_attn_allocator.available_size() * self.compress_ratio,
        )

    def alloc(self, need_size: int):
        raise NotImplementedError(
            "DeepSeek V4 HiSparse allocator does not support direct token allocation; "
            "use alloc_extend or alloc_decode instead."
        )

    def alloc_logical_only(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        """Allocate decode logical indices without allocating C4 hisparse device pages."""
        return self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )

    def alloc_extend_swa_tail(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        swa_tail_len: int,
    ):
        return self.logical_attn_allocator.alloc_extend_swa_tail(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=extend_num_tokens,
            swa_tail_len=swa_tail_len,
        )

    def alloc_device_buffer(self, allocated_indices, need_size: int):
        assert need_size % self.hisparse_page_size == 0
        hisparse_indices = self.full_to_hisparse_device_index_mapping[allocated_indices]
        self.full_to_hisparse_device_index_mapping[allocated_indices] = 0
        hisparse_indices = hisparse_indices[hisparse_indices > 0]

        device_buffer_size = need_size - self.hisparse_page_size
        P = len(hisparse_indices)
        if P > device_buffer_size + 1:
            newest_src = hisparse_indices[P - 1].clone()
            old_at_dbs = hisparse_indices[device_buffer_size].clone()
            hisparse_indices[device_buffer_size] = newest_src
            hisparse_indices[P - 1] = old_at_dbs

        if len(hisparse_indices) >= need_size:
            buffer_indices = hisparse_indices[:need_size]
            surplus = hisparse_indices[need_size:]
            if surplus.numel() > 0:
                buffer_pages = torch.unique(buffer_indices // self.hisparse_page_size)
                surplus_pages = torch.unique(surplus // self.hisparse_page_size)
                pure_surplus = surplus_pages[~torch.isin(surplus_pages, buffer_pages)]
                if pure_surplus.numel() > 0:
                    self.free_hisparse_indices(pure_surplus * self.hisparse_page_size)
        else:
            page_residual_length = len(hisparse_indices) % self.hisparse_page_size
            if page_residual_length != 0:
                hisparse_indices = torch.cat(
                    [
                        hisparse_indices,
                        torch.arange(
                            hisparse_indices[-1] + 1,
                            hisparse_indices[-1]
                            + self.hisparse_page_size
                            - page_residual_length
                            + 1,
                            device=self.device,
                        ),
                    ]
                )
            extra_indices = self.hisparse_attn_allocator.alloc(
                need_size - len(hisparse_indices)
            )
            assert (
                extra_indices is not None
            ), "Hisparse allocation failed in alloc_device_buffer"
            buffer_indices = torch.cat([hisparse_indices, extra_indices])
        return buffer_indices

    def free_hisparse_indices(self, buffer_indices: torch.Tensor):
        assert self.hisparse_attn_allocator.is_not_in_free_group
        buffer_indices = buffer_indices[buffer_indices > 0]
        self._page_ownership.assert_unclaimed(buffer_indices)
        self.hisparse_attn_allocator.free(buffer_indices)

    def release_hisparse_ownership(
        self,
        *,
        mapping_indices: torch.Tensor,
        extra_owned_coordinates: torch.Tensor | None = None,
        clear_extra_owner: Callable[[], None] | None = None,
    ) -> None:
        self._page_ownership.release(
            mapping_indices=mapping_indices,
            extra_owned_coordinates=extra_owned_coordinates,
            clear_extra_owner=clear_extra_owner,
        )

    def debug_hisparse_ownership(
        self, extra_owned_coordinates: torch.Tensor | None = None
    ) -> dict[str, object]:
        return self._page_ownership.debug_snapshot(extra_owned_coordinates)

    def claim_hisparse_ownership(self, coordinates: torch.Tensor) -> None:
        self._page_ownership.claim(coordinates)

    def release_hisparse_mapped_pages(self, page_ids: torch.Tensor) -> None:
        self._page_ownership.release_mapped_pages(page_ids)

    def rehome_temporary_hisparse_pages(
        self,
        *,
        mapping_indices: torch.Tensor,
        retained_page_ids: torch.Tensor,
        install_retained_owner: Callable[[], None],
    ) -> None:
        self._page_ownership.rehome_temporary_pages(
            mapping_indices=mapping_indices,
            retained_page_ids=retained_page_ids,
            install_retained_owner=install_retained_owner,
        )

    def get_last_loc_compressed(self, last_locs: torch.Tensor):
        return (last_locs - 3) // self.compress_ratio

    def get_last_loc_hisparse_device(self, last_locs: torch.Tensor):
        return self.hisparse_kvcache._translate_loc_to_hisparse_device(
            self.get_last_loc_compressed(last_locs)
        )

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        assert self.page_size > 1

        num_new_pages_logical = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        num_new_pages_hisparse = get_num_new_pages(
            seq_lens=seq_lens_cpu // self.compress_ratio,
            page_size=self.hisparse_page_size,
            prefix_lens=prefix_lens_cpu // self.compress_ratio,
        )
        hisparse_prefix_lens_cpu = prefix_lens_cpu // self.compress_ratio
        hisparse_seq_lens_cpu = seq_lens_cpu // self.compress_ratio
        hisparse_last_loc = self.get_last_loc_hisparse_device(last_loc)
        old_hisparse_page_ids = torch.unique(
            hisparse_last_loc[hisparse_last_loc > 0] // self.hisparse_page_size
        )
        # Direct PD admission stores the prompt C4 payload only in host memory,
        # so the shared device mapping for its final logical location is zero.
        # PagedTokenToKVPoolAllocator would interpret that zero as a valid old
        # partial-page predecessor and continue at slots 1..N in sentinel page
        # zero. Multiple requests would then overwrite the same physical slots.
        # Explicitly acquire one page for every such missing partial-page owner
        # and synthesize its predecessor before the normal paged extension.
        missing_partial_owner = (
            (hisparse_seq_lens_cpu > hisparse_prefix_lens_cpu)
            & (hisparse_prefix_lens_cpu % self.hisparse_page_size != 0)
            & (hisparse_last_loc.to(device="cpu") // self.hisparse_page_size == 0)
        )
        num_missing_partial_pages = int(missing_partial_owner.sum().item())
        if (
            num_new_pages_logical
            > self.logical_attn_allocator.available_size() // self.page_size
        ):
            return None
        if (
            num_new_pages_hisparse + num_missing_partial_pages
            > self.hisparse_attn_allocator.available_size() // self.hisparse_page_size
        ):
            if not self._ensure_hisparse_available(
                (num_new_pages_hisparse + num_missing_partial_pages)
                * self.hisparse_page_size
            ):
                return None

        logical_indices = self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        assert logical_indices is not None, "Logical allocation failed in alloc_extend"

        compressed_logical_indices = (
            self.hisparse_kvcache.translate_loc_from_full_to_compressed(logical_indices)
        )
        partial_page_indices = None
        hisparse_indices = None
        committed = False
        try:
            if num_missing_partial_pages > 0:
                partial_page_indices = self.hisparse_attn_allocator.alloc(
                    num_missing_partial_pages * self.hisparse_page_size
                )
                if partial_page_indices is None:
                    raise RuntimeError(
                        "Hisparse allocation failed for direct-PD partial-page owners"
                    )
                page_starts = partial_page_indices.reshape(
                    num_missing_partial_pages, self.hisparse_page_size
                )[:, 0]
                prefix_offsets = (
                    hisparse_prefix_lens_cpu[missing_partial_owner]
                    % self.hisparse_page_size
                ).to(device=page_starts.device)
                hisparse_last_loc = hisparse_last_loc.clone()
                hisparse_last_loc[
                    missing_partial_owner.to(hisparse_last_loc.device)
                ] = (page_starts + prefix_offsets - 1).to(hisparse_last_loc.dtype)
            hisparse_indices = self.hisparse_attn_allocator.alloc_extend(
                prefix_lens // self.compress_ratio,
                prefix_lens_cpu // self.compress_ratio,
                seq_lens // self.compress_ratio,
                seq_lens_cpu // self.compress_ratio,
                hisparse_last_loc,
                len(compressed_logical_indices),
                num_new_pages=num_new_pages_hisparse,
            )
            if hisparse_indices is None:
                raise RuntimeError("Hisparse allocation failed in alloc_extend")

            self.full_to_hisparse_device_index_mapping[compressed_logical_indices] = (
                hisparse_indices.to(torch.int64)
            )
            committed = True
        finally:
            if not committed:
                # Only pages not owned before this call may be returned.  Both
                # paged allocators can reuse the request's existing tail page,
                # so freeing the complete alloc_extend result would corrupt the
                # live prefix rather than roll the transaction back.
                self.full_to_hisparse_device_index_mapping[
                    compressed_logical_indices
                ] = 0
                physical_chunks = [
                    chunk
                    for chunk in (partial_page_indices, hisparse_indices)
                    if chunk is not None and chunk.numel() > 0
                ]
                if physical_chunks:
                    physical_page_ids = torch.unique(
                        torch.cat(physical_chunks) // self.hisparse_page_size
                    )
                    if old_hisparse_page_ids.numel() > 0:
                        physical_page_ids = physical_page_ids[
                            ~torch.isin(physical_page_ids, old_hisparse_page_ids)
                        ]
                    if physical_page_ids.numel() > 0:
                        offsets = torch.arange(
                            self.hisparse_page_size,
                            dtype=torch.int64,
                            device=physical_page_ids.device,
                        )
                        self.hisparse_attn_allocator.free(
                            (
                                physical_page_ids[:, None] * self.hisparse_page_size
                                + offsets
                            ).reshape(-1)
                        )

                logical_page_ids = torch.unique(logical_indices // self.page_size)
                old_logical_page_ids = torch.unique(
                    last_loc[last_loc > 0] // self.page_size
                )
                if old_logical_page_ids.numel() > 0:
                    logical_page_ids = logical_page_ids[
                        ~torch.isin(logical_page_ids, old_logical_page_ids)
                    ]
                if logical_page_ids.numel() > 0:
                    offsets = torch.arange(
                        self.page_size,
                        dtype=torch.int64,
                        device=logical_page_ids.device,
                    )
                    self.logical_attn_allocator.free(
                        (logical_page_ids[:, None] * self.page_size + offsets).reshape(
                            -1
                        )
                    )
        return logical_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        return self.logical_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )

    def free_compressed(self, compressed_indices: torch.Tensor):
        self.release_hisparse_ownership(mapping_indices=compressed_indices)

    def free_hisparse(self, free_indices: torch.Tensor):
        compressed_indices = (
            self.hisparse_kvcache.translate_loc_from_full_to_compressed(free_indices)
        )
        self.free_compressed(compressed_indices)

    def clear(self):
        self.logical_attn_allocator.clear()
        self.hisparse_attn_allocator.clear()

        self.full_to_hisparse_device_index_mapping[:-1].fill_(0)
        self._page_ownership.clear()
        self.is_not_in_free_group = True
        self.free_group = []

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            self.logical_attn_allocator.free(free_index)
        else:
            self.free_group.append(self._copy_for_free_group(free_index))
