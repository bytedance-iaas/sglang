import logging
from typing import List, Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def transform_index_page_table_prefill(**kwargs):
    return transform_index_page_table_prefill_ref(**kwargs)


def transform_index_page_table_decode(**kwargs):
    return transform_index_page_table_decode_ref(**kwargs)


@triton.jit
def transform_index_page_table_decode_kernel(
    page_table_ptr: torch.Tensor,
    topk_indices_ptr: torch.Tensor,
    result_ptr: torch.Tensor,
    page_size: tl.constexpr,
    max_seqlen_k: tl.constexpr,
):
    TOPK: tl.constexpr = 2048
    req_id = tl.program_id(0)
    page_table_ptr = page_table_ptr + req_id * max_seqlen_k
    topk_indices_ptr = topk_indices_ptr + req_id * TOPK
    result_ptr = result_ptr + req_id * TOPK

    offset = tl.arange(0, TOPK)  # topk should be 2048
    loaded_topk_indices = tl.load(topk_indices_ptr + offset)
    mask = loaded_topk_indices >= 0
    loaded_kv_indices = tl.load(page_table_ptr + loaded_topk_indices, mask=mask)
    tl.store(result_ptr + offset, loaded_kv_indices, mask=mask)
    tl.store(result_ptr + offset, -1, mask=~mask)


def transform_index_page_table_decode_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
) -> torch.Tensor:
    """
    Transform the page table according to topk indices for sparse topk attention.
    Args:
        page_table: [qo_len, max_seqlen_k], the original page table
        topk_indices: [qo_len, topk], the topk indices for each query position
    Returns:
        transformed_page_table: [qo_len, topk], the transformed page table
        For out-of-bound indices in topk_indices, this should be filled with -1.
    """
    assert page_size == 1
    assert page_table.shape[0] == topk_indices.shape[0]
    assert topk_indices.shape[1] == 2048
    qo_len = topk_indices.shape[0]
    max_seqlen_k = page_table.shape[1]
    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    # Launch triton kernel
    grid = (qo_len,)
    transform_index_page_table_decode_kernel[grid](
        page_table,
        topk_indices,
        result,
        page_size,
        max_seqlen_k=max_seqlen_k,
    )
    return result


def transform_index_page_table_prefill_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
) -> torch.Tensor:
    # TODO(baizhou): can be implemented with another triton kernel
    assert page_size == 1
    result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert len(extend_lens_cpu) == page_table.shape[0]
    offset = 0
    for i, l in enumerate(extend_lens_cpu):
        transform_index_page_table_decode_fast(
            page_table[i].unsqueeze(0).expand(l, -1),
            topk_indices[offset : offset + l],
            result=result[offset : offset + l],
        )
        offset += l
    assert offset == topk_indices.shape[0]
    return result


def transform_index_page_table_decode_ref(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
) -> torch.Tensor:
    assert page_size == 1
    assert page_table.shape[0] == topk_indices.shape[0]
    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert result.shape == topk_indices.shape
    torch.gather(
        page_table.to(result.dtype),
        dim=1,
        index=topk_indices.clamp(min=0),
        out=result,
    )
    result[topk_indices < 0] = -1
    return result


def transform_index_page_table_prefill_ref(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
) -> torch.Tensor:
    assert page_size == 1
    result = torch.empty_like(topk_indices, dtype=torch.int32)
    if len(extend_lens_cpu) != page_table.shape[0]:
        # CUDA-graph replay can pass a fixed-size page table buffer whose
        # leading rows are already expanded per query token. In that case the
        # per-token page table rows align directly with topk_indices.
        if page_table.shape[0] >= topk_indices.shape[0]:
            logger.warning_once(
                "DSA prefill page table row mismatch; using query-token "
                "aligned prefix. page_table_shape=%s topk_indices_shape=%s "
                "extend_lens_len=%s extend_lens_sum=%s",
                tuple(page_table.shape),
                tuple(topk_indices.shape),
                len(extend_lens_cpu),
                sum(extend_lens_cpu),
            )
            return transform_index_page_table_decode_ref(
                page_table[: topk_indices.shape[0]],
                topk_indices,
                result=result,
                page_size=page_size,
            )
        raise AssertionError(
            "DSA prefill page table row mismatch: "
            f"page_table_shape={tuple(page_table.shape)} "
            f"topk_indices_shape={tuple(topk_indices.shape)} "
            f"extend_lens_len={len(extend_lens_cpu)} "
            f"extend_lens_sum={sum(extend_lens_cpu)}"
        )
    offset = 0
    for i, l in enumerate(extend_lens_cpu):
        transform_index_page_table_decode_ref(
            page_table[i].unsqueeze(0).expand(l, -1),
            topk_indices[offset : offset + l],
            result=result[offset : offset + l],
        )
        offset += l
    assert offset == topk_indices.shape[0]
    return result


if __name__ == "__main__":
    bs, topk, max_seqlen = 10, 2048, 3000
    page_table = torch.randint(0, 100, (bs, max_seqlen), device="cuda")
    topk_indices = torch.full((bs, topk), -1, device="cuda")
    topk_indices[:, :1600] = torch.arange(1600).unsqueeze(0).repeat(bs, 1)
    ref_result = transform_index_page_table_decode_ref(page_table, topk_indices)
    result = transform_index_page_table_decode_fast(page_table, topk_indices)
    assert torch.all(result == ref_result)
    print("Passed")
