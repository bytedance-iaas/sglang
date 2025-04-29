# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import triton
import triton.language as tl


def blocksparse_flash_attn_varlen_fwd(
    q,
    k,
    v,  # (#tokens, n_heads, head_size)
    cu_seqlens_k,
    cu_seqlens_q,
    sm_scale,
    sparse_layout,
    *,
    block_size=64,
    q_block_size=None,
    max_seqlen=None
):
    # split q to blocks

    assert isinstance(sparse_layout, (list, tuple))

    _, n_heads, head_size = q.shape
    batch_size = cu_seqlens_k.size(0) - 1
    q_block_size = q_block_size or block_size

    assert q.dim() == k.dim() == v.dim() == 3
    assert q.size(1) % k.size(1) == 0
    assert q.size(2) == k.size(2)
    # TODO(linxihui): allow k, v to have different head_size
    assert k.shape == v.shape
    assert cu_seqlens_k.dim() == 1

    q_k_ratio = q.size(1) // k.size(1)

    if cu_seqlens_q is None:
        if q.size(0) == batch_size:  # decoding only
            cu_seqlens_q = torch.arange(
                0,
                batch_size + 1,
                dtype=cu_seqlens_k.dtype,
                device=cu_seqlens_k.device,
            )
        elif q.size(0) == k.size(0):
            cu_seqlens_q = cu_seqlens_k
        else:
            raise ValueError(
                "cu_seqlens_q must be specified\
                    if it mix of prefilling and decoding."
            )
    else:
        assert cu_seqlens_k.size(0) == cu_seqlens_q.size(0)

    # switch to use cpu to avoid too many kernel launches when iterated over
    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).cpu()
    k_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).cpu()

    assert torch.logical_or(
        q_lens == 1, k_lens == q_lens
    ).all(), "length of q should either be 1 (decoding) or same as k (prefilling)."

    if max_seqlen:
        assert k_lens.max() <= max_seqlen

    n_blocks = (q_lens + q_block_size - 1) // q_block_size

    q_batch_ids = torch.tensor(
        [i for i, n in enumerate(n_blocks) for _ in range(n)],
        dtype=cu_seqlens_q.dtype,
        device=cu_seqlens_q.device,
    )
    q_start_sids = torch.tensor(
        [i * q_block_size for n in n_blocks for i in range(n)],
        dtype=cu_seqlens_q.dtype,
        device=cu_seqlens_q.device,
    )

    out = q.new_empty(q.shape)
    cu_seqlens_q = cu_seqlens_q.contiguous()
    cu_seqlens_k = cu_seqlens_k.contiguous()

    layout_crow_indices, layout_col_indices = sparse_layout
    block_d = triton.next_power_of_2(head_size)

    decoding_only = (q_lens == 1).all().item()
    grid = (len(q_start_sids), n_heads, 1)

    _fwd_kernel_batch_inference[grid](
        q,
        k,
        v,
        out,
        sm_scale,
        cu_seqlens_q[:-1],
        cu_seqlens_q[1:],
        cu_seqlens_k[:-1],
        cu_seqlens_k[1:],
        q_batch_ids,
        q_start_sids,
        0,
        *q.stride(),
        0,
        *k.stride(),
        0,
        *v.stride(),
        0,
        *out.stride(),
        layout_crow_indices,
        layout_col_indices,
        *layout_crow_indices.stride(),
        *layout_col_indices.stride(),
        q_k_ratio,
        HAS_BATCH_DIM=False,
        D_HEAD=head_size,
        BLOCK_M=q_block_size,
        BLOCK_N=block_size,
        BLOCK_D=block_d,
        BLOCK_M_LOADING=(16 if decoding_only else q_block_size),  # smaller for decoding
        EVEN_D=block_d == head_size,
        num_warps=1 if decoding_only else 4,
        num_stages=3
    )

    return out


@triton.jit
def _fwd_kernel_inner(
    acc,
    l_i,
    m_i,
    q,
    Q,
    k_block_col_idx,
    layout_col_ptr,
    layout_col_stride_h,
    layout_col_stride_m,
    k_ptrs,
    v_ptrs,
    off_h,
    offs_m,
    offs_n,
    offs_d,
    stride_kt,
    stride_vt,
    sm_scale,
    k_seqlen,
    past_len,
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M_LOADING: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
    EVEN_D: tl.constexpr,
    M_LT_N: tl.constexpr,
):
    k_block_id = tl.load(
        layout_col_ptr
        + off_h * layout_col_stride_h
        + k_block_col_idx * layout_col_stride_m
    ).to(tl.int32)
    start_n = k_block_id * BLOCK_N
    if LAST_K_BLOCK:
        if EVEN_D:
            k = tl.load(
                k_ptrs + start_n * stride_kt,
                mask=offs_n[None, :] + start_n < k_seqlen,
                other=0.0,
            )
        else:
            k = tl.load(
                k_ptrs + start_n * stride_kt,
                mask=(offs_n[None, :] + start_n < k_seqlen)
                & (offs_d[:, None] < D_HEAD),
                other=0.0,
            )
    else:
        if EVEN_D:
            k = tl.load(k_ptrs + start_n * stride_kt)
        else:
            k = tl.load(
                k_ptrs + start_n * stride_kt, mask=offs_d[:, None] < D_HEAD, other=0.0
            )

    qk = tl.zeros([BLOCK_M_LOADING, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    if LAST_K_BLOCK | M_LT_N:
        qk += tl.where(
            offs_m[:, None] + past_len >= (start_n + offs_n[None, :]),
            0,
            float("-inf"),
        )

    # flash-attn2
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    p = tl.math.exp2(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    acc = acc * alpha[:, None]
    # update m_i
    m_i = m_ij
    l_i = l_i * alpha + l_ij

    p = p.to(Q.dtype.element_ty)
    # update acc
    if LAST_K_BLOCK:
        if EVEN_D:
            v = tl.load(
                v_ptrs + start_n * stride_vt,
                mask=offs_n[:, None] + start_n < k_seqlen,
                other=0.0,
            )
        else:
            v = tl.load(
                v_ptrs + start_n * stride_vt,
                mask=(offs_n[:, None] + start_n < k_seqlen)
                & (offs_d[None, :] < D_HEAD),
                other=0.0,
            )
    else:
        if EVEN_D:
            v = tl.load(v_ptrs + start_n * stride_vt)
        else:
            v = tl.load(
                v_ptrs + start_n * stride_vt, mask=offs_d[None, :] < D_HEAD, other=0.0
            )

    acc += tl.dot(p, v)

    return acc, l_i, m_i


@triton.heuristics(
    {
        "M_LT_N": lambda kwargs: kwargs["BLOCK_M"] < kwargs["BLOCK_N"],
    }
)
@triton.jit
def _fwd_kernel_batch_inference(
    Q,
    K,
    V,
    Out,
    sm_scale,
    q_batch_starts,
    q_batch_ends,
    k_batch_starts,
    k_batch_ends,
    q_batch_ids,
    q_start_sids,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_ot,
    stride_oh,
    stride_od,
    layout_crow_ptr,
    layout_col_ptr,
    layout_crow_stride_h,
    layout_crow_stride_m,
    layout_col_stride_h,
    layout_col_stride_m,
    q_k_ratio,
    HAS_BATCH_DIM: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M_LOADING: tl.constexpr,
    EVEN_D: tl.constexpr,
    M_LT_N: tl.constexpr,
):
    """
    NOTATION:
    pid: position id
    sid: storage id
    sbid: storage block id
    pbid: position block id
    offs_m, offs_n: storage offsets of m-dim(q, row) and n-dim(k, col)

    TODO(linxihui):
    Optimize grouped-attn
    """
    off_zm = tl.program_id(0)
    off_h = tl.program_id(1)

    off_h_for_kv = off_h // q_k_ratio

    if HAS_BATCH_DIM:
        off_z = tl.program_id(2)
        Q += off_z * stride_qb
        K += off_z * stride_kb
        V += off_z * stride_vb
        Out += off_z * stride_ob
        start_m = off_zm
        q_start_sid = start_m * BLOCK_M  # always 0 for decoding
    else:
        off_z = tl.load(q_batch_ids + off_zm).to(tl.int32)  # [0, 0, 0, 1]
        q_start_sid = tl.load(q_start_sids + off_zm)
        start_m = q_start_sid // BLOCK_M  # q_sbid

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M_LOADING)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_cu_start = tl.load(q_batch_starts + off_z).to(tl.int32)
    q_seqlen = tl.load(q_batch_ends + off_z).to(tl.int32) - q_cu_start
    k_cu_start = tl.load(k_batch_starts + off_z).to(tl.int32)
    k_seqlen = tl.load(k_batch_ends + off_z).to(tl.int32) - k_cu_start
    past_len = k_seqlen - q_seqlen

    Q += q_cu_start * stride_qt + off_h * stride_qh
    K += k_cu_start * stride_kt + off_h_for_kv * stride_kh
    V += k_cu_start * stride_vt + off_h_for_kv * stride_vh
    Out += q_cu_start * stride_ot + off_h * stride_oh

    q_pbid = (past_len + q_start_sid) // BLOCK_M

    if EVEN_D:
        q = tl.load(
            Q + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
            mask=offs_m[:, None] < q_seqlen,
            other=0.0,
        )
    else:
        q = tl.load(
            Q + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
            mask=(offs_m[:, None] < q_seqlen) & (offs_d[None, :] < D_HEAD),
            other=0.0,
        )

    sparse_crow_ptr = (
        layout_crow_ptr + off_h * layout_crow_stride_h + q_pbid * layout_crow_stride_m
    )

    # TODO(linxihui): load at once, with any Triton version
    # that supports `tl.split`, e.g., Triton 3.0
    k_block_start = tl.load(sparse_crow_ptr).to(tl.int32)
    k_block_end = tl.load(sparse_crow_ptr + 1).to(tl.int32)

    m_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M_LOADING, BLOCK_D], dtype=tl.float32)

    k_ptrs = K + offs_n[None, :] * stride_kt + offs_d[:, None] * stride_kd
    v_ptrs = V + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd

    sm_scale *= 1.44269504  # 1/log2 as we use base2 for exponential and logarithm

    for k_block_col_idx in range(k_block_start, k_block_end - 1):
        acc, l_i, m_i = _fwd_kernel_inner(
            acc,
            l_i,
            m_i,
            q,
            Q,
            k_block_col_idx,
            layout_col_ptr,
            layout_col_stride_h,
            layout_col_stride_m,
            k_ptrs,
            v_ptrs,
            off_h,
            offs_m,
            offs_n,
            offs_d,
            stride_kt,
            stride_vt,
            sm_scale,
            k_seqlen,
            past_len,
            False,
            BLOCK_M_LOADING,
            BLOCK_N,
            D_HEAD,
            EVEN_D,
            M_LT_N,
        )

    acc, l_i, m_i = _fwd_kernel_inner(
        acc,
        l_i,
        m_i,
        q,
        Q,
        k_block_end - 1,
        layout_col_ptr,
        layout_col_stride_h,
        layout_col_stride_m,
        k_ptrs,
        v_ptrs,
        off_h,
        offs_m,
        offs_n,
        offs_d,
        stride_kt,
        stride_vt,
        sm_scale,
        k_seqlen,
        past_len,
        True,
        BLOCK_M_LOADING,
        BLOCK_N,
        D_HEAD,
        EVEN_D,
        M_LT_N,
    )

    # flash-attn 2
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    # write output
    if EVEN_D:
        tl.store(
            Out + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od,
            acc,
            mask=offs_m[:, None] < q_seqlen,
        )
    else:
        tl.store(
            Out + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od,
            acc,
            mask=(offs_m[:, None] < q_seqlen) & (offs_d[None, :] < D_HEAD),
        )


from functools import lru_cache

IS_COMPUTE_8_OR_ABOVE = True


class csr_matrix:
    """Simple implementation of CSR matrix conversion without scipy.
    This replaced scipy.sparse.csr_matrix() previously used."""

    def __init__(self, input_array):
        if not isinstance(input_array, np.ndarray):
            raise ValueError("Input must be a NumPy array")

        self.shape = input_array.shape
        rows, cols = self.shape
        data = []
        indices = []
        indptr = [0]

        for i in range(rows):
            for j in range(cols):
                if input_array[i, j]:
                    data.append(input_array[i, j])
                    indices.append(j)
            indptr.append(len(indices))

        self.data = np.array(data)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)


def _get_sparse_attn_mask_homo_head(
    q_len: int,
    max_seqlen: int,
    dtype: torch.dtype,
    device: torch.device,
    block_size: int = 128,
    local_blocks: int = 4,
    vert_stride: int = 4,
    return_dense: bool = False,
):
    """
    :return: a tuple of 3:
        - tuple of crow_indices, col_indices representation
            of CSR format.
        - block dense mask
        - all token dense mask (be aware that it can be
            OOM if it is too big) if `return_dense==True`,
            otherwise, None
    """
    with torch.no_grad():
        num_blocks = triton.cdiv(max_seqlen, block_size)
        q_pos = torch.arange(num_blocks)[:, None]
        k_pos = torch.arange(num_blocks)[None]
        mask_vert_strided = (torch.arange(num_blocks) + 1) % vert_stride == 0
        block_mask_dense = (
            ((q_pos >= k_pos) & ((q_pos - k_pos < local_blocks) | mask_vert_strided))
            .to(device)
            .to(dtype)
        )
        num_blocks_q = triton.cdiv(q_len, block_size)
        block_mask_dense_output = dense_to_crow_col(
            block_mask_dense[-num_blocks_q:].contiguous()
        )
    if return_dense:
        mask_dense = torch.kron(
            block_mask_dense,
            block_mask_dense.new_ones((block_size, block_size)),
        )
        causal_mask = torch.tril(torch.ones(max_seqlen, max_seqlen)).type_as(
            mask_dense
        )[-q_len:]
        mask_dense = mask_dense[-q_len:, :max_seqlen] * causal_mask
        return (
            block_mask_dense_output,
            block_mask_dense,
            mask_dense,
        )
    else:
        return (
            block_mask_dense_output,
            block_mask_dense,
            None,
        )


def dense_to_crow_col(x: torch.Tensor):
    """Turning a 2D/3D torch tensor (x) to CSR rows/cols indexing.
    NOTE: col_indices padded -1
    """
    device = x.device
    pad = -1
    dim = x.dim()
    assert x.dim() in (2, 3)
    if x.dim() == 2:
        x = x[None]
    x = [csr_matrix(xi.bool().cpu().numpy()) for xi in x]
    crows = torch.vstack([torch.from_numpy(xi.indptr) for xi in x])
    cols = [torch.from_numpy(xi.indices) for xi in x]
    max_cols = max(len(xi) for xi in cols)
    cols = [torch.cat([xi, pad + xi.new_zeros(max_cols - xi.shape[0])]) for xi in cols]
    cols = torch.vstack(cols)
    if dim == 2:
        crows = crows[0]
        cols = cols[0]
    return crows.to(device), cols.to(device)


def crow_col_to_dense(
    crows: torch.Tensor, cols: torch.Tensor, dtype: torch.dtype = torch.float16
):
    dim = crows.dim()
    if dim == 1:
        crows = crows[None]
        cols = cols[None]
    device = crows.device
    crows, cols = crows.cpu(), cols.cpu()  # faster in cpu
    shape = (crows.shape[0], crows.shape[1] - 1, cols.max() + 1)
    x = torch.zeros(shape, dtype=dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            x[i, j, cols[i, crows[i, j] : crows[i, j + 1]]] = 1
    if dim == 1:
        x = x[0]
    return x.to(device)


def dense_to_ccol_row(x: torch.Tensor):
    """Similar, but to CSC format"""
    x = x.transpose(-2, -1)
    return dense_to_crow_col(x)


def ccol_row_to_dense(
    ccol: torch.Tensor, rows: torch.Tensor, dtype: torch.dtype = torch.float16
):
    return crow_col_to_dense(ccol, rows, dtype).permute(0, 2, 1).contiguous()


def get_head_sliding_step(n_heads: int, vert_stride: int, homo_head: bool = False):
    if homo_head:
        return 0
    return max(1, int(vert_stride / n_heads))


def binary_mask_to_bias(mask_dense: torch.Tensor):
    mask_dense = 1 - mask_dense
    mask_dense.masked_fill_(mask_dense.bool(), -torch.inf)
    return mask_dense


@lru_cache
def get_sparse_attn_mask(
    n_heads: int,
    q_len: int,
    max_seqlen: int,
    dtype: torch.dtype,
    device: torch.device,
    block_size: int = 64,
    local_blocks: int = 4,
    vert_stride: int = 4,
    homo_head: bool = True,
    return_dense: bool = False,
    dense_mask_type: str = "binary",
):
    """
    :param dense_mask_type: "binary" (0 for skip token, 1 for others)
        or "bias" (-inf for skip token, 0 or others)
    :return: a tuple of 3:
        - tuple of crow_indices, col_indices representation
            of CSR format.
        - block dense mask
        - all token dense mask (be aware that it can be OOM if it
            is too big) if `return_dense==True`, otherwise, None
    """
    assert dense_mask_type in ("binary", "bias")
    if homo_head:
        with torch.no_grad():
            (crow, col), block_mask_dense, mask_dense = _get_sparse_attn_mask_homo_head(
                q_len,
                max_seqlen,
                dtype,
                device,
                block_size,
                local_blocks,
                vert_stride,
                return_dense,
            )
            crow = crow[None].expand(n_heads, crow.shape[0])
            col = col[None].expand(n_heads, col.shape[0])
            if return_dense:
                mask_dense = mask_dense[None].expand(n_heads, *mask_dense.shape)
                if dense_mask_type == "bias":
                    mask_dense = binary_mask_to_bias(mask_dense)
            return (crow, col), block_mask_dense, mask_dense

    with torch.no_grad():
        num_blocks = triton.cdiv(max_seqlen, block_size)
        q_pos = torch.arange(num_blocks)[None, :, None]
        k_pos = torch.arange(num_blocks)[None, None]
        head_sliding_step = get_head_sliding_step(n_heads, vert_stride)
        mask_vert_strided = [
            (torch.arange(num_blocks) + h * head_sliding_step + 1) % vert_stride == 0
            for h in range(n_heads)
        ]
        mask_vert_strided = torch.vstack(mask_vert_strided).unsqueeze(1)
        block_mask_dense = (
            ((q_pos >= k_pos) & ((q_pos - k_pos < local_blocks) | mask_vert_strided))
            .to(device)
            .to(dtype)
        )
        num_blocks_q = triton.cdiv(q_len, block_size)
        block_mask_dense_output = block_mask_dense[:, -num_blocks_q:]
    if return_dense:
        mask_dense = torch.kron(
            block_mask_dense,
            block_mask_dense.new_ones((block_size, block_size)),
        )
        causal_mask = torch.tril(torch.ones(max_seqlen, max_seqlen)).type_as(
            mask_dense
        )[-q_len:]
        mask_dense = mask_dense[..., -q_len:, :max_seqlen] * causal_mask[None]
        if dense_mask_type == "bias":
            mask_dense = binary_mask_to_bias(mask_dense)

        return (
            dense_to_crow_col(block_mask_dense_output),
            block_mask_dense,
            mask_dense,
        )
    else:
        return (
            dense_to_crow_col(block_mask_dense_output),
            block_mask_dense,
            None,
        )


class LocalStridedBlockSparseAttn(torch.nn.Module):

    def __init__(
        self,
        n_heads,
        max_seqlen,
        local_blocks,
        vert_stride,
        block_size,
        device=None,
        dtype=None,
        homo_head=False,
        active_head_range=None,
        q_block_size=None,
        use_spda=None,
    ):
        super().__init__()
        if use_spda is None:
            use_spda = False
        device = device or (torch.cuda.current_device())
        device = torch.device(device)
        # NOTE: vllm CPU backend support BF16 instead of FP16.
        dtype = dtype or (
            torch.bfloat16
            if IS_COMPUTE_8_OR_ABOVE or device.type == "cpu"
            else torch.half
        )

        self.n_heads = n_heads
        self.max_seqlen = max_seqlen
        self.local_blocks = local_blocks
        self.vert_stride = vert_stride
        self.use_spda = use_spda
        self.dtype = dtype
        self.device = device
        self.block_size = block_size
        self.q_block_size = q_block_size
        self.homo_head = homo_head
        self.active_head_range = active_head_range
        self.head_sliding_step = get_head_sliding_step(n_heads, vert_stride, homo_head)

        sparse_layout, sparse_pattern, self.dense_attn_mask = self.get_attn_pattern(
            dtype, device
        )

        if q_block_size is not None and q_block_size != block_size:
            if q_block_size > block_size:
                assert q_block_size % block_size == 0
                blocks_to_merge = q_block_size // block_size
                shape = sparse_pattern.shape
                sparse_pattern = sparse_pattern.view(
                    shape[0], -1, blocks_to_merge, shape[-1]
                )
                sparse_pattern = sparse_pattern.sum(2)
                sparse_layout = dense_to_crow_col(sparse_pattern)
            else:
                raise ValueError(
                    "Does not support smaller q_block_size. It will be slower."
                )

        self.sparse_layout = sparse_layout

    def get_attn_pattern(self, dtype, device):
        sparse_layout, sparse_pattern, dense_attn_mask = get_sparse_attn_mask(
            self.n_heads,
            self.max_seqlen,
            self.max_seqlen,
            dtype,
            device,
            block_size=self.block_size,
            local_blocks=self.local_blocks,
            vert_stride=self.vert_stride,
            homo_head=self.homo_head,
            return_dense=self.use_spda,
            dense_mask_type="bias",
        )
        if (not self.homo_head) and (self.active_head_range is not None):
            assert isinstance(self.active_head_range, tuple)
            assert len(self.active_head_range) == 2
            h_start, h_end = self.active_head_range
            sparse_layout = tuple(x[h_start:h_end] for x in sparse_layout)
            if self.use_spda:
                dense_attn_mask = dense_attn_mask[h_start:h_end]
        return sparse_layout, sparse_pattern, dense_attn_mask

    def varlen_attn(self, q, k, v, cu_seqlens_k, cu_seqlens_q=None, sm_scale=None):
        """
        q, k, v: shape = (num_tokens, num_heads_q/kv, head_size).
        Support grouped attention, with `q[:, i*r:(i*r + r)]`
        is correspondent to `k[:, i]`, where `r` is the q/k ratio.
        cu_seqlens_k: shape=(batch_size + 1,),
        indicating segment of samples,
        e.g., `k[cu_seqlen[i]:cu_seqlne[i+1]]` is q of sample i
        cu_seqlens_q: shape=(batch_size + 1, ).
        Default None: same as cu_seqlens_k for prefilling or
        [0, 1, .., batch_size] for decoding.
        The only case you need to specify is when q is a mix of
        prefilling and decoding.
        sm_scale: softmax scale, default to 1/sqrt(head_size).

        return: tensor of shape as q.
        """
        assert (
            IS_COMPUTE_8_OR_ABOVE
        ), "Requires compute capability of 8 or above (Ampere or newer) to use \
            Triton kernel."

        sm_scale = sm_scale or 1.0 / math.sqrt(q.size(-1))

        return blocksparse_flash_attn_varlen_fwd(
            q,
            k,
            v,
            cu_seqlens_k,
            cu_seqlens_q,
            sm_scale,
            self.sparse_layout,
            block_size=self.block_size,
            q_block_size=self.q_block_size,
            max_seqlen=self.max_seqlen,
        )

    @staticmethod
    def transpose_and_pad(x, cu_seqlens, maxlen, head_repeats=1):
        """
        :param x: (total_tokens, n_heads, head_size)
        :return: (batch, n_heads, length, head_size)
        """
        x_padded = x.new_empty(
            len(cu_seqlens) - 1, x.size(1), head_repeats, maxlen, x.size(2)
        )
        cu_seqlens = cu_seqlens.cpu()
        for i, (s, e) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
            x_padded[i, :, :, : e - s].copy_(x[s:e].transpose(0, 1).unsqueeze(1))
        return x_padded.flatten(1, 2)

    @staticmethod
    def transpose_and_unpad(x_padded, cu_seqlens):
        """
        :param x_padded: (batch, n_heads, length, head_size)
        :return: (total_tokens, n_heads, head_size)
        """
        cu_seqlens = cu_seqlens.cpu()
        total_n_tokens = cu_seqlens[-1]
        x = x_padded.new_empty(total_n_tokens, x_padded.size(1), x_padded.size(3))
        for i, (s, e) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
            x[s:e].copy_(x_padded[i, :, : e - s].transpose(0, 1))
        return x

    def spda(self, q, k, v, cu_seqlens_k, cu_seqlens_q=None, sm_scale=None):
        """For CPU, V100 or other older GPUs.
        NOTE: torch SPDA supports nested tensor,
        but seems extremely slow. Choose to pad instead.
        """
        assert (
            cu_seqlens_q is None or (cu_seqlens_q == cu_seqlens_k).all()
        ), "Can only handle prompt with SPDA."
        assert q.size(0) == k.size(0), "can only handle prompt with SPDA."

        assert q.size(1) % k.size(1) == 0
        q_k_ratio = q.size(1) // k.size(1)
        sm_scale = sm_scale or 1.0 / math.sqrt(q.size(-1))
        cu_seqlens = cu_seqlens_k.cpu()
        maxlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        if (
            self.dense_attn_mask.dtype != q.dtype
            or self.dense_attn_mask.device != q.device
        ):
            _, _, self.dense_attn_mask = self.get_attn_pattern(q.dtype, q.device)
        attn_mask = self.dense_attn_mask[None, :, :maxlen, :maxlen]

        q2 = self.transpose_and_pad(q, cu_seqlens, maxlen, 1)
        k2, v2 = (
            self.transpose_and_pad(x, cu_seqlens, maxlen, q_k_ratio) for x in [k, v]
        )
        spda_output = torch.nn.functional.scaled_dot_product_attention(
            q2, k2, v2, attn_mask=attn_mask, scale=sm_scale
        )
        return self.transpose_and_unpad(spda_output, cu_seqlens)

    def forward(self, q, k, v, cu_seqlens_k, cu_seqlens_q=None, sm_scale=None):
        """Dispatch to `varlen_attn` (Ampere or newer) or
        `self.spda`(cpu, Volta, Turing or older)based on
        the type of device used and cuda compute capability.

        q, k, v: shape = (num_tokens, num_heads_q/kv, head_size).
                Support grouped attention, with `q[:, i*r:(i*r + r)]`
                is correspondent to `k[:, i]`, where `r` is the q/k ratio.
        cu_seqlens_k: shape=(batch_size + 1,), indicating segment of samples,
                    e.g., `k[cu_seqlen[i]:cu_seqlne[i+1]]` is q of sample i
        cu_seqlens_q: shape=(batch_size + 1, ).
                    Default None: same as cu_seqlens_k for prefilling or
                    [0, 1, .., batch_size] for decoding.
                    The only case you need to specify
                    is when q is a mix of prefilling
                    and decoding.
        sm_scale: softmax scale, default to 1/sqrt(head_size).

        return: tensor of shape as q.
        """
        assert k.dim() == 3
        if self.use_spda:
            return self.spda(
                q,
                k,
                v,
                cu_seqlens_k,
                cu_seqlens_q=cu_seqlens_q,
                sm_scale=sm_scale,
            )
        return self.varlen_attn(
            q, k, v, cu_seqlens_k, cu_seqlens_q=cu_seqlens_q, sm_scale=sm_scale
        )
