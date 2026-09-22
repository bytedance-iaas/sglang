"""Correctness tests for the DeepSeek-V4 (DSA indexer) JIT top-k transform v2.

The v2 kernel selects the per-row top-k of ``scores`` (ragged ``seq_lens``) and
writes the page-table transform of the selected raw indices into the output. We
validate against ``torch.topk`` with a small tolerance for boundary ties (the
fp16 coarse histogram can swap elements of equal score).

Coverage is organized around the kernel's exact dispatch so every template and
its boundaries are exercised.  The cluster templates remain covered by the
legacy matrix when enabled, but production dispatch currently routes all long
rows through Streaming until cluster-wide overflow refinement is available:

  template      per-row seq            reached when
  --------      ----------             ------------
  trivial       seq <= k
  Register2     k < seq <= 8192        max_seq <= 8192          (level 0)
  Register4     8192 < seq <= 16384    max_seq <= 16384         (level 1)
  Streaming     seq > 16384            max_seq > 16384 (level 2)
  Cluster       currently disabled     requires cluster-wide exact refinement

Historical cluster dispatch boundaries (batch 30/31 and 128/129) remain in the
matrix to prove they deterministically take the exact Streaming path. Boundary
seq lengths (8192/8193, 16384/16385, 65535/65536/65537) are included explicitly,
across k in {512,1024,2048} and identity/perm page tables.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_512_v2
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE_SIZE = 64  # c4 page size = 256 // 4
PAGE_BITS = PAGE_SIZE.bit_length() - 1
PAGE_MASK = PAGE_SIZE - 1
MAX_PERMIT_ERROR = 5
FLOOR = 65536  # kClusterFloor

# (batch, seq) chosen to land on each template and each dispatch boundary.
FIXED_CONFIGS = [
    # --- trivial (seq <= k) ---
    (8, 256),  # trivial for every k
    (16, 1024),  # trivial for k>=1024
    # --- Register2 (level 0: max_seq <= 8192) ---
    (8, 4096),
    (8, 8192),  # reg2 upper boundary
    (128, 8192),
    (300, 8192),  # batch > 128, still level 0
    # --- Register4 (level 1: 8192 < max_seq <= 16384) ---
    (8, 8193),  # just over reg2
    (64, 16384),  # reg4 upper boundary
    (256, 16384),  # batch > 128
    # --- Streaming (level 2: max_seq > 16384, non-cluster) ---
    (8, 16385),  # just over reg4 (small batch, seq < floor => non-cluster)
    (4, 32768),
    (16, 65535),  # just under floor
    (4, 65536),  # at floor (seq == floor => non-cluster)
    (100, 65536),
    # --- Cluster, fused small-batch kernel (batch <= 30, max_seq > floor) ---
    (1, 65537),  # single row just over floor
    (2, 131072),
    (8, 98304),
    (30, 131072),  # batch == pool boundary
    # --- Cluster, persistent pool + main kernel (30 < batch <= 128) ---
    (31, 131072),  # just over small-batch
    (40, 262144),  # N > pool of 30 => round-robin
    (64, 196608),
    (128, 131072),  # cluster batch upper boundary
    # --- batch > 128 => non-cluster streaming even at long ctx ---
    (129, 131072),
    (200, 262144),
]


def _assert_topk_close(
    scores_cpu, ref_raw, our_raw, bs, seq_lens, k, max_permit_error=MAX_PERMIT_ERROR
):
    """Set-compare our top-k raw indices vs torch's, tolerating equal-score ties."""
    bad = 0
    for i in range(bs):
        L = int(seq_lens[i])
        ref, our = set(ref_raw[i]), set(our_raw[i])
        more, less = our - ref, ref - our
        if more or less:
            mv = sorted(scores_cpu[i, list(more)].tolist())
            lv = sorted(scores_cpu[i, list(less)].tolist())
            if mv != lv:  # not merely a tie swap -> genuine error
                bad += len(more)
                print(
                    f"b={i} L={L} k={k}: more={list(more)[:4]} less={list(less)[:4]} mv={mv[:3]} lv={lv[:3]}"
                )
        assert len(our) == min(
            k, L
        ), f"b={i} L={L} k={k}: {len(our)} valid != {min(k, L)}"
    assert bad <= max_permit_error, f"{bad=} > {max_permit_error}"


def _make_page_table(batch, num_pages, mode, device, per_row=False):
    if mode == "identity":
        pt = torch.arange(num_pages, dtype=torch.int32, device=device)
        full = pt.unsqueeze(0).expand(batch, -1).contiguous()
        inv = pt.unsqueeze(0).expand(batch, -1).cpu()
        return full, inv
    # permutation (optionally a distinct permutation per row)
    rows = batch if per_row else 1
    full = torch.stack(
        [torch.randperm(num_pages, device=device) for _ in range(rows)]
    ).to(torch.int32)
    inv = torch.empty_like(full)
    ar = torch.arange(num_pages, dtype=torch.int32, device=device)
    for r in range(rows):
        inv[r, full[r].long()] = ar
    if not per_row:
        full = full.expand(batch, -1).contiguous()
        inv = inv.expand(batch, -1)
    return full, inv.cpu()


def _invert(out_row, inv_row):
    """Undo page_to_indices for one row's page indices (drop -1 padding)."""
    return [
        (int(inv_row[v >> PAGE_BITS]) << PAGE_BITS) | (v & PAGE_MASK)
        for v in out_row
        if v != -1
    ]


def _reference(scores, seq_lens, k):
    """torch.topk reference indices per row (trivial rows -> all positions)."""
    ref = []
    for i in range(scores.shape[0]):
        L = int(seq_lens[i])
        if L <= k:
            ref.append(list(range(L)))
        else:
            ref.append(
                torch.topk(scores[i, :L], k, sorted=False).indices.cpu().tolist()
            )
    return ref


def _run(scores, seq_lens, page_table, inv_cpu, k):
    batch = scores.shape[0]
    out = torch.full((batch, k), -1, dtype=torch.int32, device=scores.device)
    metadata = plan_topk_v2(seq_lens)
    topk_transform_512_v2(scores, seq_lens, page_table, out, PAGE_SIZE, metadata)
    torch.cuda.synchronize()
    out_cpu = out.cpu().tolist()
    return [_invert(out_cpu[i], inv_cpu[i]) for i in range(batch)]


def _run_raw(scores, seq_lens, page_table, k):
    """Run the kernel and return its optional raw (pre-transform) top-k index
    output per row, dropping -1 padding -- the selected positions themselves,
    NOT the page-table transform of them."""
    batch = scores.shape[0]
    out = torch.full((batch, k), -1, dtype=torch.int32, device=scores.device)
    raw = torch.full((batch, k), -1, dtype=torch.int32, device=scores.device)
    metadata = plan_topk_v2(seq_lens)
    topk_transform_512_v2(scores, seq_lens, page_table, out, PAGE_SIZE, metadata, raw)
    torch.cuda.synchronize()
    raw_cpu = raw.cpu().tolist()
    return [[v for v in raw_cpu[i] if v != -1] for i in range(batch)]


def _single_bin_scores(kind, batch, width, device):
    """fp32 scores whose fp16-derived coarse bin cannot order the row."""
    if kind == "distinct":
        # One coarse bin with far more than 2048 distinct fp32 values.
        # A 12-bit coarse bin spans 16 adjacent fp16 keys; this interval stays
        # in the bin beginning at 1.0 while retaining thousands of fp32 keys.
        return (
            1.0 + torch.rand(batch, width, dtype=torch.float32, device=device) * 0.001
        )
    if kind == "tiny":
        # Every value underflows when cast to fp16, while fp32 remains ordered.
        return (
            0.5 + torch.rand(batch, width, dtype=torch.float32, device=device)
        ) * 1e-30
    if kind == "ties":
        hi = torch.tensor(1.0, dtype=torch.float32, device=device).nextafter(
            torch.tensor(2.0, dtype=torch.float32, device=device)
        )
        return torch.where(
            torch.rand(batch, width, device=device) < 0.5,
            hi,
            torch.ones((), dtype=torch.float32, device=device),
        )
    raise ValueError(kind)


STRICT_OVERFLOW_CONFIGS = [
    # Register/streaming boundaries required by the GLM-5.3 K1 contract.
    (1, 8191),
    (1, 8192),
    (1, 8193),
    (1, 16383),
    (1, 16384),
    (1, 16385),
    (1, 65535),
    (1, 65536),
    (1, 65537),
    (1, 67036),
    # Actual verify/draft-expanded dispatch boundaries.  Cluster is deliberately
    # excluded until it has a cluster-wide exact refinement.
    (30, 67036),
    (31, 67036),
    (128, 67036),
    (129, 67036),
]


@pytest.mark.parametrize("batch,seq", STRICT_OVERFLOW_CONFIGS)
@torch.inference_mode()
def test_topk_v2_overflow_is_exact(batch: int, seq: int) -> None:
    """Overflowing coarse bins have zero non-tie selection errors.

    A padded backing allocation makes the score view non-contiguous while
    preserving the kernel's required 16-byte row stride.  Each row uses an
    independently permuted physical page map and the raw output is compared to
    an fp32 torch oracle with no historical five-error allowance.
    """
    torch.manual_seed(batch * 100003 + seq)
    device, k = "cuda", 2048
    width = (seq + 3) & ~3
    backing = _single_bin_scores("distinct", batch, width + 4, device)
    scores = backing[:, :width]
    # PyTorch treats the stride of a size-one leading dimension as irrelevant
    # to contiguity.  The padded row stride is still present and exercised; for
    # multi-row cases the view is also observably non-contiguous.
    assert scores.stride() == (width + 4, 1)
    assert scores.stride(0) % 4 == 0
    assert batch == 1 or not scores.is_contiguous()
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    num_pages = (width + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, _ = _make_page_table(batch, num_pages, "perm", device, per_row=True)

    our_raw = _run_raw(scores, seq_lens, page_table, k)
    ref_raw = _reference(scores, seq_lens, k)
    _assert_topk_close(
        scores.cpu(),
        ref_raw,
        our_raw,
        batch,
        seq_lens.cpu(),
        k,
        max_permit_error=0,
    )


@pytest.mark.parametrize("kind", ["tiny", "ties"])
@torch.inference_mode()
def test_topk_v2_overflow_ties_and_masked_rows(kind: str) -> None:
    """Strict overflow with repeated values, zero-length padding and tails."""
    torch.manual_seed(20326350 + (kind == "ties"))
    device, batch, width, k = "cuda", 6, 67036, 2048
    scores = _single_bin_scores(kind, batch, width + 4, device)[:, :width]
    seq_lens = torch.tensor(
        [0, 1, 2048, 2049, 65535, width], dtype=torch.int32, device=device
    )
    num_pages = (width + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, _ = _make_page_table(batch, num_pages, "perm", device, per_row=True)

    our_raw = _run_raw(scores, seq_lens, page_table, k)
    ref_raw = _reference(scores, seq_lens, k)
    _assert_topk_close(
        scores.cpu(),
        ref_raw,
        our_raw,
        batch,
        seq_lens.cpu(),
        k,
        max_permit_error=0,
    )


@pytest.mark.parametrize("batch", [30, 31, 128, 129])
@torch.inference_mode()
def test_topk_v2_overflow_cuda_graph_capture_replay(batch: int) -> None:
    """Captured exact dispatch re-reads all fixed-shape replay buffers.

    The production attention backend refreshes ``seq_lens`` and its prebuilt
    plan in-place before replay.  Exercise that contract together with dynamic
    scores and independently permuted physical-page mappings at the historical
    cluster batch boundaries used by target/draft-expanded rows.
    """
    torch.manual_seed(20326350 + batch)
    device, width, k = "cuda", 67036, 2048
    scores = _single_bin_scores("distinct", batch, width, device)
    seq_lens = torch.full((batch,), width, dtype=torch.int32, device=device)
    num_pages = (width + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv = _make_page_table(batch, num_pages, "perm", device, per_row=True)
    out = torch.full((batch, k), -1, dtype=torch.int32, device=device)
    raw = torch.full_like(out, -1)
    metadata = plan_topk_v2(seq_lens)

    # Compile and allocate all JIT state before capture.
    topk_transform_512_v2(scores, seq_lens, page_table, out, PAGE_SIZE, metadata, raw)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        topk_transform_512_v2(
            scores, seq_lens, page_table, out, PAGE_SIZE, metadata, raw
        )

    for replay in range(2):
        torch.manual_seed(20326350 + batch * 10 + replay)
        next_scores = _single_bin_scores("distinct", batch, width, device)
        next_lens = torch.full(
            (batch,), width - replay * 4, dtype=torch.int32, device=device
        )
        # Include tail and empty DP-companion rows without changing shapes.
        next_lens[-1] = 0
        if batch > 1:
            next_lens[-2] = 65535
        next_table, next_inv = _make_page_table(
            batch, num_pages, "perm", device, per_row=True
        )
        next_plan = plan_topk_v2(next_lens)
        scores.copy_(next_scores)
        seq_lens.copy_(next_lens)
        page_table.copy_(next_table)
        metadata.copy_(next_plan)
        out.fill_(-1)
        raw.fill_(-1)

        graph.replay()
        torch.cuda.synchronize()

        raw_cpu = raw.cpu().tolist()
        our_raw = [[v for v in row if v != -1] for row in raw_cpu]
        ref_raw = _reference(next_scores, next_lens, k)
        _assert_topk_close(
            next_scores.cpu(),
            ref_raw,
            our_raw,
            batch,
            next_lens.cpu(),
            k,
            max_permit_error=0,
        )
        out_cpu = out.cpu().tolist()
        mapped_raw = [_invert(out_cpu[i], next_inv[i]) for i in range(batch)]
        assert mapped_raw == our_raw


@pytest.mark.parametrize("page_mode", ["identity", "perm"])
@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("batch,seq", FIXED_CONFIGS)
@torch.inference_mode()
def test_topk_v2(batch: int, seq: int, k: int, page_mode: str) -> None:
    torch.manual_seed(batch * 100003 + seq * 7 + k)
    device = "cuda"
    # Pad the row stride to a multiple of 4 (16-byte vectorized load) while keeping
    # the exact seq_len -- this also exercises the scalar-tail path for odd seq.
    width = (seq + 3) & ~3
    scores = torch.randn(batch, width, dtype=torch.float32, device=device)[:, :seq]
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(batch, num_pages, page_mode, device)

    our_raw = _run(scores, seq_lens, page_table, inv_cpu, k)
    ref_raw = _reference(scores, seq_lens, k)
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, batch, seq_lens.cpu(), k)


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize(
    "batch,shape",
    [
        (20, "small_batch"),  # fused small-batch kernel (<= pool of 30)
        (64, "persistent"),  # persistent pool + main kernel
        (128, "persistent"),  # cluster batch boundary
    ],
)
@pytest.mark.parametrize("per_row_pt", [False, True])
@torch.inference_mode()
def test_topk_v2_ragged(batch: int, shape: str, k: int, per_row_pt: bool) -> None:
    """Ragged lengths spanning trivial..cluster in one launch, both dispatch shapes.

    ``per_row_pt`` gives each row a distinct page-table permutation, exercising
    the per-batch page_table indexing (batch_id stride) rather than a shared one.
    """
    torch.manual_seed(7777 + batch + k + int(per_row_pt))
    device = "cuda"
    seq = 262144
    scores = torch.randn(batch, seq, dtype=torch.float32, device=device)
    # span every path; guarantee at least one > floor row so cluster dispatch fires
    buckets = [max(1, k // 2), k, 4096, 12000, 40000, 65536, 98304, 262144]
    g = torch.Generator(device="cpu").manual_seed(batch + k)
    lengths = torch.tensor(
        [
            buckets[int(torch.randint(0, len(buckets), (1,), generator=g))]
            for _ in range(batch)
        ],
        dtype=torch.int32,
        device=device,
    )
    lengths[0] = max(1, k // 2)  # a trivial row
    lengths[1] = 262144  # a long (cluster) row
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(
        batch, num_pages, "perm", device, per_row=per_row_pt
    )

    our_raw = _run(scores, lengths, page_table, inv_cpu, k)
    ref_raw = _reference(scores, lengths, k)
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, batch, lengths.cpu(), k)


@pytest.mark.parametrize("page_mode", ["identity", "perm"])
@pytest.mark.parametrize(
    "batch,seq",
    [
        (8, 256),  # trivial
        (8, 4096),  # register
        (4, 131072),  # fused small-batch cluster
        (64, 131072),  # persistent cluster + main<3> epilogue
        (256, 131072),  # non-cluster streaming
    ],
)
@torch.inference_mode()
def test_topk_v2_raw_indices(batch: int, seq: int, page_mode: str) -> None:
    """The optional raw-index output must be the pre-transform position of each
    transformed output slot (out[j] == page_to_indices(raw[j])), and -1 aligns."""
    k = 512
    torch.manual_seed(batch * 131 + seq)
    device = "cuda"
    width = (seq + 3) & ~3
    scores = torch.randn(batch, width, dtype=torch.float32, device=device)[:, :seq]
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(batch, num_pages, page_mode, device)
    out = torch.full((batch, k), -1, dtype=torch.int32, device=device)
    raw = torch.full((batch, k), -1, dtype=torch.int32, device=device)

    metadata = plan_topk_v2(seq_lens)
    topk_transform_512_v2(scores, seq_lens, page_table, out, PAGE_SIZE, metadata, raw)
    torch.cuda.synchronize()

    out_cpu, raw_cpu = out.cpu().tolist(), raw.cpu().tolist()
    for i in range(batch):
        for j in range(k):
            o, r = out_cpu[i][j], raw_cpu[i][j]
            if o == -1:
                assert r == -1, f"b={i} j={j}: out=-1 but raw={r}"
            else:
                inv = (int(inv_cpu[i][o >> PAGE_BITS]) << PAGE_BITS) | (o & PAGE_MASK)
                assert r == inv, f"b={i} j={j}: raw={r} != inverse(out)={inv}"


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("batch,seq", FIXED_CONFIGS)
@torch.inference_mode()
def test_topk_v2_output_indices(batch: int, seq: int, k: int) -> None:
    """Validate the raw (pre-transform) index output DIRECTLY against torch.topk.

    Unlike ``test_topk_v2`` -- which checks the page-transformed output and inverts
    it through the page table -- this exercises the selected indices themselves, so
    it isolates the top-k selection from the page-table transform. A permuted page
    table is used so raw != out, catching any bug that leaks transformed page
    indices into the raw buffer. Covers every dispatch template/boundary.
    """
    torch.manual_seed(batch * 100003 + seq * 7 + k + 1)
    device = "cuda"
    width = (seq + 3) & ~3
    scores = torch.randn(batch, width, dtype=torch.float32, device=device)[:, :seq]
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, _ = _make_page_table(batch, num_pages, "perm", device)

    our_raw = _run_raw(scores, seq_lens, page_table, k)
    ref_raw = _reference(scores, seq_lens, k)
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, batch, seq_lens.cpu(), k)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
