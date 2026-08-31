"""Hopper execution tests for GLM-5.2 raw-FP8 TileLang sparse attention."""

import pytest
import torch

from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")

tilelang_kernel = pytest.importorskip(
    "sglang.kernels.ops.attention.dsa.tilelang_kernel"
)

requires_fp8_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="needs CUDA SM89+ for FP8 tensor-core MMA",
)

S, H, DV, ROPE, TOPK, POOL = 4, 32, 512, 64, 2048, 32768
SM_SCALE = (DV + ROPE) ** -0.5


def _relative_max_error(actual, expected):
    return (
        (actual.float() - expected.float()).abs().max() / expected.float().abs().max()
    ).item()


def _make_inputs(seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = (
        torch.randn(
            S, H, DV + ROPE, device="cuda", dtype=torch.float32, generator=generator
        )
        * 0.5
    ).to(torch.bfloat16)
    kv = (
        torch.randn(
            POOL,
            1,
            DV + ROPE,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.5
    ).to(torch.float8_e4m3fn)
    indices = torch.randint(
        1, POOL, (S, 1, TOPK), device="cuda", generator=generator
    ).to(torch.int32)
    indices[:, :, -37:] = -1
    return q, kv, indices


def _reference(q, kv, indices):
    output = torch.empty(S, H, DV, device="cuda", dtype=torch.float32)
    kv_float = kv.float().squeeze(1)
    q_float = q.to(torch.float8_e4m3fn).float()
    for row in range(S):
        selected = indices[row, 0].long()
        valid = selected >= 0
        gathered = kv_float[selected.clamp(min=0)]
        logits = (q_float[row] @ gathered.T) * SM_SCALE
        logits[:, ~valid] = float("-inf")
        probabilities = torch.softmax(logits, dim=-1)
        output[row] = probabilities @ gathered[:, :DV]
    return output


def test_cuda_sm_count_cache_is_keyed_by_device(monkeypatch):
    calls = []

    class DeviceProperties:
        def __init__(self, count):
            self.multi_processor_count = count

    def get_device_properties(device_index):
        calls.append(device_index)
        return DeviceProperties(100 + device_index)

    tilelang_kernel._cuda_sm_count.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)

    try:
        assert tilelang_kernel._cuda_sm_count(0) == 100
        assert tilelang_kernel._cuda_sm_count(1) == 101
        assert tilelang_kernel._cuda_sm_count(0) == 100
        assert calls == [0, 1]
    finally:
        # Do not leak mocked device properties into subsequent CUDA tests.
        tilelang_kernel._cuda_sm_count.cache_clear()


@requires_fp8_cuda
def test_glm52_one_hot_gather_is_exact():
    q, kv, _ = _make_inputs(0)
    indices = torch.full((S, 1, TOPK), -1, device="cuda", dtype=torch.int32)
    selected = torch.randint(1, POOL, (S,), device="cuda")
    indices[:, 0, 0] = selected.to(torch.int32)
    output = tilelang_kernel.tilelang_sparse_fwd(
        q, kv, indices, SM_SCALE, d_v=DV
    ).reshape(S, H, DV)
    expected = (
        kv.float().squeeze(1)[selected.long(), :DV].unsqueeze(1).expand_as(output)
    )
    assert _relative_max_error(output, expected) < 1e-3


@requires_fp8_cuda
def test_glm52_spread_matches_quantized_reference():
    q, kv, indices = _make_inputs(1)
    output = tilelang_kernel.tilelang_sparse_fwd(
        q, kv, indices, SM_SCALE, d_v=DV
    ).reshape(S, H, DV)
    reference = _reference(q, kv, indices)
    assert _relative_max_error(output, reference) < 0.04

    bad_indices = indices.clone()
    bad_indices[:, :, : TOPK // 2] = indices[:, :, TOPK // 2 :].flip(-1)
    bad_output = tilelang_kernel.tilelang_sparse_fwd(
        q, kv, bad_indices, SM_SCALE, d_v=DV
    ).reshape(S, H, DV)
    assert _relative_max_error(bad_output, reference) > 0.04


@requires_fp8_cuda
def test_glm52_inner_iter_two_matches_quantized_reference(monkeypatch):
    q, kv, indices = _make_inputs(2)
    monkeypatch.setattr(tilelang_kernel, "_pick_inner_iter", lambda *args: 2)

    output = tilelang_kernel.tilelang_sparse_fwd(
        q, kv, indices, SM_SCALE, d_v=DV
    ).reshape(S, H, DV)

    assert _relative_max_error(output, _reference(q, kv, indices)) < 0.04


@requires_fp8_cuda
def test_glm52_cuda_graph_capture_replays_q_cast_and_partial_combine():
    q, kv, indices = _make_inputs(3)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(2):
            tilelang_kernel.tilelang_sparse_fwd(q, kv, indices, SM_SCALE, d_v=DV)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = tilelang_kernel.tilelang_sparse_fwd(
            q, kv, indices, SM_SCALE, d_v=DV
        )

    next_q, _, next_indices = _make_inputs(4)
    q.copy_(next_q)
    indices.copy_(next_indices)
    graph.replay()
    torch.cuda.synchronize()

    assert (
        _relative_max_error(graph_output.reshape(S, H, DV), _reference(q, kv, indices))
        < 0.04
    )


@requires_fp8_cuda
def test_glm52_fp8_kv_writer_to_tilelang_attention_round_trip():
    generator = torch.Generator(device="cuda").manual_seed(5)
    q = torch.randn(
        S, H, DV + ROPE, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    cache_k_nope = torch.randn(
        TOPK, 1, DV, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    cache_k_rope = torch.randn(
        TOPK, 1, ROPE, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    raw_pool = MLATokenToKVPool.__new__(MLATokenToKVPool)
    raw_pool.use_dsa = True
    raw_pool.dtype = torch.float8_e4m3fn
    raw_pool.dsa_kv_cache_store_fp8 = False
    raw_pool.kv_lora_rank = DV
    raw_pool.qk_rope_head_dim = ROPE
    raw_pool.kv_cache_dim = DV + ROPE
    dst_buffer = torch.zeros(POOL, 1, DV + ROPE, device="cuda", dtype=torch.uint8)
    loc = torch.arange(1, TOPK + 1, device="cuda", dtype=torch.int64)
    raw_pool._write_mla_kv_buffer(dst_buffer, loc, cache_k_nope, cache_k_rope)
    kv = dst_buffer.view(torch.float8_e4m3fn)
    indices = loc.to(torch.int32).view(1, 1, TOPK).expand(S, -1, -1).clone()

    output = tilelang_kernel.tilelang_sparse_fwd(
        q, kv, indices, SM_SCALE, d_v=DV
    ).reshape(S, H, DV)

    assert _relative_max_error(output, _reference(q, kv, indices)) < 0.04
