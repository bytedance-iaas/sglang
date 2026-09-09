"""The NoPE group-scale cache retains its four scales across full/paged reads."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("dimension", [512, 528])
def test_dsa_physical_layout_registration(dimension):
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    pool = DSATokenToKVPool(
        size=128,
        page_size=64,
        kv_lora_rank=512,
        dtype=torch.float8_e4m3fn,
        qk_rope_head_dim=0,
        layer_num=2,
        device="cuda",
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=dimension,
    )
    ptrs, sizes, page_bytes = pool.get_contiguous_buf_infos()
    assert len(ptrs) == 2
    assert sizes == [192 * dimension] * 2
    assert page_bytes == [64 * dimension] * 2
    assert pool.dsa_kv_cache_store_fp8 == (dimension == 528)


@pytest.mark.parametrize("rope_dim", [0, 64])
def test_dsa_group_cache_full_paged_and_graph(rope_dim):
    from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
        dequantize_k_cache,
        dequantize_k_cache_paged,
    )
    from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache_separate

    generator = torch.Generator(device="cuda").manual_seed(528 + rope_dim)
    nope = torch.randn(18, 1, 512, device="cuda", generator=generator)
    nope.view(18, 4, 128).mul_(
        torch.tensor([0.01, 0.5, 2.0, 8.0], device="cuda")[None, :, None]
    )
    nope = nope.bfloat16()
    rope = torch.randn(18, 1, rope_dim, device="cuda", generator=generator).bfloat16()
    packed_nope, packed_rope = quantize_k_cache_separate(nope, rope)
    cache = torch.cat((packed_nope, packed_rope), -1).view(torch.float8_e4m3fn)
    values = packed_nope[:, :, :512].view(torch.float8_e4m3fn).float().view(18, 4, 128)
    scales = packed_nope[:, :, 512:].view(torch.float32).view(18, 4, 1)
    expected = torch.cat(((values * scales).bfloat16().view(18, 1, 512), rope), -1)
    actual = dequantize_k_cache(cache)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    blocked = dequantize_k_cache(cache.view(2, 9, 1, -1))
    torch.testing.assert_close(blocked.view_as(expected), expected, rtol=0, atol=0)

    pages = torch.tensor([17, 0, 3, 3, 8], device="cuda", dtype=torch.int32)
    gathered = dequantize_k_cache_paged(cache, pages)
    torch.testing.assert_close(gathered, expected[pages.long()], rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            dequantize_k_cache_paged(cache, pages)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = dequantize_k_cache_paged(cache, pages)
    pages[0] = 7
    graph.replay()
    torch.testing.assert_close(replayed, expected[pages.long()], rtol=0, atol=0)
