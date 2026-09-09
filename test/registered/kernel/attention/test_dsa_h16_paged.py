"""H16 prefill integration: real paged writer, scale bytes, relocation and replay."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not is_sm90_supported(), reason="requires SM90")


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("pool_size", [128, 32768])
@pytest.mark.parametrize("phase", ["prefill", "decode", "verify"])
def test_nope_real_pool_write_move_and_graph(fp8, pool_size, phase):
    from sglang.kernels.ops.attention.dsa.dequant_k_cache import dequantize_k_cache
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    reset_context()
    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    try:
        pool = MLATokenToKVPool(
            size=pool_size,
            page_size=1,
            dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
            use_dsa=True,
            override_kv_cache_dim=528 if fp8 else None,
        )
        heads = 16 if phase == "prefill" else 64
        layer = SimpleNamespace(
            layer_id=0,
            is_cross_attention=False,
            tp_q_head_num=heads,
            v_head_dim=512,
            head_dim=512,
            scaling=512**-0.5,
        )
        generator = torch.Generator(device="cuda").manual_seed(853)
        keys = torch.randn(32, 1, 512, device="cuda", generator=generator)
        keys.view(32, 4, 128).mul_(
            torch.tensor([0.1, 0.5, 1, 2], device="cuda")[None, :, None]
        )
        keys = keys.bfloat16()
        locs = torch.randperm(100, device="cuda", generator=generator)[:32] + 1
        pool.set_mla_kv_buffer(layer, locs, keys, None)
        q = torch.randn(4, heads, 512, device="cuda", generator=generator).bfloat16()
        # Prefix slots are non-contiguous, and live tails follow masked holes.
        indices = torch.full((4, 2051), -1, device="cuda", dtype=torch.int32)
        indices[1:, :10] = locs[:10].int()
        indices[2:, 2048:] = locs[10:13].int()
        indices[1, 20] = pool_size + 1
        pool.kv_buffer[0][0].fill_(255 if fp8 else float("nan"))

        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.dsa_prefill_impl = "cutedsl_h16"
        backend.dsa_decode_impl = "tilelang"
        backend.dsa_index_kpool = 4
        backend.dsa_kv_cache_store_fp8 = fp8
        backend.use_mha = False
        backend.use_fused_topk = True
        backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
        backend.token_to_kv_pool = pool
        backend.hisparse_coordinator = None
        backend.forward_metadata = SimpleNamespace()
        mode = {
            "prefill": ForwardMode.EXTEND,
            "decode": ForwardMode.DECODE,
            "verify": ForwardMode.TARGET_VERIFY,
        }[phase]
        batch = SimpleNamespace(forward_mode=mode)

        def run():
            forward = (
                backend.forward_decode if phase == "decode" else backend.forward_extend
            )
            return forward(
                q,
                None,
                None,
                layer,
                batch,
                save_kv_cache=False,
                topk_indices=indices,
            )

        def check(actual):
            actual = actual.view_as(q)
            cache = pool.get_key_buffer(0)
            decoded = dequantize_k_cache(cache) if fp8 else cache
            ids = indices.long()
            valid = (ids >= 0) & (ids < cache.shape[0])
            selected = decoded[ids.clamp(1, pool_size), 0].float()
            selected = torch.where(valid[:, :, None], selected, 0)
            scores = torch.einsum("qhd,qkd->qhk", q.float(), selected) * layer.scaling
            probs = (
                scores.masked_fill(~valid[:, None], -torch.inf)
                .softmax(-1)
                .nan_to_num(0)
            )
            expected = torch.einsum("qhk,qkd->qhd", probs, selected)
            assert actual.isfinite().all()
            assert (actual[0] == 0).all()
            torch.testing.assert_close(actual.float(), expected, atol=0.03, rtol=0.02)
            assert (actual.float() - expected).norm() / expected.norm() < 0.008

        check(run())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        # Accepted-state relocation must preserve the four scale words too.
        destination = torch.tensor([110, 111, 112], device="cuda", dtype=torch.int64)
        pool.move_kv_cache(destination, locs[10:13])
        indices[2:, 2048:] = destination.int()
        pool.set_mla_kv_buffer(layer, locs[:1], keys[:1] * 0.25, None)
        graph.replay()
        check(output)
    finally:
        reset_context()
