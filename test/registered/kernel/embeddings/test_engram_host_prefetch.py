"""Shared-host Engram lookup stays correct when issued on a side stream."""

import os
import socket
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

ROWS = 2048
DIM = 256
BLK = 32
N_HASH_COLS = 24


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _checkpoint_table(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    weight = (torch.randn(ROWS, DIM, generator=generator) * 3).to(torch.float8_e4m3fn)
    scale = torch.randint(
        100,
        140,
        (ROWS, DIM // BLK),
        dtype=torch.uint8,
        generator=generator,
    ).view(torch.float8_e8m0fnu)
    weight.view(torch.uint8)[0, :BLK] = (
        torch.tensor(1.0).to(torch.float8_e4m3fn).view(torch.uint8)
    )
    scale.view(torch.uint8)[0, 0] = 0
    return weight, scale


def _reference(weight, scale, ids):
    ids = ids.cpu()
    rows = weight[ids].float().unflatten(-1, (-1, BLK))
    return (rows * scale[ids].float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


class _TestProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(
            N_HASH_COLS * DIM, 3 * 32, bias=False, device="cuda", dtype=torch.bfloat16
        )
        self.linear.requires_grad_(False)

    def forward(self, x):
        return self.linear(x), None


class TestEngramHostPrefetch(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.distributed_initialized = False
        cls.model_parallel_initialized = False
        torch.cuda.set_device(0)
        port = _free_port()
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="nccl",
        )
        cls.distributed_initialized = True
        initialize_model_parallel(tensor_model_parallel_size=1, backend="nccl")
        cls.model_parallel_initialized = True

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "model_parallel_initialized", False):
            destroy_model_parallel()
        if getattr(cls, "distributed_initialized", False):
            destroy_distributed_environment()
        super().tearDownClass()

    @contextmanager
    def _embedding(self, layer_id=14, seed=0):
        from sglang.srt.layers.engram import EngramEmbedding

        weight, scale = _checkpoint_table(seed)
        embed = None
        try:
            with (
                envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(True),
                envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.override("shared"),
                get_parallel().override(tp_size=1, tp_rank=0),
                patch("sglang.srt.layers.engram.get_attention_dp_size", return_value=8),
            ):
                embed = EngramEmbedding(ROWS, DIM, layer_id=layer_id)
                embed.weight.weight_loader(embed.weight, weight)
                embed.scale.weight_loader(embed.scale, scale)
                embed.finish_load("test")
                yield embed, weight, scale
        finally:
            torch.cuda.synchronize()
            host_table = None if embed is None else embed.host_table
            if host_table is not None and host_table.registered:
                rc = torch.cuda.cudart().cudaHostUnregister(host_table.bytes.data_ptr())
                self.assertEqual(int(rc), 0)
                host_table.registered = False

    def test_side_stream_lookup_matches_sync_with_dp_attention(self):
        with self._embedding() as (embed, weight, scale):
            prefetch_stream = torch.cuda.Stream()
            ready = torch.cuda.Event()
            empty = embed(
                torch.empty((0, N_HASH_COLS), dtype=torch.int64, device="cuda")
            )
            self.assertEqual(empty.shape, (0, N_HASH_COLS, DIM))
            for seed, tokens in ((1, 1), (2, 7), (3, 32)):
                generator = torch.Generator(device="cuda").manual_seed(seed)
                ids = torch.randint(
                    ROWS,
                    (tokens, N_HASH_COLS),
                    dtype=torch.int64,
                    device="cuda",
                    generator=generator,
                )
                ids[0].zero_()
                ids[-1].fill_(ROWS - 1)
                if tokens > 2:
                    ids[1].copy_(ids[0])
                expected = _reference(weight, scale, ids)
                synchronous = embed(ids)

                main_stream = torch.cuda.current_stream()
                prefetched = embed.prefetch(ids, prefetch_stream, ready)
                main_stream.wait_event(ready)

                torch.cuda.synchronize()
                self.assertTrue(torch.equal(synchronous.cpu(), expected))
                self.assertTrue(torch.equal(prefetched.cpu(), expected))

    @staticmethod
    def _hasher(device):
        from sglang.srt.layers.engram import EngramHasher

        # Keep the real hash/commit path, but use a tiny synthetic vocabulary.
        hasher = EngramHasher.__new__(EngramHasher)
        torch.nn.Module.__init__(hasher)
        hasher.max_ngram_size = 4
        hasher.pad_id = 0
        hasher.image_token_id = None
        hasher.token_map = torch.arange(64, device=device)
        hasher.multipliers = torch.tensor(
            [[3, 5, 7, 11], [13, 17, 19, 23]], device=device
        )
        hasher.primes = torch.full((2, 3, 8), 61, device=device, dtype=torch.int64)
        hasher.offsets = (torch.arange(N_HASH_COLS, device=device) * 61).repeat(2, 1)
        hasher.init_history(40, device)
        return hasher

    def test_decode_graph_replay_live_inputs_history_and_shared_pool(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        with (
            self._embedding(layer_id=1, seed=1) as (embed1, weight1, scale1),
            self._embedding() as (embed14, weight14, scale14),
        ):
            hasher, oracle = self._hasher("cuda"), self._hasher("cpu")
            prefetch_stream = torch.cuda.Stream()
            ready1, ready14 = torch.cuda.Event(), torch.cuda.Event()
            capture_stream = torch.cuda.Stream()
            pool = torch.cuda.graph_pool_handle()
            buckets = {}
            # Largest first, matching the decode runner's shared-pool capture.
            for tokens in (32, 7, 1):
                ids = torch.zeros(tokens, dtype=torch.int64, device="cuda")
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.DECODE,
                    req_pool_indices=torch.arange(tokens, device="cuda"),
                    positions=torch.full_like(ids, 4),
                    out_cache_loc=torch.ones_like(ids),
                )

                def forward():
                    hashes = hasher(ids, batch)
                    values1 = embed1.prefetch(hashes[:, 0], prefetch_stream, ready1)
                    values14 = embed14.prefetch(hashes[:, 1], prefetch_stream, ready14)
                    # Separate joins: L1's consumer precedes L14's join. Both
                    # buffers must survive shared-pool captures and replays.
                    residual = ids.to(torch.float32)[:, None, None]
                    torch.cuda.current_stream().wait_event(ready1)
                    result1 = values1.float() + residual
                    torch.cuda.current_stream().wait_event(ready14)
                    return result1, values14.float() + residual

                capture_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(capture_stream):
                    for _ in range(3):
                        forward()
                torch.cuda.current_stream().wait_stream(capture_stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=capture_stream):
                    result = forward()
                buckets[tokens] = graph, ids, batch, result

            # Alternate buckets and include all-padding replays. Reused request
            # slots and changing histories must be read at replay, not capture.
            for step, tokens in enumerate((1, 32, 7, 1, 7, 32) * 3):
                graph, ids, batch, result = buckets[tokens]
                live = 0 if step % 4 == 0 else max(1, tokens - 2)
                cpu_ids = (torch.arange(tokens) + step * 3) % 64
                slots = (torch.arange(tokens) + step) % 40
                # Padding deliberately aliases a live slot.
                slots[live:] = slots[0]
                locations = (torch.arange(tokens) < live).to(torch.int64)
                positions = torch.full((tokens,), step % 7, dtype=torch.int64)
                history = ((torch.arange(123).reshape(41, 3) + step) % 64).to(
                    torch.int32
                )
                oracle.history.copy_(history)
                hasher.history.copy_(history)
                ids.copy_(cpu_ids)
                batch.req_pool_indices.copy_(slots)
                batch.positions.copy_(positions)
                batch.out_cache_loc.copy_(locations)
                hashes = oracle(
                    cpu_ids,
                    SimpleNamespace(
                        forward_mode=ForwardMode.DECODE,
                        req_pool_indices=slots,
                        positions=positions,
                        out_cache_loc=locations,
                    ),
                )
                graph.replay()
                for index, (weight, scale) in enumerate(
                    ((weight1, scale1), (weight14, scale14))
                ):
                    expected = (
                        _reference(weight, scale, hashes[:, index]).float()
                        + cpu_ids[:, None, None]
                    )
                    torch.testing.assert_close(
                        result[index].cpu(), expected, rtol=0, atol=0
                    )
                torch.testing.assert_close(
                    hasher.history.cpu()[:40], oracle.history[:40], rtol=0, atol=0
                )

    def test_eager_chunked_extend_history_and_tail_rows(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        with self._embedding() as (embed, weight, scale):
            hasher, oracle = self._hasher("cuda"), self._hasher("cpu")
            stream, ready = torch.cuda.Stream(), torch.cuda.Event()
            for chunk in range(3):
                lens = torch.tensor([5 + chunk, 3])
                starts = torch.tensor([0, lens[0]])
                ids = (torch.arange(lens.sum()) + chunk * 11) % 64
                positions = torch.cat(
                    (torch.arange(lens[0]) + 10 + chunk, torch.arange(lens[1]) + chunk)
                )
                # Simulate scheduler predecessors after a prefix-cache hit.
                history = (
                    None
                    if chunk
                    else torch.tensor([[3, 4, 5], [6, 7, 8]], dtype=torch.int32)
                )
                batch = dict(
                    forward_mode=ForwardMode.EXTEND,
                    req_pool_indices=torch.tensor([3, 1]),
                    positions=positions,
                    extend_seq_lens=lens,
                    extend_start_loc=starts,
                    ngram_history=history,
                    out_cache_loc=torch.ones_like(ids),
                )
                expected_hashes = oracle(ids, SimpleNamespace(**batch))
                gpu_batch = SimpleNamespace(
                    **{
                        k: v.cuda() if torch.is_tensor(v) else v
                        for k, v in batch.items()
                    }
                )
                hashes = hasher(ids.cuda(), gpu_batch)
                torch.testing.assert_close(
                    hashes.cpu(), expected_hashes, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    hasher.history.cpu(), oracle.history, rtol=0, atol=0
                )
                for rows in (
                    torch.arange(ids.numel()),
                    torch.tensor([lens[0] - 1, ids.numel() - 1]),
                ):
                    # Noncontiguous, per-request bounded-replay tails.
                    values = embed.prefetch(hashes[rows.cuda(), 1], stream, ready)
                    torch.cuda.current_stream().wait_event(ready)
                    expected = _reference(weight, scale, expected_hashes[rows, 1])
                    torch.testing.assert_close(values.cpu(), expected, rtol=0, atol=0)

    def test_model_layer_loop_graph_and_extend_match_sync(self):
        import sglang.srt.models.deepseek_v4 as model_module
        from sglang.srt.layers.engram import Engram
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            model_capture_mode,
        )

        with (
            self._embedding(layer_id=1, seed=1) as (embed1, _, _),
            self._embedding() as (embed14, _, _),
        ):
            # Exercise the production model loop, hash, lookup, WKV dispatch,
            # gate and image mask. Only attention/FFN blocks are replaced.
            engrams = {}
            for index, (layer_id, embed) in enumerate(((1, embed1), (14, embed14))):
                engram = Engram.__new__(Engram)
                torch.nn.Module.__init__(engram)
                engram.embed, engram.wkv = embed, _TestProjection()
                engram.layer_hash_index = index
                engram.eps, engram.clamp_value = 1e-6, 1e-6
                engram.q_weight = engram.k_weight = torch.ones((2, 32), device="cuda")
                engrams[layer_id] = engram
            model = model_module.DeepseekV4Model.__new__(model_module.DeepseekV4Model)
            torch.nn.Module.__init__(model)
            model.config = SimpleNamespace(
                model_type="deepseek_v41", vision_n_layers=2, image_token_id=7
            )
            model.pp_group = SimpleNamespace(world_size=1)
            model.start_layer, model.end_layer = 0, 15
            model.layers = [
                SimpleNamespace(
                    engram=engrams.get(i),
                    forward_hc_pre_from_prev=lambda **kw: (
                        kw["hidden_states"] + 0.125,
                        None,
                    ),
                )
                for i in range(15)
            ]
            model.engram_hasher = self._hasher("cuda")
            model.engram_hasher.image_token_id = 7
            model.engram_embed_prefetch_stream = stream = torch.cuda.Stream()
            model.engram_embed_prefetch_events = {
                i: torch.cuda.Event() for i in (1, 14)
            }
            model.engram_embed_prefetch_max_bytes = 128 * 1024 * 1024
            model.late_layer_start = None
            tokens = 7
            ids = torch.arange(tokens, device="cuda", dtype=torch.int64)
            hidden = torch.randn((tokens, 2, 32), device="cuda", dtype=torch.bfloat16)
            batch = SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                req_pool_indices=torch.arange(tokens, device="cuda"),
                positions=torch.full_like(ids, 4),
                out_cache_loc=torch.ones_like(ids),
            )

            def forward():
                return model._forward_layers_hc_pre_from_prev(
                    batch.positions,
                    hidden,
                    batch,
                    ids,
                    ids,
                    False,
                    [],
                )[0]

            with (
                patch.object(model_module, "is_cp_active", return_value=False),
                patch.object(
                    model_module, "check_cuda_graph_backend", return_value=True
                ),
                patch.object(
                    model_module,
                    "get_platform",
                    return_value=SimpleNamespace(is_blackwell=False),
                ),
            ):
                capture_stream = torch.cuda.Stream()
                capture_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(capture_stream):
                    for _ in range(3):
                        forward()
                torch.cuda.current_stream().wait_stream(capture_stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with (
                    model_capture_mode(),
                    torch.cuda.graph(graph, stream=capture_stream),
                ):
                    result = forward()
                for step in range(6):
                    ids.copy_((torch.arange(tokens, device="cuda") + step) % 64)
                    ids[0] = 7
                    model.engram_hasher.history.zero_()
                    model.engram_embed_prefetch_stream = None
                    expected = forward()
                    expected_history = model.engram_hasher.history.clone()
                    model.engram_hasher.history.zero_()
                    model.engram_embed_prefetch_stream = stream
                    graph.replay()
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    torch.testing.assert_close(
                        model.engram_hasher.history, expected_history, rtol=0, atol=0
                    )

                batch.forward_mode = ForwardMode.EXTEND
                batch.req_pool_indices = torch.tensor([1, 3], device="cuda")
                batch.extend_seq_lens = torch.tensor([4, 3], device="cuda")
                batch.extend_start_loc = torch.tensor([0, 4], device="cuda")
                batch.ngram_history = torch.tensor(
                    [[2, 3, 4], [5, 6, 7]], device="cuda", dtype=torch.int32
                )
                expected = forward()
                model.engram_embed_prefetch_stream = None
                actual = forward()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
