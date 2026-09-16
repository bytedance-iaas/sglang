"""Shared-host Engram lookup stays correct when issued on a side stream."""

import os
import socket
import unittest
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

    def test_side_stream_lookup_matches_sync_with_dp_attention(self):
        from sglang.srt.layers.engram import EngramEmbedding

        weight, scale = _checkpoint_table()
        embed = None
        try:
            with (
                envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(True),
                envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.override("shared"),
                envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN.override(True),
                get_parallel().override(tp_size=1, tp_rank=0),
                patch("sglang.srt.layers.engram.get_attention_dp_size", return_value=8),
            ):
                embed = EngramEmbedding(ROWS, DIM, layer_id=14)
                embed.weight.weight_loader(embed.weight, weight)
                embed.scale.weight_loader(embed.scale, scale)
                embed.finish_load("test")

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
                    prefetch_stream.wait_stream(main_stream)
                    with torch.cuda.stream(prefetch_stream):
                        prefetched = embed(ids)
                        ready.record(prefetch_stream)
                    ids.record_stream(prefetch_stream)
                    main_stream.wait_event(ready)
                    prefetched.record_stream(main_stream)

                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(synchronous.cpu(), expected))
                    self.assertTrue(torch.equal(prefetched.cpu(), expected))
        finally:
            host_table = None if embed is None else embed.host_table
            if host_table is not None and host_table.registered:
                rc = torch.cuda.cudart().cudaHostUnregister(host_table.bytes.data_ptr())
                self.assertEqual(int(rc), 0)
                host_table.registered = False


if __name__ == "__main__":
    unittest.main()
