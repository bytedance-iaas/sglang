"""Bitwise old/new KDA fold, including tracking and padded/zero-accept rows."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip("KDA fold prefetch requires CUDA", allow_module_level=True)

from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (
    commit_kda_replayssm_spec,
    commit_kda_replayssm_spec_all_layers,
)
from sglang.srt.environ import envs


@pytest.mark.parametrize(
    "layers,hv,h,k,v", [(3, 8, 2, 128, 128), (2, 4, 4, 96, 80), (2, 64, 64, 128, 128)]
)
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("track", [False, True])
def test_prefetch_bitwise(layers, hv, h, k, v, batched, track):
    torch.manual_seed(44024)
    slots, ring = 10, 16
    state = torch.randn(layers, slots, hv, v, k, device="cuda")
    rawv = torch.randn(layers, slots, hv, ring, v, device="cuda", dtype=torch.bfloat16)
    rawk = torch.randn(layers, slots, h, ring, k, device="cuda", dtype=torch.bfloat16)
    gk = -torch.rand(layers, slots, hv, ring, k, device="cuda")
    beta = torch.rand(layers, slots, hv, ring, device="cuda")
    indices = torch.tensor([1, 2, 3, -1], dtype=torch.int32, device="cuda")
    accepted = torch.tensor([0, 1, 6, 6], dtype=torch.int32, device="cuda")
    kwargs = dict(
        ssm_state_indices=indices,
        accept_lens=accepted,
        max_cache_len=ring,
        num_k_heads=h,
    )
    if track:
        kwargs.update(
            mamba_track_indices=torch.tensor(
                [6, 7, 8, -1], dtype=torch.int32, device="cuda"
            ),
            mamba_steps_to_track=torch.tensor(
                [-1, 0, 3, -1], dtype=torch.int32, device="cuda"
            ),
        )

    def run(flag):
        out = state.clone()
        with envs.SGLANG_EXPERIMENTAL_KDA_FOLD_PREFETCH.override(flag):
            if batched:
                commit_kda_replayssm_spec_all_layers(
                    out, rawv, rawk, gk, beta, **kwargs
                )
            else:
                for layer in range(layers):
                    commit_kda_replayssm_spec(
                        out[layer],
                        rawv[layer],
                        rawk[layer],
                        gk[layer],
                        beta[layer],
                        **kwargs,
                    )
        return out

    off, on = run(False), run(True)
    torch.cuda.synchronize()
    assert torch.equal(off, on)
    assert torch.equal(on[:, 1], state[:, 1])  # zero accepted prefix
    assert torch.equal(on[:, 0], state[:, 0])  # padding sentinel is never committed


def test_prefetch_graph_replay_fresh_accept_lengths():
    torch.manual_seed(53)
    state = torch.randn(2, 7, 64, 128, 128, device="cuda")
    rawv = torch.randn(2, 7, 64, 16, 128, device="cuda", dtype=torch.bfloat16)
    rawk = torch.randn_like(rawv)
    gk = -torch.rand(2, 7, 64, 16, 128, device="cuda")
    beta = torch.rand(2, 7, 64, 16, device="cuda")
    indices = torch.tensor([1, 2, -1], dtype=torch.int32, device="cuda")
    accepted = torch.tensor([6, 3, 0], dtype=torch.int32, device="cuda")
    states = [state.clone(), state.clone()]
    graphs = []
    kwargs = dict(
        rawv_cache=rawv,
        rawk_cache=rawk,
        gk_cache=gk,
        beta_cache=beta,
        ssm_state_indices=indices,
        accept_lens=accepted,
        max_cache_len=16,
        num_k_heads=64,
    )
    for flag, out in zip((False, True), states):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with envs.SGLANG_EXPERIMENTAL_KDA_FOLD_PREFETCH.override(flag):
            with torch.cuda.stream(stream):
                for _ in range(3):
                    commit_kda_replayssm_spec_all_layers(out, **kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                commit_kda_replayssm_spec_all_layers(out, **kwargs)
        graphs.append(graph)
    for accept, slots in (([1, 6, 0], [1, 2, -1]), ([0, 2, 4], [2, 3, 4])):
        accepted.copy_(torch.tensor(accept, dtype=torch.int32, device="cuda"))
        indices.copy_(torch.tensor(slots, dtype=torch.int32, device="cuda"))
        for out, graph in zip(states, graphs):
            out.copy_(state)
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(states[0], states[1])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
