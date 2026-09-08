"""KDA projection fusion follows attention TP and actual per-prefix precision."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NAMES = ("qkv_proj", "f_a_proj", "f_b_proj", "b_proj", "g_a_proj", "g_b_proj")
PREFIX = "model.layers.0.self_attn"


def _ignored_prefixes(names):
    return [
        f"{PREFIX}.{shard}"
        for name in names
        for shard in (("q_proj", "k_proj", "v_proj") if name == "qkv_proj" else (name,))
    ]


@pytest.mark.parametrize("omitted", [None, *NAMES])
def test_precision_gate_checks_every_projection(omitted):
    from sglang.srt.layers.quantization.fp8 import Fp8Config
    from sglang.srt.layers.quantization.utils import are_linear_prefixes_unquantized

    reset_context()
    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    try:
        prefixes = [f"{PREFIX}.{name}" for name in NAMES]
        config = Fp8Config(
            ignored_layers=_ignored_prefixes(n for n in NAMES if n != omitted)
        )
        assert are_linear_prefixes_unquantized(config, prefixes) == (omitted is None)
    finally:
        reset_context()


@pytest.mark.parametrize(
    "tp_size,attn_size,attn_rank", [(4, 4, 0), (4, 4, 3), (8, 1, 0)]
)
@pytest.mark.parametrize("expert_quantized", [False, True])
def test_glm_kda_projection_loading_and_outputs(
    monkeypatch, tp_size, attn_size, attn_rank, expert_quantized
):
    import sglang.srt.models.glm5_next as glm
    from sglang.srt.configs.glm5_next import Glm5NextTextConfig
    from sglang.srt.layers.quantization.fp8 import Fp8Config

    reset_context()
    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    parallel = SimpleNamespace(
        tp_size=tp_size,
        tp_rank=attn_rank,
        attn_tp_size=attn_size,
        attn_tp_rank=attn_rank,
    )
    monkeypatch.setattr(glm, "get_parallel", lambda: parallel)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        config = Glm5NextTextConfig(
            hidden_size=256,
            dtype=torch.bfloat16,
            linear_attn_config={
                "num_heads": 64,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "kda_layers": [0],
            },
        )
        qc = (
            Fp8Config(ignored_layers=_ignored_prefixes((*NAMES, "o_proj")))
            if expert_quantized
            else None
        )
        with torch.device("cuda"):
            fused = glm.Glm5NextLinearAttention(
                0, 256, config, quant_config=qc, prefix=PREFIX
            )
        assert fused.do_fuse_qkvbfg
        # Build the public unfused modules to compare equivalent loaded weights.
        with monkeypatch.context() as m:
            m.setattr(glm, "are_linear_prefixes_unquantized", lambda *a: False)
            with torch.device("cuda"):
                unfused = glm.Glm5NextLinearAttention(0, 256, config, prefix=PREFIX)
        generator = torch.Generator(device="cuda").manual_seed(5381)
        sizes = [8192, 8192, 8192, 64, 128, 128]
        mapping = [
            ("qkv_proj", "q"),
            ("qkv_proj", "k"),
            ("qkv_proj", "v"),
            ("b_proj", None),
            ("f_a_proj", None),
            ("g_a_proj", None),
        ]
        for shard, (size, (name, qkv_id)) in enumerate(zip(sizes, mapping)):
            weight = (
                torch.randn(size, 256, device="cuda", generator=generator) * 0.025
            ).bfloat16()
            fused.fused_qkvbfg_a_proj.weight_loader(
                fused.fused_qkvbfg_a_proj.weight, weight, shard
            )
            module = getattr(unfused, name)
            if qkv_id is None:
                module.weight_loader(module.weight, weight)
            else:
                module.weight_loader(module.weight, weight, qkv_id)
        for shard, name in enumerate(("f_b_proj", "g_b_proj")):
            weight = (
                torch.randn(8192, 128, device="cuda", generator=generator) * 0.025
            ).bfloat16()
            fused.fused_fg_b_proj.weight_loader(
                fused.fused_fg_b_proj.weight, weight, shard
            )
            module = getattr(unfused, name)
            module.weight_loader(module.weight, weight)
        for tokens in (1, 2, 17):
            hidden = torch.randn(
                tokens, 256, device="cuda", generator=generator
            ).bfloat16()
            expected = unfused.forward_qkvbfg(hidden, None)
            actual = fused.forward_qkvbfg_fused(hidden, None)
            for a, b in zip(actual, expected):
                assert a.shape == b.shape
                torch.testing.assert_close(a, b, atol=0.004, rtol=0.01)
        assert actual[0].shape[-1] == 3 * 8192 // attn_size
        assert actual[1].shape[-1] == 64 // attn_size
        assert actual[2].shape[-1] == 8192 // attn_size
    finally:
        torch.set_default_dtype(old_dtype)
        reset_context()
