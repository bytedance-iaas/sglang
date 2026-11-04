"""Real checkpoint weights for the standalone DSV4 regression fixture."""

import json

import operator_smoke as fixture
import torch
from safetensors import safe_open

from sglang.kernels.ops.moe.early_moe import ActiveWeights


def actual_weights(root, layer, rank):
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    h, i, s = fixture.H, fixture.I, fixture.S
    w1 = torch.zeros((s, 2 * i, h), device="cuda", dtype=torch.float8_e4m3fn)
    w2 = torch.zeros((s, h, i), device="cuda", dtype=torch.float8_e4m3fn)
    sf1 = torch.ones((s, 2 * i // 128, h // 128), device="cuda")
    sf2 = torch.ones((s, h // 128, i // 128), device="cuda")
    for slot in fixture.SLOTS:
        # Test fixture's R8 layout gives the last physical slots replicas 0..7.
        expert = (rank * s + slot) % (fixture.E - 8)
        tensors = {}
        for kind in (
            "w1.weight",
            "w1.scale",
            "w3.weight",
            "w3.scale",
            "w2.weight",
            "w2.scale",
        ):
            key = f"layers.{layer}.ffn.experts.{expert}.{kind}"
            with safe_open(str(root / index[key]), framework="pt", device="cpu") as f:
                tensors[kind] = f.get_tensor(key).cuda()
        gate_up = torch.cat((tensors["w1.weight"], tensors["w3.weight"]))
        w1[slot].copy_(
            gate_up.reshape(2, i // 8, 8, h).permute(1, 0, 2, 3).reshape(2 * i, h)
        )
        sf1[slot].copy_(torch.cat((tensors["w1.scale"], tensors["w3.scale"])))
        w2[slot].copy_(tensors["w2.weight"])
        sf2[slot].copy_(tensors["w2.scale"])
    return ActiveWeights(w1, sf1, w2, sf2)
