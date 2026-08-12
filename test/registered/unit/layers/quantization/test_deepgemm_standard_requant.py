from unittest.mock import Mock

import torch

from sglang.srt.layers.quantization import fp8_utils


def test_standard_moe_weight_requant_is_not_tied_to_deepep(monkeypatch):
    weight = torch.nn.Parameter(torch.empty(64, 128))
    scale = torch.nn.Parameter(torch.empty(1, 1))
    scale.format_ue8m0 = False
    requant = Mock()

    monkeypatch.setattr(
        "sglang.srt.model_loader.utils.should_deepgemm_weight_requant_ue8m0",
        lambda **_: True,
    )
    monkeypatch.setattr(fp8_utils, "requant_weight_ue8m0_inplace", requant)

    assert fp8_utils.requant_block_scale_ue8m0_for_deepgemm(
        weight,
        scale,
        [128, 128],
        use_deepgemm_runner=True,
        output_dtype=torch.bfloat16,
        weight_shape=weight.shape,
    )
    requant.assert_called_once_with(weight, scale, [128, 128])
    assert scale.format_ue8m0


def test_standard_moe_weight_requant_rejects_unsupported_block_shape(monkeypatch):
    weight = torch.nn.Parameter(torch.empty(64, 128))
    scale = torch.nn.Parameter(torch.empty(1, 1))
    requant = Mock()
    monkeypatch.setattr(fp8_utils, "requant_weight_ue8m0_inplace", requant)

    assert not fp8_utils.requant_block_scale_ue8m0_for_deepgemm(
        weight,
        scale,
        [64, 128],
        use_deepgemm_runner=True,
    )
    requant.assert_not_called()
