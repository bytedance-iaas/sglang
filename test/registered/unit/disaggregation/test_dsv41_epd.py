from types import SimpleNamespace
from unittest.mock import Mock

import torch
from PIL import Image

from sglang.srt.disaggregation.encoder.receiver import (
    WaitingMMRequestStatus,
    WaitingZmqRequest,
)
from sglang.srt.disaggregation.encoder.server import (
    _flatten_mm_embedding,
    _get_legacy_mm_feature,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.multimodal.deepseek_v41_image_processing import image_token_types
from sglang.srt.multimodal.encoder_preprocessing import EncoderPreprocessOutput
from sglang.srt.multimodal.processors.deepseek_v41 import DeepseekV41ImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _encoder_model() -> DeepseekV4ForCausalLM:
    model = object.__new__(DeepseekV4ForCausalLM)
    model.config = SimpleNamespace(
        vision_patch_size=14,
        vision_downsample_ratio=3,
        vision_max_n_token=1024,
        vision_min_pixels=295936,
        vision_max_wh_ratio=None,
    )
    return model


def test_encoder_preprocessing_preserves_dsv41_span_length():
    model = _encoder_model()
    output = model.preprocess_mm_for_encoder(
        [Image.new("RGB", (64, 48), color="red")],
        Modality.IMAGE,
        config={},
    )

    item = output.mm_items[0]
    grid = output["image_grid_thw"][0]
    n_vit_h, n_vit_w = grid[-2:].tolist()
    ratio = model.config.vision_downsample_ratio
    expected = len(
        image_token_types(
            (n_vit_h + ratio - 1) // ratio,
            (n_vit_w + ratio - 1) // ratio,
        )
    )

    assert item.feature.shape[0] == n_vit_h * n_vit_w
    assert item.n_vit_h == n_vit_h
    assert item.n_vit_w == n_vit_w
    assert model.get_encoder_token_count(grid, Modality.IMAGE) == expected


def test_structured_encoder_output_does_not_require_pixel_values():
    output = EncoderPreprocessOutput(
        {"image_grid_thw": torch.tensor([[1, 2, 3]])},
        mm_items=[
            SimpleNamespace(
                modality=Modality.IMAGE,
                feature=torch.ones(6, 4),
            )
        ],
    )

    assert _get_legacy_mm_feature(output, Modality.IMAGE) is None


def test_encoder_span_list_is_flattened_for_transport():
    first = torch.randn(3, 8)
    second = torch.randn(5, 8)

    actual = _flatten_mm_embedding([first, second])

    torch.testing.assert_close(actual, torch.cat([first, second]))


def test_processor_rebuilds_dsv41_epd_image_spans():
    processor = object.__new__(DeepseekV41ImageProcessor)
    processor.image_token_id = 42
    processor.hf_config = SimpleNamespace(vision_downsample_ratio=3)
    grids = torch.tensor([[1, 6, 9], [1, 9, 6]])
    counts = [
        len(image_token_types(2, 3)),
        len(image_token_types(3, 2)),
    ]
    embeddings = torch.randn(sum(counts), 8)

    output = processor.get_validated_mm_data(
        [7, 42, 8, 42, 9],
        {Modality.IMAGE: embeddings},
        img_grid_thw=grids,
    )

    assert output.input_ids == ([7] + [42] * counts[0] + [8] + [42] * counts[1] + [9])
    assert [item.offsets for item in output.mm_items] == [
        [(1, counts[0])],
        [(counts[0] + 2, sum(counts) + 1)],
    ]
    torch.testing.assert_close(
        output.mm_items[0].precomputed_embeddings, embeddings[: counts[0]]
    )
    torch.testing.assert_close(
        output.mm_items[1].precomputed_embeddings, embeddings[counts[0] :]
    )
    for item in output.mm_items:
        embedding = item.precomputed_embeddings
        assert embedding.untyped_storage().nbytes() == embedding.nbytes


def test_detached_epd_embeddings_release_receive_pool_slot_immediately():
    request = object.__new__(WaitingZmqRequest)
    mm_inputs = SimpleNamespace(input_ids=[1, 2, 3])
    request.mm_processor = SimpleNamespace(
        detach_epd_embeddings_from_pool=True,
        prefer_tokenized_input=True,
        get_validated_mm_data=Mock(return_value=mm_inputs),
    )
    request.recv_req = SimpleNamespace(
        input_ids=[1, 2, 3],
        input_text=None,
        mm_inputs=None,
    )
    request.recv_embedding_data = SimpleNamespace(
        get_mm_extra_meta=Mock(return_value={})
    )
    request._bind_pool_slot_to_mm_inputs = Mock()
    request.embedding_pool = Mock()
    request._pool_slot_id = 7
    request.embeddings_buffer = object()

    request._finish_assemble({Modality.IMAGE: torch.randn(3, 8)})
    request._cleanup_gpu_buffer()

    request._bind_pool_slot_to_mm_inputs.assert_not_called()
    request.embedding_pool.release.assert_called_once_with(7)
    assert request._pool_slot_id is None
    assert request.embeddings_buffer is None
    assert request.recv_req.mm_inputs is mm_inputs
    assert request.status == WaitingMMRequestStatus.SUCCESS


def test_language_only_prefill_uses_precomputed_embedding():
    model = object.__new__(DeepseekV4ForCausalLM)
    model.dsv41_multimodal_enabled = True
    model.vision = None
    model.config = SimpleNamespace(language_only=True, image_token_id=42)
    model.pp_group = SimpleNamespace(is_first_rank=True)
    expected = torch.randn(3, 8)
    model._prepare_mm_embeddings = Mock(return_value=expected)
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: False,
            is_target_verify=lambda: False,
            is_decode_or_idle=lambda: False,
        ),
        mm_inputs=[SimpleNamespace()],
    )
    input_ids = torch.tensor([1, 1 << 50, 2])

    canonical_ids, input_embeds = model.prepare_model_inputs(
        input_ids, forward_batch, input_embeds=None
    )

    model._prepare_mm_embeddings.assert_called_once_with(input_ids, forward_batch)
    assert input_embeds is expected
    assert canonical_ids.tolist() == [1, 42, 2]


def test_encoder_only_weight_loader_ignores_language_weights():
    model = object.__new__(DeepseekV4ForCausalLM)
    torch.nn.Module.__init__(model)
    model.encoder_only = True
    model.config = SimpleNamespace(num_hidden_layers=40)
    model.image_start = torch.nn.Parameter(torch.zeros(4))

    loaded = model.load_weights(
        iter(
            [
                ("image_start", torch.ones(4)),
                ("model.embed_tokens.weight", torch.ones(2, 4)),
            ]
        )
    )

    assert loaded == {"image_start"}
    torch.testing.assert_close(model.image_start, torch.ones(4))
