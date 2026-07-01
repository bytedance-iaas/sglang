from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    DSV4RawVerifyMetadata,
    DeepseekV4AttnBackend,
    _reshape_flashmla_query_metadata,
    _reshape_flashmla_topk_length_metadata,
)


def test_target_verify_layout_ignores_graph_bucket_padding():
    backend = object.__new__(DeepseekV4AttnBackend)
    backend.speculative_num_draft_tokens = 3
    forward_batch = SimpleNamespace(
        batch_size=8,
        _cuda_graph_raw_batch_size=2,
        spec_info=SimpleNamespace(draft_token_num=3),
        input_ids=torch.arange(8),
    )
    req_pool_indices = torch.arange(8)
    seq_lens = torch.arange(8) + 10

    layout = backend._make_target_verify_layout(
        forward_batch,
        req_pool_indices,
        seq_lens,
    )

    assert layout.active_bs == 2
    assert layout.query_len == 3
    assert layout.semantic_num_tokens == 6
    torch.testing.assert_close(layout.req_pool_indices, req_pool_indices[:2])
    torch.testing.assert_close(layout.seq_lens, seq_lens[:2])


def test_target_verify_layout_uses_unpadded_draft_tokens():
    backend = object.__new__(DeepseekV4AttnBackend)
    backend.speculative_num_draft_tokens = 3
    forward_batch = SimpleNamespace(
        batch_size=8,
        _cuda_graph_raw_batch_size=8,
        spec_info=SimpleNamespace(
            draft_token_num=3,
            draft_token=torch.arange(6),
        ),
        input_ids=torch.arange(24),
    )
    req_pool_indices = torch.arange(8)
    seq_lens = torch.arange(8) + 10

    layout = backend._make_target_verify_layout(
        forward_batch,
        req_pool_indices,
        seq_lens,
    )

    assert layout.active_bs == 2
    assert layout.query_len == 3
    assert layout.semantic_num_tokens == 6
    torch.testing.assert_close(layout.req_pool_indices, req_pool_indices[:2])
    torch.testing.assert_close(layout.seq_lens, seq_lens[:2])


def test_active_batch_size_only_prefers_draft_tokens_when_requested():
    forward_batch = SimpleNamespace(
        batch_size=8,
        _cuda_graph_raw_batch_size=8,
        spec_info=SimpleNamespace(
            draft_token_num=3,
            draft_token=torch.arange(6),
        ),
    )
    req_pool_indices = torch.arange(8)
    seq_lens = torch.arange(8) + 10

    assert (
        DeepseekV4AttnBackend._active_batch_size(
            forward_batch,
            req_pool_indices,
            seq_lens,
            prefer_draft_token=False,
        )
        == 8
    )
    assert (
        DeepseekV4AttnBackend._active_batch_size(
            forward_batch,
            req_pool_indices,
            seq_lens,
            prefer_draft_token=True,
        )
        == 2
    )


def test_active_batch_size_uses_original_batch_size_when_replay_raw_absent():
    forward_batch = SimpleNamespace(
        batch_size=8,
        _original_batch_size=7,
        spec_info=SimpleNamespace(draft_token_num=3),
    )
    req_pool_indices = torch.arange(8)
    seq_lens = torch.arange(8) + 10

    assert (
        DeepseekV4AttnBackend._active_batch_size(
            forward_batch,
            req_pool_indices,
            seq_lens,
            prefer_draft_token=True,
        )
        == 7
    )


def test_target_verify_layout_ignores_dp_padding_without_graph_marker():
    backend = object.__new__(DeepseekV4AttnBackend)
    backend.speculative_num_draft_tokens = 3
    forward_batch = SimpleNamespace(
        batch_size=66,
        _original_batch_size=62,
        spec_info=SimpleNamespace(
            draft_token_num=3,
            draft_token=torch.arange(62 * 3),
        ),
        input_ids=torch.arange(66 * 3),
    )
    req_pool_indices = torch.arange(66)
    seq_lens = torch.arange(66) + 10

    layout = backend._make_target_verify_layout(
        forward_batch,
        req_pool_indices,
        seq_lens,
    )

    assert layout.active_bs == 62
    assert layout.query_len == 3
    assert layout.semantic_num_tokens == 62 * 3
    torch.testing.assert_close(layout.req_pool_indices, req_pool_indices[:62])
    torch.testing.assert_close(layout.seq_lens, seq_lens[:62])


def test_raw_verify_metadata_copy_requires_graph_bucket_shape():
    dst = DSV4RawVerifyMetadata(
        req_pool_indices=torch.zeros(8, dtype=torch.int32),
        seq_lens=torch.zeros(8, dtype=torch.int32),
        out_cache_loc=torch.zeros(24, dtype=torch.int32),
    )
    src = DSV4RawVerifyMetadata(
        req_pool_indices=torch.arange(8, dtype=torch.int32),
        seq_lens=torch.arange(8, dtype=torch.int32) + 10,
        out_cache_loc=torch.arange(24, dtype=torch.int32),
    )

    dst.copy_(src)

    torch.testing.assert_close(dst.req_pool_indices, src.req_pool_indices)
    torch.testing.assert_close(dst.seq_lens, src.seq_lens)
    torch.testing.assert_close(dst.out_cache_loc, src.out_cache_loc)


def test_raw_verify_metadata_copy_rejects_active_shape():
    dst = DSV4RawVerifyMetadata(
        req_pool_indices=torch.zeros(8, dtype=torch.int32),
        seq_lens=torch.zeros(8, dtype=torch.int32),
        out_cache_loc=torch.zeros(24, dtype=torch.int32),
    )
    src = DSV4RawVerifyMetadata(
        req_pool_indices=torch.arange(7, dtype=torch.int32),
        seq_lens=torch.arange(7, dtype=torch.int32) + 10,
        out_cache_loc=torch.arange(21, dtype=torch.int32),
    )

    try:
        dst.copy_(src)
    except RuntimeError as exc:
        assert "metadata copy shape mismatch" in str(exc)
    else:
        raise AssertionError("Expected raw verify metadata shape mismatch")


def test_flashmla_target_verify_metadata_shapes():
    indices = torch.arange(6 * 64).view(6, 64)
    lengths = torch.tensor([5, 6, 7, 8, 9, 10])

    reshaped_indices = _reshape_flashmla_query_metadata(
        indices,
        batch_size=2,
        query_len=3,
        name="extra_indices",
        pad_value=-1,
    )
    reshaped_lengths = _reshape_flashmla_topk_length_metadata(
        lengths,
        batch_size=2,
        query_len=3,
        name="extra_topk_lengths",
        pad_value=1,
    )

    assert reshaped_indices.shape == (2, 3, 64)
    assert reshaped_lengths.shape == (2,)
    torch.testing.assert_close(reshaped_indices, indices.view(2, 3, 64))
    torch.testing.assert_close(reshaped_lengths, torch.tensor([7, 10]))
