"""Regression tests for PP-prefill hybrid KV transfer with speculative KV."""

import concurrent.futures
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.prefill import _transfer_start_layer
from sglang.srt.disaggregation.utils import (
    build_kv_layer_ids,
    build_transfer_entry_pairs,
)
from sglang.srt.distributed.utils import get_pp_indices
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool


def _full_attention_ids(num_layers=60, interval=4):
    return [i for i in range(num_layers) if i % interval == interval - 1]


def _hybrid_pool(*, start_layer=0, layer_ids=None):
    pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
    pool.start_layer = start_layer
    if layer_ids is not None:
        pool.full_attention_layer_id_mapping = layer_ids
        pool.use_mla = False
    return pool


def test_hybrid_stage_start_is_full_attention_relative():
    cfg = SimpleNamespace(full_attention_layer_ids=_full_attention_ids())

    assert (
        _transfer_start_layer(pool=_hybrid_pool(start_layer=30), hf_text_config=cfg)
        == 7
    )
    assert (
        _transfer_start_layer(pool=_hybrid_pool(start_layer=0), hf_text_config=cfg) == 0
    )


def test_draft_entries_use_reserved_layer_band():
    stage_ids = [lid for lid in _full_attention_ids() if lid >= 30]

    ids = build_kv_layer_ids(
        token_to_kv_pool=_hybrid_pool(layer_ids=stage_ids),
        draft_token_to_kv_pool=_hybrid_pool(layer_ids=[0]),
        num_draft_entries=2,
        num_hidden_layers=60,
    )

    assert ids == stage_ids + stage_ids + [60, 60]


def test_pp2_prefill_entries_pair_with_pp1_decode_layout():
    full_ids = _full_attention_ids()
    stage_ids = [lid for lid in full_ids if lid >= 30]
    src = build_kv_layer_ids(
        token_to_kv_pool=_hybrid_pool(layer_ids=stage_ids),
        draft_token_to_kv_pool=_hybrid_pool(layer_ids=[0]),
        num_draft_entries=2,
        num_hidden_layers=60,
    )
    dst = build_kv_layer_ids(
        token_to_kv_pool=_hybrid_pool(layer_ids=full_ids),
        draft_token_to_kv_pool=_hybrid_pool(layer_ids=[0]),
        num_draft_entries=2,
        num_hidden_layers=60,
    )

    pairs = build_transfer_entry_pairs(
        src, dst, len(src), len(dst), allow_positional_fallback=False
    )
    offset = len(full_ids) - len(stage_ids)
    assert pairs == (
        [(i, offset + i) for i in range(len(stage_ids))]
        + [
            (len(stage_ids) + i, len(full_ids) + offset + i)
            for i in range(len(stage_ids))
        ]
        + [
            (2 * len(stage_ids), 2 * len(full_ids)),
            (2 * len(stage_ids) + 1, 2 * len(full_ids) + 1),
        ]
    )


class _RecordingKVManager:
    get_mha_kv_ptrs_with_pp = CommonKVManager.get_mha_kv_ptrs_with_pp

    def __init__(self, *, prefill_start_layer):
        self.is_mla_backend = False
        self.is_hybrid_mla_backend = False
        self.enable_custom_mem_pool = False
        self.pp_size = 2
        self.kv_args = SimpleNamespace(prefill_start_layer=prefill_start_layer)
        self.blocks = []

    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        self.blocks.extend(transfer_blocks)
        return 0


def test_mooncake_prefers_exact_ids_over_positional_mha_slicing():
    full_ids = _full_attention_ids()
    stage_ids = full_ids[7:]
    src_ptrs = [1000 + i for i in range(2 * len(stage_ids))]
    dst_ptrs = [2000 + i for i in range(2 * len(full_ids))]
    item_lens = [10 + i for i in range(len(src_ptrs))]
    manager = _RecordingKVManager(prefill_start_layer=7)

    rc = MooncakeKVManager._send_kvcache_generic(
        manager,
        mooncake_session_id="session",
        src_data_ptrs=src_ptrs,
        dst_data_ptrs=dst_ptrs,
        item_lens=item_lens,
        prefill_data_indices=np.array([0], dtype=np.int32),
        dst_data_indices=np.array([0], dtype=np.int32),
        executor=None,
        src_layer_ids=stage_ids * 2,
        dst_layer_ids=full_ids * 2,
    )

    expected = [
        (src_ptrs[i], dst_ptrs[7 + i], item_lens[i]) for i in range(len(stage_ids))
    ] + [
        (
            src_ptrs[len(stage_ids) + i],
            dst_ptrs[len(full_ids) + 7 + i],
            item_lens[len(stage_ids) + i],
        )
        for i in range(len(stage_ids))
    ]
    assert rc == 0
    assert manager.blocks == expected


class _RecordingEngine:
    def __init__(self):
        self.transfers = []

    def batch_transfer_sync(self, session, src, dst, lengths):
        self.transfers.append((src[0], dst[0], lengths[0]))
        return 0


def test_head_sliced_transfer_pairs_target_and_draft_by_peer_layer_ids():
    manager = SimpleNamespace(
        kv_args=SimpleNamespace(
            engine_rank=0,
            kv_item_lens=[16, 16, 16],
            page_size=1,
            total_kv_head_num=2,
            kv_head_num=2,
            kv_data_ptrs=[100, 200, 300],
            kv_layer_ids=[7, 31, 60],
        ),
        attn_tp_size=1,
        pp_size=2,
        engine=_RecordingEngine(),
    )
    dst_ptrs = [1000, 2000, 3000, 4000, 5000]
    dst_ids = [3, 7, 31, 47, 60]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        rc = MooncakeKVManager.send_kvcache_slice(
            manager,
            mooncake_session_id="session",
            prefill_kv_indices=np.array([0], dtype=np.int32),
            dst_kv_ptrs=dst_ptrs,
            dst_kv_indices=np.array([0], dtype=np.int32),
            dst_tp_rank=0,
            dst_attn_tp_size=2,
            dst_kv_item_len=16,
            executor=executor,
            dst_layer_ids=dst_ids,
        )

    assert rc == 0
    assert sorted(manager.engine.transfers) == [
        (100, 2000, 8),
        (200, 3000, 8),
        (300, 5000, 8),
    ]


def test_single_stage_draft_ignores_target_pp_partition(monkeypatch):
    monkeypatch.setenv("SGLANG_PP_LAYER_PARTITION", "30,30")

    assert get_pp_indices(num_hidden_layers=60, pp_rank=0, pp_size=1) == (0, 60)
