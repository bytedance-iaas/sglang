"""DeepSeek-V4.1 vision inputs merge before interleave CP sharding."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.srt.layers.cp.base import init_cp_strategy
from sglang.srt.layers.cp.interleave_kpool import local_query_inputs
from sglang.srt.layers.cp.utils import prepare_cp_forward
from sglang.srt.managers import mm_schedule
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

HIDDEN = 8
VOCAB = 64
IMAGE_TOKEN_ID = 7
CP_SIZE = 4
CHUNKS = [(0, 7), (3, 9), (1, 5)]
IMAGE_OFFSET = (2, 8)
IMAGE_HASH = 12345
NUM_TOKENS = sum(length for _, length in CHUNKS)
PHYSICAL_ROWS = 8
IMAGE_ROWS = torch.arange(7, 13)
POSITIONS = torch.cat(
    [torch.arange(prefix, prefix + length) for prefix, length in CHUNKS]
)


def _image_span(item):
    start, end = item.offsets[0]
    rows = end - start + 1
    return torch.arange(rows * HIDDEN, dtype=torch.float32).view(rows, HIDDEN) + 100


def _pad(x):
    return torch.cat([x, x.new_zeros(PHYSICAL_ROWS - x.shape[0], *x.shape[1:])])


class _RecordingBody:
    def __init__(self, embed):
        self.embed = embed
        self.calls = []

    def get_input_embeddings(self):
        return self.embed

    def __call__(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        self.calls.append(
            SimpleNamespace(
                input_ids=input_ids.clone(),
                positions=positions.clone(),
                input_embeds=None if input_embeds is None else input_embeds.clone(),
                full_model_ids=forward_batch.model_input_ids_global.clone(),
                req_pool_indices=forward_batch.req_pool_indices.clone(),
                pp_proxy_tensors=pp_proxy_tensors,
            )
        )
        if pp_proxy_tensors is not None:
            return pp_proxy_tensors["hidden_states"]
        return input_embeds, input_embeds


class _VisionStub(DeepseekV4ForCausalLM):
    def __init__(self, embed, *, first=True, last=True):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(image_token_id=IMAGE_TOKEN_ID)
        self.dsv41_multimodal_enabled = True
        self.vision = object() if first else None
        self.model = _RecordingBody(embed)
        self.pp_group = SimpleNamespace(is_first_rank=first, is_last_rank=last)
        self.lm_head = object()
        self.capture_aux_hidden_states = False
        self.logits_calls = []

    def get_image_feature(self, items):
        return [_image_span(item) for item in items]

    def logits_processor(
        self,
        input_ids,
        hidden_states,
        lm_head,
        logits_metadata,
        aux_hidden_states=None,
        hidden_states_before_norm=None,
    ):
        self.logits_calls.append(
            SimpleNamespace(
                input_ids=input_ids.clone(),
                hidden_states=hidden_states.clone(),
                hidden_states_before_norm=hidden_states_before_norm.clone(),
            )
        )
        return SimpleNamespace()


def _build_batch():
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=torch.zeros(1),
        offsets=[IMAGE_OFFSET],
    )
    item.set_hash(IMAGE_HASH)
    ids = list(range(10, 17))
    ids += [item.pad_value] * len(IMAGE_ROWS) + [20, 21, 22]
    ids += list(range(30, 35))
    return (
        SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            mm_inputs=[
                MultimodalInputs(mm_items=[]),
                MultimodalInputs(mm_items=[item], im_token_id=IMAGE_TOKEN_ID),
                None,
            ],
            extend_prefix_lens_cpu=[prefix for prefix, _ in CHUNKS],
            extend_seq_lens_cpu=[length for _, length in CHUNKS],
            seq_lens_cpu=[prefix + length for prefix, length in CHUNKS],
            input_ids=torch.tensor(ids, dtype=torch.long),
            positions=POSITIONS.clone(),
            req_pool_indices=torch.tensor([11, 3, 27]),
            mm_input_embeds=None,
            attn_cp_metadata=None,
            global_num_tokens_cpu=None,
            out_cache_loc=None,
            input_ids_global=torch.zeros(1, dtype=torch.long),
        ),
        item,
    )


def _canonical(ids):
    result = ids.clone()
    result[IMAGE_ROWS] = IMAGE_TOKEN_ID
    return result


def _expected_embeds(embed, scheduler_ids, item):
    with torch.no_grad():
        result = embed(scheduler_ids.clamp(max=VOCAB - 1))
    result[IMAGE_ROWS] = _image_span(item)[1:7]
    return result


@contextmanager
def _cp_collectives(full, rank):
    def all_gather(output, input_tensor):
        output.zero_()
        for peer in range(CP_SIZE):
            rows = full[peer::CP_SIZE]
            output[peer * PHYSICAL_ROWS : peer * PHYSICAL_ROWS + len(rows)] = rows
        output[rank * PHYSICAL_ROWS : (rank + 1) * PHYSICAL_ROWS] = input_tensor

    with (
        patch("torch.cuda.current_stream", return_value=None),
        patch(
            "sglang.srt.layers.cp.interleave.attn_cp_all_gather_into_tensor",
            side_effect=all_gather,
        ),
        patch(
            "sglang.srt.layers.cp.interleave.is_allocation_symmetric",
            return_value=False,
        ),
        patch(
            "sglang.srt.layers.cp.interleave.use_symmetric_memory",
            return_value=torch.no_grad(),
        ),
    ):
        yield


@pytest.fixture(autouse=True)
def _cp_state():
    mm_schedule.init_mm_embedding_cache(1 << 20)
    init_cp_strategy(enable_prefill_cp=True, cp_size=CP_SIZE, cp_strategy="interleave")
    yield
    init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="interleave")


@torch.no_grad()
def test_merge_before_shard_preserves_scheduler_ids_and_physical_slots():
    torch.manual_seed(0)
    embed = nn.Embedding(VOCAB, HIDDEN)
    model = _VisionStub(embed)
    runner = EagerRunner.__new__(EagerRunner)
    runner.model_runner = SimpleNamespace(model=model)

    for rank in range(CP_SIZE):
        batch, item = _build_batch()
        scheduler_ids = batch.input_ids.clone()
        canonical = _canonical(scheduler_ids)
        full = _expected_embeds(embed, scheduler_ids, item)
        model.model.calls.clear()
        model.logits_calls.clear()
        with (
            get_parallel().override(
                attn_cp_rank=rank,
                attn_cp_size=CP_SIZE,
                attn_cp_group=object(),
                enable_cp_tp_group_sharing=False,
            ),
            _cp_collectives(full, rank),
        ):
            prepare_cp_forward(batch)
            runner._execute_extend_cp(batch, {})

        (body,) = model.model.calls
        assert torch.equal(body.input_ids, _pad(canonical[rank::CP_SIZE]))
        assert torch.equal(body.input_embeds, _pad(full[rank::CP_SIZE]))
        assert torch.equal(body.full_model_ids, canonical)
        assert body.req_pool_indices.tolist() == [11, 3, 27]
        (logits,) = model.logits_calls
        assert torch.equal(logits.input_ids, canonical)
        assert torch.equal(logits.hidden_states, full)
        assert torch.equal(batch.input_ids, scheduler_ids)
        assert not hasattr(batch, "model_input_ids_global")


@torch.no_grad()
def test_nonfirst_pp_stage_uses_proxy_hidden_without_token_embedding():
    model = _VisionStub(embed=None, first=False, last=False)
    runner = EagerRunner.__new__(EagerRunner)
    runner.model_runner = SimpleNamespace(model=model)
    batch, _ = _build_batch()
    scheduler_ids = batch.input_ids.clone()
    proxy_hidden = torch.arange(PHYSICAL_ROWS * HIDDEN, dtype=torch.float32).view(
        PHYSICAL_ROWS, HIDDEN
    )
    proxy = PPProxyTensors({"hidden_states": proxy_hidden})

    with get_parallel().override(
        attn_cp_rank=0,
        attn_cp_size=CP_SIZE,
        attn_cp_group=object(),
        enable_cp_tp_group_sharing=False,
    ):
        prepare_cp_forward(batch)
        output = runner._execute_extend_cp(batch, {"pp_proxy_tensors": proxy})

    (body,) = model.model.calls
    assert body.input_embeds is None
    assert body.pp_proxy_tensors is proxy
    assert torch.equal(body.full_model_ids, _canonical(scheduler_ids))
    assert torch.equal(output, proxy_hidden)
    assert torch.equal(batch.input_ids, scheduler_ids)


def test_local_query_mapping_uses_physical_request_pool_slots():
    batch = SimpleNamespace(
        extend_seq_lens_cpu=[2, 1, 6],
        req_pool_indices=torch.tensor([11, 3, 27]),
    )
    metadata = SimpleNamespace(
        real_page_table=torch.tensor([1]),
        dsa_seqlens_expanded=torch.tensor([2]),
        dsa_extend_seq_lens_list=[1, 1],
        indexer_seq_lens_cpu=torch.tensor([3, 9]),
    )
    with (
        get_parallel().override(attn_cp_rank=2, attn_cp_size=CP_SIZE),
        patch(
            "sglang.srt.layers.cp.interleave_kpool.is_interleave_extend",
            return_value=True,
        ),
    ):
        local = local_query_inputs(batch, metadata)

    # Rank 2 owns no row from logical request 0, but owns rows from requests
    # 1 and 2. The result must use their physical request-pool slots.
    assert local["local_req_pool_indices"].tolist() == [3, 27]
