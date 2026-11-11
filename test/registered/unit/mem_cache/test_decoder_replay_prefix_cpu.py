"""Request-aware #41303 prefix policy, independent of CUDA and native storage hashes."""

from array import array
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.common import _req_radix_key, match_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.streaming_session import StreamingSession
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _policy(bounded):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.components = {ComponentType.SWA: SimpleNamespace(sliding_window_size=128)}
    cache.decoder_swa_bounded_replay = bounded
    cache.tree_core = SimpleNamespace(has_swa_host_pool=True)
    cache.session = StreamingSession(cache)
    return cache


def _req(tokens):
    return Req(
        rid="cpu-decoder-replay",
        origin_input_text="",
        origin_input_ids=array("q", tokens),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )


@pytest.mark.parametrize("length", [0, 32, 128, 129, 260, 3500, 8192])
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("cap", [None, 0, 200])
def test_request_prefix_key_reserves_window_and_respects_existing_cap(
    length, bounded, cap
):
    policy = _policy(bounded)
    tokens = array("q", range(length))
    req = _req(tokens)
    with patch(
        "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
        return_value=False,
    ):
        key = _req_radix_key(policy, req, tokens, max_prefix_len=cap)
    expected = max(0, length - 128) if bounded else length
    if cap is not None:
        expected = min(expected, cap)
    assert len(key) == expected
    assert key.token_ids is tokens
    assert key.raw_token_ids() == tokens[:expected]


@pytest.mark.parametrize("bounded,expected", [(False, 200), (True, 132)])
def test_real_radix_match_obeys_production_unified_replay_policy(bounded, expected):
    tree = RadixCache.create_simulated()
    tree.insert(InsertParams(key=RadixKey(array("q", range(200)))))
    policy = _policy(bounded)
    tree.swa_reprefill_tail_tokens = policy.swa_reprefill_tail_tokens
    req = _req(list(range(200)) + list(range(1000, 1060)))
    with patch(
        "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
        return_value=False,
    ):
        match_kv_cache(req, tree)
    assert req.prefix_len == expected


@pytest.mark.parametrize(
    "slot_holds,aborted,expected",
    [(True, False, 0), (False, False, 128), (True, True, 128)],
)
def test_streaming_borrower_exception_does_not_detach_aborted_request(
    slot_holds, aborted, expected
):
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    policy = _policy(True)
    req = _req(range(260))
    req.session = SimpleNamespace(streaming=True, session_id="session")
    req.to_finish = FINISH_ABORT("cancelled") if aborted else None
    session = req.session
    policy.session.slots["session"] = SimpleNamespace(
        kv=SimpleNamespace(holds_kv=slot_holds)
    )
    with patch(
        "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
        return_value=False,
    ):
        assert policy.swa_reprefill_tail_tokens(req) == expected
    assert req.session is session
    assert "session" in policy.session.slots


def test_decoder_window_holdback_survives_presence_of_swa_host_pool():
    policy = _policy(True)
    with patch(
        "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
        return_value=True,
    ):
        assert policy.swa_reprefill_tail_tokens(_req(range(260))) == 128
