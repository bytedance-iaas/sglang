from contextlib import contextmanager
from unittest.mock import patch

from sglang.srt.model_executor.step_span_utils import (
    set_detailed_annotations_enabled,
)
from sglang.srt.utils.nvtx_utils import (
    PREFILL_DETAILED_RANGES,
    detailed_profile_range,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_detailed_profile_range_is_noop_without_profiler():
    set_detailed_annotations_enabled(True)
    try:
        with (
            patch(
                "sglang.srt.utils.nvtx_utils.torch.autograd._profiler_enabled",
                return_value=False,
            ),
            patch(
                "sglang.srt.utils.nvtx_utils.torch.profiler.record_function"
            ) as record_function,
        ):
            with detailed_profile_range(PREFILL_DETAILED_RANGES[0]):
                pass
            record_function.assert_not_called()
    finally:
        set_detailed_annotations_enabled(False)


def test_detailed_profile_range_is_noop_without_detailed_annotations():
    set_detailed_annotations_enabled(False)
    with (
        patch(
            "sglang.srt.utils.nvtx_utils.torch.autograd._profiler_enabled",
            return_value=True,
        ),
        patch(
            "sglang.srt.utils.nvtx_utils.torch.profiler.record_function"
        ) as record_function,
    ):
        with detailed_profile_range(PREFILL_DETAILED_RANGES[0]):
            pass
        record_function.assert_not_called()


def test_default_path_does_not_query_torch_profiler_state():
    set_detailed_annotations_enabled(False)
    with patch(
        "sglang.srt.utils.nvtx_utils.torch.autograd._profiler_enabled"
    ) as profiler_enabled:
        with detailed_profile_range(PREFILL_DETAILED_RANGES[0]):
            pass
        profiler_enabled.assert_not_called()


def test_detailed_profile_range_emits_all_prefill_categories():
    events = []

    @contextmanager
    def fake_record_function(name):
        events.append(("enter", name))
        try:
            yield
        finally:
            events.append(("exit", name))

    set_detailed_annotations_enabled(True)
    try:
        with (
            patch(
                "sglang.srt.utils.nvtx_utils.torch.autograd._profiler_enabled",
                return_value=True,
            ),
            patch(
                "sglang.srt.utils.nvtx_utils.torch.profiler.record_function",
                side_effect=fake_record_function,
            ),
        ):
            for name in PREFILL_DETAILED_RANGES:
                with detailed_profile_range(name):
                    events.append(("body", name))
    finally:
        set_detailed_annotations_enabled(False)

    assert PREFILL_DETAILED_RANGES == (
        "prefill.dsa_indexer",
        "prefill.sparse_flashmla_attention",
        "prefill.router_megamoe",
        "prefill.pp_cp_communication",
    )
    assert events == [
        event
        for name in PREFILL_DETAILED_RANGES
        for event in (("enter", name), ("body", name), ("exit", name))
    ]


def test_detailed_profile_range_preserves_return_and_exception_semantics():
    @contextmanager
    def fake_record_function(_name):
        yield

    sentinel = object()
    set_detailed_annotations_enabled(True)
    try:
        with (
            patch(
                "sglang.srt.utils.nvtx_utils.torch.autograd._profiler_enabled",
                return_value=True,
            ),
            patch(
                "sglang.srt.utils.nvtx_utils.torch.profiler.record_function",
                side_effect=fake_record_function,
            ),
            patch("torch.cuda.synchronize") as synchronize,
        ):

            def profiled_identity(value):
                with detailed_profile_range(PREFILL_DETAILED_RANGES[0]):
                    return value

            assert profiled_identity(sentinel) is sentinel
            try:
                with detailed_profile_range(PREFILL_DETAILED_RANGES[0]):
                    raise RuntimeError("sentinel")
            except RuntimeError as exc:
                assert str(exc) == "sentinel"
            else:
                raise AssertionError("profile range swallowed the body exception")
            synchronize.assert_not_called()
    finally:
        set_detailed_annotations_enabled(False)
