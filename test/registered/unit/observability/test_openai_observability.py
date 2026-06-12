"""Unit tests for OpenAI observability helpers."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except ModuleNotFoundError:
    CustomTestCase = unittest.TestCase

    def register_cpu_ci(*args, **kwargs):
        pass


register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _load_openai_observability():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "python" / "sglang" / "srt" / "openai_observability.py"

    stub_modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
    }
    stub_modules["sglang"].__path__ = [str(repo_root / "python" / "sglang")]
    stub_modules["sglang.srt"].__path__ = [str(repo_root / "python" / "sglang" / "srt")]

    previous_modules = {name: sys.modules.get(name) for name in stub_modules}
    sys.modules.update(stub_modules)

    module_name = "_test_openai_observability"
    previous_test_module = sys.modules.get(module_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module
        if previous_test_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_test_module

    return module


mod = _load_openai_observability()


class _Recorder:
    def __init__(self):
        self.values = []

    def record(self, value, attributes=None):
        self.values.append((value, attributes))


class _Counter:
    def add(self, value, attributes=None):
        pass


class _FakeSpan:
    def __init__(self):
        self.attributes = {}

    def is_recording(self):
        return True

    def set_attribute(self, name, value):
        self.attributes[name] = value

    def set_status(self, status):
        self.status = status

    def end(self):
        self.ended = True


class _FakeTracer:
    def __init__(self, span):
        self.span = span

    def start_span(self, *args, **kwargs):
        return self.span


class _Dumpable:
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def model_dump(self):
        return self._kwargs


class TestOpenAIObservability(CustomTestCase):
    def test_model_as_dict_tolerates_plain_object(self):
        plain = object()
        self.assertIs(mod.model_as_dict(plain), plain)

    def test_accumulate_stream_items_noops_when_complete_response_is_none(self):
        with (
            patch.object(mod, "is_otel_available", return_value=True),
            patch.object(
                mod,
                "model_as_dict",
                side_effect=AssertionError("should not convert discarded chunks"),
            ),
        ):
            mod.accumulate_stream_items({"model": "test-model"}, None)

    def test_accumulate_stream_items_skips_tool_calls_without_valid_index(self):
        complete_response = {"choices": [], "model": "", "usage": None, "error": None}
        chunk = {
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {"function": {"name": "missing_index"}},
                            {"index": None, "function": {"name": "none_index"}},
                            {
                                "index": "0",
                                "id": "call-0",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"city"',
                                },
                            },
                            {
                                "index": "0",
                                "function": {"arguments": ': "Paris"}'},
                            },
                        ],
                    },
                }
            ],
        }

        with patch.object(mod, "is_otel_available", return_value=True):
            mod.accumulate_stream_items(chunk, complete_response)

        tool_calls = complete_response["choices"][0]["message"]["tool_calls"]
        self.assertEqual(tool_calls[0]["id"], "call-0")
        self.assertEqual(tool_calls[0]["function"]["name"], "lookup")
        self.assertEqual(tool_calls[0]["function"]["arguments"], '{"city": "Paris"}')

    def test_accumulate_stream_items_converts_nested_choice_objects(self):
        complete_response = {"choices": [], "model": "", "usage": None, "error": None}
        chunk = {
            "model": "test-model",
            "choices": [
                _Dumpable(
                    index=0,
                    delta=_Dumpable(role="assistant", content="hello"),
                    finish_reason=None,
                )
            ],
        }

        with patch.object(mod, "is_otel_available", return_value=True):
            mod.accumulate_stream_items(chunk, complete_response)

        self.assertEqual(complete_response["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(complete_response["choices"][0]["message"]["content"], "hello")

    @unittest.skipUnless(mod._is_msgspec_available, "msgspec not installed")
    def test_accumulate_stream_items_converts_msgspec_nested_choice(self):
        class Delta(mod.msgspec.Struct, omit_defaults=True):
            reasoning_content: str | None = None
            role: str | None = None
            content: str | None = None

        class Choice(mod.msgspec.Struct):
            index: int
            delta: Delta
            logprobs: dict | None = None
            finish_reason: str | None = None

        class Chunk(mod.msgspec.Struct, omit_defaults=True):
            model: str
            choices: list[Choice]
            usage: dict | None = None

        complete_response = {"choices": [], "model": "", "usage": None, "error": None}
        chunk = Chunk("test-model", [Choice(0, Delta(role="assistant", content="hello"))])

        with patch.object(mod, "is_otel_available", return_value=True):
            mod.accumulate_stream_items(chunk, complete_response)

        self.assertEqual(complete_response["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(complete_response["choices"][0]["message"]["content"], "hello")

    def test_provider_init_disables_otel_when_initialization_fails(self):
        original = mod._is_otel_imported
        try:
            mod._is_otel_imported = True
            with (
                patch.object(mod, "init_tracer", side_effect=RuntimeError("boom")),
                patch.object(mod, "init_metrics") as init_metrics,
                self.assertLogs(mod.logger, level="WARNING"),
            ):
                provider = mod.OpenTelemetryProvider()
            self.assertIsNone(provider.tracer)
            self.assertIsNone(provider.meter)
            self.assertFalse(mod._is_otel_imported)
            init_metrics.assert_not_called()
        finally:
            mod._is_otel_imported = original

    def test_record_stream_with_usage_and_no_first_token_time_does_not_crash(self):
        span = _FakeSpan()
        provider = mod.OpenTelemetryProvider.__new__(mod.OpenTelemetryProvider)
        provider.tracer = _FakeTracer(span)
        provider.meter = None

        original_metrics_state = mod.Meters.is_metrics_inited
        original_meters = {
            name: getattr(mod.Meters, name)
            for name in (
                "chat_counter",
                "tokens_histogram",
                "chat_choice_counter",
                "chat_duration_histogram",
                "streaming_time_to_first_token",
                "streaming_time_to_generate",
                "streaming_time_per_output_token",
            )
        }
        per_token_recorder = _Recorder()
        try:
            mod.Meters.is_metrics_inited = True
            mod.Meters.chat_counter = _Counter()
            mod.Meters.tokens_histogram = _Recorder()
            mod.Meters.chat_choice_counter = _Counter()
            mod.Meters.chat_duration_histogram = _Recorder()
            mod.Meters.streaming_time_to_first_token = _Recorder()
            mod.Meters.streaming_time_to_generate = _Recorder()
            mod.Meters.streaming_time_per_output_token = per_token_recorder

            request = SimpleNamespace(
                model="test-model",
                stream=True,
                messages=[],
                max_tokens=None,
                temperature=None,
                top_p=None,
                frequency_penalty=None,
                presence_penalty=None,
                user=None,
            )

            with (
                patch.object(mod, "is_otel_available", return_value=True),
                patch.object(mod, "extract_trace_context", return_value=None),
                patch.object(mod, "should_send_prompts", return_value=False),
                patch.object(mod, "SpanKind", SimpleNamespace(SERVER="server")),
                patch.object(mod, "StatusCode", SimpleNamespace(OK="ok")),
                patch.object(mod, "Status", lambda status_code: ("status", status_code)),
                patch.object(mod.time, "time", return_value=10.0),
            ):
                provider.record(
                    "sglang_chat_completion",
                    {},
                    request,
                    {"choices": [], "model": "test-model"},
                    {"completion_tokens": 2},
                    start_time=1.0,
                    time_of_first_token=None,
                    stream=True,
                )

            self.assertEqual(per_token_recorder.values, [])
            self.assertTrue(span.ended)
        finally:
            mod.Meters.is_metrics_inited = original_metrics_state
            for name, value in original_meters.items():
                setattr(mod.Meters, name, value)


if __name__ == "__main__":
    unittest.main()
