import json
import logging
import os
import time
from collections.abc import Mapping
from enum import Enum
from importlib.metadata import version
from typing import Optional

TRACE_HEADERS = ["traceparent", "tracestate"]
LLM_USAGE_TOKEN_TYPES = ["prompt_tokens", "completion_tokens", "total_tokens"]

logger = logging.getLogger(__name__)

_is_otel_imported = False
otel_import_error_traceback: Optional[str] = None
tracer = None
meter = None
try:
    from opentelemetry.context import get_current
    from opentelemetry.context.context import Context
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
    from opentelemetry.metrics import Meter, get_meter, set_meter_provider
    from opentelemetry.sdk.environment_variables import (
        OTEL_EXPORTER_OTLP_METRICS_PROTOCOL,
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        MetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import (
        INVALID_SPAN,
        SpanKind,
        Tracer,
        get_current_span,
        get_tracer,
        set_tracer_provider,
    )
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )
    from opentelemetry.trace.status import Status, StatusCode

    _is_otel_imported = True
except ImportError:
    # Capture and format traceback to provide detailed context for the import
    # error. Only the string representation of the error is retained to avoid
    # memory leaks.
    import traceback

    otel_import_error_traceback = traceback.format_exc()

    class Context:  # type: ignore
        pass

    class BaseSpanAttributes:  # type: ignore
        pass

    class SpanKind:  # type: ignore
        pass

    class Tracer:  # type: ignore
        pass

    class Meter:  # type: ignore
        pass

    class Status:  # type: ignore
        pass

    class StatusCode:  # type: ignore
        pass

    class TraceContextTextMapPropagator:  # type: ignore
        pass


def is_otel_available() -> bool:
    return _is_otel_imported


def init_tracer(instrumenting_module_name: str) -> Optional[Tracer]:
    if not is_otel_available():
        raise ValueError(
            "OpenTelemetry is not available. Unable to initialize "
            "a tracer. Ensure OpenTelemetry packages are installed. "
            f"Original error:\n{otel_import_error_traceback}"
        )
    trace_provider = TracerProvider()

    span_exporter = get_span_exporter()
    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    set_tracer_provider(trace_provider)

    LoggingInstrumentor().instrument()

    tracer = get_tracer(instrumenting_module_name)
    return tracer


def get_span_exporter():
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "grpc")
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    elif protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,  # type: ignore
        )
    else:
        raise ValueError(f"Unsupported OTLP protocol '{protocol}' is configured")

    return OTLPSpanExporter()


def init_metrics(instrumenting_module_name: str) -> Optional[Meter]:
    if not is_otel_available():
        raise ValueError(
            "OpenTelemetry is not available. Unable to initialize "
            "a meter. Ensure OpenTelemetry packages are installed. "
            f"Original error:\n{otel_import_error_traceback}"
        )
    metric_exporter = get_metrics_exporter()
    reader = PeriodicExportingMetricReader(
        metric_exporter, export_interval_millis=30_000
    )
    metrics_provider = MeterProvider(metric_readers=[reader])
    set_meter_provider(metrics_provider)

    meter = get_meter(instrumenting_module_name)

    init_genai_metrics(meter)
    SystemMetricsInstrumentor().instrument()
    return meter


def get_metrics_exporter():
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_METRICS_PROTOCOL, "grpc")
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
    elif protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,  # type: ignore
        )
    else:
        raise ValueError(f"Unsupported OTLP protocol '{protocol}' is configured")

    return OTLPMetricExporter()


def extract_trace_context(headers: Optional[Mapping[str, str]]) -> Optional[Context]:
    if is_otel_available():
        if get_current_span() is INVALID_SPAN:
            headers = headers or {}
            return TraceContextTextMapPropagator().extract(headers)
        else:
            return get_current()
    else:
        return None


def extract_trace_headers(headers: Mapping[str, str]) -> Mapping[str, str]:

    return {h: headers[h] for h in TRACE_HEADERS if h in headers}


class Meters:
    LLM_GENERATION_CHOICES = "gen_ai.client.generation.choices"
    LLM_TOKEN_USAGE = "gen_ai.client.token.usage"
    LLM_OPERATION_DURATION = "gen_ai.client.operation.duration"
    LLM_COMPLETIONS_EXCEPTIONS = "gen_ai.chat_completions.exceptions"
    LLM_STREAMING_TIME_TO_FIRST_TOKEN = (
        "gen_ai.chat_completions.streaming_time_to_first_token"
    )
    LLM_STREAMING_TIME_TO_GENERATE = (
        "gen_ai.chat_completions.streaming_time_to_generate"
    )
    LLM_STREAMING_TIME_PER_OUTPUT_TOKEN = (
        "gen_ai.chat_completions.streaming_time_per_output_token"
    )
    LLM_CHAT_COUNT = "gen_ai.chat.count"

    LLM_EMBEDDINGS_EXCEPTIONS = "gen_ai.embeddings.exceptions"
    LLM_EMBEDDINGS_VECTOR_SIZE = "gen_ai.embeddings.vector_size"
    LLM_IMAGE_GENERATIONS_EXCEPTIONS = "gen_ai.image_generations.exceptions"
    LLM_ANTHROPIC_COMPLETION_EXCEPTIONS = "gen_ai.anthropic.completion.exceptions"

    PINECONE_DB_QUERY_DURATION = "db.pinecone.query.duration"
    PINECONE_DB_QUERY_SCORES = "db.pinecone.query.scores"
    PINECONE_DB_USAGE_READ_UNITS = "db.pinecone.usage.read_units"
    PINECONE_DB_USAGE_WRITE_UNITS = "db.pinecone.usage_write_units"

    LLM_WATSONX_COMPLETIONS_DURATION = "llm.watsonx.completions.duration"
    LLM_WATSONX_COMPLETIONS_EXCEPTIONS = "llm.watsonx.completions.exceptions"
    LLM_WATSONX_COMPLETIONS_RESPONSES = "llm.watsonx.completions.responses"
    LLM_WATSONX_COMPLETIONS_TOKENS = "llm.watsonx.completions.tokens"

    is_metrics_inited = False

    chat_counter = None
    tokens_histogram = None
    chat_choice_counter = None
    chat_duration_histogram = None
    chat_exception_counter = None
    streaming_time_to_first_token = None
    streaming_time_to_generate = None
    streaming_time_per_output_token = None


def init_genai_metrics(meter: Meter) -> None:
    if Meters.is_metrics_inited:
        return
    try:
        Meters.chat_counter = meter.create_counter(
            name=Meters.LLM_CHAT_COUNT,
            unit="time",
            description="Number of chat completions call",
        )
        Meters.tokens_histogram = meter.create_histogram(
            name=Meters.LLM_TOKEN_USAGE,
            unit="token",
            description="Measures number of input and output tokens used",
        )
        # Meters.chat_token_recoder = meter.create_observable_counter()
        Meters.chat_choice_counter = meter.create_counter(
            name=Meters.LLM_GENERATION_CHOICES,
            unit="choice",
            description="Number of choices returned by chat completions call",
        )

        Meters.chat_duration_histogram = meter.create_histogram(
            name=Meters.LLM_OPERATION_DURATION,
            unit="s",
            description="GenAI operation duration",
        )

        Meters.chat_exception_counter = meter.create_counter(
            name=Meters.LLM_COMPLETIONS_EXCEPTIONS,
            unit="time",
            description="Number of exceptions occurred during chat completions",
        )

        Meters.streaming_time_to_first_token = meter.create_histogram(
            name=Meters.LLM_STREAMING_TIME_TO_FIRST_TOKEN,
            unit="s",
            description="Time to first token in streaming chat completions",
        )
        Meters.streaming_time_to_generate = meter.create_histogram(
            name=Meters.LLM_STREAMING_TIME_TO_GENERATE,
            unit="s",
            description="Time between first token and completion in streaming chat completions",
        )
        Meters.streaming_time_per_output_token = meter.create_histogram(
            name=Meters.LLM_STREAMING_TIME_PER_OUTPUT_TOKEN,
            unit="s",
            description="Time per output token in streaming chat completions",
        )
        Meters.is_metrics_inited = True
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning("Failed to init genai metrics, error: %s", str(ex))


def set_choice_counter_metrics(choices, shared_attributes):
    if Meters.is_metrics_inited:
        for choice in choices:
            if not isinstance(choice, dict):
                choice = choice.__dict__
            if choice.get("finish_reason"):
                attributes_with_reason = {
                    **shared_attributes,
                    SpanAttributes.GEN_AI_RESPONSE_FINISH_REASON: choice.get(
                        "finish_reason"
                    ),
                }
            else:
                attributes_with_reason = shared_attributes
            Meters.chat_choice_counter.add(1, attributes=attributes_with_reason)


def set_token_counter_metrics(usage, shared_attributes):
    if Meters.is_metrics_inited:
        for name, val in usage.items():
            if name in LLM_USAGE_TOKEN_TYPES:
                attributes_with_token_type = {
                    **shared_attributes,
                    SpanAttributes.GEN_AI_TOKEN_TYPE: _token_type(name),
                }
                Meters.tokens_histogram.record(
                    val, attributes=attributes_with_token_type
                )


def metric_shared_attributes(
    response_model: str, operation: str, is_streaming: bool = False
):
    return {
        SpanAttributes.GEN_AI_SYSTEM: "dynamo",
        SpanAttributes.GEN_AI_RESPONSE_MODEL: response_model,
        "gen_ai_operation_name": operation,
        "stream": is_streaming,
    }


def _token_type(token_type: str):
    if token_type == "prompt_tokens":
        return "input"
    elif token_type == "completion_tokens":
        return "output"
    elif token_type == "total_tokens":
        return "total"

    return None


class SpanAttributes:
    # Attribute names copied from here to avoid version conflicts:
    # https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md
    GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    GEN_AI_SYSTEM = "gen_ai.system"
    GEN_AI_PROMPTS = "gen_ai.prompt"
    GEN_AI_COMPLETIONS = "gen_ai.completion"
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = (
        "gen_ai.usage.cache_creation_input_tokens"
    )
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read_input_tokens"
    GEN_AI_TOKEN_TYPE = "gen_ai.token.type"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_USER = "gen_ai.user"
    GEN_AI_HEADERS = "gen_ai.headers"
    GEN_AI_TOP_K = "gen_ai.top_k"
    GEN_AI_IS_STREAMING = "gen_ai.is_streaming"
    GEN_AI_FREQUENCY_PENALTY = "gen_ai.frequency_penalty"
    GEN_AI_PRESENCE_PENALTY = "gen_ai.presence_penalty"
    GEN_AI_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reason"
    # Attribute names added until they are added to the semantic conventions:
    GEN_AI_REQUEST_ID = "gen_ai.request.id"
    GEN_AI_REQUEST_N = "gen_ai.request.n"
    GEN_AI_USAGE_NUM_SEQUENCES = "gen_ai.usage.num_sequences"
    GEN_AI_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    GEN_AI_LATENCY_E2E = "gen_ai.latency.e2e"
    GEN_AI_LATENCY_TIME_IN_SCHEDULER = "gen_ai.latency.time_in_scheduler"
    # Time taken in the forward pass for this across all workers
    GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD = "gen_ai.latency.time_in_model_forward"
    # Time taken in the model execute function. This will include model
    # forward, block/sync across workers, cpu-gpu sync time and sampling time.
    GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE = "gen_ai.latency.time_in_model_execute"
    GEN_AI_REQUEST_TYPE = "gen_ai.request.type"
    # TTFT TPOP span
    GEN_AI_STREAMING_TIME_TO_FIRST_TOKEN = (
        "gen_ai.chat_completions.streaming_time_to_first_token"
    )
    GEN_AI_STREAMING_TIME_PER_OUTPUT_TOKEN = (
        "gen_ai.chat_completions.streaming_time_per_output_token"
    )


class LLMRequestTypeValues(Enum):
    COMPLETION = "completion"
    CHAT = "chat"
    RERANK = "rerank"
    EMBEDDING = "embedding"
    UNKNOWN = "unknown"


def contains_trace_headers(headers: Mapping[str, str]) -> bool:
    return any(h in headers for h in TRACE_HEADERS)


def log_tracing_disabled_warning() -> None:
    logger.warning("Received a request with trace context but tracing is disabled")


def set_prompts(span, messages):
    if not span.is_recording() or messages is None:
        return

    try:
        for i, msg in enumerate(messages):
            prefix = f"{SpanAttributes.GEN_AI_PROMPTS}.{i}"
            content = ""
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                content = json.dumps(msg.content)

            _set_span_attribute(span, f"{prefix}.role", msg.role)
            _set_span_attribute(span, f"{prefix}.content", content)
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning("Failed to set prompts for openai span, error: %s", str(ex))


def set_request_attributes(span, raw_request):
    if not span.is_recording():
        return

    try:
        # _set_api_attributes(span)
        _set_span_attribute(span, SpanAttributes.GEN_AI_SYSTEM, "sglang")
        _set_span_attribute(
            span, SpanAttributes.GEN_AI_REQUEST_MODEL, raw_request.model
        )
        _set_span_attribute(
            span, SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS, raw_request.max_tokens
        )
        _set_span_attribute(
            span, SpanAttributes.GEN_AI_REQUEST_TEMPERATURE, raw_request.temperature
        )
        _set_span_attribute(
            span, SpanAttributes.GEN_AI_REQUEST_TOP_P, raw_request.top_p
        )
        _set_span_attribute(
            span, SpanAttributes.GEN_AI_FREQUENCY_PENALTY, raw_request.frequency_penalty
        )
        _set_span_attribute(
            span, SpanAttributes.GEN_AI_PRESENCE_PENALTY, raw_request.presence_penalty
        )
        _set_span_attribute(span, SpanAttributes.GEN_AI_USER, raw_request.user)
        _set_span_attribute(
            span, SpanAttributes.GEN_AI_IS_STREAMING, raw_request.stream or False
        )
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning(
            "Failed to set input attributes for request span, error: %s", str(ex)
        )


def set_completions(span, choices):
    if choices is None:
        return

    for choice in choices:
        choice = model_as_dict(choice)
        index = choice.get("index")
        prefix = f"{SpanAttributes.GEN_AI_COMPLETIONS}.{index}"
        _set_span_attribute(
            span, f"{prefix}.finish_reason", choice.get("finish_reason")
        )

        message = choice.get("message")
        if not message:
            return

        _set_span_attribute(span, f"{prefix}.role", message.get("role"))
        _set_span_attribute(span, f"{prefix}.content", message.get("content"))

        function_call = message.get("function_call")
        if function_call:
            _set_span_attribute(
                span, f"{prefix}.function_call.name", function_call.get("name")
            )
            _set_span_attribute(
                span,
                f"{prefix}.function_call.arguments",
                function_call.get("arguments"),
            )

        tool_calls = message.get("tool_calls")
        if tool_calls:
            _set_span_attribute(
                span,
                f"{prefix}.function_call.name",
                tool_calls[0].get("function").get("name"),
            )
            _set_span_attribute(
                span,
                f"{prefix}.function_call.arguments",
                tool_calls[0].get("function").get("arguments"),
            )


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def set_response_attributes(span, response, usage):
    if not span.is_recording():
        return

    try:
        if not isinstance(response, dict):
            response = response.__dict__

        _set_span_attribute(
            span, SpanAttributes.GEN_AI_RESPONSE_MODEL, response.get("model")
        )

        if not usage:
            return

        if not isinstance(usage, dict):
            usage = usage.__dict__

        _set_span_attribute(
            span, SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS, usage.get("total_tokens")
        )
        _set_span_attribute(
            span,
            SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS,
            usage.get("completion_tokens"),
        )
        _set_span_attribute(
            span, SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS, usage.get("prompt_tokens")
        )

        return
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning(
            "Failed to set response attributes for response span, error: %s", str(ex)
        )


def should_send_prompts():
    return (os.getenv("TRACE_CONTENT") or "true").lower() == "true"


def model_as_dict(model):
    if version("pydantic") < "2.0.0":
        return model.dict()
    if hasattr(model, "model_dump"):
        return model.model_dump()
    elif hasattr(model, "parse"):  # Raw API response
        return model_as_dict(model.parse())
    else:
        return model


def accumulate_stream_items(item, complete_response):
    item = model_as_dict(item)
    complete_response["model"] = item.get("model")

    if item.get("error"):
        complete_response["error"] = item.get("error")

    if item.get("usage"):
        complete_response["usage"] = item.get("usage")

    if item.get("choices"):
        for choice in item.get("choices"):
            index = choice.get("index")
            if len(complete_response.get("choices")) <= index:
                complete_response["choices"].append(
                    {"index": index, "message": {"content": "", "role": ""}}
                )
            complete_choice = complete_response.get("choices")[index]
            if choice.get("finish_reason"):
                complete_choice["finish_reason"] = choice.get("finish_reason")

            delta = choice.get("delta")

            if delta.get("content"):
                complete_choice["message"]["content"] += delta.get("content")
            if delta.get("role"):
                complete_choice["message"]["role"] = delta.get("role")


class OpenTelemetryProvider:
    def __init__(self):
        self.tracer = None
        self.meter = None
        if is_otel_available():
            self.tracer = init_tracer("sglang")
            self.meter = init_metrics("sglang")

    def recordException(self, name, headers, request, exception: Exception):
        if is_otel_available():
            trace_context = extract_trace_context(headers)
            span = self.tracer.start_span(
                name=name,
                kind=SpanKind.SERVER,
                context=trace_context,
                attributes={
                    SpanAttributes.GEN_AI_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value
                },
            )
            set_request_attributes(span, request)
            if should_send_prompts():
                set_prompts(span, request.messages)
            span.set_attribute(SpanAttributes.GEN_AI_RESPONSE_MODEL, request.model)
            span.set_status(
                Status(status_code=StatusCode.ERROR, description=str(exception))
            )
            span.end()

    def record(
        self,
        name,
        headers,
        request,
        response,
        usage,
        start_time,
        time_of_first_token=None,
        stream=False,
    ):
        if is_otel_available():
            trace_context = extract_trace_context(headers)
            span = self.tracer.start_span(
                name=name,
                kind=SpanKind.SERVER,
                context=trace_context,
                attributes={
                    SpanAttributes.GEN_AI_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value
                },
            )
            set_request_attributes(span, request)
            if should_send_prompts():
                set_prompts(span, request.messages)
            span.set_attribute(SpanAttributes.GEN_AI_RESPONSE_MODEL, request.model)
            shared_attributes = metric_shared_attributes(
                response_model=request.model or None,
                operation="chat",
                is_streaming=request.stream,
            )
            if stream:
                choices = response.get("choices")
            else:
                choices = response.choices
            if Meters.is_metrics_inited:
                Meters.chat_counter.add(1, attributes=shared_attributes)
            if choices:
                set_choice_counter_metrics(choices, shared_attributes)
            # token metrics
            if usage and not isinstance(usage, dict):
                usage = usage.__dict__
            if usage:
                set_token_counter_metrics(usage, shared_attributes)

            # duration metrics
            if start_time and isinstance(start_time, (float, int)):
                duration = time.time() - start_time
            else:
                duration = None
            if (
                duration
                and isinstance(duration, (float, int))
                and Meters.is_metrics_inited
            ):
                Meters.chat_duration_histogram.record(
                    duration, attributes=shared_attributes
                )
            if Meters.is_metrics_inited and stream:
                if (
                    time_of_first_token
                    and isinstance(time_of_first_token, (float, int))
                    and time_of_first_token > start_time
                ):
                    Meters.streaming_time_to_first_token.record(
                        (time_of_first_token - start_time), attributes=shared_attributes
                    )
                    Meters.streaming_time_to_generate.record(
                        time.time() - time_of_first_token, attributes=shared_attributes
                    )

                if usage and usage.get("completion_tokens"):
                    if not isinstance(usage, dict):
                        usage = usage.__dict__
                    completion_tokens = usage.get("completion_tokens")
                    if Meters.is_metrics_inited and request.stream:
                        Meters.streaming_time_per_output_token.record(
                            (time.time() - time_of_first_token) / completion_tokens,
                            attributes=shared_attributes,
                        )
                    span.set_attribute(
                        SpanAttributes.GEN_AI_STREAMING_TIME_PER_OUTPUT_TOKEN,
                        (time.time() - time_of_first_token) / completion_tokens,
                    )

            set_response_attributes(span, response, usage)

            if should_send_prompts():
                set_completions(span, choices)

            span.set_status(Status(StatusCode.OK))
            span.end()


otel_provider = OpenTelemetryProvider()
