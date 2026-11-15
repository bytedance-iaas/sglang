import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _tool(name: str) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        ),
    )


def _call(name: str, city: str) -> str:
    return (
        f"<tool_call>{name}"
        f"<arg_key>city</arg_key><arg_value>{city}</arg_value>"
        "</tool_call>"
    )


def _collect(items: list[ToolCallItem]) -> list[tuple[str, dict]]:
    names: dict[int, str] = {}
    arguments: dict[int, str] = {}
    for item in items:
        if item.name is not None:
            names[item.tool_index] = item.name
        arguments[item.tool_index] = (
            arguments.get(item.tool_index, "") + item.parameters
        )
    return [(names[index], json.loads(arguments[index])) for index in sorted(names)]


def _parser() -> FunctionCallParser:
    return FunctionCallParser(
        tools=[_tool("weather"), _tool("forecast")],
        tool_call_parser="glm47",
    )


def test_streaming_single_call_and_end_do_not_duplicate() -> None:
    parser = _parser()
    normal, items = parser.parse_stream_chunk(_call("weather", "Beijing"))
    end_normal, end_items = parser.parse_stream_end()

    assert normal == end_normal == ""
    assert end_items == []
    assert _collect(items) == [("weather", {"city": "Beijing"})]


def test_terminal_delta_drains_two_complete_calls() -> None:
    parser = _parser()
    delta = _call("weather", "Beijing") + _call("forecast", "Shanghai")
    normal, items = parser.parse_stream_chunk(delta)
    end_normal, end_items = parser.parse_stream_end()

    assert normal == end_normal == ""
    assert end_items == []
    assert _collect(items) == [
        ("weather", {"city": "Beijing"}),
        ("forecast", {"city": "Shanghai"}),
    ]


def test_split_delta_finishes_first_and_drains_complete_second_call() -> None:
    parser = _parser()
    first = "prefix:" + _call("weather", "Beijing")[:-7]
    normal_1, items_1 = parser.parse_stream_chunk(first)
    normal_2, items_2 = parser.parse_stream_chunk(
        "l_call>" + _call("forecast", "Shanghai")
    )
    end_normal, end_items = parser.parse_stream_end()

    assert normal_1 + normal_2 + end_normal == "prefix:"
    assert end_items == []
    assert _collect(items_1 + items_2) == [
        ("weather", {"city": "Beijing"}),
        ("forecast", {"city": "Shanghai"}),
    ]


def test_complete_first_and_partial_second_waits_for_next_delta() -> None:
    parser = _parser()
    second = _call("forecast", "Shanghai")
    cut = second.index("Shanghai") + 3
    normal_1, items_1 = parser.parse_stream_chunk(
        _call("weather", "Beijing") + second[:cut]
    )
    normal_2, items_2 = parser.parse_stream_chunk(second[cut:])

    assert normal_1 + normal_2 == ""
    assert _collect(items_1 + items_2) == [
        ("weather", {"city": "Beijing"}),
        ("forecast", {"city": "Shanghai"}),
    ]
