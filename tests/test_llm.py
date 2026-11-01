from types import SimpleNamespace

import pytest

from docqa.agents import TOOLS
from docqa.llm import (
    RefusalError,
    TextBlock,
    ToolUseBlock,
    from_chat_completion,
    parse_json,
    to_chat_messages,
    to_chat_tool,
)


def completion(content=None, tool_calls=None, finish_reason="stop"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


def tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def test_tool_schema_is_translated_to_function_format():
    tool = to_chat_tool(TOOLS[0])
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "search_documents"
    assert tool["function"]["parameters"]["required"] == ["query"]


def test_tool_use_round_trip_messages():
    messages = [
        {"role": "user", "content": "What is the leave policy?"},
        {"role": "assistant", "content": [TextBlock("Searching."), ToolUseBlock("call_1", "search_documents", {"query": "leave"})]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "<passage id=\"S1\">24 days</passage>"},
                {"type": "text", "text": "Answer now."},
            ],
        },
    ]
    chat = to_chat_messages("SYSTEM", messages)
    assert [m["role"] for m in chat] == ["system", "user", "assistant", "tool", "user"]
    call = chat[2]["tool_calls"][0]
    assert call["id"] == "call_1" and call["function"]["arguments"] == '{"query": "leave"}'
    assert chat[3]["tool_call_id"] == "call_1"
    assert chat[4]["content"] == "Answer now."


def test_completion_with_tool_calls_becomes_tool_use_blocks():
    response = from_chat_completion(
        completion(tool_calls=[tool_call("c1", "search_documents", '{"query": "travel"}')], finish_reason="tool_calls")
    )
    assert response.stop_reason == "tool_use"
    assert response.content[0].type == "tool_use"
    assert response.content[0].input == {"query": "travel"}


def test_malformed_tool_arguments_become_empty_input():
    response = from_chat_completion(completion(tool_calls=[tool_call("c1", "search_documents", "{not json")]))
    assert response.content[0].input == {}


def test_text_completion_and_json_parsing():
    response = from_chat_completion(completion(content='{"intent": "summary"}'))
    assert response.stop_reason == "end_turn"
    assert parse_json(response) == {"intent": "summary"}


def test_content_filter_raises_refusal():
    with pytest.raises(RefusalError):
        from_chat_completion(completion(finish_reason="content_filter"))
