import json

import httpx

from agentrt.model import Message, ToolCall
from agentrt.openai_model import OpenAICompatibleModel, from_openai, to_openai


def test_assistant_tool_call_message_format():
    msg = Message(role="assistant", tool_calls=[ToolCall(id="a", name="f", arguments="{}")])
    assert to_openai(msg)["tool_calls"][0] == {
        "id": "a",
        "type": "function",
        "function": {"name": "f", "arguments": "{}"},
    }
    assert to_openai(Message(role="tool", content="1", tool_call_id="a"))["tool_call_id"] == "a"


def test_parses_tool_calls_and_usage():
    body = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "x", "function": {"name": "add", "arguments": '{"a": 1}'}},
                        {"id": "y", "function": {"name": "add", "arguments": {"a": 2}}},
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    response = from_openai(body)
    assert [c.arguments for c in response.tool_calls] == ['{"a": 1}', '{"a": 2}']
    assert (response.usage.input_tokens, response.usage.output_tokens) == (10, 3)


def test_missing_usage_is_not_invented():
    response = from_openai({"choices": [{"message": {"content": "hi"}}]})
    assert response.usage is None
    assert response.content == "hi"


async def test_sends_tools_and_auth():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})

    model = OpenAICompatibleModel("http://llm.test/v1", "m", api_key="k")
    model.client = httpx.AsyncClient(
        base_url="http://llm.test/v1",
        headers=model.client.headers,
        transport=httpx.MockTransport(handler),
    )
    tools = [{"name": "f", "description": "", "parameters": {"type": "object"}}]
    response = await model.complete([Message(role="user", content="hi")], tools)
    assert response.content == "done"
    assert seen["body"]["tools"][0] == {"type": "function", "function": tools[0]}
    assert seen["auth"] == "Bearer k"
