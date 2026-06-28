import json

import httpx

from agentrt.model import Message, ModelResponse, ToolCall, Usage


def to_openai(message: Message) -> dict:
    data: dict = {"role": message.role, "content": message.content}
    if message.tool_calls:
        data["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in message.tool_calls
        ]
    if message.tool_call_id:
        data["tool_call_id"] = message.tool_call_id
    return data


def from_openai(body: dict) -> ModelResponse:
    message = body["choices"][0]["message"]
    calls = []
    for c in message.get("tool_calls") or []:
        arguments = c["function"].get("arguments") or "{}"
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        calls.append(ToolCall(id=c["id"], name=c["function"]["name"], arguments=arguments))
    usage = body.get("usage")
    return ModelResponse(
        content=message.get("content"),
        tool_calls=calls,
        usage=Usage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
        if usage
        else None,
    )


class OpenAICompatibleModel:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 120.0):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout)
        self.model = model

    async def complete(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        payload: dict = {
            "model": self.model,
            "temperature": 0,
            "messages": [to_openai(m) for m in messages],
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return from_openai(response.json())
