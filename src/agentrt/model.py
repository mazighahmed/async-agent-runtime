from typing import Literal, Protocol

from pydantic import BaseModel


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = []
    tool_call_id: str | None = None


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class ModelResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = []
    usage: Usage | None = None


class Model(Protocol):
    async def complete(self, messages: list[Message], tools: list[dict]) -> ModelResponse: ...


class ScriptedModel:
    """Returns prepared responses in order. Used in tests and offline demos."""

    def __init__(self, responses: list[ModelResponse]):
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    async def complete(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        self.calls.append(list(messages))
        if not self.responses:
            raise RuntimeError("scripted model ran out of responses")
        return self.responses.pop(0)
