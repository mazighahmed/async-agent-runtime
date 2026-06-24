import asyncio
import json

from pydantic import BaseModel

from agentrt.model import ModelResponse, ToolCall, Usage
from agentrt.tools import Tool


class SleepIn(BaseModel):
    seconds: float


class EchoOut(BaseModel):
    value: str


async def sleep_fn(args: SleepIn) -> EchoOut:
    await asyncio.sleep(args.seconds)
    return EchoOut(value=f"slept {args.seconds}")


def sleep_tool(**kwargs) -> Tool:
    return Tool("sleep", "Sleep for a while", SleepIn, sleep_fn, **kwargs)


def call(name: str, args: dict | str, id: str = "c1") -> ToolCall:
    arguments = args if isinstance(args, str) else json.dumps(args)
    return ToolCall(id=id, name=name, arguments=arguments)


def tools_response(*calls: ToolCall, tokens: int = 0) -> ModelResponse:
    usage = Usage(input_tokens=tokens, output_tokens=0) if tokens else None
    return ModelResponse(tool_calls=list(calls), usage=usage)


def answer(text: str) -> ModelResponse:
    return ModelResponse(content=text)
