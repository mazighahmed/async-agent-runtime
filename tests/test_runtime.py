import asyncio
import json
import time

import pytest
from conftest import EchoOut, SleepIn, answer, call, sleep_tool, tools_response

from agentrt.model import ScriptedModel
from agentrt.runtime import Limits, Run, RunStatus, Runtime
from agentrt.tools import Tool, ToolRegistry


def runtime(responses, tools=None, **limits) -> tuple[Runtime, ScriptedModel]:
    model = ScriptedModel(responses)
    limits.setdefault("retry_backoff", 0)
    return Runtime(model, ToolRegistry(tools or [sleep_tool()]), Limits(**limits)), model


def tool_messages(model: ScriptedModel, turn: int) -> list[dict]:
    return [json.loads(m.content) for m in model.calls[turn] if m.role == "tool"]


async def test_answer_without_tools():
    rt, _ = runtime([answer("hello")])
    run = await rt.run("hi")
    assert (run.status, run.answer, run.iterations) == (RunStatus.COMPLETED, "hello", 1)


async def test_tool_result_is_sent_back_to_model():
    rt, model = runtime([tools_response(call("sleep", {"seconds": 0})), answer("ok")])
    run = await rt.run("go")
    assert run.status == RunStatus.COMPLETED
    tool_msg = [m for m in model.calls[1] if m.role == "tool"][0]
    assert tool_msg.tool_call_id == "c1"
    assert json.loads(tool_msg.content) == {"value": "slept 0.0"}


async def test_independent_tool_calls_run_concurrently():
    calls = [call("sleep", {"seconds": 0.2}, id=str(i)) for i in range(5)]
    rt, model = runtime([tools_response(*calls), answer("done")])
    start = time.perf_counter()
    run = await rt.run("go")
    assert time.perf_counter() - start < 0.6
    assert run.status == RunStatus.COMPLETED
    assert [m.tool_call_id for m in model.calls[1] if m.role == "tool"] == ["0", "1", "2", "3", "4"]


async def test_malformed_json_arguments_go_back_as_error():
    rt, model = runtime([tools_response(call("sleep", "{not json")), answer("sorry")])
    run = await rt.run("go")
    assert run.status == RunStatus.COMPLETED
    assert "invalid arguments" in tool_messages(model, 1)[0]["error"]


async def test_wrong_argument_types_are_rejected():
    rt, model = runtime([tools_response(call("sleep", {"seconds": "soon"})), answer("x")])
    await rt.run("go")
    assert "invalid arguments" in tool_messages(model, 1)[0]["error"]


async def test_unknown_tool():
    rt, model = runtime([tools_response(call("rm_rf", {})), answer("x")])
    await rt.run("go")
    assert tool_messages(model, 1)[0] == {"error": "unknown tool: rm_rf"}


async def test_failing_tool_is_retried_until_success():
    attempts = []

    async def flaky(args: SleepIn) -> EchoOut:
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("boom")
        return EchoOut(value="ok")

    tool = Tool("flaky", "", SleepIn, flaky)
    rt, model = runtime(
        [tools_response(call("flaky", {"seconds": 0})), answer("x")], [tool], tool_retries=2
    )
    run = await rt.run("go")
    assert len(attempts) == 3
    assert tool_messages(model, 1)[0] == {"value": "ok"}
    assert [e.type for e in run.trace].count("tool_error") == 2


async def test_tool_failure_after_retries_is_reported():
    async def broken(args: SleepIn) -> EchoOut:
        raise ValueError("bad input")

    tool = Tool("broken", "", SleepIn, broken, retries=1)
    rt, model = runtime([tools_response(call("broken", {"seconds": 0})), answer("x")], [tool])
    run = await rt.run("go")
    assert run.status == RunStatus.COMPLETED
    assert tool_messages(model, 1)[0] == {"error": "ValueError: bad input"}


async def test_tool_timeout():
    tool = sleep_tool(timeout=0.05, retries=0)
    rt, model = runtime([tools_response(call("sleep", {"seconds": 5})), answer("x")], [tool])
    start = time.perf_counter()
    await rt.run("go")
    assert time.perf_counter() - start < 1
    assert tool_messages(model, 1)[0] == {"error": "timed out after 0.05s"}


async def test_tool_output_is_truncated():
    rt, model = runtime(
        [tools_response(call("sleep", {"seconds": 0})), answer("x")], max_tool_output_chars=5
    )
    await rt.run("go")
    tool_msg = [m for m in model.calls[1] if m.role == "tool"][0]
    assert len(tool_msg.content) == 5


async def test_stops_at_max_iterations():
    responses = [tools_response(call("sleep", {"seconds": 0})) for _ in range(3)]
    rt, _ = runtime(responses, max_iterations=3)
    run = await rt.run("go")
    assert (run.status, run.iterations) == (RunStatus.MAX_ITERATIONS, 3)


async def test_stops_when_token_budget_is_exceeded():
    responses = [tools_response(call("sleep", {"seconds": 0}), tokens=600) for _ in range(3)]
    rt, _ = runtime(responses, max_total_tokens=1000)
    run = await rt.run("go")
    assert run.status == RunStatus.BUDGET_EXCEEDED
    assert run.iterations == 2
    assert run.input_tokens == 1200


async def test_model_error_fails_the_run():
    rt, _ = runtime([])
    run = await rt.run("go")
    assert run.status == RunStatus.FAILED
    assert "ran out of responses" in run.error


async def test_cancellation_stops_running_tools():
    finished = []

    async def slow(args: SleepIn) -> EchoOut:
        await asyncio.sleep(args.seconds)
        finished.append(1)
        return EchoOut(value="late")

    tool = Tool("slow", "", SleepIn, slow)
    rt, _ = runtime([tools_response(call("slow", {"seconds": 5})), answer("x")], [tool])

    run = Run(task="go")
    task = asyncio.create_task(rt.run("go", run))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert run.status == RunStatus.CANCELLED
    assert finished == []
