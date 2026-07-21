import time

from conftest import answer, call, sleep_tool, tools_response
from fastapi.testclient import TestClient

from agentrt.api import create_app
from agentrt.model import ScriptedModel
from agentrt.runtime import Runtime
from agentrt.tools import ToolRegistry


def client_for(responses) -> TestClient:
    runtime = Runtime(ScriptedModel(responses), ToolRegistry([sleep_tool()]))
    return TestClient(create_app(runtime))


def wait_for(client: TestClient, run_id: str, status: str) -> dict:
    for _ in range(100):
        body = client.get(f"/runs/{run_id}").json()
        if body["status"] == status:
            return body
        time.sleep(0.02)
    raise AssertionError(f"run never reached {status}: {body['status']}")


def test_run_completes_with_trace():
    responses = [tools_response(call("sleep", {"seconds": 0})), answer("all done")]
    with client_for(responses) as client:
        r = client.post("/runs", json={"task": "do it"})
        assert r.status_code == 202
        body = wait_for(client, r.json()["id"], "completed")
    assert body["answer"] == "all done"
    types = [e["type"] for e in body["trace"]]
    assert types[0] == "run_started" and types[-1] == "run_finished"
    assert "tool_finished" in types


def test_cancel_running_run():
    responses = [tools_response(call("sleep", {"seconds": 5})), answer("never")]
    with client_for(responses) as client:
        run_id = client.post("/runs", json={"task": "slow"}).json()["id"]
        wait_for(client, run_id, "running")
        time.sleep(0.05)
        assert client.post(f"/runs/{run_id}/cancel").status_code == 202
        wait_for(client, run_id, "cancelled")
        assert client.post(f"/runs/{run_id}/cancel").status_code == 409


def test_unknown_run():
    with client_for([]) as client:
        assert client.get("/runs/nope").status_code == 404
        assert client.post("/runs/nope/cancel").status_code == 404


def test_rejects_empty_task():
    with client_for([]) as client:
        assert client.post("/runs", json={"task": ""}).status_code == 422


def test_lists_tool_schemas():
    with client_for([]) as client:
        tools = client.get("/tools").json()
    assert tools[0]["name"] == "sleep"
    assert tools[0]["parameters"]["properties"]["seconds"]["type"] == "number"
