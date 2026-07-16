import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agentrt.config import Settings
from agentrt.runtime import Limits, Run, RunStatus, Runtime
from agentrt.tools import ToolRegistry


class RunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=4000)


class RunSummary(BaseModel):
    id: str
    status: RunStatus


def build_runtime(settings: Settings) -> Runtime:
    from agentrt.builtin_tools import calculator, clock
    from agentrt.openai_model import OpenAICompatibleModel
    from agentrt.rag_tool import make_rag_tool

    tools = ToolRegistry([calculator, clock])
    if settings.rag_url:
        tools.add(make_rag_tool(settings.rag_url))
    model = OpenAICompatibleModel(
        settings.model_base_url, settings.model_name, settings.model_api_key
    )
    limits = Limits(
        max_iterations=settings.max_iterations,
        max_total_tokens=settings.max_total_tokens,
        tool_timeout=settings.tool_timeout,
        tool_retries=settings.tool_retries,
    )
    return Runtime(model, tools, limits)


def create_app(runtime: Runtime | None = None) -> FastAPI:
    runs: dict[str, Run] = {}
    tasks: dict[str, asyncio.Task] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = runtime or build_runtime(Settings())
        yield
        for task in tasks.values():
            task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)

    app = FastAPI(title="agentrt", lifespan=lifespan)

    @app.get("/tools")
    def list_tools() -> list[dict]:
        return app.state.runtime.tools.schemas()

    @app.post("/runs", status_code=202)
    async def start_run(request: RunRequest) -> RunSummary:
        run = Run(task=request.task)
        runs[run.id] = run
        task = asyncio.create_task(app.state.runtime.run(request.task, run))
        tasks[run.id] = task
        task.add_done_callback(lambda _: tasks.pop(run.id, None))
        return RunSummary(id=run.id, status=run.status)

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> Run:
        if run_id not in runs:
            raise HTTPException(404, "run not found")
        return runs[run_id]

    @app.post("/runs/{run_id}/cancel", status_code=202)
    def cancel_run(run_id: str) -> RunSummary:
        if run_id not in runs:
            raise HTTPException(404, "run not found")
        task = tasks.get(run_id)
        if task is None:
            raise HTTPException(409, "run already finished")
        task.cancel()
        return RunSummary(id=run_id, status=runs[run_id].status)

    return app


app = create_app()
