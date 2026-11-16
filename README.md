# async-agent-runtime

[![Python](https://img.shields.io/badge/python-3.13.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.13.4-E92063)](https://docs.pydantic.dev/)
[![uv](https://img.shields.io/badge/uv-0.12.1-DE5FE9)](https://docs.astral.sh/uv/)
[![pytest](https://img.shields.io/badge/pytest-9.1.1-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A small async runtime for LLM tool calling. It sends a task to a model, runs the tools the
model asks for, feeds the results back and repeats until the model answers. Every step is
recorded in a trace.

This is a single tool-calling loop, not a multi-agent system.

- Tools have Pydantic input and output models; arguments from the model are validated before running.
- Tool calls from the same model turn run concurrently.
- Per-tool timeouts, bounded retries with backoff, and output truncation.
- Limits on iterations and total tokens (when the model reports usage).
- Runs can be cancelled; running tools are cancelled with them.

## Tech stack

- Python 3.13.14
- FastAPI and Uvicorn
- Pydantic and pydantic-settings
- httpx for OpenAI-compatible model requests
- uv for dependency management
- pytest and pytest-asyncio

## Run

Requires Python 3.13.14, [uv](https://docs.astral.sh/uv/) 0.12.1, and an OpenAI-compatible
endpoint with tool calling, e.g. Ollama.

```
uv sync
cp .env.example .env
ollama pull qwen3
uv run uvicorn agentrt.api:app
```

```
curl -H "content-type: application/json" -d '{"task": "What is 1234 * 5678 + 91?"}' localhost:8000/runs
curl localhost:8000/runs/<id>
curl -X POST localhost:8000/runs/<id>/cancel
curl localhost:8000/tools
```

`GET /runs/<id>` returns the status (`running`, `completed`, `failed`, `max_iterations`,
`budget_exceeded`, `cancelled`), the answer, token counts and the trace.

## Tools

- `calculator`: arithmetic only, parsed with `ast` (no `eval`)
- `current_time`: UTC time
- `search_documents`: calls the [document-intelligence-rag](https://github.com/mazighahmed/document-intelligence-rag)
  `/ask` endpoint. Enabled when `RAG_URL` is set.

Adding a tool means writing an async function that takes and returns Pydantic models and
registering a `Tool`.

## Tests

```
uv run pytest
```

Tests use a scripted model and mock HTTP transports, so no model or network is needed.

## Limitations

- Runs are kept in memory and lost on restart; nothing evicts old runs.
- No streaming.
- Token limits only work if the provider returns usage.
- The budget is checked after each model response, so a run can go one response over.
- Tool output is truncated by characters, not tokens.
