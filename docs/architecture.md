# Architecture

```
POST /runs -> Runtime.run (asyncio task)
                 |
                 v
     model.complete(messages, tool schemas)
                 |
     tool calls? --no--> completed
                 | yes
     validate args -> run tools concurrently (timeout, retries)
                 |
     append results as tool messages, loop (max iterations, token budget)
```

| Module            | Job |
|-------------------|-----|
| `tools.py`        | `Tool` (name, description, input model, async fn, timeout, retries) and `ToolRegistry` |
| `model.py`        | provider-neutral `Message`, `ToolCall`, `ModelResponse`, `Usage`; `Model` protocol; `ScriptedModel` for tests |
| `openai_model.py` | maps those types to and from OpenAI-style `/chat/completions` with httpx |
| `runtime.py`      | the loop, `Limits`, `Run` state and trace events |
| `builtin_tools.py`| calculator and clock |
| `rag_tool.py`     | HTTP adapter to the RAG service |
| `api.py`          | FastAPI: start, inspect and cancel runs, list tools |

## Failure behaviour

| Case | What happens |
|------|--------------|
| Unknown tool, bad JSON, invalid arguments | error JSON sent back to the model, no retry |
| Tool raises or times out | retried up to `retries` with exponential backoff, then error JSON sent back |
| Model call raises | run ends `failed` with the error |
| Too many iterations | run ends `max_iterations` |
| Token budget exceeded | run ends `budget_exceeded` |
| Cancel | asyncio cancellation reaches running tools, run ends `cancelled` |
| Server shutdown | running tasks are cancelled |

Tool errors go back to the model instead of failing the run, so the model can fix its
arguments or answer without the tool. Tool results are passed to the model as data; the system
prompt says not to follow instructions inside them, but that is not a guarantee.
