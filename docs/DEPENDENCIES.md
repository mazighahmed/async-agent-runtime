# Dependencies

Checked against PyPI, python.org and GitHub release pages. Cutoff: 2026-07-31.
`exclude-newer` in `pyproject.toml` keeps transitive packages under the same cutoff.

## Runtime

| Package           | Version | Released   |
|-------------------|---------|------------|
| Python            | 3.13.14 | 2026-06-10 |
| fastapi           | 0.141.1 | 2026-07-29 |
| uvicorn           | 0.52.0  | 2026-07-29 |
| pydantic          | 2.13.4  | 2026-05-06 |
| pydantic-settings | 2.14.2  | 2026-06-19 |
| httpx             | 0.28.1  | 2024-12-06 |

## Dev and tooling

| Tool               | Version | Released   |
|--------------------|---------|------------|
| uv                 | 0.12.1  | 2026-07-31 |
| hatchling          | 1.31.0  | 2026-07-08 |
| pytest             | 9.1.1   | 2026-06-19 |
| pytest-asyncio     | 1.4.0   | 2026-05-26 |
| ruff               | 0.16.1  | 2026-07-30 |
| actions/checkout   | v7.0.1  | 2026-07-20 |
| astral-sh/setup-uv | v9.0.0  | 2026-07-21 |

## Models

Any OpenAI-compatible endpoint with tool calling. The example config uses `qwen3`
through Ollama (released 2025-04).
