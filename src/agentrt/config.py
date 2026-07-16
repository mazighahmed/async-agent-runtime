from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_base_url: str = "http://localhost:11434/v1"
    model_name: str = "qwen3"
    model_api_key: str = ""
    max_iterations: int = 8
    max_total_tokens: int | None = 20_000
    tool_timeout: float = 10.0
    tool_retries: int = 2
    rag_url: str = ""
