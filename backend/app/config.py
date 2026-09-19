from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_provider: str = "ollama"  # "ollama" | "mock"
    ollama_host: str = "http://localhost:11434"
    # Per-role models; different agents may use different models.
    orchestrator_model: str = "qwen2.5:7b"
    code_model: str = "qwen2.5:7b"
    research_model: str = "qwen2.5:7b"
    ops_model: str = "qwen2.5:7b"
    judge_model: str = "qwen2.5:7b"
    summarizer_model: str = "qwen2.5:7b"

    database_url: str = "postgresql://jarvis:jarvis@localhost:5433/jarvis"  # docker-compose host port
    trace_backend: str = "postgres"  # "postgres" | "memory"

    max_agent_iterations: int = 8
    workspace_root: str = "."  # root the Code agent may read from
    docs_dir: str | None = None  # notes ingested for RAG at startup (default: <repo>/docs)
    memory_token_budget: int = 3000

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Secrets: only ever read here and injected into tool instances (tools/credentials.py).
    gmail_token: str | None = None
    calendar_token: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
