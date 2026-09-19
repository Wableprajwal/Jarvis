"""The single place that branches on provider type."""
from __future__ import annotations

from app.config import Settings, get_settings
from app.providers.base import ModelProvider


def get_provider(name: str | None = None, model: str | None = None, settings: Settings | None = None) -> ModelProvider:
    settings = settings or get_settings()
    name = (name or settings.model_provider).lower()
    if name == "mock":
        from app.providers.mock_provider import MockProvider

        return MockProvider()
    if name == "ollama":
        from app.providers.ollama_provider import OllamaProvider

        return OllamaProvider(host=settings.ollama_host, model=model or settings.orchestrator_model)
    raise ValueError(f"unknown model provider: {name!r}")
