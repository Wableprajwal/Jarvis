"""Model abstraction. Nothing outside app/providers may import a concrete model SDK.

Normalized message format (used everywhere above the provider layer):
  {"role": "system" | "user", "content": str}
  {"role": "assistant", "content": str | None, "tool_calls": [{"id", "name", "arguments"}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str | None = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ModelResponse:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # "end_turn" | "tool_use" | "max_tokens"
    usage: Usage = field(default_factory=Usage)
    # Set when the model's output could not be turned into a valid response (e.g. a malformed
    # tool-call block). The harness feeds this back to the model instead of crashing.
    error: str | None = None


class ModelProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 2048,
    ) -> ModelResponse: ...

    @abstractmethod
    def supports_native_tool_calling(self) -> bool: ...


def assistant_message(resp: ModelResponse) -> dict:
    msg: dict = {"role": "assistant", "content": resp.text}
    if resp.tool_calls:
        msg["tool_calls"] = [
            {"id": c.id, "name": c.name, "arguments": c.arguments} for c in resp.tool_calls
        ]
    return msg
