"""Deterministic provider so the harness, orchestrator, memory and eval layers are
testable with no live model."""
from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

from app.providers.base import ModelProvider, ModelResponse, ToolCall, Usage

Matcher = Callable[[list[dict]], bool]


class MockScriptExhausted(RuntimeError):
    pass


def text_response(text: str, tokens: tuple[int, int] = (10, 5)) -> ModelResponse:
    return ModelResponse(text=text, tool_calls=[], stop_reason="end_turn", usage=Usage(*tokens))


def tool_call_response(
    name: str, arguments: dict | None = None, *, text: str | None = None, call_id: str | None = None
) -> ModelResponse:
    return multi_tool_call_response([(name, arguments or {})], text=text, call_id_prefix=call_id)


def multi_tool_call_response(
    calls: list[tuple[str, dict]], *, text: str | None = None, call_id_prefix: str | None = None
) -> ModelResponse:
    prefix = call_id_prefix or "call"
    return ModelResponse(
        text=text,
        tool_calls=[ToolCall(name=n, arguments=a, id=f"{prefix}_{i}") for i, (n, a) in enumerate(calls)],
        stop_reason="tool_use",
        usage=Usage(10, 5),
    )


@dataclass
class Rule:
    """If `match(messages)` is true, reply with `response` (a ModelResponse or a callable
    building one from the messages)."""

    match: Matcher
    response: ModelResponse | Callable[[list[dict]], ModelResponse]

    @staticmethod
    def last_user_contains(needle: str) -> Matcher:
        def _m(messages: list[dict]) -> bool:
            for m in reversed(messages):
                if m["role"] == "user":
                    return needle.lower() in (m.get("content") or "").lower()
            return False

        return _m

    @staticmethod
    def last_is_tool_result() -> Matcher:
        return lambda messages: bool(messages) and messages[-1]["role"] == "tool"


class MockProvider(ModelProvider):
    """Scripted responses are consumed in order; once the script is empty (or if none was
    given) rules are tried in order; then `default`; otherwise MockScriptExhausted."""

    def __init__(
        self,
        script: list[ModelResponse | Callable[[list[dict]], ModelResponse]] | None = None,
        rules: list[Rule] | None = None,
        default: ModelResponse | None = None,
        native_tool_calling: bool = True,
    ):
        self._script = list(script or [])
        self._rules = list(rules or [])
        self._default = default
        self._native = native_tool_calling
        self.calls: list[dict] = []  # recorded {"messages", "tools", "max_tokens"} per generate()

    async def generate(self, messages: list[dict], tools: list[dict], max_tokens: int = 2048) -> ModelResponse:
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools, "max_tokens": max_tokens})
        if self._script:
            item = self._script.pop(0)
            return item(messages) if callable(item) else item
        for rule in self._rules:
            if rule.match(messages):
                return rule.response(messages) if callable(rule.response) else rule.response
        if self._default is not None:
            return self._default
        raise MockScriptExhausted(
            f"MockProvider has no scripted response for call #{len(self.calls)}"
        )

    def supports_native_tool_calling(self) -> bool:
        return self._native
