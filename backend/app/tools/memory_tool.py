"""The explicit "remember that ..." path into long-term memory."""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

from app.harness.memory import Fact, MemoryStore
from app.tools.registry import Tool, ToolResult

# The tool has no session/user parameter (the model must never choose whose memory it writes
# to), so the caller supplies the identity out-of-band. `current_user_var` is a ready-made
# provider: set it once per request, before the orchestrator runs (asyncio tasks inherit it).
current_user_var: ContextVar[str] = ContextVar("jarvis_current_user_id", default="default")


def current_user_id() -> str:
    return current_user_var.get()


UserIdProvider = Callable[[], "str | Awaitable[str]"]


class RememberFactTool(Tool):
    name = "remember_fact"
    description = (
        "Save a durable fact about the user or their project to long-term memory, so it is "
        "available in future sessions. Use when the user says 'remember that ...' or states a "
        "lasting preference. `key` is a short snake_case name (re-using a key updates that fact). "
        "Never store passwords, API keys or tokens."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "minLength": 1, "maxLength": 100, "description": "short snake_case name"},
            "value": {"type": "string", "minLength": 1, "maxLength": 1000, "description": "the fact"},
        },
        "required": ["key", "value"],
        "additionalProperties": False,
    }

    def __init__(self, memory: MemoryStore, user_id_provider: UserIdProvider = current_user_id):
        self.memory = memory
        self._user_id_provider = user_id_provider

    async def execute(self, arguments: dict) -> ToolResult:
        user_id = self._user_id_provider()
        if inspect.isawaitable(user_id):
            user_id = await user_id
        try:
            await self.memory.upsert_long_term_fact(
                user_id, Fact(key=arguments["key"], value=arguments["value"], source="explicit")
            )
        except ValueError as exc:
            return ToolResult.failure(str(exc))
        return ToolResult.success({"remembered": {"key": arguments["key"], "value": arguments["value"]}})
