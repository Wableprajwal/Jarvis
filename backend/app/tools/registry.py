"""Tool contract, per-agent allow-lists, argument validation, confirmation gating.

Capability isolation: an agent can only obtain a tool through `registry.get(agent, tool)`,
which raises unless that agent was explicitly granted the tool. A routing bug (or a model
hallucinating a tool name) therefore cannot expose e.g. `email_send` to the Code agent.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

MAX_TOOL_OUTPUT_CHARS = 8000


class ToolError(Exception):
    pass


class UnknownTool(ToolError):
    pass


class ToolNotPermitted(ToolError):
    pass


@dataclass
class ToolResult:
    ok: bool
    output: Any = None
    error: str | None = None
    pending_confirmation: bool = False
    confirmation_id: str | None = None

    @staticmethod
    def success(output: Any) -> "ToolResult":
        return ToolResult(ok=True, output=output)

    @staticmethod
    def failure(error: str, **extra: Any) -> "ToolResult":
        return ToolResult(ok=False, output=extra or None, error=error)

    def to_content(self) -> str:
        """Serialized form fed back to the model as the tool message."""
        if self.pending_confirmation:
            body: dict = {
                "status": "pending_confirmation",
                "confirmation_id": self.confirmation_id,
                "message": "This action needs explicit user approval and has NOT been performed yet.",
            }
        elif self.ok:
            body = {"status": "ok", "result": self.output}
        else:
            body = {"status": "error", "error": self.error}
            if self.output is not None:
                body["details"] = self.output
        text = json.dumps(body, default=str)
        if len(text) > MAX_TOOL_OUTPUT_CHARS:
            text = text[:MAX_TOOL_OUTPUT_CHARS] + '..."<truncated>'
        return text


class Tool(ABC):
    name: str
    description: str
    input_schema: dict
    requires_confirmation: bool = False

    @abstractmethod
    async def execute(self, arguments: dict) -> ToolResult: ...

    @property
    def schema(self) -> dict:
        """Provider-neutral tool definition; each provider translates it to its wire format."""
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


@dataclass
class PendingAction:
    id: str
    session_id: str
    agent_name: str
    tool: Tool
    arguments: dict
    status: str = "pending"  # pending | confirmed | rejected
    result: ToolResult | None = None

    def summary(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "agent": self.agent_name,
            "tool": self.tool.name,
            "arguments": self.arguments,
            "status": self.status,
        }


def sanitize_arguments(schema: dict, arguments: Any) -> Any:
    """Drop top-level `null` values for OPTIONAL properties whose schema does not allow null.
    Small local models routinely send `"limit": null` for parameters they have nothing to say
    about; that means "not provided", and rejecting it wastes a whole model round-trip. Required
    properties and properties that explicitly allow null are left untouched."""
    if not isinstance(arguments, dict):
        return arguments
    required = set(schema.get("required", []))
    props = schema.get("properties", {})

    def allows_null(spec: dict) -> bool:
        t = spec.get("type")
        return t == "null" or (isinstance(t, list) and "null" in t) or any(
            allows_null(sub) for sub in spec.get("anyOf", []) + spec.get("oneOf", [])
        )

    return {
        k: v for k, v in arguments.items()
        if not (v is None and k not in required and k in props and not allows_null(props[k]))
    }


def validate_arguments(schema: dict, arguments: Any) -> str | None:
    """Return a human/model-readable error, or None if valid."""
    if not isinstance(arguments, dict):
        return f"arguments must be a JSON object, got {type(arguments).__name__}"
    try:
        errors = sorted(Draft202012Validator(schema).iter_errors(arguments), key=lambda e: list(e.path))
    except SchemaError as exc:  # pragma: no cover - a broken tool definition is a programming error
        return f"tool schema invalid: {exc.message}"
    if not errors:
        return None
    return "; ".join(
        f"{'.'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors[:5]
    )


async def safe_execute(tool: Tool, arguments: dict) -> ToolResult:
    """Validate then execute. Never raises: every failure becomes a structured error result
    that is fed back to the model."""
    arguments = sanitize_arguments(tool.input_schema, arguments)
    problem = validate_arguments(tool.input_schema, arguments)
    if problem:
        return ToolResult.failure(f"invalid arguments for '{tool.name}': {problem}")
    try:
        result = await tool.execute(arguments)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all at the tool boundary
        return ToolResult.failure(f"tool '{tool.name}' raised {type(exc).__name__}: {exc}")
    if not isinstance(result, ToolResult):
        return ToolResult.failure(f"tool '{tool.name}' returned {type(result).__name__}, expected ToolResult")
    return result


@dataclass
class ToolRegistry:
    _tools: dict[str, Tool] = field(default_factory=dict)
    _grants: dict[str, set[str]] = field(default_factory=dict)
    _pending: dict[str, PendingAction] = field(default_factory=dict)
    # Optional output scrubber (anything with `.redact(obj) -> obj`, e.g. tools.credentials.Redactor).
    # Applied to every ToolResult that leaves dispatch()/confirm(), so a secret a tool accidentally
    # echoes never reaches the model or the trace. Fails closed: if scrubbing raises, the output is withheld.
    redactor: Any = None

    def _scrub(self, result: ToolResult) -> ToolResult:
        if self.redactor is None:
            return result
        try:
            return replace(
                result,
                output=self.redactor.redact(result.output),
                error=None if result.error is None else self.redactor.redact(result.error),
            )
        except Exception:  # noqa: BLE001 - never let an unscrubbed result through
            return ToolResult.failure("tool output withheld: it could not be safely redacted")

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def grant(self, agent_name: str, tool_names: list[str], missing_ok: bool = False) -> list[str]:
        """Allow `agent_name` to use `tool_names`. Unregistered names raise, unless `missing_ok`
        (an agent may declare tools a deployment did not register; it then simply lacks them - a
        smaller capability set, never a larger one). Returns the names that were skipped."""
        missing = [n for n in tool_names if n not in self._tools]
        if missing and not missing_ok:
            raise UnknownTool(f"cannot grant unregistered tools to '{agent_name}': {missing}")
        self._grants.setdefault(agent_name, set()).update(n for n in tool_names if n in self._tools)
        return missing

    def get(self, agent_name: str, tool_name: str) -> Tool:
        if tool_name not in self._tools:
            raise UnknownTool(f"unknown tool '{tool_name}'")
        if tool_name not in self._grants.get(agent_name, set()):
            raise ToolNotPermitted(f"agent '{agent_name}' is not permitted to use tool '{tool_name}'")
        return self._tools[tool_name]

    def tools_for(self, agent_name: str) -> list[Tool]:
        return [self._tools[n] for n in sorted(self._grants.get(agent_name, set()))]

    async def dispatch(self, agent_name: str, tool_name: str, arguments: dict, session_id: str) -> ToolResult:
        """Allow-list check -> schema validation -> confirmation gate -> execute -> redact. Never raises."""
        return self._scrub(await self._dispatch(agent_name, tool_name, arguments, session_id))

    async def _dispatch(self, agent_name: str, tool_name: str, arguments: dict, session_id: str) -> ToolResult:
        try:
            tool = self.get(agent_name, tool_name)
        except ToolError as exc:
            available = [t.name for t in self.tools_for(agent_name)]
            return ToolResult.failure(str(exc), available_tools=available)

        arguments = sanitize_arguments(tool.input_schema, arguments)
        problem = validate_arguments(tool.input_schema, arguments)
        if problem:
            return ToolResult.failure(f"invalid arguments for '{tool.name}': {problem}")

        if tool.requires_confirmation:
            action = PendingAction(uuid.uuid4().hex[:12], session_id, agent_name, tool, dict(arguments))
            self._pending[action.id] = action
            return ToolResult(ok=True, pending_confirmation=True, confirmation_id=action.id)

        return await safe_execute(tool, arguments)

    # --- confirmation gate -------------------------------------------------------------
    def pending(self, session_id: str | None = None) -> list[PendingAction]:
        return [
            a
            for a in self._pending.values()
            if a.status == "pending" and (session_id is None or a.session_id == session_id)
        ]

    async def confirm(self, confirmation_id: str, approve: bool = True) -> PendingAction:
        action = self._pending.get(confirmation_id)
        if action is None:
            raise KeyError(f"no such pending action: {confirmation_id}")
        if action.status != "pending":
            raise ValueError(f"action {confirmation_id} already {action.status}")
        if not approve:
            action.status = "rejected"
            return action
        action.status = "confirmed"
        action.result = self._scrub(await safe_execute(action.tool, action.arguments))
        return action
