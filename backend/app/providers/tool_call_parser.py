"""Tool calling for models WITHOUT native function-calling.

Injects tool schemas into the system prompt, tells the model to emit calls as a fenced ```json
block, then extracts and validates that block. Malformed output never raises: it comes back as
`ParseResult.error`, which the harness feeds to the model so it can correct itself.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field

from app.providers.base import ToolCall
from app.tools.registry import validate_arguments

_FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")

INSTRUCTIONS = """You can call tools. To call a tool, reply with ONLY a fenced json block:

```json
{{"tool": "<tool name>", "arguments": {{ ... }}}}
```

To call several independent tools at once, use a JSON array of such objects inside one block.
Do not write any other text in a reply that contains a tool call. After the tool results come back you
may call more tools or give your final answer. To give your final answer, reply in plain text with NO json block.
Only use the tools listed below and follow their argument schemas exactly.

Available tools:
{tools}"""


@dataclass
class ParseResult:
    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str | None = None  # prose outside any tool-call block
    error: str | None = None
    attempted: bool = False  # the model tried to call a tool (a block or tool-shaped JSON was present)


def render_tool_instructions(tools: list[dict]) -> str:
    listing = "\n".join(
        json.dumps({"name": t["name"], "description": t["description"], "arguments_schema": t["input_schema"]})
        for t in tools
    )
    return INSTRUCTIONS.format(tools=listing)


def render_tool_calls(calls: list[dict]) -> str:
    """Render normalized assistant tool_calls back into the prompted wire format."""
    objs = [{"tool": c["name"], "arguments": c["arguments"]} for c in calls]
    body = objs[0] if len(objs) == 1 else objs
    return "```json\n" + json.dumps(body) + "\n```"


class PromptedToolCallParser:
    def __init__(self, tools: list[dict]):
        self._schemas = {t["name"]: t["input_schema"] for t in tools}

    def parse(self, text: str | None) -> ParseResult:
        if not text or not text.strip():
            return ParseResult(text=text)

        blocks = _FENCE.findall(text)
        prose = _FENCE.sub("", text).strip() or None
        candidates: list[str] = [b.strip() for b in blocks if b.strip()]
        if not candidates:
            raw = self._bare_json(text)
            if raw is not None:
                candidates = [raw]
                prose = text.replace(raw, "").strip() or None
        if not candidates:
            return ParseResult(text=text.strip())  # plain final answer

        calls: list[ToolCall] = []
        errors: list[str] = []
        for cand in candidates:
            try:
                data = json.loads(_TRAILING_COMMA.sub(r"\1", cand))
            except json.JSONDecodeError as exc:
                errors.append(f"the json block is not valid JSON ({exc.msg} at char {exc.pos})")
                continue
            items = data if isinstance(data, list) else data.get("tool_calls", [data]) if isinstance(data, dict) else None
            if not isinstance(items, list) or not items:
                errors.append("expected a JSON object {\"tool\": ..., \"arguments\": ...} or an array of them")
                continue
            for item in items:
                call, err = self._to_call(item)
                if err:
                    errors.append(err)
                else:
                    calls.append(call)

        if errors:
            return ParseResult(tool_calls=[], text=prose, error="; ".join(errors), attempted=True)
        return ParseResult(tool_calls=calls, text=prose, attempted=True)

    def _to_call(self, item: object) -> tuple[ToolCall | None, str | None]:
        if not isinstance(item, dict):
            return None, f"tool call must be a JSON object, got {type(item).__name__}"
        name = item.get("tool") or item.get("name")
        args = item.get("arguments", item.get("parameters", {}))
        if not isinstance(name, str):
            return None, "tool call is missing the \"tool\" name"
        if name not in self._schemas:
            return None, f"unknown tool '{name}'. Available tools: {sorted(self._schemas)}"
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None, f"arguments for '{name}' must be a JSON object"
        problem = validate_arguments(self._schemas[name], args)
        if problem:
            return None, f"invalid arguments for '{name}': {problem}"
        return ToolCall(name=name, arguments=args, id=f"call_{uuid.uuid4().hex[:8]}"), None

    @staticmethod
    def _bare_json(text: str) -> str | None:
        """A model that skipped the fence but emitted a tool-shaped JSON object."""
        stripped = text.strip()
        start = stripped.find("{")
        if start < 0:
            return None
        try:
            obj, end = json.JSONDecoder().raw_decode(stripped[start:])
        except json.JSONDecodeError:
            return stripped[start:] if re.search(r'"(tool|name)"\s*:', stripped[start:]) else None
        if isinstance(obj, dict) and ("tool" in obj or "tool_calls" in obj):
            return stripped[start : start + end]
        return None
