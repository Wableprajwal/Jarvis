"""The tool-calling loop, shared by the orchestrator and every sub-agent.

Routing to a sub-agent is *also* a tool call (see orchestrator/), so this one loop serves
function calls and agent handoffs alike.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

from app.observability.trace import Stopwatch, Tracer
from app.providers.base import ModelProvider, ToolCall, Usage, assistant_message
from app.tools.registry import Tool, ToolRegistry, ToolResult

MAX_ITERATIONS_TEXT = "<max iterations reached>"
MALFORMED_OUTPUT_FEEDBACK = (
    "Your last reply could not be processed: {error}\n"
    "Correct it and try again. Either emit a well-formed tool call, or give your final answer in plain text."
)


@dataclass
class ExecutedCall:
    name: str
    arguments: dict
    ok: bool
    pending_confirmation: bool = False
    error: str | None = None


@dataclass
class AgentResult:
    text: str | None
    messages: list[dict]
    truncated: bool = False
    iterations: int = 0
    usage: Usage = field(default_factory=Usage)
    tool_calls: list[ExecutedCall] = field(default_factory=list)
    pending_confirmation_ids: list[str] = field(default_factory=list)


async def run_agent_loop(
    provider: ModelProvider,
    messages: list[dict],
    tools: list[Tool],
    registry: ToolRegistry,
    tracer: Tracer,
    session_id: str,
    agent_name: str,
    max_iterations: int = 8,
    *,
    turn_id: str | None = None,
    max_tokens: int = 2048,
) -> AgentResult:
    schemas = [t.schema for t in tools]
    result = AgentResult(text=None, messages=messages)

    for i in range(max_iterations):
        result.iterations = i + 1
        sw = Stopwatch()
        try:
            resp = await provider.generate(messages, schemas, max_tokens=max_tokens)
        except Exception as exc:
            await tracer.log(
                session_id, agent_name, "error",
                {"stage": "model_call", "iteration": i, "error": f"{type(exc).__name__}: {exc}"},
                turn_id=turn_id, latency_ms=sw.ms(),
            )
            raise
        result.usage.prompt_tokens += resp.usage.prompt_tokens
        result.usage.completion_tokens += resp.usage.completion_tokens
        await tracer.log(
            session_id, agent_name, "model_call",
            {
                "iteration": i,
                "stop_reason": resp.stop_reason,
                "text": resp.text,
                "tool_calls": [c.name for c in resp.tool_calls],
                "parse_error": resp.error,
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            },
            turn_id=turn_id, latency_ms=sw.ms(), tokens=resp.usage.total,
        )

        for call in resp.tool_calls:
            if call.id is None:
                call.id = f"call_{uuid.uuid4().hex[:8]}"
        messages.append(assistant_message(resp))

        if resp.error and not resp.tool_calls:
            messages.append({"role": "user", "content": MALFORMED_OUTPUT_FEEDBACK.format(error=resp.error)})
            continue

        if resp.tool_calls:
            # Independent calls run concurrently. If any call in the batch is `serial` (an agent
            # handoff), the whole batch runs in the order the model emitted it: a model that asks
            # for "research, then code" in one turn means the second to build on the first, and
            # running those concurrently hands the second agent a task about findings that do not
            # exist yet. Ordering the batch costs concurrency only when a handoff is involved.
            if any(registry.is_serial(call.name) for call in resp.tool_calls):
                outcomes = [
                    await _run_call(call, registry, tracer, session_id, agent_name, turn_id)
                    for call in resp.tool_calls
                ]
            else:
                outcomes = await asyncio.gather(
                    *(_run_call(call, registry, tracer, session_id, agent_name, turn_id) for call in resp.tool_calls)
                )
            for call, tool_result in zip(resp.tool_calls, outcomes, strict=True):
                result.tool_calls.append(
                    ExecutedCall(call.name, call.arguments, tool_result.ok, tool_result.pending_confirmation, tool_result.error)
                )
                if tool_result.pending_confirmation and tool_result.confirmation_id:
                    result.pending_confirmation_ids.append(tool_result.confirmation_id)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": tool_result.to_content()}
                )
            continue

        await tracer.log(
            session_id, agent_name, "final_response", {"text": resp.text, "iterations": i + 1}, turn_id=turn_id
        )
        result.text = resp.text
        return result

    await tracer.log(
        session_id, agent_name, "error",
        {"stage": "loop", "error": "max_iterations_reached", "max_iterations": max_iterations},
        turn_id=turn_id,
    )
    result.text = MAX_ITERATIONS_TEXT
    result.truncated = True
    return result


async def _run_call(
    call: ToolCall, registry: ToolRegistry, tracer: Tracer, session_id: str, agent_name: str, turn_id: str | None
) -> ToolResult:
    await tracer.log(
        session_id, agent_name, "tool_call", {"tool": call.name, "arguments": call.arguments, "call_id": call.id},
        turn_id=turn_id,
    )
    sw = Stopwatch()
    tool_result = await registry.dispatch(agent_name, call.name, call.arguments, session_id)
    await tracer.log(
        session_id, agent_name, "tool_result",
        {
            "tool": call.name,
            "call_id": call.id,
            "ok": tool_result.ok,
            "pending_confirmation": tool_result.pending_confirmation,
            "error": tool_result.error,
            "output": tool_result.output,
        },
        turn_id=turn_id, latency_ms=sw.ms(),
    )
    return tool_result
