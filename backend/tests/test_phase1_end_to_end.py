"""Phase 1 acceptance: scripted MockProvider drives user request -> orchestrator routes to
CodeAgent -> tool call executes -> final answer. No live model."""
import json

import pytest

from app.agents.code_agent import CodeAgent
from app.harness.loop import MAX_ITERATIONS_TEXT, run_agent_loop
from app.observability.trace import InMemoryTracer
from app.orchestrator.orchestrator import Orchestrator
from app.providers.mock_provider import (
    MockProvider,
    MockScriptExhausted,
    multi_tool_call_response,
    text_response,
    tool_call_response,
)
from app.tools.builtin import GetTimeTool
from app.tools.files import ReadFileTool
from app.tools.registry import ToolRegistry


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "hello.py").write_text("def add(a, b):\n    return a + b\n")
    return tmp_path


@pytest.fixture
def registry(workspace):
    reg = ToolRegistry()
    reg.register(ReadFileTool(workspace))
    reg.register(GetTimeTool())
    return reg


async def test_orchestrator_routes_to_code_agent_and_answers(registry):
    code_provider = MockProvider(
        script=[
            tool_call_response("read_file", {"path": "hello.py"}),
            text_response("`add` returns the sum of its two arguments."),
        ]
    )
    orch_provider = MockProvider(
        script=[
            tool_call_response(
                "route_to_code", {"task": "Explain hello.py", "reason": "question about local code"}
            ),
            text_response("hello.py defines add(a, b), which returns a + b."),
        ]
    )
    tracer = InMemoryTracer()
    orchestrator = Orchestrator(orch_provider, [CodeAgent(code_provider, registry)], tracer)

    result = await orchestrator.handle("What does hello.py do?", session_id="s1")

    assert result.text == "hello.py defines add(a, b), which returns a + b."
    assert result.routed_agents == ["code"]
    assert result.route_decisions[0]["reason"] == "question about local code"
    assert [c.name for c in result.agent_results[0].tool_calls] == ["read_file"]

    # The tool really executed: its output was fed back to the code agent's model.
    tool_msg = code_provider.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert "return a + b" in json.loads(tool_msg["content"])["result"]["content"]

    # The orchestrator saw the sub-agent answer as a tool result and only then finished.
    orch_last = orch_provider.calls[1]["messages"][-1]
    assert orch_last["role"] == "tool" and orch_last["name"] == "route_to_code"
    assert "sum of its two arguments" in orch_last["content"]

    # Orchestrator is offered routing tools only, never the code agent's tools.
    assert [t["name"] for t in orch_provider.calls[0]["tools"]] == ["route_to_code"]
    assert [t["name"] for t in code_provider.calls[0]["tools"]] == ["read_file"]


async def test_orchestrator_answers_directly_when_no_agent_needed(registry):
    orch = Orchestrator(
        MockProvider(script=[text_response("Hello!")]),
        [CodeAgent(MockProvider(), registry)],
        InMemoryTracer(),
    )
    result = await orch.handle("hi", "s1")
    assert result.text == "Hello!" and result.routed_agents == []


async def test_loop_iteration_cap(registry):
    looping = MockProvider(default=tool_call_response("get_time", {}))
    registry.grant("agent", ["get_time"])
    result = await run_agent_loop(
        looping, [{"role": "user", "content": "loop"}], registry.tools_for("agent"),
        registry, InMemoryTracer(), "s1", "agent", max_iterations=3,
    )
    assert result.truncated and result.text == MAX_ITERATIONS_TEXT
    assert result.iterations == 3
    assert len(looping.calls) == 3


async def test_iteration_cap_logs_error_event(registry):
    registry.grant("agent", ["get_time"])
    tracer = InMemoryTracer()
    await run_agent_loop(
        MockProvider(default=tool_call_response("get_time", {})), [], registry.tools_for("agent"),
        registry, tracer, "s1", "agent", max_iterations=2,
    )
    errors = tracer.of_type("error")
    assert len(errors) == 1 and errors[0].payload["error"] == "max_iterations_reached"


async def test_tool_failure_is_fed_back_not_raised(registry):
    registry.grant("agent", ["read_file"])
    provider = MockProvider(
        script=[
            tool_call_response("read_file", {"path": "missing.py"}),
            text_response("That file does not exist."),
        ]
    )
    result = await run_agent_loop(
        provider, [{"role": "user", "content": "read it"}], registry.tools_for("agent"),
        registry, InMemoryTracer(), "s1", "agent",
    )
    assert result.text == "That file does not exist."
    fed_back = json.loads(provider.calls[1]["messages"][-1]["content"])
    assert fed_back["status"] == "error" and "not found" in fed_back["error"]
    assert result.tool_calls[0].ok is False


async def test_invalid_arguments_rejected_before_execute(registry):
    registry.grant("agent", ["read_file"])
    provider = MockProvider(
        script=[tool_call_response("read_file", {"path": 123}), text_response("ok")]
    )
    result = await run_agent_loop(
        provider, [], registry.tools_for("agent"), registry, InMemoryTracer(), "s1", "agent"
    )
    fed_back = json.loads(provider.calls[1]["messages"][-1]["content"])
    assert fed_back["status"] == "error" and "invalid arguments" in fed_back["error"]
    assert result.text == "ok"


async def test_read_file_cannot_escape_workspace(registry):
    registry.grant("agent", ["read_file"])
    out = await registry.dispatch("agent", "read_file", {"path": "../../etc/passwd"}, "s1")
    assert not out.ok and "escapes" in out.error


async def test_parallel_tool_calls_all_answered_in_order(registry):
    registry.grant("agent", ["get_time", "read_file"])
    provider = MockProvider(
        script=[
            multi_tool_call_response([("get_time", {}), ("read_file", {"path": "hello.py"})]),
            text_response("done"),
        ]
    )
    await run_agent_loop(provider, [], registry.tools_for("agent"), registry, InMemoryTracer(), "s1", "agent")
    tail = provider.calls[1]["messages"][-2:]
    assert [m["name"] for m in tail] == ["get_time", "read_file"]
    assert all(m["role"] == "tool" for m in tail)


async def test_provider_failure_is_traced_and_raised(registry):
    tracer = InMemoryTracer()
    with pytest.raises(MockScriptExhausted):
        await run_agent_loop(MockProvider(), [], [], registry, tracer, "s1", "agent")
    assert tracer.of_type("error")[0].payload["stage"] == "model_call"
