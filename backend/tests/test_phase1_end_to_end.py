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


async def test_chained_routes_are_reported_in_call_order_not_agent_order(registry):
    """Regression: decisions were grouped per route tool, so a chain code<-research came out as [code, research]."""
    from app.agents.base import SubAgent

    class Second(SubAgent):
        name, description, system_prompt, tool_names = "aaa_first_in_list", "first alphabetically", "s", []

    class Third(SubAgent):
        name, description, system_prompt, tool_names = "zzz_last_in_list", "last alphabetically", "s", []

    agents = [
        Second(MockProvider(script=[text_response("A")]), registry),
        Third(MockProvider(script=[text_response("Z")]), registry),
    ]
    orch = Orchestrator(
        MockProvider(script=[
            tool_call_response("route_to_zzz_last_in_list", {"task": "t1"}),
            tool_call_response("route_to_aaa_first_in_list", {"task": "t2"}),
            text_response("done"),
        ]),
        agents, InMemoryTracer(),
    )
    result = await orch.handle("chain", "s1")
    assert result.routed_agents == ["zzz_last_in_list", "aaa_first_in_list"]
    assert [r.text for r in result.agent_results] == ["Z", "A"]


async def test_null_for_optional_argument_is_treated_as_not_provided(registry):
    """Regression (seen live with qwen2.5:7b): `"limit": null` for an optional param was rejected."""
    from app.tools.calendar import CalendarListTool, FakeCalendarBackend

    registry.register(CalendarListTool(FakeCalendarBackend()))
    registry.grant("agent", ["calendar_list", "read_file"])
    ok = await registry.dispatch("agent", "calendar_list", {"start": "2026-09-21", "limit": None}, "s1")
    assert ok.ok, ok.error
    # required properties are never silently dropped
    bad = await registry.dispatch("agent", "read_file", {"path": None}, "s1")
    assert not bad.ok and "invalid arguments" in bad.error


def test_sanitize_keeps_null_where_the_schema_allows_it():
    from app.tools.registry import sanitize_arguments

    schema = {"type": "object", "required": ["a"], "properties": {
        "a": {"type": "string"}, "b": {"type": ["integer", "null"]}, "c": {"type": "integer"},
        "d": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}
    assert sanitize_arguments(schema, {"a": "x", "b": None, "c": None, "d": None, "zzz": None}) == {"a": "x", "b": None, "d": None, "zzz": None}


async def test_a_failing_sub_agent_does_not_mispair_later_agents_results(registry):
    """Regression: decisions and results were two flat lists zipped together, so once one agent
    raised, every later delegation was paired with the wrong agent's transcript."""
    from app.agents.base import SubAgent

    class Boom(SubAgent):
        name, description, system_prompt, tool_names = "boom", "always fails", "s", []

        async def run(self, task, context, tracer, session_id, turn_id=None):
            raise RuntimeError("provider died mid-turn")

    class Fine(SubAgent):
        name, description, system_prompt, tool_names = "fine", "always works", "s", []

    agents = [Boom(MockProvider(), registry), Fine(MockProvider(script=[text_response("FINE")]), registry)]
    orch = Orchestrator(
        MockProvider(script=[
            tool_call_response("route_to_boom", {"task": "t1"}),
            tool_call_response("route_to_fine", {"task": "t2"}),
            text_response("done"),
        ]),
        agents, InMemoryTracer(),
    )

    result = await orch.handle("two agents, first one dies", "s1")

    assert result.routed_agents == ["boom", "fine"]
    assert [s.agent for s in result.steps] == ["boom", "fine"]
    assert result.steps[0].result is None          # boom raised
    assert result.steps[1].result.text == "FINE"   # fine's result stays attached to fine
    # the flat list still holds only real results, which is exactly why zipping it was unsafe
    assert len(result.agent_results) == 1 and len(result.route_decisions) == 2
    # the failed delegation is reported to the model as a structured error, not a crash
    assert result.text == "done"
