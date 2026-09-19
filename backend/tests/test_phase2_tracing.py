"""Phase 2 acceptance: after a run, trace_events holds an ordered, inspectable record of the turn."""
from app.agents.code_agent import CodeAgent
from app.observability.store import PostgresTracer, TraceStore
from app.orchestrator.orchestrator import Orchestrator
from app.providers.mock_provider import MockProvider, text_response, tool_call_response
from app.tools.files import ReadFileTool
from app.tools.registry import ToolRegistry


async def test_full_turn_is_recorded_in_order(db, session_id, tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    registry = ToolRegistry()
    registry.register(ReadFileTool(tmp_path))
    code = CodeAgent(
        MockProvider(script=[tool_call_response("read_file", {"path": "a.py"}), text_response("x is 1")]),
        registry,
    )
    orch = Orchestrator(
        MockProvider(
            script=[
                tool_call_response("route_to_code", {"task": "what is x in a.py", "reason": "code question"}),
                text_response("x = 1"),
            ]
        ),
        [code],
        PostgresTracer(db),
    )

    result = await orch.handle("what is x in a.py?", session_id)

    events = await TraceStore(db).get_turn(result.turn_id)
    assert [(e["agent_name"], e["event_type"]) for e in events] == [
        ("orchestrator", "model_call"),
        ("orchestrator", "tool_call"),
        ("orchestrator", "route_decision"),
        ("code", "model_call"),
        ("code", "tool_call"),
        ("code", "tool_result"),
        ("code", "model_call"),
        ("code", "final_response"),
        ("orchestrator", "tool_result"),
        ("orchestrator", "model_call"),
        ("orchestrator", "final_response"),
    ]
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs)

    route = next(e for e in events if e["event_type"] == "route_decision")
    assert route["payload"]["agent"] == "code" and route["payload"]["reason"] == "code question"
    model_calls = [e for e in events if e["event_type"] == "model_call"]
    assert all(e["latency_ms"] is not None and e["tokens"] == 15 for e in model_calls)
    tool_result = next(e for e in events if e["event_type"] == "tool_result" and e["agent_name"] == "code")
    assert tool_result["payload"]["ok"] is True and tool_result["payload"]["tool"] == "read_file"

    session_events = await TraceStore(db).get_session(session_id)
    assert len(session_events) == len(events)
    turns = await TraceStore(db).list_turns(session_id)
    assert turns[0]["routed"] == ["code"]


async def test_metrics_aggregates(db, session_id, tmp_path):
    registry = ToolRegistry()
    registry.register(ReadFileTool(tmp_path))
    code = CodeAgent(
        MockProvider(script=[tool_call_response("read_file", {"path": "nope"}), text_response("missing")]), registry
    )
    orch = Orchestrator(
        MockProvider(script=[tool_call_response("route_to_code", {"task": "t"}), text_response("done")]),
        [code],
        PostgresTracer(db),
    )
    await orch.handle("q", session_id)
    m = await TraceStore(db).metrics()
    assert {r["agent"] for r in m["latency_by_agent"]} >= {"orchestrator", "code"}
    assert any(t["tool"] == "read_file" and t["success_rate"] <= 1.0 for t in m["tool_success"])
    assert m["tokens"]["sessions"] >= 1
