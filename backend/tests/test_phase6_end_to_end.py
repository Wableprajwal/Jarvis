"""Phase 6 acceptance, end to end (MockProvider only, both stores):
 (b) a fact stored in session A (explicit remember_fact tool call, and the extraction pass) is in
     the history built for a brand-new session B of the same user - and not for another user;
 (d) SessionManager output works as `history` for a real Orchestrator: its provider receives
     the facts and the summary block in its messages."""
from app.agents.code_agent import CodeAgent
from app.harness.loop import run_agent_loop
from app.harness.memory import SUMMARY_PREFIX, Fact
from app.harness.session import FACTS_HEADER, SessionManager
from app.observability.trace import InMemoryTracer, NullTracer
from app.orchestrator.orchestrator import Orchestrator
from app.providers.mock_provider import MockProvider, text_response, tool_call_response
from app.tools.files import ReadFileTool
from app.tools.memory_tool import RememberFactTool
from app.tools.registry import ToolRegistry
from tests.test_phase6_env import env, text_of  # noqa: F401  (env is a fixture)


async def remember_via_agent_loop(memory, user_id: str, session_id: str, key: str, value: str) -> None:
    """The real path: the model emits a remember_fact tool call, the loop dispatches it."""
    registry = ToolRegistry()
    tool = RememberFactTool(memory, lambda: user_id)
    registry.register(tool)
    registry.grant("assistant", ["remember_fact"])
    provider = MockProvider(
        script=[tool_call_response("remember_fact", {"key": key, "value": value}), text_response("Got it.")]
    )
    result = await run_agent_loop(
        provider, [{"role": "user", "content": f"remember that {value}"}], registry.tools_for("assistant"),
        registry, NullTracer(), session_id, "assistant",
    )
    assert result.text == "Got it." and result.tool_calls[0].ok


async def test_fact_from_session_a_is_recalled_in_new_session_b_for_same_user_only(env):
    memory = env.make(MockProvider(default=text_response("SUMMARY")))
    sm = SessionManager(memory, token_budget=300)
    alice, bob = env.user(), env.user()

    # --- session A: one fact via the explicit tool, one via the extraction pass on close
    session_a = await sm.create_session(alice, session_id=env.session())
    await remember_via_agent_loop(memory, alice, session_a, "package_manager", "uv, never pip")
    await sm.record_turn(session_a, "We deploy everything to Fly.io from the monorepo.", "Understood.")
    extractor = MockProvider(
        script=[text_response('{"facts": [{"key": "hosting", "value": "Fly.io, deployed from a monorepo"}]}')]
    )
    extracted = await sm.close_session(session_a, extract_with=extractor)
    assert [f.key for f in extracted] == ["hosting"]

    # --- brand-new session B, same user: both facts are injected at the start
    session_b = await sm.create_session(alice, session_id=env.session())
    history_b = await sm.build_history(session_b, alice)
    assert len(history_b) == 1 and history_b[0]["role"] == "system"  # nothing but the facts: B is new
    assert history_b[0]["content"].startswith(FACTS_HEADER)
    assert "package_manager: uv, never pip" in history_b[0]["content"]
    assert "hosting: Fly.io, deployed from a monorepo" in history_b[0]["content"]
    assert "Understood." not in text_of(history_b)  # A's transcript itself does not leak into B

    # --- different user: nothing
    session_c = await sm.create_session(bob, session_id=env.session())
    assert await sm.build_history(session_c, bob) == []
    assert await memory.get_long_term_facts(bob) == []


async def test_session_history_feeds_a_real_orchestrator(env):
    """Turn after turn through Orchestrator.handle with SessionManager-built history: the model
    is shown the durable facts, then (once the token budget is exceeded) a summary block plus
    the recent turns verbatim - and the older raw turns are gone."""
    memory = env.make(MockProvider(default=text_response("SUMMARY: user is porting a Flask app to FastAPI")))
    sm = SessionManager(memory, token_budget=300)
    uid, sid = env.user(), env.session()
    await sm.create_session(uid, session_id=sid)
    await memory.upsert_long_term_fact(uid, Fact("stack", "Python + Postgres"))

    orch_provider = MockProvider(default=text_response("Sure, on it."))
    tracer = InMemoryTracer()
    orchestrator = Orchestrator(orch_provider, [], tracer)
    pad = "detail " * 25  # ~175 chars

    for i in range(7):
        message = f"Q{i} please help with step {i}. {pad}"
        history = await sm.build_history(sid, uid)
        result = await orchestrator.handle(message, sid, history=history)
        await sm.record_turn(sid, message, result.text)

    # first call: facts only, in front of the new user message, after the orchestrator's own system prompt
    first = orch_provider.calls[0]["messages"]
    assert first[0]["role"] == "system" and "Jarvis" in first[0]["content"]
    assert first[1]["role"] == "system" and first[1]["content"].startswith(FACTS_HEADER)
    assert "stack: Python + Postgres" in first[1]["content"]
    assert first[2]["role"] == "user" and first[2]["content"].startswith("Q0 ")
    assert len(first) == 3

    # last call: system prompt, facts, ONE summary block, recent turns verbatim, new user message
    last = orch_provider.calls[-1]["messages"]
    assert last[0]["role"] == "system" and last[1]["content"].startswith(FACTS_HEADER)
    summaries = [m for m in last if (m.get("content") or "").startswith(SUMMARY_PREFIX)]
    assert len(summaries) == 1 and "porting a Flask app to FastAPI" in summaries[0]["content"]
    assert last.index(summaries[0]) == 2  # right after the facts
    assert last[-1] == {"role": "user", "content": f"Q6 please help with step 6. {pad}"}
    assert last[-2] == {"role": "assistant", "content": "Sure, on it."}
    assert last[-3]["content"].startswith("Q5 ")  # recent turn kept verbatim
    text = text_of(last)
    assert "Q0 " not in text and "Q1 " not in text  # older turns replaced by the summary
    # tool/route traffic never enters history, and roles stay provider-valid
    assert all(m["role"] in ("system", "user", "assistant") for m in last)

    assert len(tracer.of_type("final_response")) == 7
    assert len(await memory.get_all_messages(sid)) == 14  # full transcript retained


async def test_history_is_valid_input_for_a_routed_turn(env, tmp_path):
    """History built from memory also works when the orchestrator actually routes to an agent."""
    memory = env.make(MockProvider(default=text_response("SUMMARY")))
    sm = SessionManager(memory, token_budget=300)
    uid, sid = env.user(), env.session()
    await sm.create_session(uid, session_id=sid)
    await sm.record_turn(sid, "My repo lives in ./svc", "Noted.")

    registry = ToolRegistry()
    registry.register(ReadFileTool(tmp_path))
    code = CodeAgent(MockProvider(script=[text_response("svc is a Go service")]), registry)
    orch_provider = MockProvider(
        script=[tool_call_response("route_to_code", {"task": "describe ./svc", "reason": "code"}), text_response("svc is Go")]
    )
    orchestrator = Orchestrator(orch_provider, [code], InMemoryTracer())
    history = await sm.build_history(sid, uid)
    result = await orchestrator.handle("what is in my repo?", sid, history=history)
    await sm.record_turn(sid, "what is in my repo?", result.text)

    assert result.routed_agents == ["code"] and result.text == "svc is Go"
    seen = text_of(orch_provider.calls[0]["messages"])
    assert "My repo lives in ./svc" in seen
    assert [m["role"] for m in await memory.get_context(sid)] == ["user", "assistant", "user", "assistant"]
