"""PHASE 4 ACCEPTANCE (multi-agent), no live model and no network:

  1. the orchestrator routes 3 distinct requests to research / code / ops
  2. a chained request Research -> Code works (retrieved facts reach the code agent's task)
  3. an allow-list violation is rejected by the registry and nothing is executed

Real Orchestrator + ToolRegistry + tools + retrieval; language models are scripted MockProviders.
"""
import re

import pytest

from app.agents.code_agent import CodeAgent
from app.agents.ops_agent import OpsAgent
from app.observability.trace import InMemoryTracer
from app.providers.mock_provider import MockProvider, text_response, tool_call_response
from tests.test_phase4_wiring import build_world, keyword_router, last_tool_message, tool_payload

EMAIL_ARGS = {"to": ["priya@example.com"], "subject": "Hi", "body": "Sending this without asking."}
EVENT_ARGS = {"title": "Rogue", "start": "2026-09-30T10:00:00Z", "end": "2026-09-30T11:00:00Z"}


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "hello.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "retry.py").write_text("def backoff(attempt):\n    raise NotImplementedError\n")
    return tmp_path


def _research_provider() -> MockProvider:
    """Searches, then answers ONLY from what the tool returned (like the real prompt demands)."""

    def answer(messages):
        citations = tool_payload(last_tool_message(messages))["result"]
        hit = next(c for c in citations if "Maximum 5 attempts" in c["text"])
        cap = re.search(r"cap is (\d+) seconds", hit["text"]).group(1)
        return text_response(f"Max 5 attempts, backoff cap {cap} seconds [{hit['ref']}].")

    return MockProvider(
        script=[tool_call_response("rag_search", {"query": "retry policy max attempts backoff cap"}), answer]
    )


# ------------------------------------------------------------------------------ 1. routing
ROUTING_CASES = [
    ("According to our notes, what is the retry policy for service calls?", "research", "rag_search"),
    ("Explain what add() in hello.py does", "code", "read_file"),
    ("What meetings do I have between 2026-09-22 and 2026-09-24?", "ops", "calendar_list"),
]


@pytest.mark.parametrize("request_text,expected_agent,expected_tool", ROUTING_CASES)
async def test_orchestrator_routes_each_request_to_the_right_agent(workspace, request_text, expected_agent, expected_tool):
    tool_args = {
        "rag_search": {"query": "retry policy max attempts backoff cap"},
        "read_file": {"path": "hello.py"},
        "calendar_list": {"start": "2026-09-22", "end": "2026-09-24"},
    }[expected_tool]
    sub_model = MockProvider(
        script=[tool_call_response(expected_tool, tool_args), text_response(f"{expected_agent} answered")]
    )
    world = await build_world(
        workspace,
        orchestrator=keyword_router({"notes": "research", "hello.py": "code", "meetings": "ops"}),
        **{expected_agent: sub_model},
    )

    result = await world.orchestrator.handle(request_text, session_id="s1")

    assert result.routed_agents == [expected_agent]
    assert result.text == f"FINAL: {expected_agent} answered"
    executed = result.agent_results[0].tool_calls
    assert [(c.name, c.ok) for c in executed] == [(expected_tool, True)]
    # only the chosen agent's model ran
    for name, provider in world.providers.items():
        if name not in ("orchestrator", expected_agent):
            assert provider.calls == []
    # the orchestrator is offered exactly the three routing tools, and each agent only its own tools
    assert sorted(t["name"] for t in world.providers["orchestrator"].calls[0]["tools"]) == [
        "route_to_code", "route_to_ops", "route_to_research",
    ]
    assert [t["name"] for t in world.providers[expected_agent].calls[0]["tools"]] == {
        "research": ["rag_search"], "code": ["read_file"],
        "ops": ["calendar_list", "calendar_create", "email_search", "email_draft", "email_send"],
    }[expected_agent]


async def test_three_requests_in_one_session_all_route_correctly(workspace):
    routed = []
    for text, agent, tool in ROUTING_CASES:
        args = {"rag_search": {"query": "retry policy"}, "read_file": {"path": "hello.py"}, "calendar_list": {}}[tool]
        world = await build_world(
            workspace,
            orchestrator=keyword_router({"notes": "research", "hello.py": "code", "meetings": "ops"}),
            **{agent: MockProvider(script=[tool_call_response(tool, args), text_response("ok")])},
        )
        routed.append((await world.orchestrator.handle(text, "sess")).routed_agents)
    assert routed == [["research"], ["code"], ["ops"]]


async def test_route_decisions_are_traced_with_reasons(workspace):
    world = await build_world(
        workspace,
        orchestrator=keyword_router({"meetings": "ops"}),
        ops=MockProvider(script=[text_response("You have 4 meetings.")]),
    )
    await world.orchestrator.handle("What meetings are on Tuesday?", "s")
    decisions = world.tracer.of_type("route_decision")
    assert [d.payload["agent"] for d in decisions] == ["ops"]
    assert decisions[0].payload["reason"] == "request is about ops"


# ------------------------------------------------------------------------------ 2. chaining
async def test_chained_research_then_code_passes_retrieved_facts_to_code_agent(workspace):
    def route_to_code_with_findings(messages):
        answer = tool_payload(last_tool_message(messages))["result"]["answer"]
        return tool_call_response(
            "route_to_code",
            {"task": f"Write a pytest for backoff() in retry.py that enforces this policy: {answer}",
             "reason": "research done, now write the test"},
        )

    orchestrator = MockProvider(
        script=[
            tool_call_response(
                "route_to_research",
                {"task": "Find our retry policy: max attempts and backoff cap.", "reason": "needs the notes first"},
            ),
            route_to_code_with_findings,
            lambda msgs: text_response(
                f"Test written: {tool_payload(last_tool_message(msgs))['result']['answer']}"
            ),
        ]
    )
    code = MockProvider(
        script=[
            tool_call_response("read_file", {"path": "retry.py"}),
            text_response("def test_backoff_caps(): assert backoff(9) <= 10  # 5 attempts max"),
        ]
    )
    world = await build_world(workspace, orchestrator=orchestrator, research=_research_provider(), code=code)

    result = await world.orchestrator.handle("Use our retry policy notes to write a test for retry.py", "s")

    assert result.routed_agents == ["research", "code"]
    research_answer = result.agent_results[0].text
    assert "5 attempts" in research_answer and "adr-0007-retry-backoff.md#" in research_answer
    # research output flowed into the code agent's task (its first user message)
    code_task = code.calls[0]["messages"][-1]
    assert code_task["role"] == "user" and research_answer in code_task["content"]
    assert "5 attempts" in code_task["content"] and "10 seconds" in code_task["content"]
    # the code agent then used its own tool, and the final answer came from it
    assert [c.name for c in result.agent_results[1].tool_calls] == ["read_file"]
    assert result.text.startswith("Test written: def test_backoff_caps")
    # ordering is visible in the trace
    agents_in_order = [e.payload["agent"] for e in world.tracer.of_type("route_decision")]
    assert agents_in_order == ["research", "code"]


async def test_research_answer_is_grounded_in_real_retrieval_output(workspace):
    research = _research_provider()
    world = await build_world(
        workspace,
        orchestrator=keyword_router({"notes": "research"}),
        research=research,
    )
    await world.orchestrator.handle("According to our notes, what is the retry policy?", "s")
    tool_msg = tool_payload(research.calls[1]["messages"][-1])
    assert tool_msg["status"] == "ok"
    assert tool_msg["result"][0]["source"] == "adr-0007-retry-backoff.md"
    # the research agent's system prompt demands grounding + the explicit not-found sentence
    system = research.calls[0]["messages"][0]["content"]
    assert "ONLY" in system and "I could not find this in the notes." in system and "[adr-0007-retry-backoff.md#1]" in system


# ------------------------------------------------------------------- 3. allow-list violation
@pytest.mark.parametrize(
    "tool,args",
    [("email_send", EMAIL_ARGS), ("calendar_create", EVENT_ARGS), ("rag_search", {"query": "x"}), ("email_draft", EMAIL_ARGS)],
)
async def test_code_agent_cannot_call_tools_outside_its_allow_list(workspace, tool, args):
    code = MockProvider(script=[tool_call_response(tool, args), text_response("I was not able to do that.")])
    world = await build_world(workspace, orchestrator=keyword_router({"hello.py": "code"}), code=code)

    result = await world.orchestrator.handle("hello.py: also email Priya the result", "s")

    # ToolNotPermitted surfaced to the model as a structured error, not a crash
    fed_back = tool_payload(code.calls[1]["messages"][-1])
    assert fed_back["status"] == "error"
    assert f"not permitted to use tool '{tool}'" in fed_back["error"]
    assert fed_back["details"]["available_tools"] == ["read_file"]
    call = result.agent_results[0].tool_calls[0]
    assert (call.name, call.ok, call.pending_confirmation) == (tool, False, False)
    assert result.text == "FINAL: I was not able to do that."
    # and nothing happened anywhere
    assert world.mail.sent == [] and world.mail.drafts == {} and len(world.calendar.events) == 4
    assert world.registry.pending() == [] and result.pending_confirmation_ids == []
    err = [e for e in world.tracer.of_type("tool_result") if e.agent_name == "code"][0]
    assert err.payload["ok"] is False and "not permitted" in err.payload["error"]


@pytest.mark.parametrize(
    "agent,tool,args",
    [
        ("research", "email_send", EMAIL_ARGS),
        ("research", "read_file", {"path": "hello.py"}),
        ("ops", "rag_search", {"query": "x"}),
        ("ops", "read_file", {"path": "hello.py"}),
        ("code", "calendar_list", {}),
        ("orchestrator", "email_send", EMAIL_ARGS),  # even the router has no direct tool access
    ],
)
async def test_every_agent_is_confined_to_its_own_tools(workspace, agent, tool, args):
    world = await build_world(workspace, orchestrator=MockProvider())
    res = await world.registry.dispatch(agent, tool, args, "s")
    assert not res.ok and not res.pending_confirmation and "not permitted" in res.error
    assert world.mail.sent == [] and world.registry.pending() == []


async def test_orchestrator_model_hallucinating_a_tool_is_refused_without_side_effects(workspace):
    orch = MockProvider(
        script=[tool_call_response("email_send", EMAIL_ARGS), text_response("I can only delegate to agents.")]
    )
    world = await build_world(workspace, orchestrator=orch)
    result = await world.orchestrator.handle("send priya an email", "s")
    fed_back = tool_payload(orch.calls[1]["messages"][-1])
    assert fed_back["status"] == "error" and "unknown tool" in fed_back["error"]  # not even registered for the router
    assert world.mail.sent == [] and result.text == "I can only delegate to agents."


async def test_hallucinated_tool_name_is_a_structured_error(workspace):
    code = MockProvider(script=[tool_call_response("delete_repository", {}), text_response("Cannot do that.")])
    world = await build_world(workspace, orchestrator=MockProvider())
    agent = CodeAgent(code, world.registry)
    result = await agent.run("wipe it", [], InMemoryTracer(), "s")
    assert tool_payload(code.calls[1]["messages"][-1])["error"] == "unknown tool 'delete_repository'"
    assert result.text == "Cannot do that."


# ------------------------------------------------------ ops: confirmation gating through the agent
async def test_ops_email_send_waits_for_confirmation_then_executes(workspace):
    ops = MockProvider(
        script=[
            tool_call_response("email_send", EMAIL_ARGS),
            lambda msgs: text_response(
                "Your email is awaiting approval (id "
                f"{tool_payload(last_tool_message(msgs))['confirmation_id']}); it has not been sent."
            ),
        ]
    )
    world = await build_world(workspace, orchestrator=keyword_router({"email": "ops"}), ops=ops)

    result = await world.orchestrator.handle("email Priya that we keep 5 attempts", "s")

    assert len(result.pending_confirmation_ids) == 1
    cid = result.pending_confirmation_ids[0]
    fed_back = tool_payload(ops.calls[1]["messages"][-1])
    assert fed_back["status"] == "pending_confirmation" and fed_back["confirmation_id"] == cid
    assert "NOT been performed" in fed_back["message"]
    assert "not been sent" in result.agent_results[0].text
    assert world.mail.sent == []  # <- not before confirmation
    assert [p.id for p in world.registry.pending("s")] == [cid]

    action = await world.registry.confirm(cid)
    assert action.status == "confirmed" and action.result.ok
    assert len(world.mail.sent) == 1 and world.mail.sent[0].to == ["priya@example.com"]  # <- only after


async def test_ops_calendar_create_rejection_leaves_calendar_untouched(workspace):
    ops = MockProvider(script=[tool_call_response("calendar_create", EVENT_ARGS), text_response("Awaiting approval.")])
    world = await build_world(workspace, orchestrator=keyword_router({"schedule": "ops"}), ops=ops)
    result = await world.orchestrator.handle("schedule a rogue meeting", "s")
    assert len(world.calendar.events) == 4
    await world.registry.confirm(result.pending_confirmation_ids[0], approve=False)
    assert len(world.calendar.events) == 4 and all(e.title != "Rogue" for e in world.calendar.events)


async def test_agent_loop_sees_ops_read_tools_execute_immediately(workspace):
    ops = MockProvider(
        script=[tool_call_response("email_search", {"query": "on call"}), text_response("Found the on-call email.")]
    )
    world = await build_world(workspace, orchestrator=MockProvider())
    result = await OpsAgent(ops, world.registry).run("find my on-call email", [], InMemoryTracer(), "s")
    payload = tool_payload(ops.calls[1]["messages"][-1])
    assert payload["status"] == "ok" and payload["result"]["messages"][0]["id"] == "msg-2"
    assert result.pending_confirmation_ids == []
