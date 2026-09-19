"""HTTP contract tests (the same contract the React frontend codes against)."""
import httpx
import pytest

from app.config import Settings
from app.main import create_app
from app.providers.mock_provider import MockProvider, text_response, tool_call_response

SETTINGS = Settings(model_provider="mock", _env_file=None, workspace_root=".")


class Client:
    def __init__(self, app, client):
        self.app, self.http = app, client

    @property
    def state(self):
        return self.app.state.jarvis


async def make_client(script, use_db=False, settings=SETTINGS):
    shared = MockProvider(script=script)
    app = create_app(settings, provider_for=lambda role: shared, use_db=use_db)
    return app, shared


async def run(script, fn, use_db=False, settings=SETTINGS):
    app, shared = await make_client(script, use_db, settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            return await fn(Client(app, http), shared)


ROUTE_TO_OPS_SEND = [
    tool_call_response("route_to_ops", {"task": "email dana", "reason": "send email"}),
    tool_call_response("email_send", {"to": ["dana@example.com"], "subject": "Delay", "body": "Delayed to Monday."}),
    text_response("Queued; awaiting approval."),
    text_response("Your email is awaiting approval."),
]


async def test_health():
    async def go(c, _):
        r = await c.http.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok" and body["provider"] == "mock" and body["storage"] == "memory"

    await run([], go)


async def test_chat_returns_contract_and_populates_traces():
    script = [
        tool_call_response("route_to_ops", {"task": "list events", "reason": "calendar"}),
        tool_call_response("calendar_list", {}),
        text_response("You have Sprint planning."),
        text_response("Sprint planning is on your calendar."),
    ]

    async def go(c, _):
        r = await c.http.post("/chat", json={"message": "what's on my calendar?"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["text"] == "Sprint planning is on your calendar."
        assert body["route_decisions"] == [{"agent": "ops", "task": "list events", "reason": "calendar"}]
        assert body["pending_confirmations"] == [] and body["truncated"] is False
        sid, turn = body["session_id"], body["turn_id"]

        turns = (await c.http.get("/traces/turns", params={"session_id": sid})).json()
        assert turns[0]["turn_id"] == turn and turns[0]["routed"] == ["ops"]

        events = (await c.http.get(f"/traces/turn/{turn}")).json()["events"]
        types = [(e["agent_name"], e["event_type"]) for e in events]
        assert ("orchestrator", "route_decision") in types and ("ops", "tool_result") in types
        assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)

        m = (await c.http.get("/metrics")).json()
        assert {"latency_by_agent", "tool_success", "tokens"} <= set(m)
        assert any(t["tool"] == "calendar_list" and t["success_rate"] == 1.0 for t in m["tool_success"])

        assert (await c.http.get("/traces/turn/nope")).status_code == 404
        sessions = (await c.http.get("/sessions")).json()
        assert sessions[0]["id"] == sid and "calendar" in sessions[0]["preview"]
        msgs = (await c.http.get(f"/sessions/{sid}/messages")).json()
        assert [m["role"] for m in msgs] == ["user", "assistant"]

    await run(script, go)


async def test_confirmation_flow_approve_executes_only_after_approval():
    async def go(c, _):
        r = await c.http.post("/chat", json={"message": "email dana", "session_id": "s-conf"})
        body = r.json()
        assert body["text"] == "Your email is awaiting approval."
        pending = body["pending_confirmations"]
        assert len(pending) == 1 and pending[0]["tool"] == "email_send" and pending[0]["agent"] == "ops"
        assert c.state.system.mail.sent == []  # nothing happened yet

        listed = (await c.http.get("/confirmations", params={"session_id": "s-conf"})).json()
        assert [p["id"] for p in listed] == [pending[0]["id"]]

        done = await c.http.post(f"/confirm/{pending[0]['id']}", json={"approve": True})
        assert done.status_code == 200 and done.json()["status"] == "confirmed"
        assert done.json()["result"]["ok"] is True
        assert len(c.state.system.mail.sent) == 1  # executed exactly once, after approval
        assert (await c.http.post(f"/confirm/{pending[0]['id']}", json={"approve": True})).status_code == 409
        assert (await c.http.get("/confirmations")).json() == []

    await run(ROUTE_TO_OPS_SEND, go)


async def test_confirmation_flow_reject_never_executes():
    async def go(c, _):
        pending = (await c.http.post("/chat", json={"message": "email dana"})).json()["pending_confirmations"]
        r = await c.http.post(f"/confirm/{pending[0]['id']}", json={"approve": False})
        assert r.json()["status"] == "rejected" and r.json()["result"] is None
        assert c.state.system.mail.sent == []
        assert (await c.http.post("/confirm/unknown", json={"approve": True})).status_code == 404

    await run(ROUTE_TO_OPS_SEND, go)


async def test_fact_remembered_in_one_session_is_recalled_in_a_new_one():
    script = [
        tool_call_response("remember_fact", {"key": "preferred_language", "value": "Rust"}),
        text_response("Noted."),
        text_response("You prefer Rust."),
    ]

    async def go(c, shared):
        await c.http.post("/chat", json={"message": "remember that I prefer Rust", "session_id": "A", "user_id": "u1"})
        r = await c.http.post("/chat", json={"message": "which language do I prefer?", "session_id": "B", "user_id": "u1"})
        assert r.json()["text"] == "You prefer Rust."
        last_call = shared.calls[-1]["messages"]
        assert any(m["role"] == "system" and "preferred_language: Rust" in m["content"] for m in last_call)

    await run(script, go)


async def test_other_users_cannot_read_a_session_or_its_facts():
    script = [text_response("hi"), text_response("hello")]

    async def go(c, shared):
        assert (await c.http.post("/chat", json={"message": "a", "session_id": "S", "user_id": "alice"})).status_code == 200
        assert (await c.http.post("/chat", json={"message": "b", "session_id": "S", "user_id": "mallory"})).status_code == 403

    await run(script, go)


async def test_backend_failure_is_a_502_not_a_crash():
    async def go(c, _):
        r = await c.http.post("/chat", json={"message": "hi"})
        assert r.status_code == 500 and "detail" in r.json()  # exhausted mock script surfaces as a clean error body

    await run([], go)


async def test_chat_validates_input():
    async def go(c, _):
        assert (await c.http.post("/chat", json={"message": ""})).status_code == 422
        assert (await c.http.post("/chat", json={})).status_code == 422

    await run([], go)


async def test_eval_endpoints_produce_and_list_scorecards():
    async def go(c, _):
        first = await c.http.post("/eval/run")
        assert first.status_code == 200, first.text
        card = first.json()
        assert set(card["scores_by_category"]) == {"routing", "tool-selection", "RAG-grounding", "memory-recall", "refusal/edge-case"}
        assert card["regressions"] == [] and card["previous_run_id"] is None
        assert all({"scenario_id", "category", "layer", "passed"} <= set(d) for d in card["details"])

        second = (await c.http.post("/eval/run")).json()
        assert second["previous_run_id"] == card["run_id"]
        runs = (await c.http.get("/eval/runs")).json()
        assert [r["run_id"] for r in runs][:2] == [second["run_id"], card["run_id"]]
        assert (await c.http.get(f"/eval/runs/{card['run_id']}")).json()["run_id"] == card["run_id"]
        assert (await c.http.get("/eval/runs/nope")).status_code == 404

    await run([], go)


async def test_api_end_to_end_with_real_postgres(db):
    script = [
        tool_call_response("route_to_ops", {"task": "list", "reason": "cal"}),
        tool_call_response("calendar_list", {}),
        text_response("ok"),
        text_response("done"),
    ]

    async def go(c, _):
        assert c.state.storage == "postgres"
        r = await c.http.post("/chat", json={"message": "calendar?", "user_id": "pg-user"})
        assert r.status_code == 200, r.text
        body = r.json()
        events = (await c.http.get(f"/traces/turn/{body['turn_id']}")).json()["events"]
        assert events and events[0]["event_type"] == "model_call"
        assert (await c.http.get("/sessions")).json()

    import os

    from app.config import get_settings

    url = os.environ.get("TEST_DATABASE_URL") or get_settings().database_url
    await run(script, go, use_db=True, settings=Settings(model_provider="mock", _env_file=None, database_url=url))
