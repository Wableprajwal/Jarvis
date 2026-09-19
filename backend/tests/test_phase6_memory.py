"""Phase 6 acceptance (short-term memory): token-budget summarization, run as contract tests
against both MemoryStore implementations. No live model: the summarizer is a MockProvider."""
import asyncio
import json

import pytest

from app.harness.memory import (
    SUMMARY_PREFIX,
    Fact,
    InMemoryMemoryStore,
    PostgresMemoryStore,
    estimate_tokens,
    message_tokens,
)
from app.observability.store import PostgresTracer, TraceStore
from app.observability.trace import InMemoryTracer
from app.providers.base import ModelResponse
from app.providers.mock_provider import MockProvider, text_response
from tests.test_phase6_env import add_turns, env, text_of, turn_messages  # noqa: F401  (env is a fixture)

BUDGET = 300  # ~3.3 turns of the default test size (90 tokens each); recent window = 40% = 120


def summarizer(*texts: str) -> MockProvider:
    if texts:
        return MockProvider(script=[text_response(t) for t in texts])
    return MockProvider(default=text_response("SUMMARY"))


def prompt_of(provider: MockProvider, call: int = 0) -> str:
    return text_of(provider.calls[call]["messages"])


def summary_blocks(ctx: list[dict]) -> list[dict]:
    return [m for m in ctx if (m.get("content") or "").startswith(SUMMARY_PREFIX)]


# --- basics ---------------------------------------------------------------------------------


async def test_append_and_get_context_roundtrip(env):
    store, sid = env.make(summarizer()), env.session()
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "name": "get_time", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "get_time", "content": '{"utc": "now"}'},
        {"role": "assistant", "content": "it is now"},
    ]
    for m in msgs:
        await store.append(sid, m)
    assert await store.get_context(sid) == msgs
    assert await store.get_all_messages(sid) == msgs
    assert await store.get_context("never-seen-" + sid) == []


async def test_get_context_returns_copies(env):
    store, sid = env.make(summarizer()), env.session()
    await store.append(sid, {"role": "user", "content": "hi"})
    (await store.get_context(sid))[0]["content"] = "mutated"
    assert (await store.get_context(sid))[0]["content"] == "hi"


async def test_invalid_role_rejected(env):
    store, sid = env.make(summarizer()), env.session()
    with pytest.raises(ValueError):
        await store.append(sid, {"role": "narrator", "content": "x"})


async def test_under_budget_never_calls_summarizer(env):
    prov = summarizer()
    store, sid = env.make(prov), env.session()
    await add_turns(store, sid, 0, 3)  # 270 tokens < 300
    await store.summarize_if_needed(sid, BUDGET)
    assert prov.calls == []
    assert len(await store.get_context(sid)) == 6


# --- acceptance (a): budget overflow -> summarize-and-replace -------------------------------


async def test_overflow_replaces_older_turns_with_one_summary_and_keeps_recent_verbatim(env):
    prov = summarizer("SUMMARY-ONE: user is debugging a parser")
    tracer = InMemoryTracer()
    store, sid = env.make(prov, tracer), env.session()
    await add_turns(store, sid, 0, 6)  # 540 tokens > 300
    before_ctx = await store.get_context(sid)
    assert summary_blocks(before_ctx) == []

    await store.summarize_if_needed(sid, BUDGET)

    assert len(prov.calls) == 1
    ctx = await store.get_context(sid)
    blocks = summary_blocks(ctx)
    assert len(blocks) == 1 and "SUMMARY-ONE" in blocks[0]["content"]
    assert ctx[0] is not None and ctx[0]["content"].startswith(SUMMARY_PREFIX)  # summary comes first
    # recent turn verbatim (in order), older turns gone from the live context
    assert ctx[1:] == turn_messages(5)
    live = text_of(ctx)
    for old in range(5):
        assert f"Q{old} " not in live and f"A{old} " not in live
    # the summarizer was actually shown the older turns (and only those)
    prompt = prompt_of(prov)
    for old in range(5):
        assert f"Q{old} " in prompt and f"A{old} " in prompt
    assert "Q5 " not in prompt
    # nothing is lost from the audit trail
    assert len(await store.get_all_messages(sid)) == 12

    (event,) = tracer.of_type("memory_summarize")
    assert event.session_id == sid
    assert event.payload["tokens_before"] > BUDGET
    assert event.payload["tokens_after"] < event.payload["tokens_before"]
    assert event.payload["messages_summarized"] == 10 and event.payload["messages_kept"] == 2
    assert event.payload["merged_prior_summary"] is False

    # after compression we are back under budget: a second call is a no-op
    await store.summarize_if_needed(sid, BUDGET)
    assert len(prov.calls) == 1


async def test_second_overflow_merges_prior_summary_instead_of_truncating(env):
    prov = summarizer("SUMMARY-ONE: about turns 0-4", "SUMMARY-TWO: about turns 0-7")
    tracer = InMemoryTracer()
    store, sid = env.make(prov, tracer), env.session()
    await add_turns(store, sid, 0, 6)
    await store.summarize_if_needed(sid, BUDGET)
    await add_turns(store, sid, 6, 9)  # summary + Q5.. grows past the budget again
    await store.summarize_if_needed(sid, BUDGET)

    assert len(prov.calls) == 2
    second = prompt_of(prov, 1)
    assert "SUMMARY-ONE" in second  # the existing summary was handed over to be merged...
    assert "Q5 " in second and "Q7 " in second  # ...together with the newly-aged turns
    assert "Q0 " not in second and "Q8 " not in second  # not re-summarizing raw old turns / recent ones

    ctx = await store.get_context(sid)
    (block,) = summary_blocks(ctx)  # still exactly ONE summary block, replaced not stacked
    assert "SUMMARY-TWO" in block["content"] and "SUMMARY-ONE" not in block["content"]
    assert ctx[1:] == turn_messages(8)
    assert [e.payload["merged_prior_summary"] for e in tracer.of_type("memory_summarize")] == [False, True]
    assert len(await store.get_all_messages(sid)) == 18


async def test_summarizer_asked_to_merge_and_capped(env):
    prov = summarizer()
    store, sid = env.make(prov), env.session()
    await add_turns(store, sid, 0, 6)
    await store.summarize_if_needed(sid, BUDGET)
    call = prov.calls[0]
    assert call["tools"] == []
    assert call["max_tokens"] < 2048  # summary size is bounded relative to the budget
    system = call["messages"][0]
    assert system["role"] == "system" and "EXISTING SUMMARY" in system["content"] + call["messages"][1]["content"]


def _tool_heavy_conversation(filler: int = 120) -> list[dict]:
    pad = "y" * filler
    return [
        {"role": "user", "content": f"U0 {pad}"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "name": "read_file", "arguments": {"path": "a.py"}},
            {"id": "c2", "name": "read_file", "arguments": {"path": "b.py"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": f"file a {pad}"},
        {"role": "tool", "tool_call_id": "c2", "name": "read_file", "content": f"file b {pad}"},
        {"role": "assistant", "content": f"A0 {pad}"},
        {"role": "user", "content": f"U1 {pad}"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c3", "name": "read_file", "arguments": {"path": "c.py"}}]},
        {"role": "tool", "tool_call_id": "c3", "name": "read_file", "content": f"file c {pad}"},
        {"role": "assistant", "content": f"A1 {pad}"},
        {"role": "user", "content": f"U2 {pad}"},
        {"role": "assistant", "content": f"A2 {pad}"},
    ]


def assert_tool_pairs_intact(messages: list[dict]) -> None:
    """Every tool result follows the assistant message that issued it, and every tool call has
    all of its results present: no orphans on either side of the cut."""
    declared: set[str] = set()
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and m.get("tool_calls"):
            ids = {c["id"] for c in m["tool_calls"]}
            declared |= ids
            following = {x["tool_call_id"] for x in messages[i + 1:] if x["role"] == "tool"}
            assert ids <= following, f"tool calls {ids - following} lost their results"
        if m["role"] == "tool":
            assert m["tool_call_id"] in declared, f"orphaned tool message {m['tool_call_id']}"
            assert i > 0 and messages[i - 1]["role"] in ("assistant", "tool")


@pytest.mark.parametrize("budget", range(60, 800, 23))
async def test_tool_call_and_result_are_never_split(env, budget):
    prov = summarizer()
    store, sid = env.make(prov), env.session()
    conversation = _tool_heavy_conversation()
    for m in conversation:
        await store.append(sid, m)
    await store.summarize_if_needed(sid, budget)

    ctx = await store.get_context(sid)
    assert_tool_pairs_intact([m for m in ctx if not (m.get("content") or "").startswith(SUMMARY_PREFIX)])
    if prov.calls:  # what was summarized is a whole-turn prefix, and the rest is kept verbatim
        kept = [m for m in ctx if not (m.get("content") or "").startswith(SUMMARY_PREFIX)]
        assert kept == conversation[len(conversation) - len(kept):]
        assert kept[0]["role"] != "tool"


def test_budget_sweep_actually_exercises_the_tool_split_hazard():
    """Sanity check on the sweep above: a naive token-only cut (keep the longest suffix that fits
    the recent window) lands on a `tool` message for some of those budgets, i.e. the test would
    catch a store that trims mid tool group."""
    conversation = _tool_heavy_conversation()
    tokens = [message_tokens(m) for m in conversation]
    hazardous = 0
    for budget in range(60, 800, 23):
        recent_budget = int(budget * 0.4)
        naive_cut = next(i for i in range(len(conversation) + 1) if sum(tokens[i:]) <= recent_budget)
        if naive_cut < len(conversation) and conversation[naive_cut]["role"] == "tool":
            hazardous += 1
    assert hazardous >= 3


async def test_single_long_agent_loop_still_summarizes_at_a_safe_boundary(env):
    pad = "z" * 200
    conv = [{"role": "user", "content": "do the big task"}]
    for i in range(4):
        conv.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"k{i}", "name": "t", "arguments": {"i": i}}]})
        conv.append({"role": "tool", "tool_call_id": f"k{i}", "name": "t", "content": f"out{i} {pad}"})
    conv.append({"role": "assistant", "content": "done"})
    prov = summarizer()
    store, sid = env.make(prov), env.session()
    for m in conv:
        await store.append(sid, m)
    await store.summarize_if_needed(sid, 200)
    assert len(prov.calls) == 1
    kept = [m for m in await store.get_context(sid) if not (m.get("content") or "").startswith(SUMMARY_PREFIX)]
    assert kept and kept[0]["role"] == "assistant"
    assert_tool_pairs_intact(kept)


async def test_single_oversized_turn_is_left_alone(env):
    prov = summarizer()
    store, sid = env.make(prov), env.session()
    await store.append(sid, {"role": "user", "content": "x" * 4000})
    await store.summarize_if_needed(sid, 100)
    assert prov.calls == []  # nothing older to compress
    assert len(await store.get_context(sid)) == 1


async def test_system_messages_are_pinned_and_never_summarized(env):
    prov = summarizer()
    store, sid = env.make(prov), env.session()
    await store.append(sid, {"role": "system", "content": "PINNED-SYSTEM-MARKER be nice"})
    await add_turns(store, sid, 0, 6)
    await store.summarize_if_needed(sid, BUDGET)
    assert len(prov.calls) == 1
    assert "PINNED-SYSTEM-MARKER" not in prompt_of(prov)
    ctx = await store.get_context(sid)
    assert ctx[0] == {"role": "system", "content": "PINNED-SYSTEM-MARKER be nice"}
    assert ctx[1]["content"].startswith(SUMMARY_PREFIX)


async def test_summary_role_is_configurable(env):
    store, sid = env.make(summarizer(), summary_role="user"), env.session()
    await add_turns(store, sid, 0, 6)
    await store.summarize_if_needed(sid, BUDGET)
    assert (await store.get_context(sid))[0]["role"] == "user"


# --- failure handling -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [MockProvider(), MockProvider(default=ModelResponse(text=None, error="boom")), MockProvider(default=text_response("   "))],
    ids=["raises", "error-field", "blank"],
)
async def test_summarizer_failure_keeps_every_message_and_traces_error(env, bad):
    tracer = InMemoryTracer()
    store, sid = env.make(bad, tracer), env.session()
    await add_turns(store, sid, 0, 6)
    await store.summarize_if_needed(sid, BUDGET)  # must not raise
    assert len(await store.get_context(sid)) == 12 and summary_blocks(await store.get_context(sid)) == []
    assert tracer.of_type("memory_summarize") == []
    (err,) = tracer.of_type("error")
    assert err.payload["stage"] == "memory_summarize"


# --- token counting -------------------------------------------------------------------------


def test_default_token_estimate_is_chars_over_four():
    assert estimate_tokens("") == 0 and estimate_tokens("abcd") == 1 and estimate_tokens("abcde") == 2
    assert message_tokens({"role": "user", "content": "abcd"}) == 4 + 1
    with_calls = {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "name": "n", "arguments": {}}]}
    assert message_tokens(with_calls) > 4


async def test_token_counter_is_injectable(env):
    wordy = "supercalifragilistic " * 3  # 3 "words" but ~63 characters
    turns = [{"role": "user" if i % 2 == 0 else "assistant", "content": wordy} for i in range(12)]

    async def run(**kw) -> int:
        prov = summarizer()
        store, sid = env.make(prov, **kw), env.session()
        for m in turns:
            await store.append(sid, m)
        await store.summarize_if_needed(sid, 100)
        return len(prov.calls)

    assert await run() == 1  # chars/4: 12 * ~20 tokens > 100
    assert await run(token_counter=lambda s: len(s.split())) == 0  # word counter: 12 * 7 < 100


# --- concurrency ----------------------------------------------------------------------------


async def test_concurrent_summarize_calls_summarize_once(env):
    prov = summarizer()
    store, sid = env.make(prov), env.session()
    await add_turns(store, sid, 0, 6)
    await asyncio.gather(*(store.summarize_if_needed(sid, BUDGET) for _ in range(4)))
    assert len(prov.calls) == 1
    assert len(summary_blocks(await store.get_context(sid))) == 1


# --- sessions & long-term facts (same contract for both stores) -----------------------------


async def test_session_lifecycle(env):
    store, sid, uid = env.make(summarizer()), env.session(), env.user()
    assert await store.get_session(sid) is None
    info = await store.ensure_session(sid, uid)
    assert info.user_id == uid and info.closed_at is None
    again = await store.ensure_session(sid, "someone-else")  # owner is never overwritten
    assert again.user_id == uid
    await store.close_session(sid)
    assert (await store.get_session(sid)).closed_at is not None


async def test_append_creates_missing_session(env):
    store, sid = env.make(summarizer()), env.session()
    await store.append(sid, {"role": "user", "content": "hello"})
    assert (await store.get_session(sid)).user_id == "default"


async def test_facts_upsert_updates_in_place_and_isolate_users(env):
    store, alice, bob = env.make(summarizer()), env.user(), env.user()
    await store.upsert_long_term_fact(alice, Fact("language", "Python", source="explicit"))
    await store.upsert_long_term_fact(alice, Fact("editor", "vim"))
    await store.upsert_long_term_fact(alice, Fact("Language", "Rust 1.80"))  # same key after normalisation
    facts = await store.get_long_term_facts(alice)
    assert [(f.key, f.value) for f in facts] == [("editor", "vim"), ("language", "Rust 1.80")]
    assert all(f.updated_at is not None for f in facts)
    assert facts[1].source is None  # the update replaced the source with the new fact's
    assert await store.get_long_term_facts(bob) == []


@pytest.mark.parametrize(
    "fact",
    [Fact("", "v"), Fact("k", "   "), Fact("api_key", "sk-abcdefghijklmnopqrstuvwx"), Fact("note", "my password is hunter22!")],
    ids=["empty-key", "blank-value", "api-key", "password"],
)
async def test_bad_or_secret_facts_are_refused(env, fact):
    store, uid = env.make(summarizer()), env.user()
    with pytest.raises(ValueError):
        await store.upsert_long_term_fact(uid, fact)
    assert await store.get_long_term_facts(uid) == []


# --- Postgres specifics ---------------------------------------------------------------------


async def test_postgres_marks_rows_summarized_and_persists_summary_and_trace(db, session_id):
    tracer = PostgresTracer(db)
    store = PostgresMemoryStore(db, summarizer("SUMMARY-PG"), tracer)
    try:
        await add_turns(store, session_id, 0, 6)
        await store.summarize_if_needed(session_id, BUDGET)

        rows = await db.pool.fetch("SELECT summarized FROM messages WHERE session_id = $1 ORDER BY id", session_id)
        assert [r["summarized"] for r in rows] == [True] * 10 + [False] * 2  # flagged, not deleted
        assert await db.pool.fetchval("SELECT summary FROM sessions WHERE id = $1", session_id) == "SUMMARY-PG"

        # a fresh store instance (e.g. after a restart) sees the same window
        fresh = PostgresMemoryStore(db, summarizer())
        ctx = await fresh.get_context(session_id)
        assert ctx[0]["content"].endswith("SUMMARY-PG") and ctx[1:] == turn_messages(5)

        events = [e for e in await TraceStore(db).get_session(session_id) if e["event_type"] == "memory_summarize"]
        assert len(events) == 1
        assert events[0]["agent_name"] == "memory"
        assert events[0]["payload"]["tokens_after"] < events[0]["payload"]["tokens_before"]
        json.dumps(events[0]["payload"])
    finally:
        await db.pool.execute("DELETE FROM sessions WHERE id = $1", session_id)
        await db.pool.execute("DELETE FROM trace_events WHERE session_id = $1", session_id)


async def test_postgres_commit_loses_race_gracefully(db, session_id):
    """If another worker already replaced the summary, our commit must change nothing."""
    store = PostgresMemoryStore(db, summarizer())
    try:
        await add_turns(store, session_id, 0, 2)
        _, rows = await store._load_active(session_id)
        await db.pool.execute("UPDATE sessions SET summary = 'someone else' WHERE id = $1", session_id)
        assert await store._commit_summary(session_id, None, "mine", [r.id for r in rows]) is False
        assert await db.pool.fetchval("SELECT count(*) FROM messages WHERE session_id=$1 AND summarized", session_id) == 0
        assert await db.pool.fetchval("SELECT summary FROM sessions WHERE id = $1", session_id) == "someone else"
    finally:
        await db.pool.execute("DELETE FROM sessions WHERE id = $1", session_id)


def test_in_memory_store_needs_no_database():
    assert isinstance(InMemoryMemoryStore(MockProvider()), InMemoryMemoryStore)
