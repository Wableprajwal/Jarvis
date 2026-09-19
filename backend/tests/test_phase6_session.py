"""Phase 6: SessionManager (history building, turn recording, fact extraction) and the
remember_fact tool. Contract tests over both stores; MockProvider only."""
import pytest

from app.harness.memory import SUMMARY_PREFIX, Fact
from app.harness.session import (
    FACTS_HEADER,
    SessionManager,
    SessionOwnershipError,
    parse_extracted_facts,
)
from app.providers.base import ModelResponse
from app.providers.mock_provider import MockProvider, text_response
from app.tools.memory_tool import RememberFactTool, current_user_id, current_user_var
from app.tools.registry import ToolRegistry
from tests.test_phase6_env import add_turns, env, text_of  # noqa: F401  (env is a fixture)

BUDGET = 300


def manager(env, summarizer=None, **kw):
    memory = env.make(summarizer or MockProvider(default=text_response("SUMMARY")))
    return SessionManager(memory, token_budget=BUDGET, **kw), memory


# --- parsing --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"facts": [{"key": "lang", "value": "Python"}]}', [("lang", "Python")]),
        ('```json\n{"facts": [{"key": "lang", "value": "Python"}]}\n```', [("lang", "Python")]),
        ('Sure! Here you go:\n{"facts": [{"key": "a", "value": "b"}]}\nHope that helps', [("a", "b")]),
        ('[{"key": "a", "value": "b"}, {"key": "c", "value": 3}]', [("a", "b"), ("c", "3")]),
        ('{"lang": "Python", "editor": "vim"}', [("lang", "Python"), ("editor", "vim")]),
        ('{"facts": []}', []),
        ('{"facts": [{"key": "a"}, {"value": "b"}, "junk", 7, {"key": "", "value": "x"}, {"key": 1, "value": "x"}]}', []),
        ('{"facts": [{"key": "lang", "value": "Py"', []),  # truncated JSON
        ("I found no durable facts.", []),
        ("", []),
        (None, []),
        ("{{{{ not json ]]]", []),
    ],
)
def test_parse_extracted_facts_is_tolerant(text, expected):
    assert parse_extracted_facts(text) == expected


def test_parse_extracted_facts_caps_count_and_length():
    many = ",".join(f'{{"key": "k{i}", "value": "{"v" * 900}"}}' for i in range(50))
    pairs = parse_extracted_facts(f'{{"facts": [{many}]}}')
    assert len(pairs) == 20 and all(len(v) == 500 for _, v in pairs)


# --- sessions & history ---------------------------------------------------------------------


async def test_create_and_get_session(env):
    sm, _ = manager(env)
    uid = env.user()
    sid = await sm.create_session(uid)
    env.session_ids.append(sid)
    info = await sm.get_session(sid)
    assert info.user_id == uid and info.closed_at is None
    assert await sm.get_session("does-not-exist-" + sid) is None
    explicit = env.session()
    assert await sm.create_session(uid, session_id=explicit) == explicit


async def test_build_history_for_empty_session_is_empty(env):
    sm, _ = manager(env)
    assert await sm.build_history(env.session(), env.user()) == []


async def test_build_history_puts_facts_first_then_context(env):
    sm, memory = manager(env)
    uid, sid = env.user(), env.session()
    await memory.upsert_long_term_fact(uid, Fact("language", "Python 3.12"))
    await memory.upsert_long_term_fact(uid, Fact("db", "Postgres"))
    await sm.record_turn(sid, "hello", "hi there", user_id=uid)
    history = await sm.build_history(sid, uid)
    assert history[0]["role"] == "system"
    assert history[0]["content"].startswith(FACTS_HEADER)
    assert "- db: Postgres" in history[0]["content"] and "- language: Python 3.12" in history[0]["content"]
    assert history[1:] == [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"}]


async def test_build_history_caps_facts_to_most_recent(env):
    sm, memory = manager(env, max_facts=2)
    uid, sid = env.user(), env.session()
    for k in ("old", "mid", "new"):
        await memory.upsert_long_term_fact(uid, Fact(k, f"{k}-value"))
    content = (await sm.build_history(sid, uid))[0]["content"]
    assert "new-value" in content and "mid-value" in content and "old-value" not in content


async def test_facts_do_not_leak_across_users_or_sessions_of_other_users(env):
    sm, memory = manager(env)
    alice, bob = env.user(), env.user()
    await memory.upsert_long_term_fact(alice, Fact("secret_project", "Apollo"))
    assert await sm.build_history(env.session(), bob) == []
    # bob cannot read alice's session (and thereby her context) by guessing its id
    alice_session = env.session()
    await sm.build_history(alice_session, alice)
    with pytest.raises(SessionOwnershipError):
        await sm.build_history(alice_session, bob)


async def test_build_history_defaults_to_the_sessions_owner(env):
    sm, memory = manager(env)
    uid = env.user()
    sid = await sm.create_session(uid, session_id=env.session())
    await memory.upsert_long_term_fact(uid, Fact("k", "owner-fact"))
    assert "owner-fact" in (await sm.build_history(sid))[0]["content"]


async def test_record_turn_persists_user_and_assistant_only(env):
    sm, memory = manager(env)
    sid = env.session()
    await sm.record_turn(sid, "q", "a")
    await sm.record_turn(sid, "q2", None)  # e.g. a truncated turn with no final answer
    assert await memory.get_all_messages(sid) == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "q2"},
    ]


async def test_record_turn_summarizes_once_over_budget(env):
    prov = MockProvider(default=text_response("SUMMARY-FROM-SESSION"))
    sm, memory = manager(env, prov)
    uid = env.user()
    sid = await sm.create_session(uid, session_id=env.session())
    pad = "p" * 160
    for i in range(6):
        await sm.record_turn(sid, f"Q{i} {pad}", f"A{i} {pad}")
        if i < 3:
            assert prov.calls == []  # still under the token budget after 3 turns
    assert len(prov.calls) >= 1
    history = await sm.build_history(sid, uid)
    blocks = [m for m in history if m["content"].startswith(SUMMARY_PREFIX)]
    assert len(blocks) == 1 and "SUMMARY-FROM-SESSION" in blocks[0]["content"]
    assert "Q0 " not in text_of(history) and f"Q5 {pad}" in text_of(history)


async def test_default_budget_comes_from_config(env):
    from app.config import get_settings

    assert SessionManager(env.make(MockProvider())).token_budget == get_settings().memory_token_budget


# --- extraction pass ------------------------------------------------------------------------


async def _session_with_chat(env, sm, uid):
    sid = await sm.create_session(uid, session_id=env.session())
    await sm.record_turn(sid, "I mostly write Go and deploy to Fly.io", "Great, noted.")
    return sid


async def test_extract_facts_upserts_for_the_session_owner(env):
    sm, memory = manager(env)
    uid = env.user()
    sid = await _session_with_chat(env, sm, uid)
    reply = '```json\n{"facts": [{"key": "Main Language", "value": "Go"}, {"key": "hosting", "value": "Fly.io"}]}\n```'
    provider = MockProvider(script=[text_response(reply)])

    saved = await sm.extract_facts(sid, provider)

    assert [f.key for f in saved] == ["Main Language", "hosting"]
    facts = {f.key: f for f in await memory.get_long_term_facts(uid)}
    assert facts["main_language"].value == "Go" and facts["hosting"].source == f"extracted:{sid}"
    # the model saw the conversation and was told to avoid secrets / transient state
    sent = text_of(provider.calls[0]["messages"])
    assert "Go and deploy to Fly.io" in sent and "NEVER secrets" in sent
    assert provider.calls[0]["tools"] == []


@pytest.mark.parametrize(
    "reply",
    ["I could not find anything.", '{"facts": [{"key": "lang"', "```json\n{{oops}}\n```", "[1, 2, 3]", ""],
)
async def test_extract_facts_tolerates_malformed_model_output(env, reply):
    sm, memory = manager(env)
    uid = env.user()
    sid = await _session_with_chat(env, sm, uid)
    assert await sm.extract_facts(sid, MockProvider(script=[text_response(reply)])) == []
    assert await memory.get_long_term_facts(uid) == []


async def test_extract_facts_ignores_provider_error_field(env):
    sm, _ = manager(env)
    sid = await _session_with_chat(env, sm, env.user())
    assert await sm.extract_facts(sid, MockProvider(script=[ModelResponse(text=None, error="bad")])) == []


async def test_extract_facts_skips_credentials_but_keeps_the_rest(env):
    sm, memory = manager(env)
    uid = env.user()
    sid = await _session_with_chat(env, sm, uid)
    reply = (
        '{"facts": [{"key": "token", "value": "ghp_abcdefghijklmnopqrstuvwxyz0123456789"},'
        ' {"key": "editor", "value": "neovim"}]}'
    )
    saved = await sm.extract_facts(sid, MockProvider(script=[text_response(reply)]))
    assert [f.key for f in saved] == ["editor"]
    assert [f.key for f in await memory.get_long_term_facts(uid)] == ["editor"]


async def test_extract_facts_on_unknown_or_empty_session_makes_no_model_call(env):
    sm, _ = manager(env)
    prov = MockProvider(default=text_response('{"facts": []}'))
    assert await sm.extract_facts("no-such-session-" + env.session(), prov) == []
    empty = await sm.create_session(env.user(), session_id=env.session())
    assert await sm.extract_facts(empty, prov) == []
    assert prov.calls == []


async def test_close_session_marks_closed_and_optionally_extracts(env):
    sm, memory = manager(env)
    uid = env.user()
    sid = await _session_with_chat(env, sm, uid)
    assert await sm.close_session(sid) == []
    assert (await sm.get_session(sid)).closed_at is not None

    sid2 = await _session_with_chat(env, sm, uid)
    prov = MockProvider(script=[text_response('{"facts": [{"key": "lang", "value": "Go"}]}')])
    saved = await sm.close_session(sid2, extract_with=prov)
    assert [f.key for f in saved] == ["lang"]
    assert (await memory.get_long_term_facts(uid))[0].value == "Go"


async def test_extraction_reads_summary_when_session_was_compressed(env):
    prov = MockProvider(default=text_response("SUMMARY: the user's stack is Rust and Kafka"))
    sm, _ = manager(env, prov)
    uid = env.user()
    sid = await sm.create_session(uid, session_id=env.session())
    pad = "p" * 160
    for i in range(6):
        await sm.record_turn(sid, f"Q{i} {pad}", f"A{i} {pad}")
    extractor = MockProvider(script=[text_response('{"facts": []}')])
    await sm.extract_facts(sid, extractor)
    sent = text_of(extractor.calls[0]["messages"])
    assert "Rust and Kafka" in sent  # older turns reach the extractor via the summary


# --- remember_fact tool ---------------------------------------------------------------------


def registry_with(tool):
    reg = ToolRegistry()
    reg.register(tool)
    reg.grant("orchestrator", ["remember_fact"])
    return reg


async def test_remember_fact_tool_stores_fact_for_the_current_user(env):
    memory = env.make(MockProvider())
    uid = env.user()
    reg = registry_with(RememberFactTool(memory, lambda: uid))

    res = await reg.dispatch("orchestrator", "remember_fact", {"key": "shell", "value": "zsh"}, "s1")

    assert res.ok and res.output == {"remembered": {"key": "shell", "value": "zsh"}}
    (fact,) = await memory.get_long_term_facts(uid)
    assert (fact.key, fact.value, fact.source) == ("shell", "zsh", "explicit")


async def test_remember_fact_tool_supports_async_user_provider_and_updates(env):
    memory = env.make(MockProvider())
    uid = env.user()

    async def who() -> str:
        return uid

    tool = RememberFactTool(memory, who)
    await tool.execute({"key": "shell", "value": "zsh"})
    await tool.execute({"key": "shell", "value": "fish"})
    assert [(f.key, f.value) for f in await memory.get_long_term_facts(uid)] == [("shell", "fish")]


@pytest.mark.parametrize("args", [{"key": "only-key"}, {"value": "only-value"}, {"key": "", "value": "v"}, {"key": "k", "value": "v", "x": 1}])
async def test_remember_fact_tool_validates_arguments(env, args):
    memory = env.make(MockProvider())
    uid = env.user()
    res = await registry_with(RememberFactTool(memory, lambda: uid)).dispatch("orchestrator", "remember_fact", args, "s1")
    assert not res.ok and "invalid arguments" in res.error
    assert await memory.get_long_term_facts(uid) == []


async def test_remember_fact_tool_refuses_credentials(env):
    memory = env.make(MockProvider())
    uid = env.user()
    tool = RememberFactTool(memory, lambda: uid)
    res = await tool.execute({"key": "openai", "value": "sk-abcdefghijklmnopqrstuvwxyz123456"})
    assert not res.ok and "credential" in res.error
    assert await memory.get_long_term_facts(uid) == []


async def test_remember_fact_tool_respects_allow_list(env):
    reg = ToolRegistry()
    reg.register(RememberFactTool(env.make(MockProvider()), lambda: "u"))
    reg.grant("orchestrator", ["remember_fact"])
    res = await reg.dispatch("code", "remember_fact", {"key": "k", "value": "v"}, "s1")
    assert not res.ok and "not permitted" in res.error


async def test_remember_fact_default_user_provider_is_a_contextvar(env):
    memory = env.make(MockProvider())
    uid = env.user()
    token = current_user_var.set(uid)
    try:
        assert current_user_id() == uid
        await RememberFactTool(memory).execute({"key": "k", "value": "v"})
    finally:
        current_user_var.reset(token)
    assert [f.key for f in await memory.get_long_term_facts(uid)] == ["k"]
