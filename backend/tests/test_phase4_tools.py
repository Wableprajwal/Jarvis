"""rag_search / calendar / email tools: validation, confirmation gating end to end, credential
plumbing. Everything runs against in-memory fakes; no model, no network."""
import json
from pathlib import Path

import pytest

from app.rag.embedder import HashingEmbedder
from app.rag.ingest import ingest_path
from app.rag.reranker import SimpleReranker
from app.rag.retriever import HybridRetriever
from app.rag.store import InMemoryVectorStore, NewChunk
from app.tools.calendar import FakeCalendarBackend, calendar_tools
from app.tools.gmail import FakeMailBackend, mail_tools
from app.tools.rag_tool import RagSearchTool
from app.tools.registry import MAX_TOOL_OUTPUT_CHARS, ToolRegistry

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"


# ---------------------------------------------------------------------------------- rag_search
@pytest.fixture
async def rag_registry():
    emb = HashingEmbedder()
    store = InMemoryVectorStore(emb.dim, emb.name)
    await ingest_path(DOCS_DIR, emb, store)
    reg = ToolRegistry()
    reg.register(RagSearchTool(HybridRetriever(store, emb, SimpleReranker())))
    reg.grant("research", ["rag_search"])
    return reg


async def test_rag_search_returns_cited_passages(rag_registry):
    res = await rag_registry.dispatch("research", "rag_search", {"query": "max retry attempts and backoff cap"}, "s")
    assert res.ok and not res.pending_confirmation
    citations = res.output
    assert isinstance(citations, list) and 1 <= len(citations) <= 5
    top = citations[0]
    assert top["source"] == "adr-0007-retry-backoff.md" and isinstance(top["chunk_id"], int)
    assert top["ref"] == f"{top['source']}#{top['chunk_id']}"
    assert "5 attempts" in top["text"] or "Maximum 5 attempts" in top["text"]
    assert {"source", "chunk_id", "text", "score"} <= set(top)


async def test_rag_search_output_survives_registry_truncation():
    emb = HashingEmbedder()
    store = InMemoryVectorStore(emb.dim, emb.name)
    for i in range(5):  # five near-maximal chunks (~2000 chars each)
        text = f"retry policy topic {i}. " + ("backoff detail. " * 125)
        await store.replace_document(f"big{i}.md", "h", [NewChunk(text, 0, {}, (await emb.embed([text]))[0])])
    reg = ToolRegistry()
    reg.register(RagSearchTool(HybridRetriever(store, emb, SimpleReranker())))
    reg.grant("r", ["rag_search"])
    res = await reg.dispatch("r", "rag_search", {"query": "retry policy backoff"}, "s")
    content = res.to_content()
    assert len(content) <= MAX_TOOL_OUTPUT_CHARS and "<truncated>" not in content.replace("…[truncated]", "")
    assert len(json.loads(content)["result"]) == 5  # still valid JSON with all five citations


async def test_rag_search_no_hits_is_an_empty_list_not_an_error():
    emb = HashingEmbedder()
    reg = ToolRegistry()
    reg.register(RagSearchTool(HybridRetriever(InMemoryVectorStore(emb.dim, emb.name), emb, SimpleReranker())))
    reg.grant("r", ["rag_search"])
    res = await reg.dispatch("r", "rag_search", {"query": "anything"}, "s")
    assert res.ok and res.output == []


@pytest.mark.parametrize(
    "args", [{}, {"query": ""}, {"query": 5}, {"query": "ok", "top_k": 3}, "not a dict"]
)
async def test_rag_search_rejects_bad_arguments(rag_registry, args):
    res = await rag_registry.dispatch("research", "rag_search", args, "s")
    assert not res.ok and ("invalid arguments" in res.error or "arguments must be" in res.error)


async def test_rag_search_blank_query_is_a_structured_error(rag_registry):
    res = await rag_registry.dispatch("research", "rag_search", {"query": "   "}, "s")
    assert not res.ok and "blank" in res.error


# ---------------------------------------------------------------------------------- ops tools
@pytest.fixture
def ops():
    cal, mail = FakeCalendarBackend(), FakeMailBackend()
    reg = ToolRegistry()
    for t in [*calendar_tools(cal), *mail_tools(mail)]:
        reg.register(t)
    reg.grant("ops", ["calendar_list", "calendar_create", "email_search", "email_draft", "email_send"])
    return reg, cal, mail


def test_confirmation_flags():
    tools = {t.name: t for t in [*calendar_tools(FakeCalendarBackend()), *mail_tools(FakeMailBackend())]}
    assert {n for n, t in tools.items() if t.requires_confirmation} == {"calendar_create", "email_send"}
    assert set(tools) == {"calendar_list", "calendar_create", "email_search", "email_draft", "email_send"}


async def test_calendar_list_reads_without_confirmation(ops):
    reg, cal, _ = ops
    res = await reg.dispatch("ops", "calendar_list", {"start": "2026-09-22", "end": "2026-09-24"}, "s")
    assert res.ok and not res.pending_confirmation
    assert [e["title"] for e in res.output["events"]] == ["1:1 with Priya", "Payments gateway incident review"]
    limited = await reg.dispatch("ops", "calendar_list", {"limit": 1}, "s")
    assert limited.output["count"] == 1


@pytest.mark.parametrize(
    "tool,args",
    [
        ("calendar_list", {"start": "next tuesday"}),
        ("calendar_list", {"limit": 0}),
        ("calendar_create", {"title": "x"}),
        ("calendar_create", {"title": "", "start": "2026-09-25T10:00", "end": "2026-09-25T11:00"}),
        ("calendar_create", {"title": "x", "start": "tomorrow", "end": "2026-09-25T11:00"}),
        ("calendar_create", {"title": "x", "start": "2026-09-25T10:00", "end": "2026-09-25T11:00", "attendees": ["nope"]}),
        ("email_search", {}),
        ("email_draft", {"to": [], "subject": "s", "body": "b"}),
        ("email_draft", {"to": ["not-an-address"], "subject": "s", "body": "b"}),
        ("email_send", {"subject": "s", "body": "b"}),  # neither draft_id nor to
        ("email_send", {"to": ["a@b.co"], "subject": "s", "body": "b", "cc": ["x@y.co"]}),
    ],
)
async def test_validation_errors_are_structured_and_never_queue_an_action(ops, tool, args):
    reg, cal, mail = ops
    res = await reg.dispatch("ops", tool, args, "s")
    assert not res.ok and not res.pending_confirmation and "invalid arguments" in res.error
    assert reg.pending() == [] and mail.sent == [] and len(cal.events) == 4


async def test_calendar_create_is_gated_then_performed_on_confirm(ops):
    reg, cal, _ = ops
    args = {"title": "Retry policy sync", "start": "2026-09-23T14:30:00Z", "end": "2026-09-23T15:00:00Z",
            "attendees": ["priya@example.com"]}
    res = await reg.dispatch("ops", "calendar_create", args, "sess")
    assert res.pending_confirmation and res.confirmation_id
    assert "NOT been performed" in res.to_content()
    assert len(cal.events) == 4 and not any(e.title == "Retry policy sync" for e in cal.events)  # nothing yet
    assert [p.tool.name for p in reg.pending("sess")] == ["calendar_create"]

    action = await reg.confirm(res.confirmation_id)
    assert action.status == "confirmed" and action.result.ok
    assert len(cal.events) == 5 and cal.events[-1].title == "Retry policy sync"
    out = action.result.output
    assert out["created"]["attendees"] == ["priya@example.com"]
    assert [e["title"] for e in out["overlapping_events"]] == ["Payments gateway incident review"]  # conflict surfaced
    assert reg.pending() == []


async def test_calendar_create_rejected_never_happens(ops):
    reg, cal, _ = ops
    res = await reg.dispatch(
        "ops", "calendar_create",
        {"title": "Nope", "start": "2026-09-25T10:00:00Z", "end": "2026-09-25T11:00:00Z"}, "s",
    )
    action = await reg.confirm(res.confirmation_id, approve=False)
    assert action.status == "rejected" and len(cal.events) == 4
    with pytest.raises(ValueError):
        await reg.confirm(res.confirmation_id)  # cannot be approved after rejection


async def test_calendar_create_end_before_start_fails_after_confirmation_without_side_effect(ops):
    reg, cal, _ = ops
    res = await reg.dispatch(
        "ops", "calendar_create",
        {"title": "Backwards", "start": "2026-09-25T11:00:00Z", "end": "2026-09-25T10:00:00Z"}, "s",
    )
    assert res.pending_confirmation  # shape is valid, semantics checked at execution
    action = await reg.confirm(res.confirmation_id)
    assert not action.result.ok and "after" in action.result.error and len(cal.events) == 4


async def test_email_search_and_draft_run_immediately_and_send_nothing(ops):
    reg, _, mail = ops
    found = await reg.dispatch("ops", "email_search", {"query": "retry adr"}, "s")
    assert found.ok and found.output["messages"][0]["id"] == "msg-1"
    draft = await reg.dispatch(
        "ops", "email_draft", {"to": ["priya@example.com"], "subject": "Re: ADR", "body": "Keeping 5 attempts."}, "s"
    )
    assert draft.ok and not draft.pending_confirmation and draft.output["sent"] is False
    assert draft.output["draft"]["id"] in mail.drafts and mail.sent == []


async def test_email_send_is_gated_then_delivered_on_confirm(ops):
    reg, _, mail = ops
    args = {"to": ["priya@example.com"], "subject": "Re: ADR-0007", "body": "Keeping 5 attempts."}
    res = await reg.dispatch("ops", "email_send", args, "sess")
    assert res.pending_confirmation
    assert mail.sent == []  # NOT sent before confirmation
    assert reg.pending("sess")[0].arguments == args

    action = await reg.confirm(res.confirmation_id)
    assert action.result.ok and action.result.output["sent"] is True
    assert len(mail.sent) == 1 and mail.sent[0].to == ["priya@example.com"] and mail.sent[0].subject == "Re: ADR-0007"


async def test_email_send_rejected_sends_nothing(ops):
    reg, _, mail = ops
    res = await reg.dispatch("ops", "email_send", {"to": ["a@b.co"], "subject": "s", "body": "b"}, "s")
    await reg.confirm(res.confirmation_id, approve=False)
    assert mail.sent == []


async def test_draft_then_send_by_draft_id(ops):
    reg, _, mail = ops
    draft = await reg.dispatch("ops", "email_draft", {"to": ["a@b.co"], "subject": "Hi", "body": "Body text"}, "s")
    did = draft.output["draft"]["id"]
    gated = await reg.dispatch("ops", "email_send", {"draft_id": did}, "s")
    assert gated.pending_confirmation and mail.sent == [] and did in mail.drafts
    action = await reg.confirm(gated.confirmation_id)
    assert action.result.ok
    assert [(m.to, m.subject, m.body) for m in mail.sent] == [(["a@b.co"], "Hi", "Body text")]
    assert did not in mail.drafts


async def test_send_with_unknown_draft_fails_cleanly_after_confirmation(ops):
    reg, _, mail = ops
    gated = await reg.dispatch("ops", "email_send", {"draft_id": "draft-999"}, "s")
    action = await reg.confirm(gated.confirmation_id)
    assert not action.result.ok and "draft-999" in action.result.error and mail.sent == []


# ---------------------------------------------------------------------------- credential plumbing
async def test_token_provider_is_used_but_never_leaks_into_results():
    secret = "ya29.SUPER-SECRET-TOKEN"
    cal, mail = FakeCalendarBackend(require_token=True), FakeMailBackend(require_token=True)
    reg = ToolRegistry()
    for t in [*calendar_tools(cal, lambda: secret), *mail_tools(mail, lambda: secret)]:
        reg.register(t)
    reg.grant("ops", ["calendar_list", "email_search", "email_send"])

    listed = await reg.dispatch("ops", "calendar_list", {}, "s")
    searched = await reg.dispatch("ops", "email_search", {"query": "retry"}, "s")
    gated = await reg.dispatch("ops", "email_send", {"to": ["a@b.co"], "subject": "s", "body": "b"}, "s")
    sent = await reg.confirm(gated.confirmation_id)
    assert listed.ok and searched.ok and sent.result.ok
    assert cal.tokens_seen == [secret] and secret in mail.tokens_seen
    for r in (listed, searched, gated, sent.result):
        assert secret not in r.to_content()


async def test_missing_credentials_fail_as_structured_errors():
    cal, mail = FakeCalendarBackend(require_token=True), FakeMailBackend(require_token=True)
    reg = ToolRegistry()
    for t in [*calendar_tools(cal, lambda: None), *mail_tools(mail)]:
        reg.register(t)
    reg.grant("ops", ["calendar_list", "email_search"])
    a = await reg.dispatch("ops", "calendar_list", {}, "s")
    b = await reg.dispatch("ops", "email_search", {"query": "x"}, "s")
    assert not a.ok and "access denied" in a.error and not b.ok and "access denied" in b.error
