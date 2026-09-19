"""Mail tools for the Ops agent (module named after the eventual Gmail integration).

`MailBackend` is the seam to a mail provider; only an in-memory fake ships, with a marked
extension point for real Gmail OAuth. Tools take the backend and an optional `token_provider`
by injection (no credential-module import); the token goes to the backend only and never into
results. `email_send` is confirmation-gated: it reaches `execute()` only via `registry.confirm`.
"""
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.tools.registry import Tool, ToolResult

TokenProvider = Callable[[], "str | None"]
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
SNIPPET_CHARS = 300


@dataclass
class Email:
    id: str
    sender: str
    to: list[str]
    subject: str
    body: str
    date: str = ""
    cc: list[str] = field(default_factory=list)

    def to_dict(self, *, snippet: bool = False) -> dict:
        body = self.body if not snippet or len(self.body) <= SNIPPET_CHARS else self.body[:SNIPPET_CHARS] + "…"
        return {
            "id": self.id, "from": self.sender, "to": self.to, "cc": self.cc, "subject": self.subject,
            "date": self.date, "body": body,
        }


class MailBackend(ABC):
    """`token` is whatever the injected token_provider returned; backends raise PermissionError
    when credentials are missing/invalid."""

    @abstractmethod
    async def search(self, query: str, limit: int, token: str | None = None) -> list[Email]: ...

    @abstractmethod
    async def create_draft(self, to: list[str], subject: str, body: str, token: str | None = None) -> Email: ...

    @abstractmethod
    async def get_draft(self, draft_id: str, token: str | None = None) -> Email | None: ...

    @abstractmethod
    async def send(
        self, to: list[str], subject: str, body: str, draft_id: str | None = None, token: str | None = None
    ) -> Email:
        """Actually deliver a message (irreversible). Removes `draft_id` if given."""


class FakeMailBackend(MailBackend):
    """In-memory mailbox: `inbox` (seeded), `drafts`, and `sent` — the list tests inspect to prove
    whether an irreversible send really happened."""

    def __init__(self, inbox: list[Email] | None = None, *, require_token: bool = False):
        self.inbox: list[Email] = list(inbox) if inbox is not None else self._seed()
        self.drafts: dict[str, Email] = {}
        self.sent: list[Email] = []
        self.require_token = require_token
        self.tokens_seen: list[str | None] = []
        self._ids = itertools.count(1)

    @staticmethod
    def _seed() -> list[Email]:
        me = "me@example.com"
        return [
            Email("msg-1", "priya@example.com", [me], "Retry policy ADR review",
                  "Hi, I read ADR-0007. Can we lower max attempts from 5 to 4 for the payments gateway?", "2026-09-16"),
            Email("msg-2", "pagerduty@example.com", [me], "You are on call next week",
                  "Your primary on-call shift starts Monday 10:00 UTC. Secondary: Marcus.", "2026-09-17"),
            Email("msg-3", "ci-bot@example.com", [me], "Build 4812 failed",
                  "Stage 'integration-tests' failed: 3 tests in test_ledger.py. Coverage gate 85% not evaluated.", "2026-09-18"),
            Email("msg-4", "hr@example.com", [me], "Benefits enrollment closes Friday",
                  "Reminder: benefits enrollment closes this Friday at 17:00.", "2026-09-15"),
        ]

    def _auth(self, token: str | None) -> None:
        self.tokens_seen.append(token)
        if self.require_token and not token:
            raise PermissionError("mail credentials are not configured")

    async def search(self, query: str, limit: int, token: str | None = None) -> list[Email]:
        self._auth(token)
        terms = [t for t in query.lower().split() if t]
        scored = []
        for m in self.inbox:
            hay = f"{m.sender} {m.subject} {m.body}".lower()
            hits = sum(t in hay for t in terms)
            if hits:
                scored.append((hits, m))
        scored.sort(key=lambda hm: (-hm[0], hm[1].id))
        return [m for _, m in scored[:limit]]

    async def create_draft(self, to, subject, body, token=None) -> Email:
        self._auth(token)
        draft = Email(f"draft-{next(self._ids)}", "me@example.com", list(to), subject, body, _today())
        self.drafts[draft.id] = draft
        return draft

    async def get_draft(self, draft_id, token=None) -> Email | None:
        self._auth(token)
        return self.drafts.get(draft_id)

    async def send(self, to, subject, body, draft_id=None, token=None) -> Email:
        self._auth(token)
        if draft_id is not None and draft_id not in self.drafts:
            raise KeyError(f"no such draft: {draft_id}")
        msg = Email(f"sent-{next(self._ids)}", "me@example.com", list(to), subject, body, _today())
        self.sent.append(msg)
        if draft_id is not None:
            del self.drafts[draft_id]
        return msg


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class GmailBackend(MailBackend):
    """EXTENSION POINT (optional, last): real Gmail via OAuth.

    Implement with the Gmail REST API (users.messages.list / users.drafts.create /
    users.messages.send) using the bearer token supplied by the injected token_provider (wired
    to app/tools/credentials.py by the app bootstrap, never imported here). Until then the
    fake backend is used everywhere.
    """

    async def search(self, query, limit, token=None):
        raise NotImplementedError("GmailBackend is not implemented; use FakeMailBackend")

    async def create_draft(self, to, subject, body, token=None):
        raise NotImplementedError("GmailBackend is not implemented; use FakeMailBackend")

    async def get_draft(self, draft_id, token=None):
        raise NotImplementedError("GmailBackend is not implemented; use FakeMailBackend")

    async def send(self, to, subject, body, draft_id=None, token=None):
        raise NotImplementedError("GmailBackend is not implemented; use FakeMailBackend")


class _MailTool(Tool):
    def __init__(self, backend: MailBackend, token_provider: TokenProvider | None = None):
        self._backend = backend
        self._token_provider = token_provider

    def _token(self) -> str | None:
        return self._token_provider() if self._token_provider else None


_ADDRESSES = {"type": "array", "items": {"type": "string", "pattern": EMAIL_PATTERN}, "minItems": 1, "maxItems": 50}


class EmailSearchTool(_MailTool):
    name = "email_search"
    description = "Search the mailbox (sender, subject, body). Returns matching messages with body snippets."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 25, "default": 10},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict) -> ToolResult:
        try:
            msgs = await self._backend.search(arguments["query"], arguments.get("limit", 10), self._token())
        except PermissionError as exc:
            return ToolResult.failure(f"mail access denied: {exc}")
        return ToolResult.success({"messages": [m.to_dict(snippet=True) for m in msgs], "count": len(msgs)})


class EmailDraftTool(_MailTool):
    name = "email_draft"
    description = "Save an email draft (NOT sent). Returns a draft id that email_send can later send."
    input_schema = {
        "type": "object",
        "properties": {
            "to": _ADDRESSES,
            "subject": {"type": "string", "minLength": 1, "maxLength": 300},
            "body": {"type": "string", "minLength": 1},
        },
        "required": ["to", "subject", "body"],
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict) -> ToolResult:
        try:
            draft = await self._backend.create_draft(
                arguments["to"], arguments["subject"], arguments["body"], self._token()
            )
        except PermissionError as exc:
            return ToolResult.failure(f"mail access denied: {exc}")
        return ToolResult.success({"draft": draft.to_dict(), "sent": False})


class EmailSendTool(_MailTool):
    name = "email_send"
    requires_confirmation = True
    description = (
        "Send an email. IRREVERSIBLE and REQUIRES USER CONFIRMATION: calling this only queues the "
        "send; nothing is sent until the user approves. Provide either `draft_id` (a draft from "
        "email_draft) or `to`, `subject` and `body`."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "draft_id": {"type": "string", "minLength": 1},
            "to": _ADDRESSES,
            "subject": {"type": "string", "minLength": 1, "maxLength": 300},
            "body": {"type": "string", "minLength": 1},
        },
        "anyOf": [{"required": ["draft_id"]}, {"required": ["to", "subject", "body"]}],
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict) -> ToolResult:
        draft_id = arguments.get("draft_id")
        to, subject, body = arguments.get("to"), arguments.get("subject"), arguments.get("body")
        token = self._token()
        try:
            if draft_id and not (to and subject and body):
                draft = await self._backend.get_draft(draft_id, token)
                if draft is None:
                    return ToolResult.failure(f"unknown draft_id '{draft_id}'; provide to/subject/body instead")
                to, subject, body = draft.to, draft.subject, draft.body
            sent = await self._backend.send(to, subject, body, draft_id, token)
        except PermissionError as exc:
            return ToolResult.failure(f"mail access denied: {exc}")
        except KeyError as exc:
            return ToolResult.failure(str(exc))
        return ToolResult.success({"sent": True, "message": sent.to_dict(snippet=True)})


def mail_tools(backend: MailBackend, token_provider: TokenProvider | None = None) -> list[Tool]:
    return [EmailSearchTool(backend, token_provider), EmailDraftTool(backend, token_provider), EmailSendTool(backend, token_provider)]
