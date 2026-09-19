"""Calendar tools for the Ops agent.

`CalendarBackend` is the seam between the tools and a calendar provider. Only an in-memory
fake ships; a real Google Calendar backend is the marked extension point below. Tools receive
the backend (and an optional `token_provider`) by injection and never import credential code:
the token is fetched at call time, handed to the backend, and never appears in results, so it
cannot reach the model or a trace payload.

`calendar_create` is confirmation-gated: the registry parks the call and only
`registry.confirm(id)` makes it reach `execute()`.
"""
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.tools.registry import Tool, ToolResult

TokenProvider = Callable[[], "str | None"]

# Cheap shape check applied by the registry BEFORE the confirmation gate; execute() re-parses.
ISO_PATTERN = r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})?)?$"
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


def parse_iso(value: str) -> datetime:
    """ISO 8601 date or datetime -> aware datetime (naive values are taken as UTC)."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


@dataclass
class CalendarEvent:
    id: str
    title: str
    start: datetime
    end: datetime
    attendees: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "title": self.title, "start": self.start.isoformat(), "end": self.end.isoformat(),
            "attendees": self.attendees, "description": self.description,
        }


class CalendarBackend(ABC):
    """`token` is whatever the injected token_provider returned (None if there is none).
    Backends that need credentials raise PermissionError when it is missing/invalid."""

    @abstractmethod
    async def list_events(
        self, start: datetime | None, end: datetime | None, limit: int, token: str | None = None
    ) -> list[CalendarEvent]: ...

    @abstractmethod
    async def create_event(
        self, title: str, start: datetime, end: datetime, attendees: list[str], description: str,
        token: str | None = None,
    ) -> CalendarEvent: ...


class FakeCalendarBackend(CalendarBackend):
    """In-memory calendar seeded with a little sample data. `require_token=True` makes it behave
    like an authenticated API (used to test credential plumbing)."""

    def __init__(self, events: list[CalendarEvent] | None = None, *, require_token: bool = False):
        self.events: list[CalendarEvent] = list(events) if events is not None else self._seed()
        self.require_token = require_token
        self.tokens_seen: list[str | None] = []
        self._ids = itertools.count(len(self.events) + 1)

    @staticmethod
    def _seed() -> list[CalendarEvent]:
        def ev(i: int, title: str, start: str, end: str, who: list[str]) -> CalendarEvent:
            return CalendarEvent(f"evt-{i}", title, parse_iso(start), parse_iso(end), who)

        return [
            ev(1, "Sprint planning", "2026-09-21T09:30:00Z", "2026-09-21T10:30:00Z", ["team@example.com"]),
            ev(2, "1:1 with Priya", "2026-09-22T15:00:00Z", "2026-09-22T15:30:00Z", ["priya@example.com"]),
            ev(3, "Payments gateway incident review", "2026-09-23T14:00:00Z", "2026-09-23T15:00:00Z", ["oncall@example.com"]),
            ev(4, "Design review: rate limiting", "2026-09-24T11:00:00Z", "2026-09-24T12:00:00Z", ["arch@example.com"]),
        ]

    def _auth(self, token: str | None) -> None:
        self.tokens_seen.append(token)
        if self.require_token and not token:
            raise PermissionError("calendar credentials are not configured")

    async def list_events(self, start, end, limit, token=None) -> list[CalendarEvent]:
        self._auth(token)
        hits = [
            e for e in sorted(self.events, key=lambda e: e.start)
            if (start is None or e.end > start) and (end is None or e.start < end)
        ]
        return hits[:limit]

    async def create_event(self, title, start, end, attendees, description, token=None) -> CalendarEvent:
        self._auth(token)
        event = CalendarEvent(f"evt-{next(self._ids)}", title, start, end, list(attendees), description)
        self.events.append(event)
        return event


class GoogleCalendarBackend(CalendarBackend):
    """EXTENSION POINT (optional, last): real Google Calendar via OAuth.

    Implement with the Calendar REST API (events.list / events.insert), using the bearer token
    supplied by the injected token_provider (wired to app/tools/credentials.py by the app
    bootstrap, never imported here). Until then the fake backend is used everywhere.
    """

    async def list_events(self, start, end, limit, token=None):
        raise NotImplementedError("GoogleCalendarBackend is not implemented; use FakeCalendarBackend")

    async def create_event(self, title, start, end, attendees, description, token=None):
        raise NotImplementedError("GoogleCalendarBackend is not implemented; use FakeCalendarBackend")


class _CalendarTool(Tool):
    def __init__(self, backend: CalendarBackend, token_provider: TokenProvider | None = None):
        self._backend = backend
        self._token_provider = token_provider

    def _token(self) -> str | None:
        return self._token_provider() if self._token_provider else None


class CalendarListTool(_CalendarTool):
    name = "calendar_list"
    description = (
        "List calendar events, soonest first. Optionally restrict to a window with ISO 8601 "
        "`start`/`end` (e.g. 2026-09-21 or 2026-09-21T09:00:00Z)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "start": {"type": "string", "pattern": ISO_PATTERN, "description": "Window start (ISO 8601)"},
            "end": {"type": "string", "pattern": ISO_PATTERN, "description": "Window end (ISO 8601)"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
        },
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict) -> ToolResult:
        try:
            start = parse_iso(arguments["start"]) if "start" in arguments else None
            end = parse_iso(arguments["end"]) if "end" in arguments else None
        except ValueError as exc:
            return ToolResult.failure(f"invalid date: {exc}")
        if start and end and end <= start:
            return ToolResult.failure("`end` must be after `start`")
        try:
            events = await self._backend.list_events(start, end, arguments.get("limit", 20), self._token())
        except PermissionError as exc:
            return ToolResult.failure(f"calendar access denied: {exc}")
        return ToolResult.success({"events": [e.to_dict() for e in events], "count": len(events)})


class CalendarCreateTool(_CalendarTool):
    name = "calendar_create"
    requires_confirmation = True
    description = (
        "Create a calendar event. REQUIRES USER CONFIRMATION: calling this only queues the event; "
        "it is not created until the user approves."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1, "maxLength": 200},
            "start": {"type": "string", "pattern": ISO_PATTERN, "description": "ISO 8601 start"},
            "end": {"type": "string", "pattern": ISO_PATTERN, "description": "ISO 8601 end, after start"},
            "attendees": {"type": "array", "items": {"type": "string", "pattern": EMAIL_PATTERN}, "maxItems": 50},
            "description": {"type": "string", "maxLength": 2000},
        },
        "required": ["title", "start", "end"],
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict) -> ToolResult:
        try:
            start, end = parse_iso(arguments["start"]), parse_iso(arguments["end"])
        except ValueError as exc:
            return ToolResult.failure(f"invalid date: {exc}")
        if end <= start:
            return ToolResult.failure("`end` must be after `start`")
        token = self._token()
        try:
            conflicts = [e.to_dict() for e in await self._backend.list_events(start, end, 50, token)]
            event = await self._backend.create_event(
                arguments["title"], start, end, arguments.get("attendees", []),
                arguments.get("description", ""), token,
            )
        except PermissionError as exc:
            return ToolResult.failure(f"calendar access denied: {exc}")
        return ToolResult.success({"created": event.to_dict(), "overlapping_events": conflicts})


def calendar_tools(backend: CalendarBackend, token_provider: TokenProvider | None = None) -> list[Tool]:
    return [CalendarListTool(backend, token_provider), CalendarCreateTool(backend, token_provider)]
