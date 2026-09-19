"""Memory: a short-term rolling window per session + long-term facts per user.

One `MemoryStore` interface, two implementations (`PostgresMemoryStore`, `InMemoryMemoryStore`).
Everything above this module (SessionManager, the orchestrator, agents) depends only on the
interface, so swapping the storage backend never touches the harness.

Design notes
------------
* Trigger is TOKEN-BUDGET based, not turn-count based. Turns vary wildly in size (a one-line
  "thanks" vs. a 6 KB file read fed back as a tool result), so "summarize every N turns" either
  fires far too early or lets one huge turn blow the context window. The token budget is the
  quantity the model's context window actually constrains, so it is what we meter.
* Summarize-and-REPLACE, not truncate: when the active context (summary + unsummarized
  messages) exceeds the budget, the older turns are folded, together with any existing summary,
  into ONE new summary block by a model call; a recent window (a fraction of the budget) stays
  verbatim. Summarized rows are flagged, not deleted, so the full transcript remains auditable.
* Cuts land on turn boundaries so an assistant `tool_calls` message is never separated from its
  `tool` results (an orphaned tool message makes providers reject the request). System messages
  are pinned and never summarized.
* Both implementations share `_SummarizingMemoryStore`, which owns ALL of that behaviour and
  delegates only raw storage to subclasses, so they cannot drift apart.
"""
from __future__ import annotations

import asyncio
import copy
import json
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from app.db import Database
from app.observability.trace import NullTracer, Stopwatch, Tracer
from app.providers.base import ModelProvider

MEMORY_AGENT_NAME = "memory"
VALID_ROLES = {"system", "user", "assistant", "tool"}

SUMMARY_PREFIX = (
    "Summary of the earlier part of this conversation (older turns were compressed to save "
    "context; the most recent turns follow verbatim):\n"
)

SUMMARIZER_SYSTEM_PROMPT = """You compress the older part of a conversation between a user and an AI engineering assistant into a running summary, so those turns can be dropped from the context window.

Preserve: the user's goals and requests, decisions made, concrete facts and names (files, paths, functions, versions, config values, commands), errors and how they were resolved, open questions or pending work, and the user's stated preferences.
Drop: pleasantries, and verbatim tool output (keep only the conclusion drawn from it).
If an EXISTING SUMMARY is given, merge it with the new turns: your output REPLACES it, so keep everything from it that still matters. Never just repeat it and never discard it wholesale.
Write short plain prose or bullet points, at most about {target_tokens} tokens. Output only the summary text."""

_TOOL_RESULT_RENDER_CHARS = 1500

# --- token counting -----------------------------------------------------------------------

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Cheap default heuristic: ~4 characters per token (ceil). Inject a real tokenizer's
    `len(tokenizer.encode(text))` via the store constructor when accuracy matters."""
    return (len(text) + 3) // 4


def message_tokens(message: dict, counter: TokenCounter = estimate_tokens) -> int:
    n = 4  # per-message framing overhead (role markers etc.)
    n += counter(message.get("content") or "")
    if message.get("tool_calls"):
        n += counter(json.dumps(message["tool_calls"], default=str))
    return n


# --- data types ---------------------------------------------------------------------------


@dataclass
class Fact:
    key: str
    value: str
    source: str | None = None
    updated_at: datetime | None = None


@dataclass
class SessionInfo:
    id: str
    user_id: str
    created_at: datetime | None = None
    closed_at: datetime | None = None
    summary: str | None = None


def normalize_fact(fact: Fact) -> Fact:
    """Keys are identifiers ("preferred_language"), so normalise them: upserting
    "Preferred Language" must update, not duplicate, "preferred_language"."""
    key = "_".join((fact.key or "").strip().lower().split())
    value = (fact.value or "").strip()
    if not key or not value:
        raise ValueError("a fact needs a non-empty key and value")
    if looks_like_secret(f"{key} {value}") or looks_like_secret(f"{key}: {value}"):
        raise ValueError("fact looks like it contains a credential; refusing to store it")
    return Fact(key=key, value=value, source=fact.source, updated_at=fact.updated_at)


_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),  # OpenAI / Anthropic style keys
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub tokens
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),  # Slack tokens
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}", re.IGNORECASE),
    re.compile(r"(?i)\b(password|passwd|secret|api[_ -]?key|token)\b\s*(is|=|:)\s*\S{6,}"),
]


def looks_like_secret(text: str) -> bool:
    """Facts are injected into every future prompt, so credentials must never be stored as
    facts (Phase 5 rule: no secret in a message/trace payload). Best-effort pattern screen."""
    return any(p.search(text) for p in _SECRET_PATTERNS)


def validate_message(message: dict) -> None:
    role = message.get("role") if isinstance(message, dict) else None
    if role not in VALID_ROLES:
        raise ValueError(f"message needs a role in {sorted(VALID_ROLES)}, got {role!r}")


# --- interface ----------------------------------------------------------------------------


class MemoryStore(ABC):
    """Spec section 7 interface (the five methods) plus small session-lifecycle helpers."""

    # --- short-term ---
    @abstractmethod
    async def get_context(self, session_id: str) -> list[dict]:
        """[pinned system messages] + [one summary block, if any] + verbatim recent messages,
        oldest -> newest, in the normalized message format: ready to pass as `history`."""

    @abstractmethod
    async def append(self, session_id: str, message: dict) -> None: ...

    @abstractmethod
    async def summarize_if_needed(self, session_id: str, token_budget: int) -> None:
        """Compress older turns into the session summary when active context tokens exceed
        `token_budget`. No-op otherwise. Never raises for summarizer failures."""

    # --- long-term ---
    @abstractmethod
    async def get_long_term_facts(self, user_id: str) -> list[Fact]: ...

    @abstractmethod
    async def upsert_long_term_fact(self, user_id: str, fact: Fact) -> None: ...

    # --- session lifecycle helpers ---
    @abstractmethod
    async def ensure_session(self, session_id: str, user_id: str = "default") -> SessionInfo:
        """Create the session if missing; otherwise return the existing one unchanged
        (an existing session's owner is never overwritten)."""

    @abstractmethod
    async def get_session(self, session_id: str) -> SessionInfo | None: ...

    @abstractmethod
    async def close_session(self, session_id: str) -> None: ...

    @abstractmethod
    async def get_all_messages(self, session_id: str) -> list[dict]:
        """The complete transcript, including messages already folded into the summary."""


# --- shared behaviour ---------------------------------------------------------------------


@dataclass
class _Row:
    id: int
    message: dict
    tokens: int


def choose_cut(rows: list[_Row], recent_budget: int) -> int | None:
    """Index into `rows` (>= 1): rows[:cut] are summarized, rows[cut:] stay verbatim.

    A cut is only legal in front of a `user` message (a turn boundary) - or, when the window
    holds a single long agent loop with no later user message, in front of any non-`tool`
    message. A `tool` message can never start the kept window, so an assistant `tool_calls`
    message and all of its results always land on the same side. Among legal cuts we keep as
    much verbatim history as fits `recent_budget`; if not even the newest turn fits we still
    keep that newest turn whole.
    """
    n = len(rows)
    if n < 2:
        return None
    legal = [i for i in range(1, n) if rows[i].message["role"] == "user"]
    if not legal:
        legal = [i for i in range(1, n) if rows[i].message["role"] != "tool"]
    if not legal:
        return None
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + rows[i].tokens
    for i in legal:  # ascending: the earliest cut keeps the most verbatim history
        if suffix[i] <= recent_budget:
            return i
    return legal[-1]


def render_transcript(messages: list[dict]) -> str:
    lines: list[str] = []
    for m in messages:
        role = m["role"]
        content = (m.get("content") or "").strip()
        if role == "tool":
            if len(content) > _TOOL_RESULT_RENDER_CHARS:
                content = content[:_TOOL_RESULT_RENDER_CHARS] + " ...[truncated]"
            lines.append(f"[tool result: {m.get('name', '?')}] {content}")
            continue
        if content:
            lines.append(f"[{role}] {content}")
        for call in m.get("tool_calls") or []:
            args = json.dumps(call.get("arguments"), default=str)
            if len(args) > 300:
                args = args[:300] + "..."
            lines.append(f"[assistant called tool] {call.get('name')}({args})")
    return "\n".join(lines)


class _SummarizingMemoryStore(MemoryStore):
    """Owns context assembly + the summarization algorithm; subclasses provide storage only."""

    def __init__(
        self,
        summarizer: ModelProvider,
        tracer: Tracer | None = None,
        token_counter: TokenCounter = estimate_tokens,
        *,
        recent_fraction: float = 0.4,
        summary_role: str = "system",
    ):
        if not 0 < recent_fraction < 1:
            raise ValueError("recent_fraction must be in (0, 1)")
        if summary_role not in ("system", "user"):
            raise ValueError("summary_role must be 'system' or 'user'")
        self.summarizer = summarizer
        self.tracer = tracer or NullTracer()
        self.token_counter = token_counter
        self.recent_fraction = recent_fraction
        self.summary_role = summary_role
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    # -- storage primitives -------------------------------------------------------------
    @abstractmethod
    async def _load_active(self, session_id: str) -> tuple[str | None, list[_Row]]:
        """(current summary text, unsummarized rows oldest -> newest)."""

    @abstractmethod
    async def _commit_summary(
        self, session_id: str, old_summary: str | None, new_summary: str, row_ids: list[int]
    ) -> bool:
        """Atomically store `new_summary` and flag `row_ids` summarized. Must return False and
        change nothing if the stored summary is no longer `old_summary` (a concurrent
        summarizer got there first) or any row was already flagged."""

    # -- context ------------------------------------------------------------------------
    def _summary_block(self, summary: str) -> dict:
        return {"role": self.summary_role, "content": SUMMARY_PREFIX + summary}

    def _active_tokens(self, summary: str | None, rows: list[_Row]) -> int:
        total = sum(r.tokens for r in rows)
        if summary:
            total += message_tokens(self._summary_block(summary), self.token_counter)
        return total

    async def get_context(self, session_id: str) -> list[dict]:
        summary, rows = await self._load_active(session_id)
        pinned = [r.message for r in rows if r.message["role"] == "system"]
        recent = [r.message for r in rows if r.message["role"] != "system"]
        out = [*pinned]
        if summary:
            out.append(self._summary_block(summary))
        out.extend(recent)
        return copy.deepcopy(out)

    # -- summarization ------------------------------------------------------------------
    async def summarize_if_needed(self, session_id: str, token_budget: int) -> None:
        async with self._locks[session_id]:
            summary, rows = await self._load_active(session_id)
            before = self._active_tokens(summary, rows)
            if before <= token_budget:
                return

            pinned = [r for r in rows if r.message["role"] == "system"]  # never summarized
            body = [r for r in rows if r.message["role"] != "system"]
            cut = choose_cut(body, int(token_budget * self.recent_fraction))
            if cut is None:
                return  # a single oversized turn: nothing older to compress
            older, kept = body[:cut], body[cut:]

            target = max(64, token_budget // 4)
            sw = Stopwatch()
            try:
                resp = await self.summarizer.generate(
                    [
                        {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT.format(target_tokens=target)},
                        {
                            "role": "user",
                            "content": (
                                f"EXISTING SUMMARY:\n{summary or '(none)'}\n\n"
                                f"OLDER TURNS TO FOLD IN:\n{render_transcript([r.message for r in older])}\n\n"
                                "Write the updated summary."
                            ),
                        },
                    ],
                    [],
                    max_tokens=int(target * 1.5) + 16,
                )
                new_summary = (resp.text or "").strip()
                if resp.error or not new_summary:
                    raise RuntimeError(resp.error or "summarizer returned no text")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - never break the chat turn; keep every message
                await self.tracer.log(
                    session_id, MEMORY_AGENT_NAME, "error",
                    {"stage": "memory_summarize", "error": f"{type(exc).__name__}: {exc}"},
                    latency_ms=sw.ms(),
                )
                return

            latency = sw.ms()
            if not await self._commit_summary(session_id, summary, new_summary, [r.id for r in older]):
                return  # lost a race with another summarizer; its result stands
            after = (
                sum(r.tokens for r in pinned)
                + sum(r.tokens for r in kept)
                + message_tokens(self._summary_block(new_summary), self.token_counter)
            )
            await self.tracer.log(
                session_id, MEMORY_AGENT_NAME, "memory_summarize",
                {
                    "token_budget": token_budget,
                    "tokens_before": before,
                    "tokens_after": after,
                    "messages_summarized": len(older),
                    "messages_kept": len(kept),
                    "merged_prior_summary": bool(summary),
                },
                latency_ms=latency, tokens=resp.usage.total or None,
            )


# --- in-memory implementation -------------------------------------------------------------


class InMemoryMemoryStore(_SummarizingMemoryStore):
    """Same behaviour as the Postgres store, no I/O: for fast unit tests and eval replay."""

    def __init__(self, summarizer: ModelProvider, tracer: Tracer | None = None,
                 token_counter: TokenCounter = estimate_tokens, **kwargs):
        super().__init__(summarizer, tracer, token_counter, **kwargs)
        self._sessions: dict[str, SessionInfo] = {}
        self._rows: dict[str, list[_Row]] = defaultdict(list)
        self._summarized: dict[str, set[int]] = defaultdict(set)
        self._facts: dict[str, dict[str, Fact]] = defaultdict(dict)
        self._next_id = 1

    async def ensure_session(self, session_id: str, user_id: str = "default") -> SessionInfo:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionInfo(session_id, user_id, datetime.now(timezone.utc))
        return copy.copy(self._sessions[session_id])

    async def get_session(self, session_id: str) -> SessionInfo | None:
        s = self._sessions.get(session_id)
        return copy.copy(s) if s else None

    async def close_session(self, session_id: str) -> None:
        if session_id in self._sessions and self._sessions[session_id].closed_at is None:
            self._sessions[session_id].closed_at = datetime.now(timezone.utc)

    async def append(self, session_id: str, message: dict) -> None:
        validate_message(message)
        await self.ensure_session(session_id)
        row = _Row(self._next_id, copy.deepcopy(message), message_tokens(message, self.token_counter))
        self._next_id += 1
        self._rows[session_id].append(row)

    async def get_all_messages(self, session_id: str) -> list[dict]:
        return copy.deepcopy([r.message for r in self._rows.get(session_id, [])])

    async def _load_active(self, session_id: str) -> tuple[str | None, list[_Row]]:
        session = self._sessions.get(session_id)
        done = self._summarized[session_id]
        rows = [r for r in self._rows.get(session_id, []) if r.id not in done]
        return (session.summary if session else None), rows

    async def _commit_summary(self, session_id, old_summary, new_summary, row_ids) -> bool:
        session = self._sessions[session_id]
        if session.summary != old_summary or set(row_ids) & self._summarized[session_id]:
            return False
        session.summary = new_summary
        self._summarized[session_id].update(row_ids)
        return True

    async def get_long_term_facts(self, user_id: str) -> list[Fact]:
        return [copy.copy(f) for _, f in sorted(self._facts.get(user_id, {}).items())]

    async def upsert_long_term_fact(self, user_id: str, fact: Fact) -> None:
        f = normalize_fact(fact)
        f.updated_at = datetime.now(timezone.utc)
        self._facts[user_id][f.key] = f


# --- Postgres implementation --------------------------------------------------------------


class _LostRace(Exception):
    pass


class PostgresMemoryStore(_SummarizingMemoryStore):
    """Rolling window in Postgres: `messages` (JSONB message + token count + `summarized` flag),
    `sessions.summary` (the single merged summary block) and `facts`."""

    def __init__(self, db: Database, summarizer: ModelProvider, tracer: Tracer | None = None,
                 token_counter: TokenCounter = estimate_tokens, **kwargs):
        super().__init__(summarizer, tracer, token_counter, **kwargs)
        self.db = db

    @staticmethod
    def _session(row) -> SessionInfo:
        return SessionInfo(row["id"], row["user_id"], row["created_at"], row["closed_at"], row["summary"])

    async def ensure_session(self, session_id: str, user_id: str = "default") -> SessionInfo:
        async with self.db.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO sessions (id, user_id) VALUES ($1, $2) ON CONFLICT (id) DO NOTHING", session_id, user_id
            )
            return self._session(await conn.fetchrow("SELECT * FROM sessions WHERE id = $1", session_id))

    async def get_session(self, session_id: str) -> SessionInfo | None:
        row = await self.db.pool.fetchrow("SELECT * FROM sessions WHERE id = $1", session_id)
        return self._session(row) if row else None

    async def close_session(self, session_id: str) -> None:
        await self.db.pool.execute(
            "UPDATE sessions SET closed_at = now() WHERE id = $1 AND closed_at IS NULL", session_id
        )

    async def append(self, session_id: str, message: dict) -> None:
        validate_message(message)
        tokens = message_tokens(message, self.token_counter)
        async with self.db.pool.acquire() as conn, conn.transaction():
            await conn.execute("INSERT INTO sessions (id) VALUES ($1) ON CONFLICT (id) DO NOTHING", session_id)
            await conn.execute(
                "INSERT INTO messages (session_id, message, tokens) VALUES ($1, $2::jsonb, $3)",
                session_id, json.dumps(message, default=str), tokens,
            )

    async def get_all_messages(self, session_id: str) -> list[dict]:
        rows = await self.db.pool.fetch("SELECT message FROM messages WHERE session_id = $1 ORDER BY id", session_id)
        return [json.loads(r["message"]) for r in rows]

    async def _load_active(self, session_id: str) -> tuple[str | None, list[_Row]]:
        async with self.db.pool.acquire() as conn:
            summary = await conn.fetchval("SELECT summary FROM sessions WHERE id = $1", session_id)
            rows = await conn.fetch(
                "SELECT id, message, tokens FROM messages WHERE session_id = $1 AND NOT summarized ORDER BY id",
                session_id,
            )
        return summary, [_Row(r["id"], json.loads(r["message"]), r["tokens"]) for r in rows]

    async def _commit_summary(self, session_id, old_summary, new_summary, row_ids) -> bool:
        try:
            async with self.db.pool.acquire() as conn, conn.transaction():
                current = await conn.fetchval("SELECT summary FROM sessions WHERE id = $1 FOR UPDATE", session_id)
                if current != old_summary:
                    raise _LostRace
                flagged = await conn.fetch(
                    "UPDATE messages SET summarized = TRUE "
                    "WHERE session_id = $1 AND id = ANY($2::bigint[]) AND NOT summarized RETURNING id",
                    session_id, row_ids,
                )
                if len(flagged) != len(row_ids):
                    raise _LostRace
                await conn.execute("UPDATE sessions SET summary = $2 WHERE id = $1", session_id, new_summary)
            return True
        except _LostRace:
            return False

    async def get_long_term_facts(self, user_id: str) -> list[Fact]:
        rows = await self.db.pool.fetch(
            "SELECT key, value, source, updated_at FROM facts WHERE user_id = $1 ORDER BY key", user_id
        )
        return [Fact(r["key"], r["value"], r["source"], r["updated_at"]) for r in rows]

    async def upsert_long_term_fact(self, user_id: str, fact: Fact) -> None:
        f = normalize_fact(fact)
        await self.db.pool.execute(
            """INSERT INTO facts (user_id, key, value, source) VALUES ($1, $2, $3, $4)
               ON CONFLICT (user_id, key)
               DO UPDATE SET value = EXCLUDED.value, source = EXCLUDED.source, updated_at = now()""",
            user_id, f.key, f.value, f.source,
        )
