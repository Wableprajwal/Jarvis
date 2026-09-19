"""Shared fixtures/helpers for the Phase 6 (memory) tests.

`env` is parametrized over both MemoryStore implementations, so every test that uses it is a
contract test that must pass identically for InMemory and Postgres. Postgres-backed params
skip when no database is reachable; created rows are removed at teardown."""
import os
import uuid

import pytest

from app.config import get_settings
from app.db import Database
from app.harness.memory import InMemoryMemoryStore, MemoryStore, PostgresMemoryStore
from app.observability.trace import Tracer
from app.providers.base import ModelProvider


class Env:
    def __init__(self, kind: str, db: Database | None):
        self.kind = kind
        self.db = db
        self.session_ids: list[str] = []
        self.user_ids: list[str] = []

    def make(self, summarizer: ModelProvider, tracer: Tracer | None = None, **kwargs) -> MemoryStore:
        if self.kind == "memory":
            return InMemoryMemoryStore(summarizer, tracer, **kwargs)
        return PostgresMemoryStore(self.db, summarizer, tracer, **kwargs)

    def session(self) -> str:
        sid = f"p6-{uuid.uuid4().hex[:10]}"
        self.session_ids.append(sid)
        return sid

    def user(self) -> str:
        uid = f"p6user-{uuid.uuid4().hex[:10]}"
        self.user_ids.append(uid)
        return uid

    async def cleanup(self) -> None:
        if self.db is None:
            return
        await self.db.pool.execute("DELETE FROM sessions WHERE id = ANY($1::text[])", self.session_ids)
        await self.db.pool.execute("DELETE FROM facts WHERE user_id = ANY($1::text[])", self.user_ids)
        await self.db.pool.execute("DELETE FROM trace_events WHERE session_id = ANY($1::text[])", self.session_ids)


@pytest.fixture(params=["memory", "postgres"])
async def env(request):
    db = None
    if request.param == "postgres":
        url = os.environ.get("TEST_DATABASE_URL") or get_settings().database_url
        try:
            db = await Database.connect(url)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"postgres not available: {exc}")
    e = Env(request.param, db)
    yield e
    await e.cleanup()
    if db is not None:
        await db.close()


def turn_messages(i: int, filler: int = 160) -> list[dict]:
    """One user/assistant exchange; ~(filler+10)/4 tokens each with the default counter."""
    pad = "x" * filler
    return [
        {"role": "user", "content": f"Q{i} {pad}"},
        {"role": "assistant", "content": f"A{i} {pad}"},
    ]


async def add_turns(store: MemoryStore, session_id: str, start: int, stop: int, filler: int = 160) -> None:
    for i in range(start, stop):
        for m in turn_messages(i, filler):
            await store.append(session_id, m)


def text_of(messages: list[dict]) -> str:
    return "\n".join(m.get("content") or "" for m in messages)
