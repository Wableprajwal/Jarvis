"""Postgres-backed tracer plus the read side used by the API, trace view, dashboard and evals."""
from __future__ import annotations

import json

from app.db import Database
from app.observability.trace import TraceEvent, Tracer

_COLUMNS = "seq, id, session_id, agent_name, turn_id, event_type, payload, latency_ms, tokens, ts"


def _row_to_dict(row) -> dict:
    d = dict(row)
    d["payload"] = json.loads(d["payload"]) if isinstance(d["payload"], str) else d["payload"]
    d["ts"] = d["ts"].isoformat()
    return d


class PostgresTracer(Tracer):
    def __init__(self, db: Database):
        self._db = db

    async def record(self, event: TraceEvent) -> None:
        seq = await self._db.pool.fetchval(
            """INSERT INTO trace_events (id, session_id, agent_name, turn_id, event_type, payload, latency_ms, tokens, ts)
               VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9) RETURNING seq""",
            event.id, event.session_id, event.agent_name, event.turn_id, event.event_type,
            json.dumps(event.payload, default=str), event.latency_ms, event.tokens, event.ts,
        )
        event.seq = seq


class TraceStore:
    def __init__(self, db: Database):
        self._db = db

    async def get_turn(self, turn_id: str) -> list[dict]:
        rows = await self._db.pool.fetch(
            f"SELECT {_COLUMNS} FROM trace_events WHERE turn_id = $1 ORDER BY seq", turn_id
        )
        return [_row_to_dict(r) for r in rows]

    async def get_session(self, session_id: str, limit: int = 1000) -> list[dict]:
        rows = await self._db.pool.fetch(
            f"SELECT {_COLUMNS} FROM trace_events WHERE session_id = $1 ORDER BY seq LIMIT $2",
            session_id, limit,
        )
        return [_row_to_dict(r) for r in rows]

    async def list_turns(self, session_id: str | None = None, limit: int = 50) -> list[dict]:
        rows = await self._db.pool.fetch(
            """SELECT turn_id, session_id, min(ts) AS started, count(*) AS events,
                      coalesce(sum(tokens), 0) AS tokens,
                      array_agg(payload->>'agent' ORDER BY seq) FILTER (WHERE event_type = 'route_decision') AS routed
               FROM trace_events
               WHERE turn_id IS NOT NULL AND ($1::text IS NULL OR session_id = $1)
               GROUP BY turn_id, session_id ORDER BY min(seq) DESC LIMIT $2""",
            session_id, limit,
        )
        return [
            {**dict(r), "started": r["started"].isoformat(), "routed": r["routed"] or []} for r in rows
        ]

    async def metrics(self) -> dict:
        pool = self._db.pool
        latency = await pool.fetch(
            """SELECT agent_name,
                      count(*) AS calls,
                      percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,
                      percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms
               FROM trace_events WHERE event_type = 'model_call' AND latency_ms IS NOT NULL
               GROUP BY agent_name ORDER BY agent_name"""
        )
        tools = await pool.fetch(
            """SELECT payload->>'tool' AS tool,
                      count(*) AS calls,
                      count(*) FILTER (WHERE (payload->>'ok')::boolean) AS ok
               FROM trace_events WHERE event_type = 'tool_result'
               GROUP BY payload->>'tool' ORDER BY 1"""
        )
        tokens = await pool.fetchrow(
            """SELECT count(*) AS sessions, coalesce(avg(t), 0) AS avg_tokens_per_session
               FROM (SELECT session_id, sum(tokens) AS t FROM trace_events GROUP BY session_id) s"""
        )
        return {
            "latency_by_agent": [
                {"agent": r["agent_name"], "calls": r["calls"], "p50_ms": r["p50_ms"], "p95_ms": r["p95_ms"]}
                for r in latency
            ],
            "tool_success": [
                {"tool": r["tool"], "calls": r["calls"], "ok": r["ok"], "success_rate": r["ok"] / r["calls"]}
                for r in tools
            ],
            "tokens": {"sessions": tokens["sessions"], "avg_tokens_per_session": float(tokens["avg_tokens_per_session"])},
        }
