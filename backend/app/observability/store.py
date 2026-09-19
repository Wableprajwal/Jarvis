"""Postgres-backed tracer plus the read side used by the API, trace view, dashboard and evals."""
from __future__ import annotations

import json

from app.db import Database
from app.observability.trace import InMemoryTracer, TraceEvent, Tracer

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


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


class InMemoryTraceStore(InMemoryTracer):
    """Tracer + the same read API as TraceStore, for running without Postgres."""

    def _dicts(self, events) -> list[dict]:
        return [e.to_dict() for e in sorted(events, key=lambda e: e.seq)]

    async def get_turn(self, turn_id: str) -> list[dict]:
        return self._dicts(e for e in self.events if e.turn_id == turn_id)

    async def get_session(self, session_id: str, limit: int = 1000) -> list[dict]:
        return self._dicts(e for e in self.events if e.session_id == session_id)[:limit]

    async def list_turns(self, session_id: str | None = None, limit: int = 50) -> list[dict]:
        turns: dict[str, list[TraceEvent]] = {}
        for e in self.events:
            if e.turn_id and (session_id is None or e.session_id == session_id):
                turns.setdefault(e.turn_id, []).append(e)
        out = [
            {
                "turn_id": t, "session_id": evs[0].session_id, "started": min(e.ts for e in evs).isoformat(),
                "events": len(evs), "tokens": sum(e.tokens or 0 for e in evs),
                "routed": [e.payload.get("agent") for e in sorted(evs, key=lambda e: e.seq) if e.event_type == "route_decision"],
            }
            for t, evs in turns.items()
        ]
        return sorted(out, key=lambda d: d["started"], reverse=True)[:limit]

    async def metrics(self) -> dict:
        by_agent: dict[str, list[float]] = {}
        tools: dict[str, list[bool]] = {}
        per_session: dict[str, int] = {}
        for e in self.events:
            if e.event_type == "model_call" and e.latency_ms is not None:
                by_agent.setdefault(e.agent_name, []).append(e.latency_ms)
            if e.event_type == "tool_result":
                tools.setdefault(e.payload.get("tool", "?"), []).append(bool(e.payload.get("ok")))
            per_session[e.session_id] = per_session.get(e.session_id, 0) + (e.tokens or 0)
        return {
            "latency_by_agent": [
                {"agent": a, "calls": len(v), "p50_ms": _pct(v, 0.5), "p95_ms": _pct(v, 0.95)} for a, v in sorted(by_agent.items())
            ],
            "tool_success": [
                {"tool": t, "calls": len(v), "ok": sum(v), "success_rate": sum(v) / len(v)} for t, v in sorted(tools.items())
            ],
            "tokens": {
                "sessions": len(per_session),
                "avg_tokens_per_session": (sum(per_session.values()) / len(per_session)) if per_session else 0.0,
            },
        }
