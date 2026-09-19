"""Structured tracing. Every loop iteration, tool call and routing decision becomes a
TraceEvent. The same records feed the UI trace view, the metrics dashboard and the eval
harness (which inspects *how* an answer was produced)."""
from __future__ import annotations

import itertools
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

EVENT_TYPES = {
    "route_decision",
    "model_call",
    "tool_call",
    "tool_result",
    "memory_summarize",
    "final_response",
    "error",
}


@dataclass
class TraceEvent:
    session_id: str
    agent_name: str
    turn_id: str | None
    event_type: str
    payload: dict = field(default_factory=dict)
    latency_ms: float | None = None
    tokens: int | None = None
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    seq: int = 0  # monotonic per-process ordering; ts alone can collide

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        return d


_seq = itertools.count(1)


class Tracer(ABC):
    @abstractmethod
    async def record(self, event: TraceEvent) -> None: ...

    async def log(
        self,
        session_id: str,
        agent_name: str,
        event_type: str,
        payload: dict | None = None,
        *,
        turn_id: str | None = None,
        latency_ms: float | None = None,
        tokens: int | None = None,
    ) -> TraceEvent:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown trace event type: {event_type}")
        event = TraceEvent(
            session_id=session_id,
            agent_name=agent_name,
            turn_id=turn_id,
            event_type=event_type,
            payload=payload or {},
            latency_ms=latency_ms,
            tokens=tokens,
            seq=next(_seq),
        )
        await self.record(event)
        return event


class NullTracer(Tracer):
    async def record(self, event: TraceEvent) -> None:
        return None


class InMemoryTracer(Tracer):
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    async def record(self, event: TraceEvent) -> None:
        self.events.append(event)

    def of_type(self, event_type: str) -> list[TraceEvent]:
        return [e for e in self.events if e.event_type == event_type]


class Stopwatch:
    def __init__(self) -> None:
        self._t = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self._t) * 1000.0
