"""FastAPI app: /chat, /traces, /metrics, /confirmations, /confirm, /eval/*, /sessions, /health."""
from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.bootstrap import ProviderFor, System, build_retriever, build_system, default_provider_for
from app.config import Settings, get_settings
from app.db import Database
from app.eval.judge import Judge
from app.eval.runner import EvalRunner, EvalRunStore, InMemoryEvalStore, PostgresEvalStore
from app.eval.simulated import HeuristicJudgeProvider
from app.harness.memory import InMemoryMemoryStore, PostgresMemoryStore
from app.observability.store import InMemoryTraceStore, PostgresTracer, TraceStore
from app.observability.trace import Tracer
from app.providers.ollama_provider import ProviderError
from app.tools.memory_tool import current_user_var

logger = logging.getLogger("jarvis")
DEFAULT_DOCS = Path(__file__).resolve().parents[2] / "docs"


@dataclass
class AppState:
    settings: Settings
    system: System
    tracer: Tracer
    traces: object  # TraceStore | InMemoryTraceStore (same read API)
    eval_store: EvalRunStore
    retriever: object | None
    provider_for: ProviderFor
    db: Database | None = None
    storage: str = "memory"
    eval_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def create_state(settings: Settings, provider_for: ProviderFor | None = None, use_db: bool | None = None) -> AppState:
    provider_for = provider_for or default_provider_for(settings)
    db: Database | None = None
    if use_db is None:
        use_db = settings.trace_backend == "postgres"
    if use_db:
        try:
            db = await Database.connect(settings.database_url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Postgres unavailable (%s); falling back to in-memory storage", exc)

    if db:
        tracer: Tracer = PostgresTracer(db)
        traces = TraceStore(db)
        memory = PostgresMemoryStore(db, provider_for("summarizer"), tracer)
        eval_store: EvalRunStore = PostgresEvalStore(db)
    else:
        tracer = traces = InMemoryTraceStore()
        memory = InMemoryMemoryStore(provider_for("summarizer"), tracer)
        eval_store = InMemoryEvalStore()

    retriever = None
    docs = Path(settings.docs_dir) if settings.docs_dir else DEFAULT_DOCS
    try:
        kind = "offline" if settings.model_provider == "mock" else ("postgres" if db else "memory")
        retriever = await build_retriever(kind, docs, db=db, settings=settings)
    except Exception as exc:  # noqa: BLE001 - the app still works without RAG; the Research agent just lacks rag_search
        logger.warning("RAG unavailable (%s); Research agent will have no rag_search tool", exc)

    system = build_system(
        settings, tracer=tracer, provider_for=provider_for, memory=memory, retriever=retriever,
        extra_code_tools=code_exec_tools(settings),
    )
    return AppState(settings, system, tracer, traces, eval_store, retriever, provider_for, db, "postgres" if db else "memory")


def code_exec_tools(settings: Settings) -> list:
    return []


# ------------------------------------------------------------------------------- schemas
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    session_id: str | None = None
    user_id: str = "default"


class ConfirmRequest(BaseModel):
    approve: bool


def create_app(settings: Settings | None = None, *, provider_for: ProviderFor | None = None, use_db: bool | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.jarvis = await create_state(settings, provider_for, use_db)
        yield
        if app.state.jarvis.db:
            await app.state.jarvis.db.close()

    app = FastAPI(title="Jarvis", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"], allow_headers=["*"],
    )

    def st() -> AppState:
        return app.state.jarvis

    @app.get("/health")
    async def health():
        s = st()
        return {
            "status": "ok", "provider": s.settings.model_provider, "model": s.settings.orchestrator_model,
            "storage": s.storage, "rag": s.retriever is not None,
        }

    # ---- chat
    @app.post("/chat")
    async def chat(req: ChatRequest):
        s = st()
        session_id = req.session_id or uuid.uuid4().hex[:12]
        try:
            await s.system.sessions.get_or_create_session(session_id, req.user_id)
        except ValueError as exc:  # session owned by another user
            raise HTTPException(403, str(exc)) from exc
        current_user_var.set(req.user_id)
        try:
            history = await s.system.sessions.build_history(session_id, req.user_id)
            result = await s.system.orchestrator.handle(req.message, session_id, history)
        except ProviderError as exc:
            raise HTTPException(502, f"model backend error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat failed")
            raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
        await s.system.sessions.record_turn(session_id, req.message, result.text, user_id=req.user_id)
        return {
            "session_id": session_id,
            "turn_id": result.turn_id,
            "text": result.text,
            "truncated": result.truncated,
            "route_decisions": result.route_decisions,
            "pending_confirmations": [a.summary() for a in s.system.registry.pending(session_id)],
        }

    @app.post("/sessions/{session_id}/close")
    async def close_session(session_id: str):
        s = st()
        facts = await s.system.sessions.close_session(session_id, extract_with=s.provider_for("summarizer"))
        return {"session_id": session_id, "facts_extracted": [{"key": f.key, "value": f.value} for f in facts]}

    @app.get("/sessions")
    async def sessions(limit: int = Query(50, ge=1, le=200)):
        s = st()
        out = []
        for info in await s.system.memory.list_sessions(limit):
            msgs = await s.system.memory.get_all_messages(info.id)
            first = next((m["content"] for m in msgs if m["role"] == "user"), "")
            out.append({
                "id": info.id, "user_id": info.user_id,
                "created_at": info.created_at.isoformat() if info.created_at else None, "preview": (first or "")[:80],
            })
        return out

    @app.get("/sessions/{session_id}/messages")
    async def session_messages(session_id: str):
        s = st()
        if await s.system.memory.get_session(session_id) is None:
            raise HTTPException(404, "no such session")
        return [
            {"role": m["role"], "content": m.get("content"), "created_at": None}
            for m in await s.system.memory.get_all_messages(session_id) if m["role"] in ("user", "assistant") and m.get("content")
        ]

    # ---- traces / metrics
    @app.get("/traces/turns")
    async def trace_turns(session_id: str | None = None, limit: int = Query(50, ge=1, le=500)):
        return await st().traces.list_turns(session_id, limit)

    @app.get("/traces/turn/{turn_id}")
    async def trace_turn(turn_id: str):
        events = await st().traces.get_turn(turn_id)
        if not events:
            raise HTTPException(404, "no such turn")
        return {"turn_id": turn_id, "events": events}

    @app.get("/traces/session/{session_id}")
    async def trace_session(session_id: str):
        return {"session_id": session_id, "events": await st().traces.get_session(session_id)}

    @app.get("/metrics")
    async def metrics():
        return await st().traces.metrics()

    # ---- confirmation gate
    @app.get("/confirmations")
    async def confirmations(session_id: str | None = None):
        return [a.summary() for a in st().system.registry.pending(session_id)]

    @app.post("/confirm/{confirmation_id}")
    async def confirm(confirmation_id: str, req: ConfirmRequest):
        try:
            action = await st().system.registry.confirm(confirmation_id, req.approve)
        except KeyError:
            raise HTTPException(404, "no such pending action") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        result = None
        if action.result is not None:
            result = {"ok": action.result.ok, "output": action.result.output, "error": action.result.error}
        return {"id": action.id, "status": action.status, "result": result}

    # ---- eval
    @app.post("/eval/run")
    async def eval_run():
        s = st()
        if s.eval_lock.locked():
            raise HTTPException(409, "an eval run is already in progress")
        async with s.eval_lock:
            mock = s.settings.model_provider == "mock"
            judge = Judge(HeuristicJudgeProvider() if mock else s.provider_for("judge"))
            runner = EvalRunner(
                s.settings, mode="mock" if mock else "live", retriever=s.retriever, judge=judge, store=s.eval_store,
                extra_code_tools=(lambda: code_exec_tools(s.settings)),
            )
            card = await runner.run()
        return card.to_dict()

    @app.get("/eval/runs")
    async def eval_runs(limit: int = Query(20, ge=1, le=100)):
        return await st().eval_store.list(limit)

    @app.get("/eval/runs/{run_id}")
    async def eval_run_get(run_id: str):
        run = await st().eval_store.get(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        return run

    return app


app = create_app()
