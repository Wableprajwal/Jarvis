"""FastAPI app: /chat, /traces, /metrics, /confirmations, /confirm, /eval/*, /sessions, /health."""
from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.bootstrap import ProviderFor, System, build_retriever, build_system, code_exec_tools, default_provider_for
from app.config import Settings, get_settings
from app.db import Database
from app.eval.judge import Judge
from app.eval.runner import EvalRunner, EvalRunStore, InMemoryEvalStore, PostgresEvalStore
from app.eval.simulated import HeuristicJudgeProvider
from app.harness.memory import InMemoryMemoryStore, PostgresMemoryStore
from app.observability.store import InMemoryTraceStore, PostgresTracer, TraceStore
from app.observability.trace import Tracer
from app.providers.ollama_provider import ProviderError
from app.tools.credentials import CredentialStore, RedactingProvider, RedactingTracer
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
    # Eval runs are background jobs: a live run takes ~15 minutes, far past what a browser or a
    # reverse proxy will hold a single request open for (nginx defaulted to 504 after 60 s).
    eval_jobs: dict[str, dict] = field(default_factory=dict)
    eval_tasks: set[asyncio.Task] = field(default_factory=set)


async def create_state(settings: Settings, provider_for: ProviderFor | None = None, use_db: bool | None = None) -> AppState:
    # Credential isolation (Phase 5): secrets never reach a prompt, a tool schema or a trace payload.
    # provider_for and the tracer are wrapped here, once, so every consumer built from them below -
    # memory's summarizer, every agent, the judge, the registry's tool results - is covered.
    credentials = CredentialStore.from_settings(settings)
    redactor = credentials.redactor()
    base_provider_for = provider_for or default_provider_for(settings)
    provider_for = lambda role: RedactingProvider(base_provider_for(role), redactor)  # noqa: E731

    db: Database | None = None
    if use_db is None:
        use_db = settings.trace_backend == "postgres"
    if use_db:
        try:
            db = await Database.connect(settings.database_url)
        except Exception as exc:  # noqa: BLE001 - Postgres optional: fall back to in-memory storage
            logger.warning("Postgres unavailable (%s); falling back to in-memory storage", exc)

    if db:
        raw_tracer: Tracer = PostgresTracer(db)
        traces = TraceStore(db)  # reads rows that were already scrubbed on write, below
        eval_store: EvalRunStore = PostgresEvalStore(db)
    else:
        raw_tracer = traces = InMemoryTraceStore()
        eval_store = InMemoryEvalStore()
    tracer: Tracer = RedactingTracer(raw_tracer, redactor)
    memory = (PostgresMemoryStore(db, provider_for("summarizer"), tracer) if db
              else InMemoryMemoryStore(provider_for("summarizer"), tracer))

    retriever = None
    docs = Path(settings.docs_dir) if settings.docs_dir else DEFAULT_DOCS
    try:
        kind = "offline" if settings.model_provider == "mock" else ("postgres" if db else "memory")
        retriever = await build_retriever(kind, docs, db=db, settings=settings)
    except Exception as exc:  # noqa: BLE001 - RAG optional: the Research agent simply has no rag_search tool
        logger.warning("RAG unavailable (%s); Research agent will have no rag_search tool", exc)

    system = build_system(
        settings, tracer=tracer, provider_for=provider_for, memory=memory, retriever=retriever,
        extra_code_tools=code_exec_tools(settings),
    )
    system.registry.redactor = redactor  # scrubs ToolResult output/error before it reaches the model or a trace
    return AppState(settings, system, tracer, traces, eval_store, retriever, provider_for, db, "postgres" if db else "memory")


def _require_owner(info, user_id: str | None) -> None:
    """`user_id` is a namespacing key, not authentication - this app has no authN, so a caller can
    claim any id. It is enforced anyway so the read endpoints agree with POST /chat (which refuses a
    session owned by someone else) instead of quietly being the way around it. Omitting `user_id`
    keeps the single-user dev/demo case ergonomic; see DECISIONS.md for what real auth would add."""
    if user_id is not None and info.user_id != user_id:
        raise HTTPException(403, f"session belongs to another user ({info.user_id})")


# ------------------------------------------------------------------------------- schemas
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    # Client-chosen identifiers end up in URLs, logs, trace rows and the UI; keep them bounded and
    # printable. They are namespacing keys, not secrets or auth (see _require_owner).
    session_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[\w .:@+-]+$")
    user_id: str = Field(default="default", min_length=1, max_length=128, pattern=r"^[\w .:@+-]+$")


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
        except Exception as exc:
            logger.exception("chat failed")
            raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
        try:
            await s.system.sessions.record_turn(session_id, req.message, result.text, user_id=req.user_id)
        except Exception:
            logger.exception("could not persist turn for session %s", session_id)
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
    async def sessions(limit: int = Query(50, ge=1, le=200), user_id: str | None = None):
        s = st()
        return [
            {
                "id": i.id, "user_id": i.user_id,
                "created_at": i.created_at.isoformat() if i.created_at else None,
                "preview": i.preview or "",
            }
            for i in await s.system.memory.list_sessions(limit, user_id)
        ]

    @app.get("/sessions/{session_id}/messages")
    async def session_messages(session_id: str, user_id: str | None = None):
        s = st()
        info = await s.system.memory.get_session(session_id)
        if info is None:
            raise HTTPException(404, "no such session")
        _require_owner(info, user_id)
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
    async def trace_session(session_id: str, user_id: str | None = None):
        s = st()
        info = await s.system.memory.get_session(session_id)
        if info is not None:
            _require_owner(info, user_id)
        return {"session_id": session_id, "events": await s.traces.get_session(session_id)}

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
    MAX_EVAL_JOBS_KEPT = 20

    async def _run_eval_job(job: dict) -> None:
        """Runs with `eval_lock` already held by the endpoint that created the job; releases it."""
        s = st()
        try:
            try:
                mock = s.settings.model_provider == "mock"
                judge = Judge(HeuristicJudgeProvider() if mock else s.provider_for("judge"))
                runner = EvalRunner(
                    s.settings, mode="mock" if mock else "live", retriever=s.retriever, judge=judge, store=s.eval_store,
                    extra_code_tools=(lambda: code_exec_tools(s.settings)),
                )
                card = await runner.run()
            except Exception as exc:
                logger.exception("eval job %s failed", job["job_id"])
                job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            else:
                job.update(status="done", run_id=card.run_id, scorecard=card.to_dict())
            finally:
                job["finished_at"] = datetime.now(UTC).isoformat()
        finally:
            s.eval_lock.release()

    @app.post("/eval/run", status_code=202)
    async def eval_run():
        """Start an eval run. Returns immediately with a job id; poll GET /eval/jobs/{job_id}."""
        s = st()
        # Take the lock HERE, not inside the task: a task only acquires it after it is scheduled,
        # so two quick POSTs would both pass a `locked()` check and the second would queue instead
        # of being refused. With no await between the check and the acquire, this is atomic.
        if s.eval_lock.locked():
            raise HTTPException(409, "an eval run is already in progress")
        await s.eval_lock.acquire()
        job = {
            "job_id": uuid.uuid4().hex[:12], "status": "running",
            "mode": "mock" if s.settings.model_provider == "mock" else "live",
            "started_at": datetime.now(UTC).isoformat(), "finished_at": None, "run_id": None, "error": None,
        }
        s.eval_jobs[job["job_id"]] = job
        for old_id in list(s.eval_jobs)[: max(0, len(s.eval_jobs) - MAX_EVAL_JOBS_KEPT)]:
            if s.eval_jobs[old_id]["status"] != "running":
                del s.eval_jobs[old_id]
        task = asyncio.create_task(_run_eval_job(job))
        s.eval_tasks.add(task)
        task.add_done_callback(s.eval_tasks.discard)
        return {k: v for k, v in job.items() if k != "scorecard"}

    @app.get("/eval/jobs/{job_id}")
    async def eval_job(job_id: str):
        job = st().eval_jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such eval job")
        return job

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
