"""Composition root: the only module that knows every concrete class. Everything it builds
depends on abstractions (ModelProvider, VectorStore, MemoryStore, Tracer, Tool, backends)."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.agents.base import SubAgent
from app.agents.code_agent import CodeAgent
from app.agents.ops_agent import OpsAgent
from app.agents.research_agent import ResearchAgent
from app.config import Settings, get_settings
from app.harness.memory import InMemoryMemoryStore, MemoryStore
from app.harness.session import SessionManager
from app.observability.trace import Tracer
from app.orchestrator.orchestrator import Orchestrator
from app.providers.base import ModelProvider
from app.providers.factory import get_provider
from app.rag.retriever import HybridRetriever
from app.tools.builtin import GetTimeTool
from app.tools.calendar import CalendarBackend, FakeCalendarBackend, calendar_tools
from app.tools.files import GenerateTestsTool, GitDiffSummarizeTool, ReadFileTool
from app.tools.gmail import FakeMailBackend, MailBackend, mail_tools
from app.tools.memory_tool import RememberFactTool
from app.tools.rag_tool import RagSearchTool
from app.tools.registry import Tool, ToolRegistry

ROLES = ("orchestrator", "code", "research", "ops", "summarizer", "judge")
ProviderFor = Callable[[str], ModelProvider]


@dataclass
class System:
    settings: Settings
    orchestrator: Orchestrator
    agents: list[SubAgent]
    registry: ToolRegistry
    tracer: Tracer
    memory: MemoryStore
    sessions: SessionManager
    mail: MailBackend
    calendar: CalendarBackend
    providers: dict[str, ModelProvider] = field(default_factory=dict)

    def fingerprint(self) -> dict:
        """Everything a prompt/skill/harness/model change could alter. Its hash identifies an eval
        run's configuration, so any such change shows up as a new config_hash."""
        return {
            "provider": self.settings.model_provider,
            "models": {r: getattr(self.settings, f"{r}_model", None) for r in ROLES},
            "orchestrator_prompt": self.orchestrator.system_prompt(),
            "agents": {
                a.name: {
                    "description": a.description,
                    "system_prompt": a.system_prompt,
                    "tools": sorted(t.name for t in self.registry.tools_for(a.name)),
                }
                for a in self.agents
            },
            "tools": {
                t.name: {"description": t.description, "schema": t.input_schema, "gated": t.requires_confirmation}
                for a in self.agents
                for t in self.registry.tools_for(a.name)
            },
        }

    def config_hash(self) -> str:
        blob = json.dumps(self.fingerprint(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def default_provider_for(settings: Settings) -> ProviderFor:
    def provider_for(role: str) -> ModelProvider:
        return get_provider(settings.model_provider, model=getattr(settings, f"{role}_model"), settings=settings)

    return provider_for


def build_system(
    settings: Settings | None = None,
    *,
    tracer: Tracer,
    provider_for: ProviderFor | None = None,
    memory: MemoryStore | None = None,
    retriever: HybridRetriever | None = None,
    mail: MailBackend | None = None,
    calendar: CalendarBackend | None = None,
    workspace_root: str | Path | None = None,
    extra_code_tools: list[Tool] | None = None,
) -> System:
    settings = settings or get_settings()
    provider_for = provider_for or default_provider_for(settings)
    providers = {role: provider_for(role) for role in ROLES}
    workspace = Path(workspace_root or settings.workspace_root).resolve()
    mail = mail or FakeMailBackend()
    calendar = calendar or FakeCalendarBackend()
    memory = memory or InMemoryMemoryStore(providers["summarizer"], tracer)

    registry = ToolRegistry()
    tools: list[Tool] = [
        GetTimeTool(),
        ReadFileTool(workspace),
        GitDiffSummarizeTool(workspace),
        GenerateTestsTool(workspace, providers["code"]),
        *(extra_code_tools or []),  # e.g. the sandboxed code_exec tool
        *calendar_tools(calendar),
        *mail_tools(mail),
    ]
    if retriever is not None:
        tools.append(RagSearchTool(retriever))
    for tool in tools:
        registry.register(tool)

    n = settings.max_agent_iterations
    agents: list[SubAgent] = [
        CodeAgent(providers["code"], registry, n),
        ResearchAgent(providers["research"], registry, n),
        OpsAgent(providers["ops"], registry, n),
    ]
    orchestrator = Orchestrator(
        providers["orchestrator"], agents, tracer, n, extra_tools=[RememberFactTool(memory)]
    )
    return System(
        settings=settings, orchestrator=orchestrator, agents=agents, registry=registry, tracer=tracer,
        memory=memory, sessions=SessionManager(memory, settings.memory_token_budget),
        mail=mail, calendar=calendar, providers=providers,
    )


async def build_retriever(
    kind: str, docs_dir: str | Path, *, db=None, settings: Settings | None = None
) -> HybridRetriever:
    """kind: "offline"  in-memory store + HashingEmbedder + SimpleReranker (deterministic, no downloads)
             "memory"   in-memory store + the real bge embedder + cross-encoder (no Postgres)
             "postgres" Postgres store (pgvector if available, else array store) + real models"""
    from app.rag.embedder import HashingEmbedder, SentenceTransformerEmbedder
    from app.rag.ingest import ingest_path
    from app.rag.reranker import CrossEncoderReranker, SimpleReranker
    from app.rag.store import InMemoryVectorStore, create_vector_store

    settings = settings or get_settings()
    if kind == "offline":
        embedder, reranker = HashingEmbedder(), SimpleReranker()
    else:
        embedder, reranker = SentenceTransformerEmbedder(settings.embedding_model), CrossEncoderReranker(settings.reranker_model)
    if kind == "postgres":
        if db is None:
            raise ValueError("kind='postgres' needs a Database")
        store = await create_vector_store(db.pool, embedder.dim, embedder.name)
    else:
        store = InMemoryVectorStore(embedder.dim, embedder.name)
        await store.init()
    if Path(docs_dir).exists():
        await ingest_path(docs_dir, embedder, store)
    return HybridRetriever(store, embedder, reranker)
