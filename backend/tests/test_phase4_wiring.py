"""Hand-built multi-agent wiring for the Phase 4 acceptance tests (the app's real bootstrap is
separate). Real Orchestrator, real ToolRegistry, real tools, real retrieval over docs/ with the
offline HashingEmbedder; only the language models are scripted MockProviders."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.agents.code_agent import CodeAgent
from app.agents.ops_agent import OpsAgent
from app.agents.research_agent import ResearchAgent
from app.observability.trace import InMemoryTracer
from app.orchestrator.orchestrator import Orchestrator
from app.providers.mock_provider import MockProvider, Rule, text_response, tool_call_response
from app.rag.embedder import HashingEmbedder
from app.rag.ingest import ingest_path
from app.rag.reranker import SimpleReranker
from app.rag.retriever import HybridRetriever
from app.rag.store import InMemoryVectorStore
from app.tools.calendar import FakeCalendarBackend, calendar_tools
from app.tools.files import ReadFileTool
from app.tools.gmail import FakeMailBackend, mail_tools
from app.tools.rag_tool import RagSearchTool
from app.tools.registry import ToolRegistry

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"


@dataclass
class World:
    registry: ToolRegistry
    orchestrator: Orchestrator
    tracer: InMemoryTracer
    calendar: FakeCalendarBackend
    mail: FakeMailBackend
    providers: dict[str, MockProvider] = field(default_factory=dict)


def tool_payload(message: dict) -> dict:
    """Decode a `role: tool` message the harness fed back to a model."""
    assert message["role"] == "tool", message
    return json.loads(message["content"])


def last_tool_message(messages: list[dict]) -> dict:
    return next(m for m in reversed(messages) if m["role"] == "tool")


async def build_world(
    workspace: Path,
    *,
    orchestrator: MockProvider,
    research: MockProvider | None = None,
    code: MockProvider | None = None,
    ops: MockProvider | None = None,
) -> World:
    embedder = HashingEmbedder()
    store = InMemoryVectorStore(embedder.dim, embedder.name)
    await ingest_path(DOCS_DIR, embedder, store)
    retriever = HybridRetriever(store, embedder, SimpleReranker())

    calendar_backend, mail_backend = FakeCalendarBackend(), FakeMailBackend()
    registry = ToolRegistry()
    for tool in [
        ReadFileTool(workspace),
        RagSearchTool(retriever),
        *calendar_tools(calendar_backend),
        *mail_tools(mail_backend),
    ]:
        registry.register(tool)

    providers = {
        "orchestrator": orchestrator,
        "research": research or MockProvider(),
        "code": code or MockProvider(),
        "ops": ops or MockProvider(),
    }
    tracer = InMemoryTracer()
    agents = [
        ResearchAgent(providers["research"], registry),
        CodeAgent(providers["code"], registry),
        OpsAgent(providers["ops"], registry),
    ]
    return World(registry, Orchestrator(orchestrator, agents, tracer), tracer, calendar_backend, mail_backend, providers)


def keyword_router(routes: dict[str, str]) -> MockProvider:
    """Orchestrator model stand-in: routes by keyword in the latest user message, then relays the
    sub-agent's answer. `routes` maps keyword -> agent name."""
    rules = [
        Rule(
            Rule.last_is_tool_result(),
            lambda msgs: text_response(f"FINAL: {tool_payload(last_tool_message(msgs))['result']['answer']}"),
        )
    ]
    for keyword, agent in routes.items():
        rules.append(
            Rule(
                Rule.last_user_contains(keyword),
                lambda msgs, agent=agent: tool_call_response(
                    f"route_to_{agent}",
                    {"task": next(m["content"] for m in reversed(msgs) if m["role"] == "user"),
                     "reason": f"request is about {agent}"},
                ),
            )
        )
    return MockProvider(rules=rules)
