from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from app.agents.base import SubAgent
from app.harness.loop import AgentResult, run_agent_loop
from app.observability.trace import Tracer
from app.orchestrator.routing import ROUTER_NAME, RouteTool
from app.providers.base import ModelProvider
from app.tools.registry import ToolRegistry

DEFAULT_SYSTEM_PROMPT = """You are Jarvis, the orchestrator of a team of specialist agents for engineers.
You do not do specialist work yourself. You decide which agent should, by calling its route_to_<agent> tool.

Available agents:
{agents}

Rules:
- Call exactly the agent(s) the request needs. Write a complete, self-contained `task` for each call (the agent sees nothing else) and a one-sentence `reason`.
- If a task needs several agents in order (e.g. research first, then code), call them one after another and pass the earlier agent's findings into the next task.
- Independent sub-tasks may be delegated in the same turn.
- If the request is plain conversation that needs no agent, answer directly.
- When agents have answered, give the user one clear final answer. Never claim an action was performed if an agent reported it is pending user confirmation - say it is awaiting approval.
"""


@dataclass
class OrchestratorResult:
    text: str | None
    turn_id: str
    route_decisions: list[dict] = field(default_factory=list)
    agent_results: list[AgentResult] = field(default_factory=list)
    pending_confirmation_ids: list[str] = field(default_factory=list)
    truncated: bool = False
    loop: AgentResult | None = None

    @property
    def routed_agents(self) -> list[str]:
        return [d["agent"] for d in self.route_decisions]


class Orchestrator:
    def __init__(
        self,
        provider: ModelProvider,
        agents: list[SubAgent],
        tracer: Tracer,
        max_iterations: int = 8,
        system_prompt: str | None = None,
    ):
        self.provider = provider
        self.agents = agents
        self.tracer = tracer
        self.max_iterations = max_iterations
        self._system_prompt = system_prompt

    def system_prompt(self) -> str:
        if self._system_prompt:
            return self._system_prompt
        listing = "\n".join(f"- {a.name}: {a.description}" for a in self.agents)
        return DEFAULT_SYSTEM_PROMPT.format(agents=listing)

    async def handle(
        self,
        user_message: str,
        session_id: str,
        history: list[dict] | None = None,
        turn_id: str | None = None,
    ) -> OrchestratorResult:
        turn_id = turn_id or uuid.uuid4().hex[:12]
        # Route tools are built per request: they close over this turn's tracer/session so every
        # sub-agent event lands in the same trace, and live in a private registry so the
        # orchestrator's allow-list contains routing tools and nothing else.
        registry = ToolRegistry()
        route_tools = [RouteTool(a, self.tracer, session_id, turn_id) for a in self.agents]
        for t in route_tools:
            registry.register(t)
        registry.grant(ROUTER_NAME, [t.name for t in route_tools])

        messages = [
            {"role": "system", "content": self.system_prompt()},
            *(history or []),
            {"role": "user", "content": user_message},
        ]
        loop_result = await run_agent_loop(
            self.provider, messages, registry.tools_for(ROUTER_NAME), registry, self.tracer,
            session_id, ROUTER_NAME, self.max_iterations, turn_id=turn_id,
        )
        decisions = [d for t in route_tools for d in t.decisions]
        agent_results = [r for t in route_tools for r in t.results]
        return OrchestratorResult(
            text=loop_result.text,
            turn_id=turn_id,
            route_decisions=decisions,
            agent_results=agent_results,
            pending_confirmation_ids=[i for r in agent_results for i in r.pending_confirmation_ids],
            truncated=loop_result.truncated,
            loop=loop_result,
        )
